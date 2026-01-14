using UnityEngine;
using System.Collections;
using System.Collections.Generic;
using Helpers;
using UnityEngine.Assertions;
using UnityEngine.Rendering;

public class RayTraceManager : MonoBehaviour
{
    [Header("Simulation")]
    public ComputeShader shader;
    [Range(10, 100000)] public int rayCount = 1000;
    [Range(1, 10)] public int maxBounces = 5;
    public float speedOfSound = 343f;
    public bool dynamicObstacles = false;

    [Header("Audio Settings")]
    public AudioClip inputClip;
    public AudioManager audioManager;
    // public ComputeShader convolutionShader;
    public float inputGain = 1.0f;
    public int sampleRate = 48000;
    public bool loop = false;
    public int chunkSize = 256;

    [Header("Accum Settings")]
    [Range(0.1f, 5.0f)] public float reverbDuration = 5f;

    [Header("Scene")]
    public Transform source, listener;
    [Range(0.1f, 5f)] public float listenerRadius = 0.5f;
    public List<GameObject> obstacleObjects;

    [Header("Debug")]
    public bool showDebugTexture = true;
    public Vector2 debugTextureSize = new Vector2(1024, 512);
    [Range(1, 500000)] public float debugGain = 1000.0f;
    [Range(5, 100)] public int debugRayCount = 100;
    public int currentAccumCount;
    public float lastProcessingLatency;

    ComputeBuffer wallBuffer, hitBuffer, debugBuffer, argsBuffer, spectrogramBuffer;
    ComputeBuffer spectrogramBufferPing, spectrogramBufferPong;
    ComputeBuffer fftBuffer, ifftBuffer;
    int activeSpectrogramIndex = 0;
    int numFrequencyBins;
    Vector2[] currentChunkFFT;

    Vector4[] debugRayPaths;
    RenderTexture spectrogramTexture; // For debugging
    List<Segment> activeSegments;
    int accumFrames = 0;
    int samplesSinceLastChunk = 0;
    int chunkSamples;
    int nextStreamingOffset = 0;
    float[] fullInputSamples;

    struct RayInfo { public float timeDelay, energy, frequency; public Vector2 hitPoint; };

    void Start()
    {
        Debug.Log($"[Start] Validating shader...");
        if (shader == null)
        {
            Debug.LogError("[Start] Shader is NULL!");
            return;
        }
        Debug.Log($"[Start] Shader: {shader.name}");
        
        Assert.IsTrue(sampleRate == inputClip.frequency, $"SampleRate ({sampleRate}) != input frequency ({inputClip.frequency})");
        Assert.IsNotNull(audioManager);
        numFrequencyBins = chunkSize / 2; // Nyquist limit
        Debug.Log($"[Start] numFrequencyBins = {numFrequencyBins}");
        UpdateGeometry();
    }

    void Update()
    {
        if (!source || !listener || !shader) return;
        RunSimulation();

        if (Input.GetKeyDown(KeyCode.Space) && audioManager)
        {
            StartStreaming();
        }
        if (Input.GetKeyDown(KeyCode.R)) { Reset(); }
        if (Input.GetKeyDown(KeyCode.Q)) { QueueDebugSineWave(); }
    }

    void FixedUpdate()
    {
        if (!audioManager || !shader) return;
        if (dynamicObstacles) UpdateGeometry();

        if (nextStreamingOffset >= 0)
        {
            currentAccumCount = accumFrames;
            
            // Convert fixed delta to samples (exact)
            int samplesThisFrame = Mathf.RoundToInt(Time.fixedDeltaTime * sampleRate);
            samplesSinceLastChunk += samplesThisFrame;
            Debug.Log($"[FixedUpdate] samplesThisFrame: {samplesThisFrame}, samplesSinceLastChunk: {samplesSinceLastChunk}, nextStreamingOffset: {nextStreamingOffset}");

            // Dispatch when we've accumulated exactly one chunk worth of samples
            if (samplesSinceLastChunk >= chunkSamples)
            {
                if (nextStreamingOffset >= inputClip.samples)
                {
                    if (loop)
                        nextStreamingOffset = 0;
                    else
                        nextStreamingOffset = -1; // Stop processing
                }
                else
                {
                    ComputeBuffer frozenBuffer = GetActiveSpectrogramBuffer();
                    
                    Debug.Log($"[FixedUpdate] Starting ProcessAndQueueChunk for offset {nextStreamingOffset}, chunkSamples {chunkSamples}, accumFrames {accumFrames}");
                    StartCoroutine(ProcessAndQueueChunk(nextStreamingOffset, chunkSamples, Mathf.Max(1, accumFrames), frozenBuffer));
                    
                    activeSpectrogramIndex = 1 - activeSpectrogramIndex;
                    nextStreamingOffset += chunkSamples;
                    Reset();
                    samplesSinceLastChunk -= chunkSamples;
                }
            }
        }
    }

    void StartStreaming()
    {
        Assert.IsNotNull(inputClip, "RayTraceManager: inputClip is null");
        
        nextStreamingOffset = 0;
        samplesSinceLastChunk = 0;

        chunkSamples = chunkSize; // Use chunkSize directly (samples per chunk)
        fullInputSamples = LoadMonoSamples(inputClip);
        Reset();
        audioManager.StartStreaming(sampleRate);
    }

    void Reset()
    {
        accumFrames = 0;
        int timeSteps = (int)(sampleRate * reverbDuration / chunkSize);
        int totalSize = timeSteps * numFrequencyBins;
        ComputeBuffer activeBuffer = GetActiveSpectrogramBuffer();

        Debug.Log($"[Reset] Finding ClearSpectrogram kernel...");
        int k = shader.FindKernel("ClearSpectrogram");
        Debug.Log($"[Reset] ClearSpectrogram kernel index: {k}");
        shader.SetInt("SpectrogramSize", totalSize);
        shader.SetBuffer(k, "Spectrogram", activeBuffer);
        ComputeHelper.Dispatch(shader, totalSize, 1, 1, k);
        Debug.Log($"[Reset] ClearSpectrogram dispatched successfully");
    }

    void RunSimulation()
    {
        int spectrogramLength = (int)(sampleRate * reverbDuration);

        // Lazy initialization / Hot-reload safety
        ComputeHelper.CreateStructuredBuffer<Vector4>(ref debugBuffer, debugRayCount * (maxBounces + 1));
        ComputeHelper.CreateAppendBuffer<RayInfo>(ref hitBuffer, rayCount * maxBounces);
        ComputeBuffer activeBuffer = GetActiveSpectrogramBuffer();

        if (wallBuffer == null || !wallBuffer.IsValid()) UpdateGeometry();
        if (argsBuffer == null || !argsBuffer.IsValid()) argsBuffer = new ComputeBuffer(1, sizeof(int) * 4, ComputeBufferType.IndirectArguments);
        if (spectrogramTexture == null)
        {
            spectrogramTexture = new RenderTexture(1024, 256, 0);
            spectrogramTexture.enableRandomWrite = true;
            spectrogramTexture.filterMode = FilterMode.Point;
            spectrogramTexture.Create();
        }

        // Initialize FFT data with default values if not yet computed
        if (currentChunkFFT == null || currentChunkFFT.Length != numFrequencyBins)
        {   
            Debug.Log($"[RunSimulation] Initializing currentChunkFFT with default values.");
            currentChunkFFT = new Vector2[numFrequencyBins];
            // Default: flat frequency response
            for (int i = 0; i < numFrequencyBins; i++)
                currentChunkFFT[i] = new Vector2(1.0f / numFrequencyBins, 0);
        }

        // Create or update FFT buffer for shader
        ComputeHelper.CreateStructuredBuffer(ref fftBuffer, currentChunkFFT);

        int k = shader.FindKernel("Trace");
        hitBuffer.SetCounterValue(0);
        shader.SetVector("sourcePos", source.position);
        shader.SetVector("listenerPos", listener.position);
        shader.SetFloat("listenerRadius", listenerRadius);
        shader.SetFloat("speedOfSound", speedOfSound);
        shader.SetFloat("inputGain", inputGain);
        shader.SetInt("maxBounceCount", maxBounces);
        shader.SetInt("rngStateOffset", Time.frameCount);
        shader.SetInt("numWalls", activeSegments.Count);
        shader.SetInt("rayCount", rayCount);
        shader.SetInt("debugRayCount", debugRayCount);
        shader.SetInt("accumFrames", accumFrames);
        shader.SetInt("NumFrequencyBins", numFrequencyBins);
        shader.SetInt("SampleRate", sampleRate);
        shader.SetBuffer(k, "walls", wallBuffer);
        shader.SetBuffer(k, "rayInfoBuffer", hitBuffer);
        shader.SetBuffer(k, "debugRays", debugBuffer);
        shader.SetBuffer(k, "FFTData", fftBuffer);
        ComputeHelper.Dispatch(shader, rayCount, 1, 1, k);

        AsyncGPUReadback.Request(debugBuffer, (request) => {
            if (request.hasError) return;
            debugRayPaths = request.GetData<Vector4>().ToArray();
        });

        ComputeBuffer.CopyCount(hitBuffer, argsBuffer, 0);
        AsyncGPUReadback.Request(argsBuffer, (request) => {
            if (request.hasError) return;
            int[] args = request.GetData<int>().ToArray();
            int hitCount = args[0];
            OnSimulationFinished(hitCount, spectrogramLength);
        });
    }

    ComputeBuffer GetActiveSpectrogramBuffer() {
        int timeSteps = (int)(sampleRate * reverbDuration / chunkSize);
        int totalSize = timeSteps * numFrequencyBins;
        ComputeHelper.CreateStructuredBuffer<float>(ref spectrogramBufferPing, totalSize);
        ComputeHelper.CreateStructuredBuffer<float>(ref spectrogramBufferPong, totalSize);
        return (activeSpectrogramIndex == 0) ? spectrogramBufferPing : spectrogramBufferPong;
    }

    void OnSimulationFinished(int hitCount, int spectrogramLength)
    {
        if (hitCount > 0)
        {
            int kp = shader.FindKernel("ProcessHits");
            shader.SetInt("SampleRate", sampleRate);
            shader.SetInt("ChunkSize", chunkSize);
            shader.SetInt("HitCount", hitCount);
            shader.SetInt("NumFrequencyBins", numFrequencyBins);
            shader.SetBuffer(kp, "RawHits", hitBuffer);
            shader.SetBuffer(kp, "Spectrogram", GetActiveSpectrogramBuffer());
            ComputeHelper.Dispatch(shader, hitCount, 1, 1, kp);
        }

        accumFrames++;

        int timeSteps = (int)(sampleRate * reverbDuration / chunkSize);
        shader.SetInt("accumCount", accumFrames);
        shader.SetInt("TimeSteps", timeSteps);
        shader.SetInt("NumFrequencyBins", numFrequencyBins);

        int kd = shader.FindKernel("DrawSpectrogram");
        shader.SetTexture(kd, "DebugTexture", spectrogramTexture);
        shader.SetBuffer(kd, "Spectrogram", GetActiveSpectrogramBuffer());
        shader.SetInt("TexWidth", spectrogramTexture.width);
        shader.SetInt("TexHeight", spectrogramTexture.height);
        shader.SetFloat("DebugGain", debugGain);
        ComputeHelper.Dispatch(shader, spectrogramTexture.width, spectrogramTexture.height, 1, kd);
    }

    void UpdateGeometry()
    {
        activeSegments = SceneToData2D.GetSegmentsFromColliders(obstacleObjects);
        ComputeHelper.CreateStructuredBuffer(ref wallBuffer, activeSegments);
    }

    float[] LoadMonoSamples(AudioClip clip)
    {
        float[] raw = new float[clip.samples * clip.channels];
        clip.GetData(raw, 0);
        if (clip.channels == 1) return raw;
        
        float[] mono = new float[clip.samples];
        for (int i = 0; i < clip.samples; i++)
        {
            float sum = 0;
            for (int c = 0; c < clip.channels; c++) sum += raw[i * clip.channels + c];
            mono[i] = sum / clip.channels;
        }
        return mono;
    }

    void QueueDebugSineWave()
    {
        if (audioManager == null)
        {
            Debug.LogWarning("[QueueDebugSineWave] AudioManager is null.");
            return;
        }

        // Generate 0.5 second sine wave at 440 Hz (A4 note)
        int duration = sampleRate / 2; // 0.5 seconds
        float frequency = 440f;
        float[] sineWave = new float[duration];
        
        for (int i = 0; i < duration; i++)
        {
            float t = i / (float)sampleRate;
            sineWave[i] = Mathf.Sin(2 * Mathf.PI * frequency * t) * 0.5f; // 50% volume
        }

        // Queue directly - AudioManager writes at current position
        audioManager.QueueAudioChunk(sineWave);
        Debug.Log($"[QueueDebugSineWave] Queued sine wave at writeHead {audioManager.WriteHead}");
    }

    IEnumerator ProcessAndQueueChunk(int sampleOffset, int chunkSamples, int accumCount, ComputeBuffer spectrogram)
    {
        float start = Time.realtimeSinceStartup;
        int inputLen = Mathf.Min(chunkSize, fullInputSamples.Length - sampleOffset);
        if (inputLen <= 0) yield break;

        // Extract chunk
        float[] chunk = new float[chunkSize];
        System.Array.Copy(fullInputSamples, sampleOffset, chunk, 0, Mathf.Min(inputLen, chunkSize));

        // Convert to complex numbers for FFT (real, imaginary)
        Vector2[] complexChunk = new Vector2[chunkSize];
        for (int i = 0; i < chunkSize; i++)
            complexChunk[i] = new Vector2(chunk[i], 0);

        // Create FFT buffer and compute FFT using Common.hlsl kernel
        var tempFFTBuffer = new ComputeBuffer(chunkSize, sizeof(float) * 2);
        tempFFTBuffer.SetData(complexChunk);
        
        Debug.Log($"[ProcessAndQueueChunk] Finding FFT kernel...");
        int kFFT = shader.FindKernel("FFT");
        Debug.Log($"[ProcessAndQueueChunk] FFT kernel index: {kFFT}");
        shader.SetBuffer(kFFT, "Data", tempFFTBuffer);
        shader.Dispatch(kFFT, 1, 1, 1);
        Debug.Log($"[ProcessAndQueueChunk] FFT dispatched");

        // Read back FFT results
        var fftReadback = AsyncGPUReadback.Request(tempFFTBuffer);
        while (!fftReadback.done) yield return null;
        
        if (!fftReadback.hasError)
        {
            Vector2[] fftData = fftReadback.GetData<Vector2>().ToArray();
            
            // Store first half (positive frequencies) for next frame's ray tracing
            currentChunkFFT = new Vector2[numFrequencyBins];
            for (int i = 0; i < numFrequencyBins; i++)
                currentChunkFFT[i] = fftData[i];
        }
        
        tempFFTBuffer.Release();

        // Reconstruct audio from spectrogram using IFFT on current time slice
        int timeSteps = (int)(sampleRate * reverbDuration / chunkSize);
        int currentTimeStep = sampleOffset / chunkSize;
        
        // Read back the spectrogram data
        var spectrogramReadback = AsyncGPUReadback.Request(spectrogram);
        while (!spectrogramReadback.done) yield return null;
        if (spectrogramReadback.hasError) yield break;

        float[] spectrogramData = spectrogramReadback.GetData<float>().ToArray();
        
        // Extract frequency bins for current time step only
        Vector2[] complexOutput = new Vector2[chunkSize];
        for (int i = 0; i < chunkSize; i++)
            complexOutput[i] = Vector2.zero;
            
        // Fill positive frequencies from spectrogram
        for (int f = 0; f < numFrequencyBins; f++)
        {
            int idx = currentTimeStep * numFrequencyBins + f;
            if (idx < spectrogramData.Length)
            {
                float magnitude = spectrogramData[idx] / Mathf.Max(1, accumCount);
                complexOutput[f] = new Vector2(magnitude, 0);
            }
        }
        
        // Mirror for negative frequencies (conjugate symmetry for real IFFT output)
        for (int f = 1; f < numFrequencyBins; f++)
        {
            int mirrorIdx = chunkSize - f;
            if (mirrorIdx < chunkSize)
                complexOutput[mirrorIdx] = new Vector2(complexOutput[f].x, -complexOutput[f].y);
        }

        // Perform IFFT to get time-domain audio for this chunk
        var tempIFFTBuffer = new ComputeBuffer(chunkSize, sizeof(float) * 2);
        tempIFFTBuffer.SetData(complexOutput);
        
        int kIFFT = shader.FindKernel("IFFT");
        shader.SetBuffer(kIFFT, "Data", tempIFFTBuffer);
        shader.Dispatch(kIFFT, 1, 1, 1);

        // Read back IFFT results
        var ifftReadback = AsyncGPUReadback.Request(tempIFFTBuffer);
        while (!ifftReadback.done) yield return null;
        
        float[] result = new float[chunkSize];
        if (!ifftReadback.hasError)
        {
            Vector2[] ifftData = ifftReadback.GetData<Vector2>().ToArray();
            for (int i = 0; i < chunkSize; i++)
                result[i] = ifftData[i].x; // Take real part
        }
        
        tempIFFTBuffer.Release();

        // Debug: stats
        float max = 0, sum = 0;
        for (int i = 0; i < result.Length; i++)
        {
            float abs = Mathf.Abs(result[i]);
            if (abs > max) max = abs;
            sum += abs;
        }
        float avg = result.Length > 0 ? sum / result.Length : 0;
        Debug.Log($"[IFFT Result] TimeStep: {currentTimeStep}/{timeSteps}, Max: {max:F6}, Avg: {avg:F6}, AccumFrames: {accumCount}");

        lastProcessingLatency = Time.realtimeSinceStartup - start;

        // Queue the processed chunk to AudioManager
        Debug.Log($"[ProcessAndQueueChunk] Queuing audio chunk at sampleOffset {sampleOffset}");
        audioManager.QueueAudioChunk(result, sampleOffset);
    }

    void OnGUI()
    {
        if (showDebugTexture && spectrogramTexture)
            GUI.DrawTexture(new Rect(10, 10, debugTextureSize.x, debugTextureSize.y), spectrogramTexture);
    }

    void OnDrawGizmos()
    {
        if (!source || !listener) return;
        Gizmos.color = Color.green; Gizmos.DrawWireSphere(source.position, 0.2f);
        Gizmos.color = Color.cyan; Gizmos.DrawWireSphere(listener.position, listenerRadius);
        if (activeSegments != null) { Gizmos.color = Color.red; foreach (var seg in activeSegments) Gizmos.DrawLine(seg.start, seg.end); }
        if (debugRayPaths != null)
        {
            int stride = maxBounces + 1; float z = source.position.z - 5.0f;
            if (debugRayPaths.Length < debugRayCount * stride) return;

            for (int i = 0; i < debugRayCount; i++)
            {
                for (int b = 0; b < maxBounces - 1; b++)
                {
                    Vector4 p1 = debugRayPaths[i * stride + b], p2 = debugRayPaths[i * stride + b + 1];
                    if (p2.sqrMagnitude == 0) break;
                    Gizmos.color = Color.Lerp(new Color(1, 0.5f, 0, 0.1f), new Color(1, 1, 0, 0.8f), p1.z);
                    Gizmos.DrawLine(new Vector3(p1.x, p1.y, z), new Vector3(p2.x, p2.y, z));
                }
            }
        }
    }

    void OnDestroy()
    {
        ComputeHelper.Release(wallBuffer, hitBuffer, debugBuffer, spectrogramBufferPing, spectrogramBufferPong, argsBuffer, fftBuffer, ifftBuffer); spectrogramTexture?.Release();
    }
}