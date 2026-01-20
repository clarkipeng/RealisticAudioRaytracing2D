using UnityEngine;
using System.Collections.Generic;
using Helpers;
using UnityEngine.Rendering;

public class RayTraceManager : MonoBehaviour
{
    [Header("Shaders")]
    public ComputeShader raytraceShader;

    [Header("Simulation")]
    [Range(10, 100000)] public int rayCount = 1000;
    [Range(1, 10)] public int maxBounces = 5;
    public float speedOfSound = 343f;
    public bool dynamicObstacles = false;

    [Header("Audio")]
    public AudioClip inputClip;
    public AudioManager audioManager;
    public int sampleRate = 48000;
    public int chunkSamples = 128;
    [Range(0.1f, 10f)] public float inputGain = 1.0f;
    [Range(0.1f, 5.0f)] public float reverbDuration = 5f;
    public bool loop = true;

    [Header("Scene")]
    public Transform source, listener;
    [Range(0.1f, 5f)] public float listenerRadius = 0.5f;
    public List<GameObject> obstacleObjects;

    [Header("Debug")]
    public bool showDebugTexture = true;
    [Range(5, 100)] public int debugRayCount = 100;
    [Range(1, 10000)] public float waveformGain = 1000.0f;

    ComputeBuffer wallBuffer, hitBuffer, debugBuffer, spectrogramBufferPing, spectrogramBufferPong, argsBuffer;
    RenderTexture spectrogramTexture;
    RenderTexture waveformTexture;
    List<Segment> activeSegments;
    Vector4[] debugRayPaths;
    float[] fullInputSamples;
    int activeSpectrogramIndex;
    int accumFrames;
    int chunkSamples;

    struct RayInfo { public float timeDelay, energy; public Vector2 hitPoint; }

    void Start()
    {
        UpdateGeometry();
        ResetSpectrogram();
        audioManager.StartStreaming(sampleRate);
    }

    void Update()
    {
        if (Input.GetKeyDown(KeyCode.R)) { ResetSpectrogram(); accumFrames = 0; }
        if (Input.GetKeyDown(KeyCode.Q)) QueueSineWave(440f, 1.0f);
        if (Input.GetKeyDown(KeyCode.Space)) StartStreaming();
    }

    void FixedUpdate()
    {
        if (!source || !listener || !raytraceShader) return;
        
        // Simulation should be run before each audio chunk
        // RunSimulation(); 
    }

    void QueueSineWave(float frequency, float duration)
    {
        int totalSamples = Mathf.CeilToInt(sampleRate * duration);
        float[] samples = new float[totalSamples];
        for (int i = 0; i < totalSamples; i++)
            samples[i] = Mathf.Sin(2 * Mathf.PI * frequency * i / sampleRate);
        
        audioManager.QueueAudioChunk(samples);
    }

    void StartStreaming()
    {
        fullInputSamples = LoadSample(inputClip);
        ResetSpectrogram();
        float delayBetweenChunks = (float)chunkSamples / sampleRate;
        StartCoroutine(StreamChunks(delayBetweenChunks));
    }

    // Breaks the full input samples into chunks and processes them and plays them over time
    IEnumerator StreamChunks(float delayBetweenChunks)
    {
        int totalSamples = fullInputSamples.Length;
        int offset = 0;

        while (offset < totalSamples)
        {
            int samplesToProcess = Mathf.Min(chunkSamples, totalSamples - offset);
            float[] chunk = new float[samplesToProcess];
            System.Array.Copy(fullInputSamples, offset, chunk, 0, samplesToProcess);
            offset += samplesToProcess;

            // Simulation should be ran here
            // First the simulation takes the FFT of the chunk, then runs the raytracing
            // based on the distribution of frequencies in the chunk
            RunSimulation(chunk);
        }   
    }

    float[] LoadSample(AudioClip clip)
    {
        float[] raw = new float[clip.samples * clip.channels];
        clip.GetData(raw, 0);
        
        // Convert to mono
        float[] mono = new float[clip.samples];
        for (int i = 0; i < clip.samples; i++)
        {
            float sum = 0;
            for (int c = 0; c < clip.channels; c++) sum += raw[i * clip.channels + c];
            mono[i] = sum / clip.channels;
        }
        
        // Resample if needed
        if (clip.frequency == sampleRate) return mono;
        
        float ratio = (float)clip.frequency / sampleRate;
        int newLength = Mathf.RoundToInt(clip.samples / ratio);
        float[] resampled = new float[newLength];
        
        for (int i = 0; i < newLength; i++)
        {
            float srcIdx = i * ratio;
            int idx0 = Mathf.FloorToInt(srcIdx);
            int idx1 = Mathf.Min(idx0 + 1, mono.Length - 1);
            float t = srcIdx - idx0;
            resampled[i] = Mathf.Lerp(mono[idx0], mono[idx1], t);
        }
        
        Debug.Log($"Resampled audio from {clip.frequency}Hz to {sampleRate}Hz ({clip.samples} -> {newLength} samples)");
        return resampled;
    }

    void ResetSpectrogram()
    {
        int len = Mathf.CeilToInt(sampleRate * reverbDuration);
        activeSpectrogramIndex = 0;
        ComputeHelper.CreateStructuredBuffer<float>(ref spectrogramBufferPing, len);
        ComputeHelper.CreateStructuredBuffer<float>(ref spectrogramBufferPong, len);
    }

    void RunSimulation(float[] inputSamples)
    {

        // Compute FFT of input samples
        int k_fft = raytraceShader.FindKernel("FFT");
        ComputeBuffer inputBuffer = ComputeHelper.CreateStructuredBuffer(inputSamples);
        ComputeBuffer fftBuffer = ComputeHelper.CreateStructuredBuffer<float>(inputSamples.Length);


        int spectrogramSize = (int)(sampleRate * reverbDuration);

        ComputeHelper.CreateStructuredBuffer<Vector4>(ref debugBuffer, debugRayCount * (maxBounces + 1));
        ComputeHelper.CreateAppendBuffer<RayInfo>(ref hitBuffer, rayCount * maxBounces);
        
        if (wallBuffer == null || !wallBuffer.IsValid()) UpdateGeometry();
        if (argsBuffer == null || !argsBuffer.IsValid()) argsBuffer = new ComputeBuffer(1, sizeof(int) * 4, ComputeBufferType.IndirectArguments);
        if (spectrogramTexture == null) {
            spectrogramTexture = new RenderTexture(1024, 256, 0) {
                enableRandomWrite = true, filterMode = FilterMode.Point 
            };
            spectrogramTexture.Create();
        }

        int k = raytraceShader.FindKernel("Trace");
        hitBuffer.SetCounterValue(0);

        raytraceShader.SetVector("sourcePos", source.position);
        raytraceShader.SetVector("listenerPos", listener.position);

        raytraceShader.SetFloat("listenerRadius", listenerRadius);
        raytraceShader.SetFloat("speedOfSound", speedOfSound);
        raytraceShader.SetFloat("inputGain", inputGain);

        raytraceShader.SetInt("maxBounceCount", maxBounces);
        raytraceShader.SetInt("rngStateOffset", Time.frameCount);
        raytraceShader.SetInt("numWalls", activeSegments.Count);
        raytraceShader.SetInt("rayCount", rayCount);
        raytraceShader.SetInt("debugRayCount", debugRayCount);
        raytraceShader.SetInt("accumFrames", accumFrames);

        raytraceShader.SetBuffer(k, "walls", wallBuffer);
        raytraceShader.SetBuffer(k, "rayInfoBuffer", hitBuffer);
        raytraceShader.SetBuffer(k, "debugRays", debugBuffer);

        ComputeHelper.Dispatch(raytraceShader, rayCount, 1, 1, k);

        AsyncGPUReadback.Request(debugBuffer, r => { if (!r.hasError) debugRayPaths = r.GetData<Vector4>().ToArray(); });
        ComputeBuffer.CopyCount(hitBuffer, argsBuffer, 0);
        AsyncGPUReadback.Request(argsBuffer, r => { if (!r.hasError) OnSimulationFinished(r.GetData<int>().ToArray()[0], spectrogramSize); });
    }

    void OnSimulationFinished(int hitCount, int spectrogramSize)
    {
        if (hitBuffer == null || spectrogramTexture == null) return;
        
        // Process Hits
        if (hitCount > 0)
        {
            int kp = raytraceShader.FindKernel("ProcessHits");
            raytraceShader.SetInt("SampleRate", sampleRate);
            raytraceShader.SetInt("HitCount", hitCount);
            raytraceShader.SetBuffer(kp, "RawHits", hitBuffer);
            raytraceShader.SetBuffer(kp, "Spectrogram", GetActiveSpectrogramBuffer());
            ComputeHelper.Dispatch(raytraceShader, hitCount, 1, 1, kp);
        }
        accumFrames++;


        // TODO: Draw Spectrogram
    }

    ComputeBuffer GetActiveSpectrogramBuffer()
    {
        return activeSpectrogramIndex == 0 ? spectrogramBufferPing : spectrogramBufferPong;
    }

    void UpdateGeometry()
    {
        activeSegments = SceneToData2D.GetSegmentsFromColliders(obstacleObjects);
        ComputeHelper.CreateStructuredBuffer(ref wallBuffer, activeSegments);
    }

    void OnGUI()
    {
        if (showDebugTexture)
        {
            if (spectrogramTexture != null)
            {
                float w = Screen.width * 0.4f, h = Screen.height * 0.15f;
                GUI.DrawTexture(new Rect(10, 10, w, h), spectrogramTexture);
            }

            if (waveformTexture != null)
            {
                float w = Screen.width * 0.4f, h = Screen.height * 0.15f;
                GUI.DrawTexture(new Rect(10, 20 + Screen.height * 0.15f, w, h), waveformTexture);
            }
        }
    }

    void OnDrawGizmos()
    {
        if (!source || !listener) return;
        Gizmos.color = Color.green; Gizmos.DrawWireSphere(source.position, 0.2f);
        Gizmos.color = Color.cyan; Gizmos.DrawWireSphere(listener.position, listenerRadius);
        if (activeSegments != null) { Gizmos.color = Color.red; foreach (var seg in activeSegments) Gizmos.DrawLine(seg.start, seg.end); }
        if (debugRayPaths != null && debugRayPaths.Length >= debugRayCount * (maxBounces + 1))
        {
            int stride = maxBounces + 1;
            for (int i = 0; i < debugRayCount; i++)
                for (int b = 0; b < maxBounces - 1; b++)
                {
                    Vector4 p1 = debugRayPaths[i * stride + b], p2 = debugRayPaths[i * stride + b + 1];
                    if (p2.sqrMagnitude == 0) break;
                    Gizmos.color = Color.Lerp(new Color(1, 0.5f, 0, 0.1f), new Color(1, 1, 0, 0.8f), p1.z);
                    Gizmos.DrawLine(new Vector3(p1.x, p1.y, source.position.z - 5), new Vector3(p2.x, p2.y, source.position.z - 5));
                }
        }
    }

    void OnDestroy() => ComputeHelper.Release(wallBuffer, hitBuffer, debugBuffer, spectrogramBufferPing, spectrogramBufferPong, argsBuffer);
}