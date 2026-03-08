using UnityEngine;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using Helpers;
using UnityEngine.Rendering;

public class RayTraceManager : MonoBehaviour
{
    [Header("Shaders")]
    public ComputeShader raytraceShader;

    [Header("Simulation")]
    [Range(10, 100000)] public int rayCount = 1000;
    [Range(0, 10)] public int maxBounces = 5;
    public float speedOfSound = 343f;
    public bool dynamicObstacles = false;

    [Header("Audio")]
    public AudioClip inputClip;
    public AudioManager audioManager;
    public int sampleRate = 48000;
    public int chunkSamples = 1024;
    [Range(0.1f, 1000000f)] public float inputGain = 1f;

    public bool loopAudio = false;

    [Header("UI & Visualization")]
    public float reverbDuration = 5f;
    public bool loop = false;

    [Header("Scene")]
    public Transform source, listener;
    [Range(0.1f, 5f)] public float listenerRadius = 0.5f;
    public List<GameObject> obstacleObjects;

    [Header("Debug")]
    public bool showDebugTexture = true;
    [Range(5, 100)] public int debugRayCount = 100;
    [Range(1, 1000000)] public float waveformGain = 1000.0f;
    [Range(1, 1000000)] public float spectrogramGain = 1000.0f;

    ComputeBuffer wallBuffer,
        hitBuffer,
        debugBuffer,
        spectrogramBufferPing,
        spectrogramBufferPong,
        waveformOutBuffer,
        argsBuffer,
        frequencyDistributionBuffer,
        frequencyBinCountsBuffer,
        inputBuffer;

    double lastTime = 0.0;

    struct FFTChunk {
        public int[] frequencyDistribution;
        public Vector2[] spectrum;
        public float[] binCounts; // per-bin ray count for correct normalization
    }
    List<FFTChunk> precomputedChunks = new List<FFTChunk>();
    int currentChunkIndex = 0;
    bool isStreaming = false;
    float delayBetweenChunks;
    float nextChunkTime = 0f;
    int startingPoint = -1;

    RenderTexture spectrogramTexture;
    public RenderTexture waveformTexture;

    List<Segment> activeSegments;
    Vector4[] debugRayPaths;
    float[] fullInputSamples;
    int activeSpectrogramIndex;
    int accumFrames;
    int spectrogramSize;

    float[] outputWaveform;

    struct RayInfo { public float timeDelay, energy; public int frequencyBin; public float roomFreqHz; public Vector2 hitPoint; }
    
    Vector3 lastSourcePos;
    Vector3 lastListenerPos;
    Vector3 sourceVel;
    Vector3 listenerVel;

    void Start()
    {
        spectrogramSize = Mathf.CeilToInt(sampleRate * reverbDuration);
        UpdateGeometry();
        ResetSpectrogram();

        if (spectrogramTexture == null) {
            spectrogramTexture = new RenderTexture(1024, 256, 0) {
                enableRandomWrite = true, filterMode = FilterMode.Point 
            };
            spectrogramTexture.Create();
        }

        if (waveformTexture == null) {
            waveformTexture = new RenderTexture(1024, 256, 0) {
                enableRandomWrite = true, filterMode = FilterMode.Point 
            };
            waveformTexture.Create();
        }

        audioManager.StartStreaming(sampleRate);
    }

    void Update()
    {
        float dt = Time.deltaTime > 0 ? Time.deltaTime : 0.016f;
        sourceVel = (source.position - lastSourcePos) / dt;
        listenerVel = (listener.position - lastListenerPos) / dt;
        lastSourcePos = source.position;
        lastListenerPos = listener.position;

        if (Input.GetKeyDown(KeyCode.R)) { Debug.Log("R pressed - Resetting"); ResetSpectrogram(); accumFrames = 0; }
        if (Input.GetKeyDown(KeyCode.Q)) { Debug.Log("Q pressed - Queueing sine"); QueueSineWave(440f, 1.0f); }
        if (Input.GetKeyDown(KeyCode.Space)) {
            Debug.Log("Space pressed - Starting stream");
            StartStreaming();
        }

        // Debug: Check if ANY key is pressed
        if (Input.anyKeyDown) Debug.Log($"A key was pressed this frame");

        if (spectrogramTexture == null || raytraceShader == null) {
            Debug.LogWarning("Spectrogram texture or raytrace shader not assigned.");
            return;
        }

        // Debug.Log($"Update: isStreaming={isStreaming}, currentChunkIndex={currentChunkIndex}, precomputedChunks={precomputedChunks.Count}, nextChunkTime={nextChunkTime:F2}");
        if (isStreaming && currentChunkIndex < precomputedChunks.Count)
        {
            if (Time.realtimeSinceStartup >= nextChunkTime)
            {
                float iterationStartTime = Time.realtimeSinceStartup;
                FFTChunk chunk = precomputedChunks[currentChunkIndex];
                
                RunRaytracing(chunk);
                
                RunSimulation();

                currentChunkIndex++;

                float processingTime = Time.realtimeSinceStartup - iterationStartTime;
                float remainingWait = delayBetweenChunks - processingTime;
                
                if (remainingWait > 0) {
                    nextChunkTime = Time.realtimeSinceStartup + remainingWait;
                } else {
                    nextChunkTime = Time.realtimeSinceStartup; // Catch up
                }
            }
        }
        else if (isStreaming && currentChunkIndex >= precomputedChunks.Count)
        {
            if (loopAudio)
            {
                currentChunkIndex = 0;
                startingPoint = -1;
            }
            else
            {
                isStreaming = false;
                Debug.Log("Finished streaming and processing chunks!");
            }
        }

        double timeNow = Time.realtimeSinceStartup;
        float deltaTime = (float)(timeNow - lastTime);
        // Debug.Log($"Time delta: {deltaTime:F4}s");
        lastTime = timeNow;
    }

    void QueueSineWave(float frequency, float duration)
    {
        int totalSamples = Mathf.CeilToInt(sampleRate * duration);
        float[] samples = new float[totalSamples];
        for (int i = 0; i < totalSamples; i++)
            samples[i] = Mathf.Sin(2 * Mathf.PI * frequency * i / sampleRate);

        audioManager.QueueAudioChunk(samples);
    }

    void ResetSpectrogram()
    {
        int len = Mathf.CeilToInt(sampleRate * reverbDuration);
        activeSpectrogramIndex = 0;
        ComputeHelper.CreateStructuredBuffer<Vector2>(ref spectrogramBufferPing, len);
        ComputeHelper.CreateStructuredBuffer<Vector2>(ref spectrogramBufferPong, len);
    }

    ComputeBuffer GetActiveSpectrogramBuffer()
    {
        return activeSpectrogramIndex == 0 ? spectrogramBufferPing : spectrogramBufferPong;
    }

    void StartStreaming()
    {
        Debug.Log("Starting audio streaming and processing.");
        fullInputSamples = LoadSample(inputClip);
        ResetSpectrogram();
        
        delayBetweenChunks = (float)chunkSamples / sampleRate;
        PrecomputeAllFFTs();
        
        currentChunkIndex = 0;
        startingPoint = -1;
        isStreaming = true;
        nextChunkTime = Time.realtimeSinceStartup;
    }

    void PrecomputeAllFFTs()
    {
        precomputedChunks.Clear();
        int totalSamples = fullInputSamples.Length;
        int offset = 0;

        int k_fft = raytraceShader.FindKernel("FFT");

        while (offset < totalSamples)
        {
            int samplesToProcess = Mathf.Min(chunkSamples, totalSamples - offset);
            float[] chunk = new float[chunkSamples];
            System.Array.Copy(fullInputSamples, offset, chunk, 0, samplesToProcess);

            Vector2[] complexSamples = new Vector2[chunkSamples];
            for (int i = 0; i < chunkSamples; i++)
            {
                complexSamples[i] = new Vector2(chunk[i], 0f);
            }

            ComputeHelper.CreateStructuredBuffer(ref inputBuffer, complexSamples);

            raytraceShader.SetBuffer(k_fft, "Data", inputBuffer);
            ComputeHelper.Dispatch(raytraceShader, 1, 1, 1, k_fft);

            Vector2[] fftResult = new Vector2[chunkSamples];
            inputBuffer.GetData(fftResult);
            
            var (freqDist, binCounts) = BuildFrequencyDistribution(fftResult);
            
            precomputedChunks.Add(new FFTChunk {
                frequencyDistribution = freqDist,
                spectrum = fftResult,
                binCounts = binCounts
            });

            offset += samplesToProcess;
            ComputeHelper.Release(inputBuffer);
        }
        Debug.Log($"Precomputed {precomputedChunks.Count} FFT chunks.");
    }

    void TestSpectrogramBuffer()
    {
        int kp = raytraceShader.FindKernel("TestSetSpectrogramAtCertainFreq");
        raytraceShader.SetInt("SpectrogramSize", spectrogramSize);
        raytraceShader.SetInt("ChunkSamples", chunkSamples);
        raytraceShader.SetInt("TestSetSpectrogramFreqBin", 16);
        raytraceShader.SetBuffer(kp, "Spectrogram", GetActiveSpectrogramBuffer());

        ComputeHelper.Dispatch(raytraceShader, spectrogramSize / chunkSamples, chunkSamples, 1, kp);
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

        // Debug.Log($"Resampled audio from {clip.frequency}Hz to {sampleRate}Hz ({clip.samples} -> {newLength} samples)");
        return resampled;
    }

    ComputeBuffer initialSpectrumBuffer;

    void RunRaytracing(FFTChunk chunk)
    {
        ComputeHelper.CreateStructuredBuffer(ref frequencyDistributionBuffer, chunk.frequencyDistribution);
        ComputeHelper.CreateStructuredBuffer(ref initialSpectrumBuffer, chunk.spectrum);
        ComputeHelper.CreateStructuredBuffer(ref frequencyBinCountsBuffer, chunk.binCounts);

        ComputeHelper.CreateStructuredBuffer<Vector4>(ref debugBuffer, debugRayCount * (maxBounces + 1));
        ComputeHelper.CreateAppendBuffer<RayInfo>(ref hitBuffer, rayCount * maxBounces);

        if (wallBuffer == null || !wallBuffer.IsValid()) UpdateGeometry();
        if (argsBuffer == null || !argsBuffer.IsValid()) argsBuffer = new ComputeBuffer(1, sizeof(int) * 4, ComputeBufferType.IndirectArguments);

        int k = raytraceShader.FindKernel("Trace");
        hitBuffer.SetCounterValue(0);

        raytraceShader.SetVector("sourcePos", source.position);
        raytraceShader.SetVector("listenerPos", listener.position);
        raytraceShader.SetVector("sourceVel", sourceVel);
        raytraceShader.SetVector("listenerVel", listenerVel);

        raytraceShader.SetFloat("listenerRadius", listenerRadius);
        raytraceShader.SetFloat("speedOfSound", speedOfSound);
        raytraceShader.SetFloat("inputGain", inputGain);
        raytraceShader.SetFloat("diffractionFactor", 0.5f);

        raytraceShader.SetInt("maxBounceCount", maxBounces);
        raytraceShader.SetInt("rngStateOffset", Time.frameCount);
        raytraceShader.SetInt("numWalls", activeSegments.Count);
        raytraceShader.SetInt("rayCount", rayCount);
        raytraceShader.SetInt("debugRayCount", debugRayCount);
        raytraceShader.SetInt("accumFrames", accumFrames);
        raytraceShader.SetInt("frequencyDistributionSize", chunk.frequencyDistribution.Length);

        raytraceShader.SetBuffer(k, "walls", wallBuffer);
        raytraceShader.SetBuffer(k, "rayInfoBuffer", hitBuffer);
        raytraceShader.SetBuffer(k, "debugRays", debugBuffer);
        raytraceShader.SetBuffer(k, "FrequencyDistribution", frequencyDistributionBuffer);

        ComputeHelper.Dispatch(raytraceShader, rayCount, 1, 1, k);

        AsyncGPUReadback.Request(debugBuffer, r => { if (!r.hasError) debugRayPaths = r.GetData<Vector4>().ToArray(); });
        ComputeBuffer.CopyCount(hitBuffer, argsBuffer, 0);

        int[] argsData = new int[4];  // Match the actual buffer size
        argsBuffer.GetData(argsData);
        int hitCount = argsData[0];  // Only the first int is the counter

        // Ray hit processing
        if (hitBuffer == null || spectrogramTexture == null) return;

        // Process Hits
        if (hitCount > 0)
        {

            int safeHitCount = Mathf.Min(hitCount, hitBuffer.count);
            RayInfo[] hitData = new RayInfo[safeHitCount];
            hitBuffer.GetData(hitData, 0, 0, safeHitCount);
            // int[] test2 = new int[safeHitCount];
            // for (int i = 0; i < safeHitCount; i++)
            // {
            //     test2[i] = hitData[i].frequencyBin;
            // }
            // // Count occurrences of each unique integer
            // Dictionary<int, int> counts = new Dictionary<int, int>();
            // foreach (int val in test2)
            // {
            //     if (counts.ContainsKey(val))
            //         counts[val]++;
            //     else
            //         counts[val] = 1;
            // }

            // string countsStr2 = string.Join(", ", counts.OrderBy(kv => kv.Key).Select(kv => $"{kv.Key}:{kv.Value}"));
            // Debug.Log($"Frequency Distribution Counts After Hits: {countsStr2}");


            // Debug.Log($"Processing {safeHitCount} ray hits into spectrogram.");
            int kp = raytraceShader.FindKernel("ProcessHits");
            raytraceShader.SetInt("SampleRate", sampleRate);
            raytraceShader.SetInt("ChunkSamples", chunkSamples);
            raytraceShader.SetInt("HitCount", safeHitCount);
            raytraceShader.SetBuffer(kp, "RawHits", hitBuffer);
            raytraceShader.SetBuffer(kp, "Spectrogram", GetActiveSpectrogramBuffer());
            raytraceShader.SetBuffer(kp, "InitialSpectrum", initialSpectrumBuffer);
            raytraceShader.SetBuffer(kp, "FrequencyBinCounts", frequencyBinCountsBuffer);
            ComputeHelper.Dispatch(raytraceShader, hitCount, 1, 1, kp);
        }
        accumFrames++;
    }

    // Build a weighted frequency distribution from the FFT result (Bins)
    // Returns (distribution, perBinCounts) where perBinCounts[i] = number of rays assigned to bin i
    (int[], float[]) BuildFrequencyDistribution(Vector2[] fftResult)
    {
        // Only use first half of FFT (positive frequencies)
        int halfSize = fftResult.Length / 2;

        // Calculate magnitudes for each frequency bin
        float[] magnitudes = new float[halfSize];
        float totalMagnitude = 0f;

        for (int i = 0; i < halfSize; i++)
        {
            magnitudes[i] = Mathf.Sqrt(fftResult[i].x * fftResult[i].x + fftResult[i].y * fftResult[i].y);
            totalMagnitude += magnitudes[i];
        }

        // Normalize magnitudes
        if (totalMagnitude > 0)
        {
            for (int i = 0; i < halfSize; i++)
                magnitudes[i] /= totalMagnitude;
        }

        // Build weighted distribution buffer (rayCount entries)
        // Each entry is a frequency bin index, with frequencies appearing proportional to their magnitude
        // Use FloorToInt to avoid overflowing rayCount from rounding-up accumulation
        int[] distribution = new int[rayCount];
        int distributionIndex = 0;

        for (int i = 0; i < halfSize && distributionIndex < rayCount; i++)
        {
            int count = Mathf.FloorToInt(magnitudes[i] * rayCount);
            for (int j = 0; j < count && distributionIndex < rayCount; j++)
            {
                distribution[distributionIndex++] = i;
            }
        }

        // Fill remaining slots proportionally (from largest-magnitude bins)
        // instead of purely random, to keep the distribution deterministic
        int fillIdx = 0;
        while (distributionIndex < rayCount)
        {
            // Find the bin with largest fractional remainder
            float bestRemainder = -1f;
            int bestBin = 0;
            for (int i = 0; i < halfSize; i++)
            {
                float exact = magnitudes[i] * rayCount;
                float remainder = exact - Mathf.Floor(exact);
                if (remainder > bestRemainder)
                {
                    bestRemainder = remainder;
                    bestBin = i;
                }
            }
            distribution[distributionIndex++] = bestBin;
            // Zero out the remainder for this bin so we pick the next best
            magnitudes[bestBin] = Mathf.Floor(magnitudes[bestBin] * rayCount) / (float)rayCount;
            fillIdx++;
        }

        Shuffle(distribution);

        // Compute actual per-bin counts for correct normalization on the GPU
        // This prevents the "magnitude squaring" bug where the spectrum is applied twice
        // (once via weighted ray allocation, once via InitialSpectrum multiplication)
        float[] perBinCounts = new float[halfSize];
        for (int i = 0; i < rayCount; i++)
            perBinCounts[distribution[i]] += 1f;

        return (distribution, perBinCounts);
    }

    void RunSimulation()
    {
        // Read the accumulated spectrogram from the current active buffer
        ComputeBuffer currentBuffer = GetActiveSpectrogramBuffer();

        int kd = raytraceShader.FindKernel("DrawSpectrogram");
        raytraceShader.SetInt("SpectrogramSize", spectrogramSize);
        raytraceShader.SetInt("ChunkSamples", chunkSamples);
        raytraceShader.SetInt("TexWidth", spectrogramTexture.width);
        raytraceShader.SetInt("TexHeight", spectrogramTexture.height);
        raytraceShader.SetInt("accumCount", accumFrames);
        raytraceShader.SetFloat("DebugGain", spectrogramGain);
        raytraceShader.SetTexture(kd, "DebugTexture", spectrogramTexture);
        raytraceShader.SetBuffer(kd, "Spectrogram", currentBuffer);
        ComputeHelper.Dispatch(raytraceShader, spectrogramTexture.width, spectrogramTexture.height, 1, kd);

        if (chunkSamples != 1024)
        {
            Debug.LogWarning($"SpectrogramToWaveformIFFT kernel assumes chunkSamples==1024 (WINDOW_SIZE). Current chunkSamples={chunkSamples}. Skipping GPU IFFT.");
        }
        else
        {
            int timeSteps = spectrogramSize / chunkSamples;
            if (timeSteps <= 0)
            {
                Debug.LogWarning($"SpectrogramToWaveformIFFT: timeSteps was {timeSteps} (spectrogramSize={spectrogramSize}, chunkSamples={chunkSamples}). Skipping.");
                return;
            }

            // Allocate/reuse output buffer
            if (waveformOutBuffer == null || !waveformOutBuffer.IsValid() || waveformOutBuffer.count != spectrogramSize)
            {
                ComputeHelper.Release(waveformOutBuffer);
                waveformOutBuffer = new ComputeBuffer(spectrogramSize, sizeof(float));
            }

            if (outputWaveform == null || outputWaveform.Length != spectrogramSize)
                outputWaveform = new float[spectrogramSize];

            int k_stifft = raytraceShader.FindKernel("SpectrogramToWaveformIFFT");
            raytraceShader.SetInt("SpectrogramSize", spectrogramSize);
            raytraceShader.SetInt("ChunkSamples", chunkSamples);
            raytraceShader.SetBuffer(k_stifft, "Spectrogram", currentBuffer);
            raytraceShader.SetBuffer(k_stifft, "WaveformOut", waveformOutBuffer);

            // IMPORTANT: this kernel's X dimension is thread-groups (one group per time frame).
            // ComputeHelper.Dispatch interprets the parameter as *threads/iterations* and would under-dispatch.
            raytraceShader.Dispatch(k_stifft, timeSteps, 1, 1);

            // Clear any tail samples (spectrogramSize may not be divisible by chunkSamples)
            int written = timeSteps * chunkSamples;
            int tail = spectrogramSize - written;
            if (tail > 0)
            {
                float[] zeros = new float[tail];
                waveformOutBuffer.SetData(zeros, 0, written, tail);
            }

            // Read back and queue
            waveformOutBuffer.GetData(outputWaveform);
            if (startingPoint == -1)
            {
                // First chunk: place at the audio manager's safe write position
                (int, int) result = audioManager.QueueAudioChunk(outputWaveform);
                startingPoint = result.Item1 + chunkSamples;
            }
            else {
                // Subsequent chunks: always advance by exactly chunkSamples
                // to maintain correct overlap-add alignment
                audioManager.QueueAudioChunk(outputWaveform, 0, startingPoint);
                startingPoint += chunkSamples;
            }
        }

        // Draw Waveform for Debugging
        if (waveformTexture != null)
        {
            ComputeBuffer waveformBuffer = ComputeHelper.CreateStructuredBuffer(outputWaveform);
            int kw = raytraceShader.FindKernel("DrawWaveform");
            raytraceShader.SetInt("WaveformLength", outputWaveform.Length);
            raytraceShader.SetInt("TexWidth", waveformTexture.width);
            raytraceShader.SetInt("TexHeight", waveformTexture.height);
            raytraceShader.SetFloat("DebugGain", waveformGain);
            raytraceShader.SetBuffer(kw, "WaveformData", waveformBuffer);
            raytraceShader.SetTexture(kw, "DebugTexture", waveformTexture);
            ComputeHelper.Dispatch(raytraceShader, waveformTexture.width, waveformTexture.height, 1, kw);
        }

        // Switch buffers for next chunk (ping-pong)
        activeSpectrogramIndex = 1 - activeSpectrogramIndex;

        // Clear the new active buffer for fresh accumulation
        int kClear = raytraceShader.FindKernel("ClearSpectrogram");

        raytraceShader.SetInt("SpectrogramSize", spectrogramSize);
        raytraceShader.SetBuffer(kClear, "Spectrogram", GetActiveSpectrogramBuffer());
        ComputeHelper.Dispatch(raytraceShader, spectrogramSize, 1, 1, kClear);
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
            // Create a style for labels with visible text
            GUIStyle labelStyle = new GUIStyle(GUI.skin.label);
            labelStyle.normal.textColor = Color.white;
            labelStyle.fontSize = 10;
            labelStyle.fontStyle = FontStyle.Bold;

            if (spectrogramTexture != null)
            {
                float w = Screen.width * 0.4f, h = Screen.height * 0.15f;
                GUI.DrawTexture(new Rect(10, 10, w, h), spectrogramTexture);

                Matrix4x4 matrixBackup = GUI.matrix;
                GUIUtility.RotateAroundPivot(90, new Vector2(10 + w + 5, 10));
                GUI.Label(new Rect(10 + w + 5, 10, h, 20), "Spectrogram", labelStyle);
                GUI.matrix = matrixBackup;
            }

            if (waveformTexture != null)
            {
                float w = Screen.width * 0.4f, h = Screen.height * 0.15f;
                float yPos = 20 + Screen.height * 0.15f;
                GUI.DrawTexture(new Rect(10, yPos, w, h), waveformTexture);

                Matrix4x4 matrixBackup = GUI.matrix;
                GUIUtility.RotateAroundPivot(90, new Vector2(10 + w + 5, yPos));
                GUI.Label(new Rect(10 + w + 5, yPos, h, 20), "Waveform", labelStyle);
                GUI.matrix = matrixBackup;
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

    void Shuffle<T>(T[] array)
    {
        for (int i = array.Length - 1; i > 0; i--)
        {
            int j = Random.Range(0, i + 1);
            T temp = array[i];
            array[i] = array[j];
            array[j] = temp;
        }
    }
    void OnDestroy() => ComputeHelper.Release(wallBuffer, hitBuffer, debugBuffer, spectrogramBufferPing, spectrogramBufferPong, waveformOutBuffer, argsBuffer, frequencyDistributionBuffer, frequencyBinCountsBuffer, inputBuffer, initialSpectrumBuffer);
}