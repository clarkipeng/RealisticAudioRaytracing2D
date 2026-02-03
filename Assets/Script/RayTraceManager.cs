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
    [Range(0.1f, 1000000f)] public float inputGain = 1.0f;
    [Range(0.1f, 5.0f)] public float reverbDuration = 5f;
    public bool loop = true;

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
        frequencyDistributionBuffer;
    
    RenderTexture spectrogramTexture;
    public RenderTexture waveformTexture;

    public RenderTexture FFTTexture;
    public RenderTexture SpectrogramFirstFreqBinTexture;
    List<Segment> activeSegments;
    Vector4[] debugRayPaths;
    float[] fullInputSamples;
    int activeSpectrogramIndex;
    int accumFrames;
    int spectrogramSize;

    int? audioWritePos = null;

    float[] outputWaveform;

    struct RayInfo { public float timeDelay, energy; public int frequencyBin; public Vector2 hitPoint; }

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

        if (FFTTexture == null) {
            FFTTexture = new RenderTexture(1024, 256, 0) {
                enableRandomWrite = true, filterMode = FilterMode.Point 
            };
            FFTTexture.Create();
        }

        if (SpectrogramFirstFreqBinTexture == null) {
            SpectrogramFirstFreqBinTexture = new RenderTexture(1024, 256, 0) {
                enableRandomWrite = true, filterMode = FilterMode.Point 
            };
            SpectrogramFirstFreqBinTexture.Create();
        }

        audioManager.StartStreaming(sampleRate);
    }

    void Update()
    {
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

        // ===== Kernel Call: Draw Spectrogram =====
        // int kd = raytraceShader.FindKernel("DrawSpectrogram");
        // raytraceShader.SetInt("SpectrogramSize", spectrogramSize);
        // raytraceShader.SetInt("ChunkSamples", chunkSamples);
        // raytraceShader.SetInt("TexWidth", spectrogramTexture.width);
        // raytraceShader.SetInt("TexHeight", spectrogramTexture.height);
        // raytraceShader.SetInt("accumCount", accumFrames);
        // raytraceShader.SetFloat("DebugGain", spectrogramGain);
        // raytraceShader.SetTexture(kd, "DebugTexture", spectrogramTexture);
        // raytraceShader.SetBuffer(kd, "Spectrogram", GetActiveSpectrogramBuffer());
        // ComputeHelper.Dispatch(raytraceShader, spectrogramTexture.width, spectrogramTexture.height, 1, kd);

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

    void ResetSpectrogram()
    {
        int len = Mathf.CeilToInt(sampleRate * reverbDuration);
        activeSpectrogramIndex = 0;
        ComputeHelper.CreateStructuredBuffer<float>(ref spectrogramBufferPing, len);
        ComputeHelper.CreateStructuredBuffer<float>(ref spectrogramBufferPong, len);
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
        float delayBetweenChunks = (float)chunkSamples / sampleRate;

        StartCoroutine(StreamChunks(delayBetweenChunks));
        
    }

    // Breaks the full input samples into chunks and processes them and plays them over time
    IEnumerator StreamChunks(float delayBetweenChunks)
    {
        int totalSamples = fullInputSamples.Length;
        int offset = 0;
        float lastIterationTime = Time.realtimeSinceStartup;

        while (offset < totalSamples)
        {
            float iterationStartTime = Time.realtimeSinceStartup;
            float timeSinceLastIteration = iterationStartTime - lastIterationTime;
            Debug.Log($"StreamChunks iteration - Time since last call: {timeSinceLastIteration:F4}s (expected: {delayBetweenChunks:F4}s)");
            lastIterationTime = iterationStartTime;

            int samplesToProcess = Mathf.Min(chunkSamples, totalSamples - offset);
            float[] chunk = new float[samplesToProcess];
            System.Array.Copy(fullInputSamples, offset, chunk, 0, samplesToProcess);
            offset += samplesToProcess;

            // Simulation should be ran here
            // First the simulation takes the FFT of the chunk, then runs the raytracing
            // based on the distribution of frequencies in the chunk
            RunSimulation(chunk);
            // TestSpectrogramBuffer(); // TEMP for testing

            // After simulation is done, we need to process the spectrogram
            // and queue it to the audio manager
            ProcessAndQueueSpectrogram(audioWritePos);
            
            // Calculate remaining time to wait, accounting for processing duration
            float processingTime = Time.realtimeSinceStartup - iterationStartTime;
            float remainingWait = delayBetweenChunks - processingTime;
            
            // yield return new WaitForSeconds(delayBetweenChunks);
            if (remainingWait > 0)
            {
                yield return new WaitForSeconds(remainingWait);
            }
            else
            {
                // Processing took longer than the chunk duration - we're falling behind
                Debug.LogWarning($"Processing took {processingTime:F4}s, exceeding chunk duration {delayBetweenChunks:F4}s by {-remainingWait:F4}s");
                yield return null; // Still yield to prevent freezing, but continue immediately
            }
        }   
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

    void RunSimulation(float[] inputSamples)
    {
        // Debug.Log($"Running simulation for chunk of {inputSamples.Length} samples.");
        // Compute FFT of input samples
        int k_fft = raytraceShader.FindKernel("FFT");
        
        // Convert float samples to complex numbers (Vector2)
        Vector2[] complexSamples = new Vector2[inputSamples.Length];
        for (int i = 0; i < inputSamples.Length; i++)
        {
            complexSamples[i] = new Vector2(inputSamples[i], 0f); // real part = sample, imaginary part = 0
        }
        
        ComputeBuffer inputBuffer = ComputeHelper.CreateStructuredBuffer(complexSamples);

        raytraceShader.SetBuffer(k_fft, "Data", inputBuffer);
        ComputeHelper.Dispatch(raytraceShader, inputSamples.Length / chunkSamples, 1, 1, k_fft);

        // Synchronous readback (blocks until GPU finishes)
        Vector2[] fftResult = new Vector2[inputSamples.Length];
        inputBuffer.GetData(fftResult);

        // Get FFT max magnitude for debugging
        float maxMagnitude = 0f;
        for (int i = 0; i < fftResult.Length / 2; i++)
        {
            float mag = Mathf.Sqrt(fftResult[i].x * fftResult[i].x + fftResult[i].y * fftResult[i].y);
            if (mag > maxMagnitude) maxMagnitude = mag;
        }

        // ===== Draw FFT for Debugging =====
        int kdfft = raytraceShader.FindKernel("DrawFFTDebug");
        raytraceShader.SetInt("FFTDataLength", inputSamples.Length);
        raytraceShader.SetFloat("FFTDebugGain", 1000f);
        raytraceShader.SetFloat("FFTMaxValue", maxMagnitude);

        raytraceShader.SetBuffer(kdfft, "FFTData", inputBuffer);
        raytraceShader.SetTexture(kdfft, "FFTDebugTexture", FFTTexture);
        ComputeHelper.Dispatch(raytraceShader, FFTTexture.width, FFTTexture.height, 1, kdfft);
        ComputeHelper.Release(inputBuffer);

        // Build weighted frequency distribution from FFT magnitudes
        int[] frequencyDistribution = BuildFrequencyDistribution(fftResult);

        ComputeHelper.CreateStructuredBuffer(ref frequencyDistributionBuffer, frequencyDistribution);

        int[] test = new int[frequencyDistribution.Length];
        frequencyDistributionBuffer.GetData(test);
        
        // Count occurrences of each unique integer
        Dictionary<int, int> counts_old = new Dictionary<int, int>();
        foreach (int val in test)
        {
            if (counts_old.ContainsKey(val))
                counts_old[val]++;
            else
                counts_old[val] = 1;
        }
        
        string countsStr = string.Join(", ", counts_old.OrderBy(kv => kv.Key).Select(kv => $"{kv.Key}:{kv.Value}"));
        Debug.Log($"Frequency Distribution Counts before hits: {countsStr}");

        ComputeHelper.CreateStructuredBuffer<Vector4>(ref debugBuffer, debugRayCount * (maxBounces + 1));
        ComputeHelper.CreateAppendBuffer<RayInfo>(ref hitBuffer, rayCount * maxBounces);
        
        if (wallBuffer == null || !wallBuffer.IsValid()) UpdateGeometry();
        if (argsBuffer == null || !argsBuffer.IsValid()) argsBuffer = new ComputeBuffer(1, sizeof(int) * 4, ComputeBufferType.IndirectArguments);

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
        raytraceShader.SetInt("frequencyDistributionSize", frequencyDistribution.Length);

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
            int[] test2 = new int[safeHitCount];
            for (int i = 0; i < safeHitCount; i++)
            {
                test2[i] = hitData[i].frequencyBin;
            }
            // Count occurrences of each unique integer
            Dictionary<int, int> counts = new Dictionary<int, int>();
            foreach (int val in test2)
            {
                if (counts.ContainsKey(val))
                    counts[val]++;
                else
                    counts[val] = 1;
            }
            
            string countsStr2 = string.Join(", ", counts.OrderBy(kv => kv.Key).Select(kv => $"{kv.Key}:{kv.Value}"));
            Debug.Log($"Frequency Distribution Counts After Hits: {countsStr2}");


            Debug.Log($"Processing {safeHitCount} ray hits into spectrogram.");
            int kp = raytraceShader.FindKernel("ProcessHits");
            raytraceShader.SetInt("SampleRate", sampleRate);
            raytraceShader.SetInt("ChunkSamples", chunkSamples);
            raytraceShader.SetInt("HitCount", safeHitCount);
            raytraceShader.SetBuffer(kp, "RawHits", hitBuffer);
            raytraceShader.SetBuffer(kp, "Spectrogram", GetActiveSpectrogramBuffer());
            ComputeHelper.Dispatch(raytraceShader, hitCount, 1, 1, kp);


            // Read first time interval of spectrogram and visualize it
            float[] firstTimeIntervalData = new float[chunkSamples];
            GetActiveSpectrogramBuffer().GetData(firstTimeIntervalData, 0, 0, chunkSamples);

            // Convert real spectrogram values to complex format for DrawFFTDebug
            Vector2[] spectrogramComplex = new Vector2[chunkSamples];
            float spectrogramMaxValue = 0f;
            for (int i = 0; i < chunkSamples; i++)
            {
                spectrogramComplex[i] = new Vector2(firstTimeIntervalData[i], 0f);
                if (firstTimeIntervalData[i] > spectrogramMaxValue)
                    spectrogramMaxValue = firstTimeIntervalData[i];
            }

            ComputeBuffer spectrogramDebugBuffer = ComputeHelper.CreateStructuredBuffer(spectrogramComplex);

            int kSpectrogramDebug = raytraceShader.FindKernel("DrawFFTDebug");
            raytraceShader.SetInt("FFTDataLength", chunkSamples);
            raytraceShader.SetFloat("FFTDebugGain", spectrogramGain);
            raytraceShader.SetFloat("FFTMaxValue", spectrogramMaxValue > 0f ? spectrogramMaxValue : 1f);

            raytraceShader.SetBuffer(kSpectrogramDebug, "FFTData", spectrogramDebugBuffer);
            raytraceShader.SetTexture(kSpectrogramDebug, "FFTDebugTexture", SpectrogramFirstFreqBinTexture);
            ComputeHelper.Dispatch(raytraceShader, SpectrogramFirstFreqBinTexture.width, SpectrogramFirstFreqBinTexture.height, 1, kSpectrogramDebug);

            ComputeHelper.Release(spectrogramDebugBuffer);

            // Hit Statistics
            // int safeHitCount = Mathf.Min(hitCount, hitBuffer.count);
            // RayInfo[] hitData = new RayInfo[safeHitCount];
            // hitBuffer.GetData(hitData, 0, 0, safeHitCount);

            // float totalEnergy = 0f;
            // for (int i = 0; i < hitData.Length; i++) {
            //     totalEnergy += hitData[i].energy;
            // }
            // Debug.Log($"Total energy from hits: {totalEnergy}");
            // Debug.Log($"Average energy per hit: {totalEnergy / hitData.Length}");
        }

        accumFrames++;
    }

    // Build a weighted frequency distribution from the FFT result (Bins)
    int[] BuildFrequencyDistribution(Vector2[] fftResult)
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
        
        // Track counts for each frequency bin for debugging
        
        // Build weighted distribution buffer (rayCount entries)
        // Each entry is a frequency bin index, with frequencies appearing proportional to their magnitude
        
        int[] binCounts = new int[halfSize];
        int[] distribution = new int[rayCount];
        int distributionIndex = 0;

        int test_total = 0;
        
        for (int i = 0; i < halfSize && distributionIndex < rayCount; i++)
        {
            // Convert bin index to frequency
            int frequencyBin = i;
            
            // Add this frequency proportional to its normalized magnitude
            int count = Mathf.RoundToInt(magnitudes[i] * rayCount);
            binCounts[i] = count;
            test_total += count;
            
            for (int j = 0; j < count && distributionIndex < rayCount; j++)
            {
                distribution[distributionIndex++] = frequencyBin;
            }
        }

        Debug.Log($"Filled {distributionIndex}, we have: {rayCount}");
        
        // Fill any remaining slots with audible frequencies (if rounding left gaps)
        int randomFills = 0;
        while (distributionIndex < rayCount)
        {
            int randomBin = Random.Range(0, halfSize);
            distribution[distributionIndex++] = randomBin;
            randomFills++;
        }

        // Debug log the distribution counts
        // Debug.Log($"=== Frequency Distribution Counts (Total Rays: {rayCount}, Total Assigned: {test_total}, Random Fills: {randomFills}) ===");
        // // Shuffle(distribution);
        // Debug.Log($"[{string.Join(", ", binCounts)}]");
        
        
        return distribution;
        // return magnitudes;
    }

    void ProcessAndQueueSpectrogram(int? audioWritePos = null)
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
            audioManager.QueueAudioChunk(outputWaveform, 0, audioWritePos);
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

            if (FFTTexture != null)
            {
                float w = Screen.width * 0.4f, h = Screen.height * 0.15f;
                float yPos = 40 + Screen.height * 0.45f;
                GUI.DrawTexture(new Rect(10, yPos, w, h), FFTTexture);
                
                Matrix4x4 matrixBackup = GUI.matrix;
                GUIUtility.RotateAroundPivot(90, new Vector2(10 + w + 5, yPos));
                GUI.Label(new Rect(10 + w + 5, yPos, h, 20), "FFT Input", labelStyle);
                GUI.matrix = matrixBackup;
            }

            if (SpectrogramFirstFreqBinTexture != null)
            {
                float w = Screen.width * 0.4f, h = Screen.height * 0.15f;
                float yPos = 50 + Screen.height * 0.60f;
                GUI.DrawTexture(new Rect(10, yPos, w, h), SpectrogramFirstFreqBinTexture);
                
                Matrix4x4 matrixBackup = GUI.matrix;
                GUIUtility.RotateAroundPivot(90, new Vector2(10 + w + 5, yPos));
                GUI.Label(new Rect(10 + w + 5, yPos, h, 20), "First Freq Bin", labelStyle);
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

    void OnDestroy() => ComputeHelper.Release(wallBuffer, hitBuffer, debugBuffer, spectrogramBufferPing, spectrogramBufferPong, waveformOutBuffer, argsBuffer, frequencyDistributionBuffer);
}