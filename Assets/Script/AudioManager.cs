using UnityEngine;
using System.Collections.Generic;
using System.IO;
using Helpers;

public class AudioManager : MonoBehaviour
{
    [Header("Debug")]
    public int writeHead;
    public int readHead_Debug;
    public int bufferCapacity;
    public long writeSampleCursor_Debug;
    public long readSampleCursor_Debug;
    public int writeLoopCount_Debug;
    public int readLoopCount_Debug;
    public int underrunCount_Debug;
    
    [Header("Waveform Visualization")]
    public ComputeShader shader;
    public RayTraceManager rayTraceManager;
    public bool showWaveform = true;
    public Vector2 waveformTextureSize = new Vector2(1024, 256);
    [Range(0.1f, 100f)] public float waveformGain = 10.0f;
    public RenderTexture waveformTexture;

    int safetyBuffer = 4096; // Minimum samples between read and write heads to avoid underrun

    float[] ringBuffer;
    int readHead, sampleRate, bufferSize;
    long readSampleCursor;
    long writeSampleCursor;
    long nextUnderrunLogSample;
    readonly object bufferLock = new object();
    ComputeBuffer ringBufferGPU;

    const string WaveformKernelName = "DrawRingBufferWaveform";

    public bool paused = false;

    // Recording
    bool isRecording;
    List<float> recordingBuffer;

    public int SampleRate => sampleRate;
    public int WriteHead => writeHead;
    public int ReadHead { get { lock (bufferLock) return readHead; } }
    public bool IsRecording => isRecording;

    void Awake()
    {
        sampleRate = AudioSettings.outputSampleRate;
        bufferSize = sampleRate * 16; // 16 second buffer
        ringBuffer = new float[bufferSize];
        
        var src = gameObject.AddComponent<AudioSource>();
        src.playOnAwake = true;
        src.loop = true;
        var clip = AudioClip.Create("RingBuffer", sampleRate, 1, sampleRate, false);
        clip.SetData(new float[sampleRate], 0);
        src.clip = clip;
        src.Play();
    }

    void Start()
    {
        if (shader == null && rayTraceManager != null)
            shader = rayTraceManager.raytraceShader;

        if (waveformTexture == null) {
            waveformTexture = new RenderTexture((int)waveformTextureSize.x, (int)waveformTextureSize.y, 0) {
                enableRandomWrite = true, filterMode = FilterMode.Point 
            };
            waveformTexture.Create();
        }
    }

    public void StartStreaming(int audioSampleRate)
    {
        Debug.Log($"AudioManager starting streaming with sample rate {audioSampleRate}.");
        sampleRate = audioSampleRate;
        lock (bufferLock) 
        { 
            readHead = 0;
            writeHead = 0;
            readSampleCursor = 0;
            writeSampleCursor = 0;
            underrunCount_Debug = 0;
            System.Array.Clear(ringBuffer, 0, ringBuffer.Length); 
        }
    }

    /// <summary>
    /// Add audio chunk to ring buffer just ahead of read position for immediate playback
    /// </summary>
    public (long, int) QueueAudioChunk(float[] audio, int audioStartPos=0, long? forcedWritePos = null)
    {
        if (audio == null || audio.Length == 0)
            return (writeSampleCursor, audioStartPos);

        lock (bufferLock)
        {
            long safeStart = readSampleCursor + safetyBuffer;
            long writePos = forcedWritePos ?? safeStart;

            if (writePos < safeStart)
            {
                long skippedSamples = safeStart - writePos;
                audioStartPos += skippedSamples > int.MaxValue ? int.MaxValue : (int)skippedSamples;
                writePos = safeStart;
                LogUnderrun($"Audio computation is behind playback by {skippedSamples} samples; skipping ahead to avoid writing in the past.");
            }

            audioStartPos = Mathf.Clamp(audioStartPos, 0, audio.Length);
            int samplesToWrite = audio.Length - audioStartPos;
            if (samplesToWrite <= 0)
                return (writePos, audioStartPos);

            long maxWritableEnd = readSampleCursor + bufferSize;
            if (writePos + samplesToWrite > maxWritableEnd)
            {
                samplesToWrite = Mathf.Max(0, (int)(maxWritableEnd - writePos));
                Debug.LogWarning($"Audio queue is more than one ring buffer ahead; truncating chunk to {samplesToWrite} samples.");
            }

            for (int i = 0; i < samplesToWrite; i++)
            {
                int idx = Mod(writePos + i, bufferSize);
                ringBuffer[idx] += audio[i + audioStartPos];
            }

            writeSampleCursor = System.Math.Max(writeSampleCursor, writePos + samplesToWrite);
            writeHead = Mod(writeSampleCursor, bufferSize);
            return (writePos, audioStartPos);
        }
    }

    static int Mod(long value, int modulus) => (int)((value % modulus + modulus) % modulus);

    void OnAudioFilterRead(float[] data, int channels)
    {
        if (paused) return;
        lock (bufferLock)
        {
            int samplesPerChannel = data.Length / channels;
            for (int i = 0; i < samplesPerChannel; i++)
            {
                if (readSampleCursor >= writeSampleCursor)
                    LogUnderrun($"Audio read cursor passed write cursor at sample {readSampleCursor}. Computation is too slow or audio has not been queued yet.");

                float s = ringBuffer[readHead];
                ringBuffer[readHead] = 0; // Clear after reading
                readHead++;
                if (readHead >= bufferSize) readHead -= bufferSize;
                readSampleCursor++;
                for (int c = 0; c < channels; c++)
                    data[i * channels + c] = s;

                if (isRecording)
                    recordingBuffer.Add(s);
            }
        }
    }

    void Update()
    {
        bufferCapacity = bufferSize;
        readHead_Debug = readHead;
        readSampleCursor_Debug = readSampleCursor;
        writeSampleCursor_Debug = writeSampleCursor;
        readLoopCount_Debug = bufferSize > 0 ? (int)(readSampleCursor / bufferSize) : 0;
        writeLoopCount_Debug = bufferSize > 0 ? (int)(writeSampleCursor / bufferSize) : 0;

        if (Input.GetKeyDown(KeyCode.R))
            ToggleRecording();

        if (Input.GetKeyDown(KeyCode.P))
            TogglePause();

        if (showWaveform && shader != null)
            DrawCyclingWaveform();
    }

    public void TogglePause()
    {
        paused = !paused;
        Debug.Log(paused ? "Audio paused." : "Audio resumed.");
    }

    void LogUnderrun(string message)
    {
        underrunCount_Debug++;
        if (readSampleCursor < nextUnderrunLogSample)
            return;

        nextUnderrunLogSample = readSampleCursor + sampleRate / 2;
        Debug.LogWarning(message);
    }

    public void ToggleRecording()
    {
        if (isRecording)
        {
            StopRecordingAndSave();
            return;
        }

        recordingBuffer = new List<float>();
        isRecording = true;
        Debug.Log("Recording started.");
    }

    public void StopRecordingAndSave()
    {
        if (!isRecording) return;

        isRecording = false;
        Debug.Log($"Recording stopped. {recordingBuffer.Count} samples captured.");
        SaveRecordingToWav();
    }

    void SaveRecordingToWav()
    {
        if (recordingBuffer == null || recordingBuffer.Count == 0)
        {
            Debug.LogWarning("No recording data to save.");
            return;
        }

        float[] samples = recordingBuffer.ToArray();
        string path = Path.Combine(Application.dataPath, $"recording_{System.DateTime.Now:yyyyMMdd_HHmmss}.wav");

        using (var fs = new FileStream(path, FileMode.Create))
        using (var writer = new BinaryWriter(fs))
        {
            int channels = 1;
            int bitsPerSample = 16;
            int byteRate = sampleRate * channels * bitsPerSample / 8;
            int blockAlign = channels * bitsPerSample / 8;
            int dataSize = samples.Length * blockAlign;

            // RIFF header
            writer.Write(System.Text.Encoding.ASCII.GetBytes("RIFF"));
            writer.Write(36 + dataSize);
            writer.Write(System.Text.Encoding.ASCII.GetBytes("WAVE"));

            // fmt chunk
            writer.Write(System.Text.Encoding.ASCII.GetBytes("fmt "));
            writer.Write(16);
            writer.Write((short)1); // PCM
            writer.Write((short)channels);
            writer.Write(sampleRate);
            writer.Write(byteRate);
            writer.Write((short)blockAlign);
            writer.Write((short)bitsPerSample);

            // data chunk
            writer.Write(System.Text.Encoding.ASCII.GetBytes("data"));
            writer.Write(dataSize);

            for (int i = 0; i < samples.Length; i++)
            {
                float clamped = Mathf.Clamp(samples[i], -1f, 1f);
                writer.Write((short)(clamped * 32767f));
            }
        }

        Debug.Log($"Recording saved to {path}");
    }
    
    void DrawCyclingWaveform()
    {
        if (shader == null)
            return;
        if (!shader.HasKernel(WaveformKernelName))
        {
            Debug.LogWarning($"Waveform kernel '{WaveformKernelName}' not found on shader '{shader.name}'.");
            return;
        }

        // Lazy initialize waveform texture
        if (waveformTexture == null || waveformTexture.width != (int)waveformTextureSize.x || waveformTexture.height != (int)waveformTextureSize.y)
        {
            if (waveformTexture != null)
                waveformTexture.Release();
            
            waveformTexture = new RenderTexture((int)waveformTextureSize.x, (int)waveformTextureSize.y, 0);
            waveformTexture.enableRandomWrite = true;
            waveformTexture.filterMode = FilterMode.Point;
            waveformTexture.Create();
            
            // Clear to visible color
            RenderTexture.active = waveformTexture;
            UnityEngine.GL.Clear(true, true, new Color(0.05f, 0.05f, 0.1f, 1));
            RenderTexture.active = null;
        }
        
        // Create or update GPU buffer
        if (ringBufferGPU == null || ringBufferGPU.count != bufferSize)
        {
            ringBufferGPU?.Release();
            ringBufferGPU = new ComputeBuffer(bufferSize, sizeof(float));
        }
        
        // Copy ring buffer to GPU
        lock (bufferLock)
        {
            ringBufferGPU.SetData(ringBuffer);
        }
        
        // Dispatch compute shader
        int kernel = shader.FindKernel(WaveformKernelName);
        shader.SetBuffer(kernel, "RingBufferData", ringBufferGPU);
        shader.SetInt("RingBufferSize", bufferSize);
        shader.SetInt("RingBufferReadHead", readHead);
        shader.SetInt("RingBufferWriteHead", writeHead);
        shader.SetTexture(kernel, "DebugTexture", waveformTexture);
        shader.SetInt("TexWidth", waveformTexture.width);
        shader.SetInt("TexHeight", waveformTexture.height);
        shader.SetFloat("WaveformGain", waveformGain);
        
        ComputeHelper.Dispatch(shader, waveformTexture.width, waveformTexture.height, 1, kernel);
    }
    
    void OnGUI()
    {
        if (showWaveform && waveformTexture != null)
        {
            float w = Screen.width * 0.4f;
            float h = Screen.height * 0.15f;

            // Position below RayTraceManager's two textures using the same on-screen size
            float yOffset = 10;
            if (rayTraceManager != null && rayTraceManager.showDebugTexture)
            {
                yOffset = 30 + h * 2;
            }
            GUI.DrawTexture(new Rect(10, yOffset, w, h), waveformTexture);

        }
    }
    
    void OnDestroy()
    {
        ringBufferGPU?.Release();
        if (waveformTexture != null)
        {
            waveformTexture.Release();
            waveformTexture = null;
        }
    }
}
