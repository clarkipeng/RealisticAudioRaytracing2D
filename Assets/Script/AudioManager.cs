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
    readonly object bufferLock = new object();
    ComputeBuffer ringBufferGPU;

    const string WaveformKernelName = "DrawRingBufferWaveform";

    // Recording
    bool isRecording;
    List<float> recordingBuffer;

    public int SampleRate => sampleRate;
    public int WriteHead => writeHead;
    public int ReadHead { get { lock (bufferLock) return readHead; } }

    void Awake()
    {
        sampleRate = AudioSettings.outputSampleRate;
        bufferSize = sampleRate * 8; // 8 second buffer
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
            System.Array.Clear(ringBuffer, 0, ringBuffer.Length); 
        }
    }

    /// <summary>
    /// Add audio chunk to ring buffer just ahead of read position for immediate playback
    /// </summary>
    public (int, int) QueueAudioChunk(float[] audio, int audioStartPos=0, int? forcedWritePos = null)
    {
        lock (bufferLock)
        {
            int safeStart = (readHead + safetyBuffer) % bufferSize;
            int writePos = safeStart;
            if (forcedWritePos.HasValue) {
                writePos = (forcedWritePos.Value % bufferSize);

                if ( (((safeStart + bufferSize / 2) % bufferSize) > safeStart) && (writePos < safeStart || writePos > (safeStart + bufferSize / 2) % bufferSize)
                    || (((safeStart + bufferSize / 2) % bufferSize) < safeStart) && (writePos < safeStart && writePos > (safeStart + bufferSize / 2) % bufferSize))
                {   
                    if (writePos < safeStart)
                    {
                        audioStartPos += safeStart - writePos;
                    }
                    else if (writePos > safeStart) {
                        audioStartPos += bufferSize - writePos + safeStart;
                    }
                    writePos = safeStart;
                }
            }
            
            for (int i = 0; i < audio.Length - audioStartPos; i++)
            {
                int idx = (writePos + i) % bufferSize;
                ringBuffer[idx] += audio[i + audioStartPos];
            }
            writeHead = (writePos + audio.Length) % bufferSize;
            return (writePos, audioStartPos);
        }
    }

    void OnAudioFilterRead(float[] data, int channels)
    {
        lock (bufferLock)
        {
            int samplesPerChannel = data.Length / channels;
            for (int i = 0; i < samplesPerChannel; i++)
            {
                float s = ringBuffer[readHead];
                ringBuffer[readHead] = 0; // Clear after reading
                readHead++;
                if (readHead >= bufferSize) readHead -= bufferSize;
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

        if (Input.GetKeyDown(KeyCode.R))
        {
            if (!isRecording)
            {
                recordingBuffer = new List<float>();
                isRecording = true;
                Debug.Log("Recording started.");
            }
            else
            {
                isRecording = false;
                Debug.Log($"Recording stopped. {recordingBuffer.Count} samples captured.");
                SaveRecordingToWav();
            }
        }

        if (showWaveform && shader != null)
            DrawCyclingWaveform();
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
