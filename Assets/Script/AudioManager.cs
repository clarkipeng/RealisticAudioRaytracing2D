using UnityEngine;

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
    public Vector2 waveformTextureSize = new Vector2(1024, 512);
    [Range(0.1f, 100f)] public float waveformGain = 10.0f;
    public RenderTexture waveformTexture;

    float[] ringBuffer;
    int readHead, sampleRate, bufferSize;
    readonly object bufferLock = new object();
    ComputeBuffer ringBufferGPU;

    public int SampleRate => sampleRate;
    public int WriteHead => writeHead;
    public int ReadHead { get { lock (bufferLock) return readHead; } }

    void Awake()
    {
        sampleRate = AudioSettings.outputSampleRate;
        bufferSize = sampleRate * 4; // 4 second buffer
        ringBuffer = new float[bufferSize];
        
        var src = gameObject.AddComponent<AudioSource>();
        src.playOnAwake = true;
        src.loop = true;
        var clip = AudioClip.Create("RingBuffer", sampleRate, 1, sampleRate, false);
        clip.SetData(new float[sampleRate], 0);
        src.clip = clip;
        src.Play();
    }

    public void StartStreaming(int audioSampleRate)
    {
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
    public int QueueAudioChunk(float[] audio, int offset = 0, int? forcedWritePos = null)
    {
        if (audio == null || audio.Length == 0) return writeHead;

        lock (bufferLock)
        {

            int writePos = (readHead + /*safetyBuffer*/ 0 + offset) % bufferSize;
            if (forcedWritePos.HasValue)
                writePos = forcedWritePos.Value;
            
            
            for (int i = 0; i < audio.Length; i++)
            {
                ringBuffer[(writePos + i) % bufferSize] += audio[i];
            }
            
            // Update writeHead to track latest write position
            writeHead = (writePos + audio.Length) % bufferSize;
        }
        return writeHead;
    }

    void OnAudioFilterRead(float[] data, int channels)
    {
        lock (bufferLock)
        {
            for (int i = 0; i < data.Length / channels; i++)
            {
                float s = ringBuffer[readHead];
                ringBuffer[readHead] = 0; // Clear after reading
                readHead = (readHead + 1) % bufferSize;
                for (int c = 0; c < channels; c++) 
                    data[i * channels + c] = s;
            }
        }
    }

    void Update()
    {
        bufferCapacity = bufferSize;
        readHead_Debug = readHead;
        
        if (showWaveform && shader != null)
            DrawCyclingWaveform();
    }
    
    void DrawCyclingWaveform()
    {
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
        int kernel = shader.FindKernel("DrawRingBufferWaveform");
        shader.SetBuffer(kernel, "RingBufferData", ringBufferGPU);
        shader.SetInt("RingBufferSize", bufferSize);
        shader.SetInt("RingBufferReadHead", readHead);
        shader.SetInt("RingBufferWriteHead", writeHead);
        shader.SetTexture(kernel, "DebugTexture", waveformTexture);
        shader.SetInt("TexWidth", waveformTexture.width);
        shader.SetInt("TexHeight", waveformTexture.height);
        shader.SetFloat("WaveformGain", waveformGain);
        
        int threadGroupsX = Mathf.CeilToInt(waveformTexture.width / 8.0f);
        int threadGroupsY = Mathf.CeilToInt(waveformTexture.height / 8.0f);
        shader.Dispatch(kernel, threadGroupsX, threadGroupsY, 1);
    }
    
    void OnGUI()
    {
        if (showWaveform && waveformTexture != null)
        {
            // Position below RayTraceManager's two textures
            float yOffset = 10;
            if (rayTraceManager != null && rayTraceManager.showDebugTexture)
            {
                // Stack below the two RayTraceManager textures
                yOffset = 30 + rayTraceManager.debugTextureSize.y * 2;
            }
            
            GUI.DrawTexture(new Rect(10, yOffset, waveformTextureSize.x, waveformTextureSize.y), waveformTexture);
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
