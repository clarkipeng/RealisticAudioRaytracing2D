using UnityEngine;
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

    float[] ringBuffer;
    int readHead, sampleRate, bufferSize;
    readonly object bufferLock = new object();
    ComputeBuffer ringBufferGPU;

    const string WaveformKernelName = "DrawRingBufferWaveform";

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
                int idx = writePos + i;
                if (idx >= bufferSize)
                    idx -= bufferSize;
                ringBuffer[idx] += audio[i];
            }
            writeHead = (writePos + audio.Length) % bufferSize;
        }
        return writeHead;
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
