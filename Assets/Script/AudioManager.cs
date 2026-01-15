using UnityEngine;

public class AudioManager : MonoBehaviour
{
    [Header("Debug")]
    public int writeHead;
    public int readHead_Debug;
    public int bufferCapacity;

    float[] ringBuffer;
    int readHead, sampleRate, bufferSize;
    readonly object bufferLock = new object();

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
    public void QueueAudioChunk(float[] audio, int offset = 0)
    {
        if (audio == null || audio.Length == 0) return;

        lock (bufferLock)
        {
            // Write ahead of readHead by a small safety buffer (e.g., 1 audio frame ~20ms)
            int safetyBuffer = sampleRate / 50; // 20ms
            int writePos = (readHead + safetyBuffer + offset) % bufferSize;
            
            for (int i = 0; i < audio.Length; i++)
            {
                ringBuffer[(writePos + i) % bufferSize] += audio[i];
            }
            
            // Update writeHead to track latest write position
            writeHead = (writePos + audio.Length) % bufferSize;
        }
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
    }
}
