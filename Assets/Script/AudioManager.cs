using UnityEngine;

public class AudioManager : MonoBehaviour
{
    [Header("Debug")]
    public int queuedSamples;
    public int bufferCapacity;

    float[] ringBuffer;
    int readHead, sampleRate, bufferSize;
    readonly object bufferLock = new object();
    bool isStreaming;

    public bool IsStreaming => isStreaming;
    public int SampleRate => sampleRate;

    void Awake()
    {
        sampleRate = AudioSettings.outputSampleRate;
        bufferSize = sampleRate * 4;
        ringBuffer = new float[bufferSize];
        
        var src = gameObject.AddComponent<AudioSource>();
        src.playOnAwake = false;
        src.loop = true;
        var clip = AudioClip.Create("Silent", sampleRate, 1, sampleRate, false);
        clip.SetData(new float[sampleRate], 0);
        src.clip = clip;
    }

    public void StartStreaming(int audioSampleRate)
    {
        sampleRate = audioSampleRate;
        if (isStreaming) StopStreaming();
        lock (bufferLock) 
        { 
            readHead = 0; 
            System.Array.Clear(ringBuffer, 0, ringBuffer.Length); 
        }
        isStreaming = true;
        GetComponent<AudioSource>().Play();
    }

    public void StopStreaming()
    {
        if (!isStreaming) return;
        isStreaming = false;
        GetComponent<AudioSource>()?.Stop();
    }

    public void QueueAudioChunk(float[] processedAudio, int sampleOffset)
    {
        if (!isStreaming || processedAudio == null || processedAudio.Length == 0) return;

        lock (bufferLock)
        {
            int writePos = sampleOffset % bufferSize;
            for (int i = 0; i < processedAudio.Length; i++)
                ringBuffer[(writePos + i) % bufferSize] += processedAudio[i];
        }
    }

    void OnAudioFilterRead(float[] data, int channels)
    {
        if (!isStreaming) return;
        lock (bufferLock)
        {
            for (int i = 0; i < data.Length / channels; i++)
            {
                float s = ringBuffer[readHead];
                ringBuffer[readHead] = 0;
                readHead = (readHead + 1) % bufferSize;
                for (int c = 0; c < channels; c++) data[i * channels + c] = s;
            }
        }
    }

    void Update()
    {
        lock (bufferLock)
        {
            queuedSamples = (readHead <= readHead) 
                ? readHead - readHead 
                : (bufferSize - readHead) + readHead;
            bufferCapacity = bufferSize;
        }
    }

    void OnDestroy() => StopStreaming();
}
