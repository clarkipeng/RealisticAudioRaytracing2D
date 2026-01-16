using UnityEngine;

public class AudioManager : MonoBehaviour
{
    [Range(0.05f, 1.0f)] public float chunkDuration = 0.1f;

    float[] ringBuffer;
    int readHead, sampleRate, bufferSize;
    readonly object bufferLock = new object();
    bool isStreaming;

    public bool IsStreaming => isStreaming;

    void Awake()
    {
        sampleRate = AudioSettings.outputSampleRate;
        
        var src = gameObject.AddComponent<AudioSource>();
        src.playOnAwake = false;
        src.loop = true;
        var clip = AudioClip.Create("Silent", sampleRate, 1, sampleRate, false);
        clip.SetData(new float[sampleRate], 0);
        src.clip = clip;
    }

    public void StartStreaming(float reverbDuration)
    {
        if (isStreaming) StopStreaming();
        
        bufferSize = Mathf.CeilToInt(sampleRate * (reverbDuration + 1f));
        ringBuffer = new float[bufferSize];
        readHead = 0;
        
        isStreaming = true;
        GetComponent<AudioSource>().Play();
    }

    public void StopStreaming()
    {
        if (!isStreaming) return;
        isStreaming = false;
        GetComponent<AudioSource>()?.Stop();
    }

    public void PushSamples(float[] samples, int sampleOffset)
    {
        if (!isStreaming || ringBuffer == null) return;
        lock (bufferLock)
        {
            int writePos = sampleOffset % bufferSize;
            for (int i = 0; i < samples.Length; i++)
                ringBuffer[(writePos + i) % bufferSize] += samples[i];
        }
    }

    void OnAudioFilterRead(float[] data, int channels)
    {
        if (!isStreaming || ringBuffer == null) return;
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

    void OnDestroy() => StopStreaming();
}
