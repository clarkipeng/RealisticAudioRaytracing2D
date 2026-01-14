using UnityEngine;
using UnityEngine.Rendering;
using System.Collections;
using Helpers;
using UnityEngine.Assertions;

public class AudioManager : MonoBehaviour
{
    [Range(0.05f, 1.0f)] public float chunkDuration = 0.1f;
    
    [Header("References")]
    public ComputeShader convolutionShader;
    
    [Header("Debug")]
    public int currentAccumCount;
    public float lastProcessingLatency;

    float[] ringBuffer, fullInputSamples;
    int readHead, sampleRate, bufferSize;
    readonly object bufferLock = new object();
    bool isStreaming;
    ComputeBuffer irBuffer;

    public bool IsStreaming => isStreaming;

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

    public void StartStreaming(AudioClip inputClip, ComputeBuffer impulseResponseBuffer, int audioSampleRate)
    {
        Assert.IsNotNull(inputClip, "AudioManager: inputClip is null");
        Assert.IsNotNull(impulseResponseBuffer, "AudioManager: impulseResponseBuffer is null");
        Assert.IsNotNull(convolutionShader, "AudioManager: convolutionShader not assigned");
        
        irBuffer = impulseResponseBuffer;
        sampleRate = audioSampleRate;
        if (isStreaming) StopStreaming();
        fullInputSamples = LoadMonoSamples(inputClip);
        lock (bufferLock) { readHead = 0; System.Array.Clear(ringBuffer, 0, ringBuffer.Length); }
        isStreaming = true;
        GetComponent<AudioSource>().Play();
    }

    public void StopStreaming()
    {
        if (!isStreaming) return;
        isStreaming = false;
        GetComponent<AudioSource>()?.Stop();
    }

    float[] LoadMonoSamples(AudioClip clip)
    {
        float[] raw = new float[clip.samples * clip.channels];
        clip.GetData(raw, 0);
        if (clip.channels == 1) return raw;
        
        float[] mono = new float[clip.samples];
        for (int i = 0; i < clip.samples; i++)
        {
            float sum = 0;
            for (int c = 0; c < clip.channels; c++) sum += raw[i * clip.channels + c];
            mono[i] = sum / clip.channels;
        }
        return mono;
    }

    public void ProcessNextChunk(int sampleOffset, int chunkSampleCount, int accumCount, ComputeBuffer ir)
    {
        if (isStreaming) StartCoroutine(ProcessAndPush(sampleOffset, chunkSampleCount, accumCount, ir ?? irBuffer));
    }

    IEnumerator ProcessAndPush(int sampleOffset, int chunkSamples, int accumCount, ComputeBuffer ir)
    {
        float start = Time.realtimeSinceStartup;
        int inputLen = Mathf.Min(chunkSamples, fullInputSamples.Length - sampleOffset);
        if (inputLen <= 0) yield break;

        int irLen = ir.count, outputLen = inputLen + irLen;
        float[] chunk = new float[inputLen];
        System.Array.Copy(fullInputSamples, sampleOffset, chunk, 0, inputLen);

        var inputBuf = new ComputeBuffer(inputLen, sizeof(float));
        var outputBuf = new ComputeBuffer(outputLen, sizeof(float));
        inputBuf.SetData(chunk);

        int k = convolutionShader.FindKernel("AudioConvolve");
        convolutionShader.SetInt("InputLength", inputLen);
        convolutionShader.SetInt("IRLength", irLen);
        convolutionShader.SetInt("accumCount", accumCount);
        convolutionShader.SetBuffer(k, "InputAudio", inputBuf);
        convolutionShader.SetBuffer(k, "ImpulseResponse", ir);
        convolutionShader.SetBuffer(k, "OutputAudio", outputBuf);
        ComputeHelper.Dispatch(convolutionShader, outputLen, 1, 1, k);

        float[] result = new float[outputLen];
        var req = AsyncGPUReadback.Request(outputBuf);
        while (!req.done) yield return null;

        if (req.hasError) { inputBuf.Release(); outputBuf.Release(); yield break; }

        req.GetData<float>().CopyTo(result);
        lastProcessingLatency = Time.realtimeSinceStartup - start;
        inputBuf.Release();
        outputBuf.Release();

        lock (bufferLock)
        {
            int writePos = sampleOffset % bufferSize;
            for (int i = 0; i < outputLen; i++)
                ringBuffer[(writePos + i) % bufferSize] += result[i];
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

    void OnDestroy() => StopStreaming();
}
