using UnityEngine;
using UnityEngine.Rendering;
using System.Collections;
using Helpers;
using UnityEngine.Assertions;
using System.Collections.Generic;

public class AudioManager : MonoBehaviour
{
    [Range(0.05f, 1.0f)] public float chunkDuration = 0.1f;
    
    [Header("References")]
    public ComputeShader convolutionShader;
    
    [Header("Debug")]
    public int currentAccumCount;
    public float lastProcessingLatency;

    float[] ringBuffer;
    int readHead, writeHead, sampleRate, bufferSize;
    readonly object bufferLock = new object();
    bool isStreaming;
    ComputeBuffer irBuffer;

    public bool IsStreaming => isStreaming;

    struct Source
    {
        public float[] samples;
        public int offset;
        public float volume;
        public bool loop;
        public int clipLength;
    }
    List<Source> sources;

    void Awake()
    {
        sampleRate = AudioSettings.outputSampleRate;
        bufferSize = sampleRate * 4;
        ringBuffer = new float[bufferSize];

        // Ensure AudioSource is attached and initialized
        var src = gameObject.GetComponent<UnityEngine.AudioSource>();
        if (src == null)
            src = gameObject.AddComponent<UnityEngine.AudioSource>();
        src.loop = true;
        var clip = AudioClip.Create("Silent", sampleRate, 1, sampleRate, false);
        clip.SetData(new float[sampleRate], 0);
        src.clip = clip;
    }

    /// Stream multiple Unity AudioSources
    public void StartStreaming(List<AudioSource> audioSources, ComputeBuffer impulseResponseBuffer, int audioSampleRate)
    {
        Assert.IsNotNull(impulseResponseBuffer, "AudioManager: impulseResponseBuffer is null");
        Assert.IsNotNull(convolutionShader, "AudioManager: convolutionShader not assigned");
        Assert.IsTrue(audioSources.Count > 0, "AudioManager: no audio sources provided");
        
        irBuffer = impulseResponseBuffer;
        sampleRate = audioSampleRate;
        if (isStreaming) StopStreaming();
        
        // iterates and makes a list of the audio sources
        sources = new List<Source>();
        foreach (var src in audioSources)
        {
            if (src == null || src.inputClip == null) continue;
            
            sources.Add(new Source
            {
                samples = LoadMonoSamples(src.inputClip),   // Use Unity AudioSource clip
                offset = 0,
                volume = src.volume,                    // Use Unity AudioSource volume
                loop = src.loop,                        // Use Unity AudioSource loop
                clipLength = src.inputClip.samples
            });
        }
        
        lock (bufferLock) { readHead = 0; writeHead = 0; System.Array.Clear(ringBuffer, 0, ringBuffer.Length); }
        isStreaming = true;
        GetComponent<UnityEngine.AudioSource>().Play();
    }

    public void StopStreaming()
    {
        if (!isStreaming) return;
        isStreaming = false;
        GetComponent<UnityEngine.AudioSource>()?.Stop();
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

    /// AI helped write, but apparently this mixes all of the next audio chunks into one thing
    public void ProcessNextChunk(int chunkSampleCount, int accumCount, ComputeBuffer ir)
    {
        if (isStreaming) StartCoroutine(ProcessAndPush(chunkSampleCount, accumCount, ir ?? irBuffer));
    }

    IEnumerator ProcessAndPush(int chunkSamples, int accumCount, ComputeBuffer ir)
    {
        float start = Time.realtimeSinceStartup;
        if (sources == null || sources.Count == 0) yield break;

        // array for mixed audio
        float[] mixBuffer = new float[chunkSamples];
        
        // loops for each of the sources
        for (int i = 0; i < sources.Count; i++)
        {
            Source src = sources[i];
            
            // for each of samples in a chunk
            for (int s = 0; s < chunkSamples; s++)
            {
                // find the what to play, offset + current sample
                int idx = src.offset + s;
                
                // if the offset makes the audio too long, if loop wrap around to beginning, else stop
                if (src.loop)
                {
                    idx = idx % src.clipLength;  // Wrap around
                }
                else if (idx >= src.clipLength)
                {
                    continue;
                }
                
                // uh this is the math, takes the sample and adds it to the buffer array
                mixBuffer[s] += src.samples[idx] * src.volume;
            }
            
            // move to the next offset location
            src.offset += chunkSamples;
            if (src.loop) src.offset %= src.clipLength;
            sources[i] = src;  // Structs are value types, so we must write back
        }

        int irLen = ir.count, outputLen = chunkSamples + irLen;

        var inputBuf = new ComputeBuffer(chunkSamples, sizeof(float));
        var outputBuf = new ComputeBuffer(outputLen, sizeof(float));
        inputBuf.SetData(mixBuffer);

        int k = convolutionShader.FindKernel("AudioConvolve");
        convolutionShader.SetInt("InputLength", chunkSamples);
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

        // this iterates all of the sources instead of just one
        lock (bufferLock)
        {
            for (int i = 0; i < outputLen; i++)
                ringBuffer[(writeHead + i) % bufferSize] += result[i];
            writeHead = (writeHead + chunkSamples) % bufferSize;
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
