using UnityEngine;
using UnityEngine.Rendering;
using System.Collections;
using System.Collections.Generic;
using Helpers;
using UnityEngine.Assertions;

public class AudioManager : MonoBehaviour
{
    [Header("Streaming Settings")]
    [Tooltip("Duration of each audio chunk in seconds")]
    [Range(0.1f, 2.0f)] public float chunkDuration = 0.5f;
    [Tooltip("Maximum number of simultaneous audio sources")]
    [Range(2, 64)] public int audioSourcePoolSize = 64;

    [Header("References")]
    public ComputeShader convolutionShader;

    // Audio source pool
    struct SourceMeta { public AudioSource src; public double busyUntil; }
    private SourceMeta[] sourceMetas;

    // Streaming state
    private float[] fullInputSamples;
    private bool isStreaming;
    private double nextScheduledTime;
    private int nextSourceIdx = 0;

    // Impulse response buffer (set externally by Acoustic2D)
    private ComputeBuffer irBuffer;
    private int sampleRate;

    public int currentAccumCount = 1;

    public bool IsStreaming => isStreaming;

    void Awake()
    {
        InitializeAudioPool();
    }

    void InitializeAudioPool()
    {
        if (sourceMetas != null) return;

        sourceMetas = new SourceMeta[audioSourcePoolSize];

        for (int i = 0; i < audioSourcePoolSize; i++)
        {
            GameObject child = new GameObject($"StreamingAudioSource_{i}");
            child.transform.SetParent(transform);
            AudioSource s = child.AddComponent<AudioSource>();
            s.playOnAwake = false;
            s.spatialBlend = 0f;
            sourceMetas[i] = new SourceMeta { src = s, busyUntil = 0 };
        }
    }

    /// <summary>
    /// Start streaming audio with polyphonic playback.
    /// Each chunk is convolved with the current impulse response and played with overlapping tails.
    /// </summary>
    public void StartStreaming(AudioClip inputClip, ComputeBuffer impulseResponseBuffer, int audioSampleRate)
    {
        Assert.IsNotNull(inputClip);
        Assert.IsNotNull(impulseResponseBuffer);
        Assert.IsTrue(audioSampleRate > 0);
        irBuffer = impulseResponseBuffer;
        sampleRate = audioSampleRate;

        if (isStreaming)
        {
            StopStreaming();
        }

        // Load and convert to mono
        fullInputSamples = LoadMonoSamples(inputClip);
        sampleRate = inputClip.frequency;
        isStreaming = true;

        nextScheduledTime = AudioSettings.dspTime;
        Debug.Log($"AudioManager: Starting streaming playback ({fullInputSamples.Length} samples) first chunk scheduled at {nextScheduledTime:F3}");
    }

    public void StopStreaming()
    {
        if (!isStreaming) return;
        isStreaming = false;
        Debug.Log("AudioManager: Streaming stopping gracefully");
    }

    private void KillAllAudio()
    {
        isStreaming = false;
        if (sourceMetas == null) return;
        foreach (var meta in sourceMetas)
        {
            if (meta.src != null) meta.src.Stop();
        }
    }

    private float[] LoadMonoSamples(AudioClip clip)
    {
        float[] rawSamples = new float[clip.samples * clip.channels];
        clip.GetData(rawSamples, 0);

        if (clip.channels == 1)
        {
            return rawSamples;
        }

        // Convert to mono by averaging channels
        float[] monoSamples = new float[clip.samples];
        int channels = clip.channels;
        for (int i = 0; i < clip.samples; i++)
        {
            float sum = 0;
            for (int c = 0; c < channels; c++)
            {
                sum += rawSamples[i * channels + c];
            }
            monoSamples[i] = sum / channels;
        }
        return monoSamples;
    }

    /// <summary>
    /// Processes a single chunk of audio using the CURRENT state of the impulse response.
    /// Should be called by RayTraceManager.
    /// </summary>
    public void ProcessNextChunk(int sampleOffset, int chunkSampleCount)
    {
        if (!isStreaming) return;
        StartCoroutine(ProcessAndPlayChunk(sampleOffset, chunkSampleCount));
    }

    private IEnumerator ProcessAndPlayChunk(int sampleOffset, int chunkSampleCount)
    {
        int inputLen = Mathf.Min(chunkSampleCount, fullInputSamples.Length - sampleOffset);
        if (inputLen <= 0) yield break;

        int irLen = irBuffer.count;
        int outputLen = inputLen + irLen;

        float[] chunkInput = new float[inputLen];
        System.Array.Copy(fullInputSamples, sampleOffset, chunkInput, 0, inputLen);

        ComputeBuffer inputBuf = new ComputeBuffer(inputLen, sizeof(float));
        ComputeBuffer outputBuf = new ComputeBuffer(outputLen, sizeof(float));
        inputBuf.SetData(chunkInput);

        int kConv = convolutionShader.FindKernel("AudioConvolve");
        convolutionShader.SetInt("InputLength", inputLen);
        convolutionShader.SetInt("IRLength", irLen);
        convolutionShader.SetInt("accumCount", currentAccumCount);
        convolutionShader.SetBuffer(kConv, "InputAudio", inputBuf);
        convolutionShader.SetBuffer(kConv, "ImpulseResponse", irBuffer);
        convolutionShader.SetBuffer(kConv, "OutputAudio", outputBuf);

        ComputeHelper.Dispatch(convolutionShader, outputLen, 1, 1, kConv);

        // Async GPU readback to avoid blocking main thread
        float[] resultData = new float[outputLen];
        var request = AsyncGPUReadback.Request(outputBuf);

        while (!request.done)
        {
            yield return null;
        }

        if (request.hasError)
        {
            Debug.LogError("AudioManager: GPU readback failed!");
            inputBuf.Release();
            outputBuf.Release();
            yield break;
        }

        request.GetData<float>().CopyTo(resultData);

        // Release GPU buffers
        inputBuf.Release();
        outputBuf.Release();

        // Get an available audio source
        int metaIdx = nextSourceIdx;
        nextSourceIdx = (nextSourceIdx + 1) % sourceMetas.Length;
        double currentDspTime = AudioSettings.dspTime;
        if (currentDspTime > sourceMetas[metaIdx].busyUntil)
        {
            AudioSource src = sourceMetas[metaIdx].src;
            AudioClip clip = AudioClip.Create($"Chunk_{sampleOffset}", outputLen, 1, sampleRate, false);
            clip.SetData(resultData, 0);
            src.clip = clip;

            if (nextScheduledTime < currentDspTime)
            {
                Debug.LogWarning($"AudioManager: Chunk late by {currentDspTime - nextScheduledTime:F4}s. If audio is stuttering, increase lookahead.");
            }

            // Schedule precise playback
            src.PlayScheduled(nextScheduledTime);

            // Mark as busy for the FULL duration (input + tail)
            sourceMetas[metaIdx].busyUntil = nextScheduledTime + (double)outputLen / sampleRate;

            // Increment the playback timeline for the NEXT chunk
            nextScheduledTime += (double)inputLen / sampleRate;
        }
        else
        {
            Debug.LogWarning("AudioManager: No available audio source (all busy)!");
        }
    }



    void OnDestroy()
    {
        KillAllAudio();

        if (sourceMetas != null)
        {
            foreach (var meta in sourceMetas)
            {
                if (meta.src != null)
                {
                    Destroy(meta.src.gameObject);
                }
            }
        }
    }
}
