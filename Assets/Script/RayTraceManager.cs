using UnityEngine;
using System.Collections.Generic;
using Helpers;
using UnityEngine.Rendering;

public class RayTraceManager : MonoBehaviour
{
    [Header("Shaders")]
    public ComputeShader raytraceShader;
    public ComputeShader convolutionShader;

    [Header("Simulation")]
    [Range(10, 100000)] public int rayCount = 1000;
    [Range(1, 10)] public int maxBounces = 5;
    public float speedOfSound = 343f;
    public bool dynamicObstacles = false;

    [Header("Audio")]
    public AudioClip inputClip;
    public AudioManager audioManager;
    public int sampleRate = 48000;
    [Range(0.1f, 10f)] public float inputGain = 1.0f;
    [Range(0.1f, 5.0f)] public float reverbDuration = 2f;
    public bool loop = true;

    [Header("Scene")]
    public Transform source, listener;
    [Range(0.1f, 5f)] public float listenerRadius = 0.5f;
    public List<GameObject> obstacleObjects;

    [Header("Debug")]
    public bool showDebugTexture = true;
    [Range(5, 100)] public int debugRayCount = 100;
    [Range(1, 10000)] public float waveformGain = 1000.0f;

    ComputeBuffer wallBuffer, hitBuffer, debugBuffer, irBufferPing, irBufferPong, argsBuffer;
    RenderTexture irTexture;
    List<Segment> activeSegments;
    Vector4[] debugRayPaths;
    float[] fullInputSamples;
    int activeIRIndex, accumFrames, samplesSinceLastChunk, chunkSamples, nextStreamingOffset;

    struct RayInfo { public float timeDelay, energy; public Vector2 hitPoint; }

    void Start()
    {
        UpdateGeometry();
    }

    void Update()
    {
        if (!source || !listener || !raytraceShader) return;
        RunSimulation();

        if (Input.GetKeyDown(KeyCode.Space) && audioManager)
        {
            Debug.Log($"Space pressed. IsStreaming: {audioManager.IsStreaming}");
            if (audioManager.IsStreaming) audioManager.StopStreaming();
            else StartStreaming();
        }
        if (Input.GetKeyDown(KeyCode.R)) { ResetIR(); audioManager?.StopStreaming(); }
    }

    void FixedUpdate()
    {
        if (!audioManager || !audioManager.IsStreaming) return;
        if (dynamicObstacles) UpdateGeometry();

        int samplesThisFrame = Mathf.RoundToInt(Time.fixedDeltaTime * sampleRate);
        samplesSinceLastChunk += samplesThisFrame;

        if (samplesSinceLastChunk >= chunkSamples)
        {
            if (nextStreamingOffset >= fullInputSamples.Length)
            {
                if (loop) nextStreamingOffset = 0;
                else audioManager.StopStreaming();
            }
            
            if (audioManager.IsStreaming)
            {
                StartCoroutine(ProcessChunk(nextStreamingOffset, chunkSamples, Mathf.Max(1, accumFrames), GetActiveIRBuffer()));
                activeIRIndex = 1 - activeIRIndex;
                nextStreamingOffset += chunkSamples;
                ResetIR();
                samplesSinceLastChunk -= chunkSamples;
            }
        }
    }

    System.Collections.IEnumerator ProcessChunk(int sampleOffset, int chunkLen, int accumCount, ComputeBuffer ir)
    {
        int inputLen = Mathf.Min(chunkLen, fullInputSamples.Length - sampleOffset);
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

        inputBuf.Release();
        outputBuf.Release();
        if (req.hasError) yield break;

        req.GetData<float>().CopyTo(result);
        audioManager.PushSamples(result, sampleOffset);
    }

    void StartStreaming()
    {
        nextStreamingOffset = 0;
        samplesSinceLastChunk = 0;
        chunkSamples = Mathf.RoundToInt(sampleRate * audioManager.chunkDuration);
        fullInputSamples = LoadSample(inputClip);
        ResetIR();
        audioManager.StartStreaming(reverbDuration);
    }

    float[] LoadSample(AudioClip clip)
    {
        float[] raw = new float[clip.samples * clip.channels];
        clip.GetData(raw, 0);
        
        // Convert to mono
        float[] mono = new float[clip.samples];
        for (int i = 0; i < clip.samples; i++)
        {
            float sum = 0;
            for (int c = 0; c < clip.channels; c++) sum += raw[i * clip.channels + c];
            mono[i] = sum / clip.channels;
        }
        
        // Resample if needed
        if (clip.frequency == sampleRate) return mono;
        
        float ratio = (float)clip.frequency / sampleRate;
        int newLength = Mathf.RoundToInt(clip.samples / ratio);
        float[] resampled = new float[newLength];
        
        for (int i = 0; i < newLength; i++)
        {
            float srcIdx = i * ratio;
            int idx0 = Mathf.FloorToInt(srcIdx);
            int idx1 = Mathf.Min(idx0 + 1, mono.Length - 1);
            float t = srcIdx - idx0;
            resampled[i] = Mathf.Lerp(mono[idx0], mono[idx1], t);
        }
        
        Debug.Log($"Resampled audio from {clip.frequency}Hz to {sampleRate}Hz ({clip.samples} -> {newLength} samples)");
        return resampled;
    }

    void ResetIR()
    {
        accumFrames = 0;
        int len = (int)(sampleRate * reverbDuration);
        int k = raytraceShader.FindKernel("ClearImpulse");
        raytraceShader.SetInt("ImpulseLength", len);
        raytraceShader.SetBuffer(k, "ImpulseResponse", GetActiveIRBuffer());
        ComputeHelper.Dispatch(raytraceShader, len, 1, 1, k);
    }

    void RunSimulation()
    {
        int irLength = (int)(sampleRate * reverbDuration);

        ComputeHelper.CreateStructuredBuffer<Vector4>(ref debugBuffer, debugRayCount * (maxBounces + 1));
        ComputeHelper.CreateAppendBuffer<RayInfo>(ref hitBuffer, rayCount * maxBounces);
        if (wallBuffer == null || !wallBuffer.IsValid()) UpdateGeometry();
        if (argsBuffer == null || !argsBuffer.IsValid()) argsBuffer = new ComputeBuffer(1, sizeof(int) * 4, ComputeBufferType.IndirectArguments);
        if (irTexture == null) { irTexture = new RenderTexture(1024, 256, 0) { enableRandomWrite = true, filterMode = FilterMode.Point }; irTexture.Create(); }

        int k = raytraceShader.FindKernel("Trace");
        hitBuffer.SetCounterValue(0);
        raytraceShader.SetVector("sourcePos", source.position);
        raytraceShader.SetVector("listenerPos", listener.position);
        raytraceShader.SetFloat("listenerRadius", listenerRadius);
        raytraceShader.SetFloat("speedOfSound", speedOfSound);
        raytraceShader.SetFloat("inputGain", inputGain);
        raytraceShader.SetInt("maxBounceCount", maxBounces);
        raytraceShader.SetInt("rngStateOffset", Time.frameCount);
        raytraceShader.SetInt("numWalls", activeSegments.Count);
        raytraceShader.SetInt("rayCount", rayCount);
        raytraceShader.SetInt("debugRayCount", debugRayCount);
        raytraceShader.SetInt("accumFrames", accumFrames);
        raytraceShader.SetBuffer(k, "walls", wallBuffer);
        raytraceShader.SetBuffer(k, "rayInfoBuffer", hitBuffer);
        raytraceShader.SetBuffer(k, "debugRays", debugBuffer);
        ComputeHelper.Dispatch(raytraceShader, rayCount, 1, 1, k);

        AsyncGPUReadback.Request(debugBuffer, r => { if (!r.hasError) debugRayPaths = r.GetData<Vector4>().ToArray(); });
        ComputeBuffer.CopyCount(hitBuffer, argsBuffer, 0);
        AsyncGPUReadback.Request(argsBuffer, r => { if (!r.hasError) OnSimulationFinished(r.GetData<int>().ToArray()[0], irLength); });
    }

    ComputeBuffer GetActiveIRBuffer()
    {
        int len = (int)(sampleRate * reverbDuration);
        ComputeHelper.CreateStructuredBuffer<float>(ref irBufferPing, len);
        ComputeHelper.CreateStructuredBuffer<float>(ref irBufferPong, len);
        return activeIRIndex == 0 ? irBufferPing : irBufferPong;
    }

    void OnSimulationFinished(int hitCount, int irLength)
    {
        if (hitBuffer == null || irTexture == null) return;
        
        if (hitCount > 0)
        {
            int kp = raytraceShader.FindKernel("ProcessHits");
            raytraceShader.SetInt("SampleRate", sampleRate);
            raytraceShader.SetInt("HitCount", hitCount);
            raytraceShader.SetBuffer(kp, "RawHits", hitBuffer);
            raytraceShader.SetBuffer(kp, "ImpulseResponse", GetActiveIRBuffer());
            ComputeHelper.Dispatch(raytraceShader, hitCount, 1, 1, kp);
        }
        accumFrames++;

        int kd = raytraceShader.FindKernel("DrawIR");
        raytraceShader.SetInt("accumCount", accumFrames);
        raytraceShader.SetInt("ImpulseLength", irLength);
        raytraceShader.SetTexture(kd, "DebugTexture", irTexture);
        raytraceShader.SetBuffer(kd, "ImpulseResponse", GetActiveIRBuffer());
        raytraceShader.SetInt("TexWidth", irTexture.width);
        raytraceShader.SetInt("TexHeight", irTexture.height);
        raytraceShader.SetFloat("DebugGain", waveformGain);
        ComputeHelper.Dispatch(raytraceShader, irTexture.width, irTexture.height, 1, kd);
    }

    void UpdateGeometry()
    {
        activeSegments = SceneToData2D.GetSegmentsFromColliders(obstacleObjects);
        ComputeHelper.CreateStructuredBuffer(ref wallBuffer, activeSegments);
    }

    void OnGUI()
    {
        if (showDebugTexture && irTexture)
        {
            float w = Screen.width * 0.4f, h = Screen.height * 0.15f;
            GUI.DrawTexture(new Rect(10, 10, w, h), irTexture);
        }
    }

    void OnDrawGizmos()
    {
        if (!source || !listener) return;
        Gizmos.color = Color.green; Gizmos.DrawWireSphere(source.position, 0.2f);
        Gizmos.color = Color.cyan; Gizmos.DrawWireSphere(listener.position, listenerRadius);
        if (activeSegments != null) { Gizmos.color = Color.red; foreach (var seg in activeSegments) Gizmos.DrawLine(seg.start, seg.end); }
        if (debugRayPaths != null && debugRayPaths.Length >= debugRayCount * (maxBounces + 1))
        {
            int stride = maxBounces + 1;
            for (int i = 0; i < debugRayCount; i++)
                for (int b = 0; b < maxBounces - 1; b++)
                {
                    Vector4 p1 = debugRayPaths[i * stride + b], p2 = debugRayPaths[i * stride + b + 1];
                    if (p2.sqrMagnitude == 0) break;
                    Gizmos.color = Color.Lerp(new Color(1, 0.5f, 0, 0.1f), new Color(1, 1, 0, 0.8f), p1.z);
                    Gizmos.DrawLine(new Vector3(p1.x, p1.y, source.position.z - 5), new Vector3(p2.x, p2.y, source.position.z - 5));
                }
        }
    }

    void OnDestroy() => ComputeHelper.Release(wallBuffer, hitBuffer, debugBuffer, irBufferPing, irBufferPong, argsBuffer);
}