using UnityEngine;
using System.Collections.Generic;
using Helpers;
using UnityEngine.Assertions;

public class RayTraceManager : MonoBehaviour
{
    [Header("Simulation")]
    public ComputeShader shader;
    [Range(10, 100000)] public int rayCount = 1000;
    [Range(1, 10)] public int maxBounces = 5;
    public float speedOfSound = 343f;
    public bool dynamicObstacles = false;

    [Header("Audio Settings")]
    public AudioClip inputClip;
    public AudioManager audioManager;
    public float inputGain = 1.0f;
    public int sampleRate = 48000;

    [Header("Accum Settings")]
    [Range(0.1f, 5.0f)] public float reverbDuration = 5f;
    [Range(0f, 1f)] public float lookaheadSeconds = 0.1f;
    ComputeBuffer argsBuffer;

    [Header("Scene")]
    public Transform source, listener;
    [Range(0.1f, 5f)] public float listenerRadius = 0.5f;
    public List<GameObject> obstacleObjects;

    [Header("Debug")]
    public bool showDebugTexture = true;
    public Vector2 debugTextureSize = new Vector2(512, 128);
    [Range(1, 5000)] public float waveformGain = 1000.0f;
    [Range(5, 100)] public int debugRayCount = 100;

    ComputeBuffer wallBuffer, hitBuffer, debugBuffer, irBuffer, argsBuffer;
    Vector4[] debugRayPaths;
    RenderTexture irTexture;
    List<Segment> activeSegments;
    int accumFrames = 0;
    float streamingTimer = 0;
    int nextStreamingOffset = 0;

    struct RayInfo { public float timeDelay, energy; public Vector2 hitPoint; };

    void Start()
    {
        Assert.IsTrue(sampleRate == inputClip.frequency, $"SampleRate ({sampleRate}) != input frequency ({inputClip.frequency})");
        Assert.IsNotNull(audioManager);
        UpdateGeometry();
        if (irTexture == null)
        {
            irTexture = new RenderTexture(1024, 256, 0);
            irTexture.enableRandomWrite = true;
            irTexture.filterMode = FilterMode.Point;
            irTexture.Create();
        }

        int irLength = (int)(sampleRate * reverbDuration);
        ComputeHelper.CreateStructuredBuffer<float>(ref irBuffer, irLength);
        
        int kClear = shader.FindKernel("ClearImpulse");
        shader.SetInt("ImpulseLength", irLength);
        shader.SetBuffer(kClear, "ImpulseResponse", irBuffer);
        shader.SetFloat("inputGain", inputGain);
        ComputeHelper.Dispatch(shader, irLength, 1, 1, kClear);
    }

    void Update()
    {
        if (!source || !listener || !shader) return;
        if (dynamicObstacles) UpdateGeometry();

        RunSimulation();

        if (audioManager.IsStreaming)
        {
            audioManager.currentAccumCount = accumFrames;
            streamingTimer += Time.deltaTime;

            // Use a small lookahead to ensure we start processing early enough to get results
            float lookaheadThreshold = Mathf.Max(0, audioManager.chunkDuration - lookaheadSeconds);

            if (streamingTimer >= lookaheadThreshold)
            {
                if (nextStreamingOffset >= inputClip.samples)
                {
                    audioManager.StopStreaming();
                }
                else
                {
                    audioManager.ProcessNextChunk(nextStreamingOffset, (int)(sampleRate * audioManager.chunkDuration));
                    nextStreamingOffset += (int)(sampleRate * audioManager.chunkDuration);
                    ResetIR();
                    streamingTimer -= audioManager.chunkDuration;
                }
            }
        }

        if (Input.GetKeyDown(KeyCode.Space) && audioManager)
        {
            if (audioManager.IsStreaming) audioManager.StopStreaming();
            else StartStreaming();
        }
        if (Input.GetKeyDown(KeyCode.R)) { ResetIR(); audioManager?.StopStreaming(); }
    }

    void StartStreaming()
    {
        nextStreamingOffset = 0;
        streamingTimer = 0;
        ResetIR();
        audioManager.StartStreaming(inputClip, irBuffer, sampleRate);
    }

    void ResetIR()
    {
        accumFrames = 0;
        int len = (int)(sampleRate * reverbDuration);
        int k = shader.FindKernel("ClearImpulse");
        shader.SetInt("ImpulseLength", len);
        shader.SetBuffer(k, "ImpulseResponse", irBuffer);
        ComputeHelper.Dispatch(shader, len, 1, 1, k);
    }

    void RunSimulation()
    {
        int len = (int)(sampleRate * reverbDuration);
        argsBuffer = new ComputeBuffer(1, sizeof(int) * 4, ComputeBufferType.IndirectArguments);

        int k = shader.FindKernel("Trace");
        hitBuffer.SetCounterValue(0);
        shader.SetVector("sourcePos", source.position);
        shader.SetVector("listenerPos", listener.position);
        shader.SetFloat("listenerRadius", listenerRadius);
        shader.SetFloat("speedOfSound", speedOfSound);
        shader.SetFloat("inputGain", inputGain);
        shader.SetInt("maxBounceCount", maxBounces);
        shader.SetInt("rngStateOffset", Time.frameCount);
        shader.SetInt("numWalls", activeSegments.Count);
        shader.SetInt("rayCount", rayCount);
        shader.SetInt("debugRayCount", debugRayCount);
        shader.SetInt("accumFrames", accumFrames);
        shader.SetBuffer(k, "walls", wallBuffer);
        shader.SetBuffer(k, "rayInfoBuffer", hitBuffer);
        shader.SetBuffer(k, "debugRays", debugBuffer);
        ComputeHelper.Dispatch(shader, rayCount, 1, 1, k);

        debugRayPaths = ComputeHelper.ReadbackData<Vector4>(debugBuffer);
        ComputeBuffer.CopyCount(hitBuffer, argsBuffer, 0);
        int[] args = new int[4]; argsBuffer.GetData(args);
        int hitCount = args[0];

        accumFrames++;
        shader.SetInt("accumCount", accumFrames);

        if (hitCount > 0)
        {
            int kp = shader.FindKernel("ProcessHits");
            shader.SetInt("SampleRate", sampleRate);
            shader.SetInt("ImpulseLength", len);
            shader.SetInt("HitCount", hitCount);
            shader.SetBuffer(kp, "RawHits", hitBuffer);
            shader.SetBuffer(kp, "ImpulseResponse", irBuffer);
            ComputeHelper.Dispatch(shader, hitCount, 1, 1, kp);
        }

        int kd = shader.FindKernel("DrawIR");
        shader.SetTexture(kd, "DebugTexture", irTexture);
        shader.SetBuffer(kd, "ImpulseResponse", irBuffer);
        shader.SetInt("TexWidth", irTexture.width);
        shader.SetInt("TexHeight", irTexture.height);
        shader.SetInt("ImpulseLength", len);
        shader.SetFloat("DebugGain", waveformGain);
        ComputeHelper.Dispatch(shader, irTexture.width, irTexture.height, 1, kd);
    }

    void UpdateGeometry()
    {
        activeSegments = SceneToData2D.GetSegmentsFromColliders(obstacleObjects);
        ComputeHelper.CreateStructuredBuffer(ref wallBuffer, activeSegments);
    }

    void OnGUI()
    {
        if (showDebugTexture && irTexture)
            GUI.DrawTexture(new Rect(10, 10, debugTextureSize.x, debugTextureSize.y), irTexture);
    }

    void OnDrawGizmos()
    {
        if (!source || !listener) return;
        Gizmos.color = Color.green; Gizmos.DrawWireSphere(source.position, 0.2f);
        Gizmos.color = Color.cyan; Gizmos.DrawWireSphere(listener.position, listenerRadius);
        if (activeSegments != null) { Gizmos.color = Color.red; foreach (var seg in activeSegments) Gizmos.DrawLine(seg.start, seg.end); }
        if (debugRayPaths != null)
        {
            int stride = maxBounces + 1; float z = source.position.z - 5.0f;
            for (int i = 0; i < debugRayCount; i++)
            {
                for (int b = 0; b < maxBounces - 1; b++)
                {
                    Vector4 p1 = debugRayPaths[i * stride + b], p2 = debugRayPaths[i * stride + b + 1];
                    if (p2.sqrMagnitude == 0) break;
                    Gizmos.color = Color.Lerp(new Color(1, 0.5f, 0, 0.1f), new Color(1, 1, 0, 0.8f), p1.z);
                    Gizmos.DrawLine(new Vector3(p1.x, p1.y, z), new Vector3(p2.x, p2.y, z));
                }
            }
        }
    }

    void OnDestroy()
    {
        ComputeHelper.Release(wallBuffer, hitBuffer, debugBuffer, irBuffer, argsBuffer); irTexture?.Release();
    }
}