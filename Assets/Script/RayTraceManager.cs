using UnityEngine;
using System.Collections.Generic;
using Helpers;
using UnityEngine.Assertions;
using UnityEngine.Rendering;

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
    public bool loop = true;

    [Header("Accum Settings")]
    [Range(0.1f, 5.0f)] public float reverbDuration = 5f;
    [Range(0f, 1f)] public float lookaheadSeconds = 0.1f;

    [Header("Scene")]
    public Transform source, listener;
    [Range(0.1f, 5f)] public float listenerRadius = 0.5f;
    public List<GameObject> obstacleObjects;

    [Header("Debug")]
    public bool showDebugTexture = true;
    public Vector2 debugTextureSize = new Vector2(1024, 512);
    [Range(1, 5000)] public float waveformGain = 1000.0f;
    [Range(5, 100)] public int debugRayCount = 100;

    ComputeBuffer wallBuffer, hitBuffer, debugBuffer, irBufferPing, irBufferPong, argsBuffer;
    int activeIRIndex = 0;
    Vector4[] debugRayPaths;
    RenderTexture irTexture;
    List<Segment> activeSegments;
    int accumFrames = 0;
    int samplesSinceLastChunk = 0;
    int chunkSamples; // Exact samples per chunk
    int nextStreamingOffset = 0;

    struct RayInfo { public float timeDelay, energy; public Vector2 hitPoint; };

    void Start()
    {
        Assert.IsTrue(sampleRate == inputClip.frequency, $"SampleRate ({sampleRate}) != input frequency ({inputClip.frequency})");
        Assert.IsNotNull(audioManager);
        UpdateGeometry();
    }

    void Update()
    {
        if (!source || !listener || !shader) return;
        RunSimulation();

        if (Input.GetKeyDown(KeyCode.Space) && audioManager)
        {
            if (audioManager.IsStreaming) audioManager.StopStreaming();
            else StartStreaming();
        }
        if (Input.GetKeyDown(KeyCode.R)) { ResetIR(); audioManager?.StopStreaming(); }
    }

    void FixedUpdate()
    {
        if (!audioManager || !shader) return;
        if (dynamicObstacles) UpdateGeometry();

        if (audioManager.IsStreaming)
        {
            audioManager.currentAccumCount = accumFrames;
            
            // Convert fixed delta to samples (exact)
            int samplesThisFrame = Mathf.RoundToInt(Time.fixedDeltaTime * sampleRate);
            samplesSinceLastChunk += samplesThisFrame;

            // Dispatch when we've accumulated exactly one chunk worth of samples
            if (samplesSinceLastChunk >= chunkSamples)
            {
                if (nextStreamingOffset >= inputClip.samples)
                {
                    if (loop)
                        nextStreamingOffset = 0;
                    else
                        audioManager.StopStreaming();
                }
                else
                {
                    ComputeBuffer frozenBuffer = GetActiveIRBuffer();
                    audioManager.ProcessNextChunk(nextStreamingOffset, chunkSamples, Mathf.Max(1, accumFrames), frozenBuffer);
                    
                    activeIRIndex = 1 - activeIRIndex;
                    nextStreamingOffset += chunkSamples;
                    ResetIR();
                    samplesSinceLastChunk -= chunkSamples;
                }
            }
        }
    }

    void StartStreaming()
    {
        nextStreamingOffset = 0;
        samplesSinceLastChunk = 0;
        chunkSamples = Mathf.RoundToInt(sampleRate * audioManager.chunkDuration); // Calculate once, exact
        ResetIR();
        audioManager.StartStreaming(inputClip, GetActiveIRBuffer(), sampleRate);
    }

    void ResetIR()
    {
        accumFrames = 0;
        int len = (int)(sampleRate * reverbDuration);
        ComputeBuffer activeBuffer = GetActiveIRBuffer();

        int k = shader.FindKernel("ClearImpulse");
        shader.SetInt("ImpulseLength", len);
        shader.SetBuffer(k, "ImpulseResponse", activeBuffer);
        ComputeHelper.Dispatch(shader, len, 1, 1, k);
    }

    void RunSimulation()
    {
        int irLength = (int)(sampleRate * reverbDuration);

        // Lazy initialization / Hot-reload safety
        ComputeHelper.CreateStructuredBuffer<Vector4>(ref debugBuffer, debugRayCount * (maxBounces + 1));
        ComputeHelper.CreateAppendBuffer<RayInfo>(ref hitBuffer, rayCount * maxBounces);
        ComputeBuffer activeBuffer = GetActiveIRBuffer();

        if (wallBuffer == null || !wallBuffer.IsValid()) UpdateGeometry();
        if (argsBuffer == null || !argsBuffer.IsValid()) argsBuffer = new ComputeBuffer(1, sizeof(int) * 4, ComputeBufferType.IndirectArguments);
        if (irTexture == null)
        {
            irTexture = new RenderTexture(1024, 256, 0);
            irTexture.enableRandomWrite = true;
            irTexture.filterMode = FilterMode.Point;
            irTexture.Create();
        }

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

        AsyncGPUReadback.Request(debugBuffer, (request) => {
            if (request.hasError) return;
            debugRayPaths = request.GetData<Vector4>().ToArray();
        });

        ComputeBuffer.CopyCount(hitBuffer, argsBuffer, 0);
        AsyncGPUReadback.Request(argsBuffer, (request) => {
            if (request.hasError) return;
            int[] args = request.GetData<int>().ToArray();
            int hitCount = args[0];
            OnSimulationFinished(hitCount, irLength);
        });
    }

    ComputeBuffer GetActiveIRBuffer() {
        int len = (int)(sampleRate * reverbDuration);
        ComputeHelper.CreateStructuredBuffer<float>(ref irBufferPing, len);
        ComputeHelper.CreateStructuredBuffer<float>(ref irBufferPong, len);
        return (activeIRIndex == 0) ? irBufferPing : irBufferPong;
    }

    void OnSimulationFinished(int hitCount, int irLength)
    {
        if (hitCount > 0)
        {
            int kp = shader.FindKernel("ProcessHits");
            shader.SetInt("SampleRate", sampleRate);
            shader.SetInt("HitCount", hitCount);
            shader.SetBuffer(kp, "RawHits", hitBuffer);
            shader.SetBuffer(kp, "ImpulseResponse", GetActiveIRBuffer());
            ComputeHelper.Dispatch(shader, hitCount, 1, 1, kp);
        }

        accumFrames++;

        shader.SetInt("accumCount", accumFrames);
        shader.SetInt("ImpulseLength", irLength);

        int kd = shader.FindKernel("DrawIR");
        shader.SetTexture(kd, "DebugTexture", irTexture);
        shader.SetBuffer(kd, "ImpulseResponse", GetActiveIRBuffer());
        shader.SetInt("TexWidth", irTexture.width);
        shader.SetInt("TexHeight", irTexture.height);
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
            if (debugRayPaths.Length < debugRayCount * stride) return;

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
        ComputeHelper.Release(wallBuffer, hitBuffer, debugBuffer, irBufferPing, irBufferPong, argsBuffer); irTexture?.Release();
    }
}