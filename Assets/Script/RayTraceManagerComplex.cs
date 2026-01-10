using UnityEngine;
using UnityEditor;
using System.Collections.Generic;
using Helpers;
using UnityEngine.Assertions;


public class RayTraceManagerComplex : MonoBehaviour
{
    [Header("Simulation Settings")]
    public ComputeShader shader;
    [Range(10, 100000)] public int rayCount = 1000;
    [Range(5, 100)] public int debugRayCount = 100;
    [Range(1, 10)] public int maxBounces = 5;
    public float speedOfSound = 343f;
    public bool dynamicObstacles = false;

    [Header("Accum Settings")]
    public int sampleRate = 44100;
    ComputeBuffer argsBuffer;

    [Header("Audio Settings")]
    public AudioClip inputClip;
    public float inputGain = 1.0f;
    public bool bakeOnSpace = true;
    [Range(0.1f, 5.0f)] public float reverbDuration = 0.5f;

    [Header("Scene Objects")]
    public Transform source;
    public Transform listener;
    [Range(0.1f, 5f)] public float listenerRadius = 0.5f;
    public List<GameObject> obstacleObjects;


    [Header("Debug Visualization")]
    public bool showDebugTexture = true;
    public Vector2 debugTextureSize = new Vector2(512, 128);
    [Range(1, 100)] public float waveformGain = 50.0f;

    ComputeBuffer wallBuffer;
    ComputeBuffer hitBuffer;

    ComputeBuffer debugBuffer;
    ComputeBuffer irBuffer;

    ComputeBuffer inputAudioBuffer;
    ComputeBuffer outputAudioBuffer;

    Vector4[] debugRayPaths;
    RenderTexture irTexture;

    List<Segment> activeSegments;
    private int accumFrames = 0;

    struct RayInfo { public float timeDelay; public float energy; public Vector2 hitPoint; };

    void Start()
    {
        Assert.IsTrue(sampleRate == inputClip.frequency);
        UpdateGeometry();
        if (irTexture == null)
        {
            irTexture = new RenderTexture(1024, 256, 0);
            irTexture.enableRandomWrite = true;
            irTexture.filterMode = FilterMode.Point;
            irTexture.Create();
        }
    }

    void Update()
    {
        if (source == null || listener == null || shader == null) return;

        if (dynamicObstacles) UpdateGeometry();

        RunSimulation();
        if (bakeOnSpace && Input.GetKeyDown(KeyCode.Space))
        {
            BakeAudio();
        }
        if (Input.GetKeyDown(KeyCode.R))
        {
            accumFrames = 0;
            int irLength = (int)(sampleRate * reverbDuration);
            int kClear = shader.FindKernel("ClearImpulse");
            shader.SetInt("ImpulseLength", irLength);
            shader.SetBuffer(kClear, "ImpulseResponse", irBuffer);
            shader.SetFloat("inputGain", inputGain);
            ComputeHelper.Dispatch(shader, irLength, 1, 1, kClear);
        }
    }

    void RunSimulation()
    {
        int irLength = (int)(sampleRate * reverbDuration);

        ComputeHelper.CreateStructuredBuffer<Vector4>(ref debugBuffer, debugRayCount * (maxBounces + 1));
        ComputeHelper.CreateStructuredBuffer<float>(ref irBuffer, irLength);
        ComputeHelper.CreateAppendBuffer<RayInfo>(ref hitBuffer, rayCount * maxBounces);
        if (argsBuffer == null)
            argsBuffer = new ComputeBuffer(1, sizeof(int) * 4, ComputeBufferType.IndirectArguments);


        int kernel = shader.FindKernel("Trace");
        hitBuffer.SetCounterValue(0);

        shader.SetVector("sourcePos", new Vector2(source.position.x, source.position.y));
        shader.SetVector("listenerPos", new Vector2(listener.position.x, listener.position.y));
        shader.SetFloat("listenerRadius", listenerRadius);
        shader.SetFloat("speedOfSound", speedOfSound);
        shader.SetInt("maxBounceCount", maxBounces);
        shader.SetInt("rngStateOffset", Time.frameCount);
        shader.SetInt("numWalls", activeSegments.Count);
        shader.SetInt("rayCount", rayCount);
        shader.SetInt("debugRayCount", debugRayCount);
        shader.SetInt("accumFrames", accumFrames);

        shader.SetBuffer(kernel, "walls", wallBuffer);
        shader.SetBuffer(kernel, "rayInfoBuffer", hitBuffer);
        shader.SetBuffer(kernel, "debugRays", debugBuffer);

        ComputeHelper.Dispatch(shader, rayCount, 1, 1, kernel);

        debugRayPaths = ComputeHelper.ReadbackData<Vector4>(debugBuffer);

        ComputeBuffer.CopyCount(hitBuffer, argsBuffer, 0);

        int[] args = new int[4];
        argsBuffer.GetData(args);
        int hitCount = args[0];

        accumFrames++;
        shader.SetInt("accumCount", accumFrames);

        if (hitCount > 0)
        {
            int kProc = shader.FindKernel("ProcessHits");
            shader.SetInt("SampleRate", sampleRate);
            shader.SetInt("ImpulseLength", irLength);
            shader.SetInt("HitCount", hitCount);
            shader.SetBuffer(kProc, "RawHits", hitBuffer);
            shader.SetBuffer(kProc, "ImpulseResponse", irBuffer);
            ComputeHelper.Dispatch(shader, hitCount, 1, 1, kProc);
        }

        int kDraw = shader.FindKernel("DrawIR");
        shader.SetTexture(kDraw, "DebugTexture", irTexture);
        shader.SetBuffer(kDraw, "ImpulseResponse", irBuffer);
        shader.SetInt("TexWidth", irTexture.width);
        shader.SetInt("TexHeight", irTexture.height);
        shader.SetInt("ImpulseLength", irLength);
        shader.SetFloat("DebugGain", waveformGain);

        ComputeHelper.Dispatch(shader, irTexture.width, irTexture.height, 1, kDraw);
    }

    void BakeAudio()
    {
        if (inputClip == null) { Debug.LogError("Assign an Input Clip!"); return; }

        Debug.Log("Convolving on GPU...");

        float[] rawSamples = new float[inputClip.samples * inputClip.channels];
        inputClip.GetData(rawSamples, 0);

        float[] monoSamples = new float[inputClip.samples];
        int channels = inputClip.channels;
        for (int i = 0; i < inputClip.samples; i++)
        {
            float sum = 0;
            for (int c = 0; c < channels; c++)
            {
                sum += rawSamples[i * channels + c];
            }
            monoSamples[i] = sum / channels;
        }

        ComputeHelper.CreateStructuredBuffer<float>(ref inputAudioBuffer, monoSamples.Length);
        inputAudioBuffer.SetData(monoSamples);

        int irLen = irBuffer.count;
        int outputLen = monoSamples.Length + irLen;
        ComputeHelper.CreateStructuredBuffer<float>(ref outputAudioBuffer, outputLen);

        int kConv = shader.FindKernel("AudioConvolve");
        shader.SetInt("InputLength", monoSamples.Length);
        shader.SetInt("IRLength", irLen);

        shader.SetBuffer(kConv, "InputAudio", inputAudioBuffer);
        shader.SetBuffer(kConv, "ImpulseResponse", irBuffer);
        shader.SetBuffer(kConv, "OutputAudio", outputAudioBuffer);

        ComputeHelper.Dispatch(shader, outputLen, 1, 1, kConv);

        float[] resultData = new float[outputLen];
        outputAudioBuffer.GetData(resultData);

        PlayResult(resultData);
    }
    void PlayResult(float[] data)
    {
        float maxVol = 0f;
        foreach (float f in data) if (Mathf.Abs(f) > maxVol) maxVol = Mathf.Abs(f);
        if (maxVol > 0.0001f)
        {
            float scaler = 1.0f / maxVol;
            for (int i = 0; i < data.Length; i++) data[i] *= scaler;
        }

        AudioClip result = AudioClip.Create("ReverbResult", data.Length, 1, sampleRate, false);
        result.SetData(data, 0);

        AudioSource source = GetComponent<AudioSource>();
        source.clip = result;
        source.Play();
        Debug.Log("Playing Result!");
    }

    void UpdateGeometry()
    {
        activeSegments = SceneToData2D.GetSegmentsFromColliders(obstacleObjects);
        Assert.IsTrue(activeSegments.Count != 0, "MUST HAVE OBJECTS IN SCENE");
        ComputeHelper.CreateStructuredBuffer(ref wallBuffer, activeSegments);
    }

    void OnGUI()
    {
        if (showDebugTexture && irTexture != null)
        {
            GUI.DrawTexture(new Rect(10, 10, debugTextureSize[0], debugTextureSize[1]), irTexture);
        }
    }

    void OnDrawGizmos()
    {
        if (source == null || listener == null) return;

        Gizmos.color = Color.green; Gizmos.DrawWireSphere(source.position, 0.2f);
        Gizmos.color = Color.cyan; Gizmos.DrawWireSphere(listener.position, listenerRadius);

        if (activeSegments != null)
        {
            Gizmos.color = Color.red;
            foreach (var seg in activeSegments)
            {
                Gizmos.DrawLine(seg.start, seg.end);
                Gizmos.DrawLine((seg.start + seg.end) * 0.5f, (seg.start + seg.end) * 0.5f + seg.normal * 0.2f);
            }
        }

        if (debugRayPaths != null)
        {
            int stride = maxBounces + 1;
            float z = source.position.z - 5.0f;

            for (int i = 0; i < debugRayCount; i++)
            {
                for (int b = 0; b < maxBounces; b++)
                {
                    if (i * stride + b >= debugRayPaths.Length)
                    {
                        break;
                    }
                    Vector4 p1 = debugRayPaths[i * stride + b];
                    Vector4 p2 = debugRayPaths[i * stride + b + 1];

                    if (p2.sqrMagnitude == 0) break;

                    Vector3 start = new Vector3(p1.x, p1.y, z);
                    Vector3 end = new Vector3(p2.x, p2.y, z);

                    float energy = p1.z;
                    float width = Mathf.Lerp(0.5f, 5.0f, energy);
                    Color col = Color.Lerp(new Color(1, 0.5f, 0, 0.1f), new Color(1, 1, 0, 0.8f), energy);

                    Gizmos.color = col;
                    Gizmos.DrawLine(start, end);
                }
            }
        }
    }

    void OnDestroy()
    {
        ComputeHelper.Release(wallBuffer, hitBuffer, debugBuffer, irBuffer, argsBuffer, inputAudioBuffer, outputAudioBuffer);
        if (irTexture != null) irTexture.Release();
    }
}