using UnityEngine;
using UnityEditor;
using System.Collections.Generic;
using Seb.Helpers;

public class Acoustic2D : MonoBehaviour
{
    [Header("Simulation Settings")]
    public ComputeShader shader;
    [Range(10, 5000)] public int rayCount = 1000;
    [Range(5, 100)] public int debugLines = 100;
    [Range(1, 10)] public int maxBounces = 5;
    public float speedOfSound = 343f;
    public bool dynamicObstacles = false;

    [Header("Scene Objects")]
    public Transform source;
    public Transform listener;
    [Range(0.1f, 5f)] public float listenerRadius = 0.5f;
    public List<GameObject> obstacleObjects;

    ComputeBuffer wallBuffer;
    ComputeBuffer hitBuffer;
    ComputeBuffer debugRayBuffer;

    Vector4[] debugRayPaths;
    List<Segment> activeSegments;

    struct AcousticHit { public float timeDelay; public float energy; public Vector2 hitPoint; };

    void Start()
    {
        UpdateGeometry();
    }

    void Update()
    {
        if (source == null || listener == null || shader == null) return;

        if (dynamicObstacles) UpdateGeometry();

        RunSimulation();
    }

    void UpdateGeometry()
    {
        activeSegments = SceneToData2D.GetSegmentsFromColliders(obstacleObjects);

        ComputeHelper.CreateStructuredBuffer(ref wallBuffer, activeSegments);
    }

    void RunSimulation()
    {
        ComputeHelper.CreateAppendBuffer<AcousticHit>(ref hitBuffer, rayCount * maxBounces);
        ComputeHelper.CreateStructuredBuffer<Vector4>(ref debugRayBuffer, debugLines * (maxBounces + 1));

        int kernel = shader.FindKernel("CSMain");
        hitBuffer.SetCounterValue(0);

        shader.SetVector("sourcePos", source.position);
        shader.SetVector("listenerPos", listener.position);
        shader.SetFloat("listenerRadius", listenerRadius);
        shader.SetFloat("speedOfSound", speedOfSound);
        shader.SetInt("maxBounceCount", maxBounces);
        shader.SetInt("rngStateOffset", Time.frameCount);
        shader.SetInt("numWalls", activeSegments.Count);

        shader.SetBuffer(kernel, "walls", wallBuffer);
        shader.SetBuffer(kernel, "hits", hitBuffer);
        shader.SetBuffer(kernel, "debugRays", debugRayBuffer);

        ComputeHelper.Dispatch(shader, rayCount, 1, 1, kernel);

        debugRayPaths = ComputeHelper.ReadbackData<Vector4>(debugRayBuffer);
    }

    void OnDrawGizmos()
    {
        if (source == null || listener == null) return;

        // Draw Source & Listener
        Gizmos.color = Color.green; Gizmos.DrawWireSphere(source.position, 0.2f);
        Gizmos.color = Color.cyan; Gizmos.DrawWireSphere(listener.position, listenerRadius);

        // Draw Walls (Visualizing what the Ray Tracer actually sees)
        if (activeSegments != null)
        {
            Gizmos.color = Color.red;
            foreach (var seg in activeSegments)
            {
                Gizmos.DrawLine(seg.start, seg.end);
                Gizmos.DrawLine((seg.start + seg.end) * 0.5f, (seg.start + seg.end) * 0.5f + seg.normal * 0.2f);
            }
        }

        // #if UNITY_EDITOR
        if (debugRayPaths != null)
        {
            int stride = maxBounces + 1;
            float z = source.position.z;

            for (int i = 0; i < debugLines; i++)
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
                    // Handles.color = col;
                    // Handles.DrawAAPolyLine(width, start, end);
                }
            }
        }
        // #endif
    }

    void OnDestroy()
    {
        ComputeHelper.Release(wallBuffer, hitBuffer, debugRayBuffer);
    }
}