using UnityEngine;
using System.Collections.Generic;
using System.Runtime.InteropServices;


public struct AudioMat
{
    public Vector4 absorptionLow;
    public Vector4 absorptionHigh;
    public Vector4 scatteringLow;
    public Vector4 scatteringHigh;
    public Vector4 transmissionLow;
    public Vector4 transmissionHigh;
    public Vector4 iorLow;
    public Vector4 iorHigh;
    public Vector4 dampingLow;
    public Vector4 dampingHigh;
}
[StructLayout(LayoutKind.Sequential)]
public struct Segment
{
    public Vector2 start;
    public Vector2 end;
    public Vector2 normal;
    public AudioMat mat;
}

public static class SceneToData2D
{
    private const int CIRCLE_RESOLUTION = 32;
    public static List<Segment> GetSegmentsFromColliders(List<GameObject> objects)
    {
        List<Segment> allSegments = new List<Segment>();
        if (objects == null) return allSegments;

        foreach (var obj in objects)
        {
            if (obj == null) continue;

            AudioMat mat = ResolveMaterial(obj);
            Collider2D col = obj.GetComponent<Collider2D>();
            if (col != null && col.enabled)
            {
                AddCollider2D(col, allSegments, mat);
                continue;
            }

            Collider col3D = obj.GetComponent<Collider>();
            if (col3D != null && col3D.enabled)
            {
                AddCollider3D(col3D, allSegments, mat);
                continue;
            }

            MeshFilter meshFilter = obj.GetComponent<MeshFilter>();
            if (meshFilter != null && meshFilter.sharedMesh != null)
            {
                AddMesh3DLongSide(meshFilter, allSegments, mat);
                continue;
            }

            Debug.LogWarning($"{obj.name} has no supported 2D collider, 3D collider, or mesh.");
        }
        return allSegments;
    }

    private static void AddCollider2D(Collider2D col, List<Segment> allSegments, AudioMat mat)
    {
        if (col is PolygonCollider2D poly)
        {
            for (int i = 0; i < poly.pathCount; i++)
                AddLoopToSegments(col.transform, poly.GetPath(i), allSegments, mat);
        }
        else if (col is BoxCollider2D box)
        {
            Vector2 h = box.size * 0.5f;
            Vector2 o = box.offset;
            AddLoopToSegments(col.transform, new[] {
                o + new Vector2(-h.x, -h.y),
                o + new Vector2(h.x, -h.y),
                o + new Vector2(h.x, h.y),
                o + new Vector2(-h.x, h.y)
            }, allSegments, mat);
        }
        else if (col is CircleCollider2D circle)
        {
            Vector2[] points = new Vector2[CIRCLE_RESOLUTION];
            for (int i = 0; i < CIRCLE_RESOLUTION; i++)
            {
                float angle = (i / (float)CIRCLE_RESOLUTION) * Mathf.PI * 2;
                points[i] = circle.offset + new Vector2(Mathf.Cos(angle), Mathf.Sin(angle)) * circle.radius;
            }
            AddLoopToSegments(col.transform, points, allSegments, mat);
        }
        else
        {
            Debug.LogWarning($"{col.GetType()} collider is not supported yet.");
        }
    }

    private static void AddCollider3D(Collider col, List<Segment> allSegments, AudioMat mat)
    {
        if (col is BoxCollider box)
        {
            AddBoxCollider3DLongSide(box, allSegments, mat);
            return;
        }

        Debug.LogWarning($"{col.GetType()} collider is not supported for acoustic 2D projection yet.");
    }

    private static void AddBoxCollider3DLongSide(BoxCollider box, List<Segment> allSegments, AudioMat mat)
    {
        Transform trans = box.transform;
        Vector3 center = trans.TransformPoint(box.center);
        Vector3 right = trans.TransformVector(Vector3.right * box.size.x);
        Vector3 up = trans.TransformVector(Vector3.up * box.size.y);
        Vector3 forward = trans.TransformVector(Vector3.forward * box.size.z);

        AddSegmentFromProjectedAxis(box.name, center, right, up, forward, allSegments, mat);
    }

    private static void AddMesh3DLongSide(MeshFilter meshFilter, List<Segment> allSegments, AudioMat mat)
    {
        Bounds bounds = meshFilter.sharedMesh.bounds;
        Transform trans = meshFilter.transform;
        Vector3 center = trans.TransformPoint(bounds.center);
        Vector3 right = trans.TransformVector(Vector3.right * bounds.size.x);
        Vector3 up = trans.TransformVector(Vector3.up * bounds.size.y);
        Vector3 forward = trans.TransformVector(Vector3.forward * bounds.size.z);

        AddSegmentFromProjectedAxis(meshFilter.name, center, right, up, forward, allSegments, mat);
    }

    private static void AddSegmentFromProjectedAxis(
        string objectName,
        Vector3 center,
        Vector3 right,
        Vector3 up,
        Vector3 forward,
        List<Segment> allSegments,
        AudioMat mat)
    {
        Vector2 center2D = new Vector2(center.x, center.y);
        Vector2 longAxis = LongestProjectedAxis(right, up, forward);
        if (longAxis.sqrMagnitude < 1e-6f)
        {
            Debug.LogWarning($"{objectName} projects to a near-zero 2D wall length.");
            return;
        }

        Segment seg = new Segment();
        Vector2 half = longAxis * 0.5f;
        seg.start = center2D - half;
        seg.end = center2D + half;

        Vector2 dir = longAxis.normalized;
        seg.normal = new Vector2(dir.y, -dir.x);
        seg.mat = mat;
        allSegments.Add(seg);
    }

    private static Vector2 LongestProjectedAxis(Vector3 a, Vector3 b, Vector3 c)
    {
        Vector2 axisA = new Vector2(a.x, a.y);
        Vector2 axisB = new Vector2(b.x, b.y);
        Vector2 axisC = new Vector2(c.x, c.y);

        Vector2 longest = axisA;
        if (axisB.sqrMagnitude > longest.sqrMagnitude) longest = axisB;
        if (axisC.sqrMagnitude > longest.sqrMagnitude) longest = axisC;
        return longest;
    }

    private static void AddLoopToSegments(Transform trans, Vector2[] localPoints, List<Segment> outSegments, AudioMat material)
    {
        Vector3 s = trans.lossyScale;
        float winding = Mathf.Sign(s.x * s.y);

        for (int i = 0; i < localPoints.Length; i++)
        {
            Vector2 p1 = localPoints[i];
            Vector2 p2 = localPoints[(i + 1) % localPoints.Length];

            Segment seg = new Segment();
            seg.start = trans.TransformPoint(p1);
            seg.end = trans.TransformPoint(p2);

            Vector2 dir = (seg.end - seg.start).normalized;
            seg.normal = new Vector2(dir.y, -dir.x) * winding;
            seg.mat = material;

            outSegments.Add(seg);
        }
    }
    private static AudioMat ResolveMaterial(GameObject obj)
    {
        AcousticSurface surface = obj.GetComponent<AcousticSurface>();
        if (surface == null || surface.material == null)
        {
            Debug.LogWarning($"{obj.name} has no AcousticSurface material. Using default concrete-like material.");
            return new AudioMat
            {
                absorptionLow = new Vector4(0.01f, 0.01f, 0.015f, 0.02f),
                absorptionHigh = new Vector4(0.02f, 0.03f, 0.04f, 0.05f),
                scatteringLow = new Vector4(0.2f, 0.25f, 0.3f, 0.35f),
                scatteringHigh = new Vector4(0.4f, 0.45f, 0.5f, 0.55f),
                transmissionLow = new Vector4(0.03f, 0.02f, 0.015f, 0.01f),
                transmissionHigh = new Vector4(0.008f, 0.006f, 0.004f, 0.002f),
                iorLow = new Vector4(343f / 5405f, 343f / 5405f, 343f / 5405f, 343f / 5405f),
                iorHigh = new Vector4(343f / 5405f, 343f / 5405f, 343f / 5405f, 343f / 5405f),
                dampingLow = new Vector4(0.02f, 0.025f, 0.03f, 0.035f),
                dampingHigh = new Vector4(0.04f, 0.05f, 0.065f, 0.08f)
            };
        }

        AudioMatData data = surface.material.GetShaderData();
        return new AudioMat {
            absorptionLow = data.absorptionLow,
            absorptionHigh = data.absorptionHigh,
            scatteringLow = data.scatteringLow,
            scatteringHigh = data.scatteringHigh,
            transmissionLow = data.transmissionLow,
            transmissionHigh = data.transmissionHigh,
            iorLow = data.iorLow,
            iorHigh = data.iorHigh,
            dampingLow = data.dampingLow,
            dampingHigh = data.dampingHigh
        };
    }
}
