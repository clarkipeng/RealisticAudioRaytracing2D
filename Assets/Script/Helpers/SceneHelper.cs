using UnityEngine;
using UnityEngine.Experimental.Rendering;
using System.Collections.Generic;
using System;
using System.Runtime.InteropServices;


public struct AudioMat
{
    public float absorption;
    public float scattering;
    public float transmission;
    public float ior;
    public float damping;
    public float padding;
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
    private const int CAPSULE_RESOLUTION = 16;

    public static List<Segment> GetSegmentsFromColliders(List<GameObject> objects)
    {
        List<Segment> allSegments = new List<Segment>();

        foreach (var obj in objects)
        {
            Collider2D col = obj.GetComponent<Collider2D>();
            if (col == null || !col.enabled) continue;

            List<Vector2> worldPoints = new List<Vector2>();
            AudioMat mat = ResolveMaterial(obj);

            if (col is PolygonCollider2D poly)
            {
                for (int i = 0; i < poly.pathCount; i++)
                {
                    Vector2[] path = poly.GetPath(i);
                    AddLoopToSegments(obj.transform, path, allSegments, mat);
                }
            }
            else if (col is BoxCollider2D box)
            {
                Vector2 h = box.size * 0.5f;
                Vector2 o = box.offset;
                worldPoints.Add(o + new Vector2(-h.x, -h.y));
                worldPoints.Add(o + new Vector2(h.x, -h.y));
                worldPoints.Add(o + new Vector2(h.x, h.y));
                worldPoints.Add(o + new Vector2(-h.x, h.y));
                AddLoopToSegments(obj.transform, worldPoints.ToArray(), allSegments, mat);
            }
            else if (col is CircleCollider2D circle)
            {
                float radius = circle.radius;
                Vector2 offset = circle.offset;
                for (int i = 0; i < CIRCLE_RESOLUTION; i++)
                {
                    float angle = (i / (float)CIRCLE_RESOLUTION) * Mathf.PI * 2;
                    worldPoints.Add(offset + new Vector2(Mathf.Cos(angle), Mathf.Sin(angle)) * radius);
                }
                AddLoopToSegments(obj.transform, worldPoints.ToArray(), allSegments, mat);
            }
            else
            {
                Debug.Log(col.GetType() + "COLLIDER NOT SUPPORTED YET");
            }
        }
        return allSegments;
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
        //default material can be added
        AcousticSurface surface = obj.GetComponent<AcousticSurface>();
        AudioMaterial mat = surface.material;
        AudioMatData data = mat.GetShaderData();
        AudioMat ret = new AudioMat();
        ret.absorption = data.absorption;
        ret.scattering = data.scattering;
        ret.transmission = data.transmission;
        ret.ior = data.ior;
        ret.damping = data.damping;
        return ret;
    }
}