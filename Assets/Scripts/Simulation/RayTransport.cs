using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using UnityEngine;

// Shared layouts and geometry preparation of the path method's transport (MethodTransport) and its audio
// backends (GpuAudio, SpectralAudio, the CPU mixer): the exact arrivals' slot layout and row format, the late
// histogram of a solved frame, and the material, volume and edge tables built from the scene.
internal static class RayTransport
{
    public const int PathStride = 48;
    // The same prefix ranges own transport paths, CPU groups and GPU FFT rows: each source/listener pair owns
    // ArrivalSlots exact arrivals (the direct sound), one path per FFT bin.
    internal sealed class Layout
    {
        internal readonly Vector2Int[] Ranges,DirectSites;
        internal readonly int[] Owners,RayCounts,GroupLimits;
        internal readonly int Capacity,MaximumSlots;
        internal Layout(AcousticTake.Scene scene,TakeRender.Settings settings)
        {
            Ranges=new Vector2Int[checked(scene.sources.Length*scene.listeners.Length)];
            RayCounts=new int[Ranges.Length];
            GroupLimits=new int[Ranges.Length];
            var direct=new List<Vector2Int>();
            for(int s=0;s<scene.sources.Length;s++)
            {
                int sites=scene.planes==null?1:Array.Find(scene.planes,p=>p.source==s)?.samples??1;
                int slots=checked(settings.FrequencyBins*TakeRender.Settings.ArrivalSlots);
                MaximumSlots=Math.Max(MaximumSlots,slots);
                for(int l=0;l<scene.listeners.Length;l++)
                {
                    int pair=s*scene.listeners.Length+l;Ranges[pair]=new Vector2Int(Capacity,slots);Capacity=checked(Capacity+slots);
                    GroupLimits[pair]=TakeRender.Settings.ArrivalSlots;
                    for(int n=0;n<sites;n++)direct.Add(new Vector2Int(pair,n));
                }
            }
            Owners=new int[Capacity];for(int p=0;p<Ranges.Length;p++)Array.Fill(Owners,p,Ranges[p].x,Ranges[p].y);
            DirectSites=direct.ToArray();
        }
    }
    // One solved frame's histograms, read by the audio backends (PathReverb, GpuAudio.Path).
    internal readonly struct LateFrame
    {
        internal readonly MethodTransport Transport;
        internal readonly int Frame;
        internal LateFrame(MethodTransport transport,int frame){Transport=transport;Frame=frame;}
    }
    [StructLayout(LayoutKind.Sequential, Pack = 4)]
    public struct Path
    {
        public Vector4 direction; // xyz arrival direction; w = d(propagation seconds)/d(receiver second)
        public float delaySeconds, pressureGain;
        public uint frequencyBin, identity;
        public float pressureImag;
        public uint emissionPhase, birthLow, birthHigh; // High phase bit selects finite burst; otherwise legacy 16-bit phase.
        public long Birth => (long)(((ulong)birthHigh<<32)|birthLow);
    }
    [StructLayout(LayoutKind.Sequential, Pack = 4)]
    internal struct Material
    {
        public Vector4 surface; // absorption, scattering, transmission, IOR
        public Vector4 medium;  // damping; reserved
    }
    [StructLayout(LayoutKind.Sequential, Pack = 4)]
    internal struct Edge
    {
        public Vector4 a, b, face, normal; // face.w: exterior wedge angle; normal points into air.
        public uint owner, firstFace, secondFace, pad;
    }
    [StructLayout(LayoutKind.Sequential, Pack = 4)]
    internal struct Volume
    {
        public Vector4 minimum, maximum; // w: winding correction, enclosed volume/area (relative units)
        public uint start, count, owner, material;
    }
    // Closed, consistently wound meshes/loops have thickness. Open sheets only
    // apply their surface coefficients. This is derived once, not saved in takes.
    internal static Volume[] Volumes(AcousticTake.Scene scene, AcousticTake.Primitive[] geometry, Material[] materials,
                            int bins)
    {
        var result = new List<Volume>();
        for (int owner = 0; owner < scene.objects.Length; owner++)
        {
            var obj = scene.objects[owner];
            if (obj.primitiveCount == 0)
                continue;
            int start = obj.primitiveStart, end = start + obj.primitiveCount;
            var min = geometry[start].a;
            var max = min;
            for (int i = start; i < end; i++)
            {
                var p = geometry[i];
                min = Vector3.Min(min, Vector3.Min(p.a, p.b));
                max = Vector3.Max(max, Vector3.Max(p.a, p.b));
                min = Vector3.Min(min, p.c);
                max = Vector3.Max(max, p.c);
            }
            // Weld floating-point seams from curved colliders for topology only.
            float tolerance = Mathf.Max(1e-7f, (max - min).magnitude * 1e-6f);
            var vertices = new Dictionary<Vector3Int, int>();
            int Vertex(Vector3 p)
            {
                p = (p - min) / tolerance;
                var key = new Vector3Int(Mathf.RoundToInt(p.x), Mathf.RoundToInt(p.y), Mathf.RoundToInt(p.z));
                if (!vertices.TryGetValue(key, out int id))
                {
                    id = vertices.Count;
                    vertices.Add(key, id);
                }
                return id;
            }
            var edges = new Dictionary<(int, int), (int count, int winding)>();
            void Edge(int a, int b)
            {
                var key = a < b ? (a, b) : (b, a);
                edges.TryGetValue(key, out var value);
                edges[key] = (value.count + 1, value.winding + (a < b ? 1 : -1));
            }
            double signed = 0;
            for (int i = start; i < end; i++)
            {
                var p = geometry[i];
                int a = Vertex(p.a), b = Vertex(p.b), c = Vertex(p.c);
                Edge(a, b);
                Edge(b, c);
                Edge(c, a);
                signed += Vector3.Dot(p.a - min, Vector3.Cross(p.b - min, p.c - min));
            }
            bool closed = signed != 0;
            foreach (var edge in edges.Values)
                closed &= edge.count == 2 && edge.winding == 0;
            if (!closed)
                continue;
            int material = geometry[start].materialIndex;
            for (int i = start + 1; i < end; i++)
                for (int b = 0; b < bins; b++)
                {
                    var a = materials[material * bins + b];
                    var other = materials[geometry[i].materialIndex * bins + b];
                    if (a.surface.w != other.surface.w || a.medium.x != other.medium.x)
                        throw new ArgumentException(obj.name +
                                                    (": a closed acoustic object must share interior " +
                                                     "IOR/damping across its surface materials."));
                }
            result.Add(new Volume {
                minimum = new Vector4(min.x, min.y, min.z, signed > 0 ? 1 : -1),
                maximum = new Vector4(max.x, max.y, max.z, (float)Math.Abs(signed)), start = (uint)start,
                count = (uint)obj.primitiveCount, owner = (uint)owner, material = (uint)material
            });
        }
        return result.ToArray();
    }
    internal static float Sample(AcousticTake.MaterialDefinition material, float[] values, float frequency,
                        float fallback, bool signed=false)
    {
        if (values == null || values.Length == 0)
            return fallback;
        var knots = material.frequencies;
        if (knots == null || knots.Length != values.Length)
            throw new ArgumentException("Material coefficient grids do not match.");
        int upper = 0;
        for (int i = 0; i < values.Length; i++)
        {
            if (!float.IsFinite(values[i]) || (!signed && values[i] < 0) || !float.IsFinite(knots[i]) || knots[i] < 0 ||
                (i > 0 && knots[i] <= knots[i - 1]))
                throw new ArgumentException("Invalid material coefficient curve.");
        }
        while (upper < knots.Length - 1 && frequency > knots[upper])
            upper++;
        int lower = Math.Max(0, upper - 1);
        return Mathf.Lerp(values[lower], values[upper],
                          lower == upper ? 0 : Mathf.InverseLerp(knots[lower], knots[upper], frequency));
    }
    internal static Edge[] Edges(AcousticTake.Primitive[] geometry, Material[] materials, int bins,
                        Volume[] volumes)
    {
        var winding = new Dictionary<uint, float>();
        foreach (var volume in volumes)
            winding[volume.owner] = volume.minimum.w;
        var candidates = new Dictionary<(int, Vector3, Vector3),
                                        (Vector3 normal, int first, int second, bool keep, int count)>();
        int Compare(Vector3 a, Vector3 b)
        {
            int r = a.x.CompareTo(b.x);
            if (r == 0)
                r = a.y.CompareTo(b.y);
            return r == 0 ? a.z.CompareTo(b.z) : r;
        }
        void Add(int face, Vector3 a, Vector3 b, Vector3 normal)
        {
            if (Compare(a, b) > 0)
            {
                var swap = a;
                a = b;
                b = swap;
            }
            var key = (geometry[face].objectIndex, a, b);
            if (candidates.TryGetValue(key, out var old))
                candidates[key] =
                    (old.normal, old.first, face,
                     old.count == 1 && Mathf.Abs(Vector3.Dot(old.normal, normal)) < .99999f, old.count + 1);
            else
                candidates.Add(key, (normal, face, face, true, 1));
        }
        for (int i = 0; i < geometry.Length; i++)
        {
            var p = geometry[i];
            var n = Vector3.Cross(p.b - p.a, p.c - p.a).normalized;
            Add(i, p.a, p.b, n);
            Add(i, p.b, p.c, n);
            Add(i, p.c, p.a, n);
        }
        bool Opaque(int face)
        {
            int start = geometry[face].materialIndex * bins;
            for (int bin = 0; bin < bins; bin++)
            {
                var m = materials[start + bin].surface;
                if ((1 - m.x) * m.z < 1)
                    return true;
            }
            return false;
        }
        Vector3 Side(int face, Vector3 a, Vector3 b)
        {
            var p = geometry[face];
            var axis = (b - a).normalized;
            var side = p.a + p.b + p.c - 3 * a;
            return (side - axis * Vector3.Dot(side, axis)).normalized;
        }
        var result = new List<Edge>();
        foreach (var item in candidates)
        {
            var value = item.Value;
            if (!value.keep || (!Opaque(value.first) && !Opaque(value.second)))
                continue;
            var a = item.Key.Item2;
            var b = item.Key.Item3;
            var side = Side(value.first, a, b);
            var normal = value.normal;
            if (winding.TryGetValue((uint)item.Key.Item1, out float sign))
                normal *= sign;
            float angle = 2 * Mathf.PI;
            if (value.first != value.second)
            {
                var other = Side(value.second, a, b);
                angle = Mathf.Atan2(Vector3.Dot(other, normal), Vector3.Dot(other, side));
                if (angle <= 0)
                    angle += 2 * Mathf.PI;
            }
            result.Add(new Edge { owner = (uint)item.Key.Item1, a = a, b = b,
                                  face = new Vector4(side.x, side.y, side.z, angle), normal = normal,
                                  firstFace = (uint)value.first, secondFace = (uint)value.second });
        }
        result.Sort((a, b) =>
                    {
                        int r = a.owner.CompareTo(b.owner);
                        if (r == 0)
                            r = Compare(a.a, b.a);
                        return r == 0 ? Compare(a.b, b.b) : r;
                    });
        return result.ToArray();
    }
}
