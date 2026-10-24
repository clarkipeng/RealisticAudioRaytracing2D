using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using UnityEngine;
using UnityEngine.Rendering;

// Shared 2D/3D GPU transport: direct arrivals and sampled surface/edge paths.
internal sealed class RayTransport : IDisposable
{
    public const int PathStride = 48;
    [StructLayout(LayoutKind.Sequential, Pack = 4)]
    public struct Path
    {
        public Vector4 direction; // xyz arrival direction; w = d(propagation seconds)/d(receiver second)
        public float delaySeconds, pressureGain;
        public uint frequencyBin, identity;
        public float pressureImag;
        public uint emissionPhase, padding1, padding2; // 16-bit cyclic PCM phase; separate from physical delay.
    }
    [StructLayout(LayoutKind.Sequential, Pack = 4)]
    struct Material
    {
        public Vector4 surface; // absorption, scattering, transmission, IOR
        public Vector4 medium;  // damping; reserved
    }
    [StructLayout(LayoutKind.Sequential, Pack = 4)]
    struct Edge
    {
        public Vector4 a, b, face, normal; // face.w: exterior wedge angle; normal points into air.
        public uint owner, firstFace, secondFace, pad;
    }
    readonly ComputeShader shader;
    readonly CommandBuffer commands;
    readonly int directKernel, traceKernel, pairs, rayCount, frequencyBins, directSamples;
    [StructLayout(LayoutKind.Sequential, Pack = 4)]
    struct Volume
    {
        public Vector4 minimum, maximum; // w: winding correction, enclosed volume/area (relative units)
        public uint start, count, owner, material;
    }
    readonly ComputeBuffer materialBuffer, edgeBuffer, volumeBuffer, pathBuffer, enabledBuffer,
        edgeRangeBuffer, edgeIndexBuffer, emissionRanges, emissionPoints;
    readonly bool[] everActive;
    readonly uint[] enabledPairs;
    readonly Action<AsyncGPUReadbackRequest> completion;
    AsyncGPUReadbackRequest request;
    bool pending, failed;
    volatile bool completed;
    public readonly Path[] Paths;
    public int PathsPerFrame { get; }
    public int SlotsPerPair { get; }
    internal ComputeBuffer Buffer => pathBuffer;
    public long CpuBytes => Paths.Length * (long)PathStride + enabledPairs.Length * 4L + everActive.Length;
    public long GpuBytes { get; }
    public RayTransport(AcousticTake.Scene scene, AcousticTake.Primitive[] geometry,
                        TakeRender.Settings settings, ComputeShader shader, ComputeBuffer primitives,
                        ComputeBuffer states, ComputeBuffer sources, ComputeBuffer listeners,
                        ComputeBuffer bvhNodes, ComputeBuffer bvhIndices, int rate, int maximumFrames,
                        bool readback = true)
    {
        if (Marshal.SizeOf<Path>() != PathStride || Marshal.SizeOf<Material>() != 32 ||
            Marshal.SizeOf<Edge>() != 80 || Marshal.SizeOf<Volume>() != 48)
            throw new InvalidOperationException("Spectral GPU ABI mismatch.");
        this.shader = shader;
        rayCount = settings.RaysPerPair;
        frequencyBins = settings.fftSize / 2 + 1;
        pairs = checked(scene.sources.Length * scene.listeners.Length);
        everActive = new bool[scene.sources.Length];
        enabledPairs = new uint[checked(pairs * maximumFrames)];
        int bins = settings.fftSize / 2 + 1;
        foreach(var material in scene.materials)material.ValidateAdmittance();
        SlotsPerPair = settings.SlotsPerPair(scene);
        directSamples = AudioSourcePlane.MaximumSamples(scene);
        PathsPerFrame = checked(pairs * SlotsPerPair);
        Paths = new Path[readback ? checked(PathsPerFrame * maximumFrames) : 0];
        completion = OnComplete;
        bool planar = scene.dimension == AcousticTake.Dimension.Planar2D;
        try
        {
            commands = new CommandBuffer {name="Acoustics GPU transport"};
            var points=new List<Vector4>();var ranges=new int[scene.sources.Length*4];
            for(int s=0;s<scene.sources.Length;s++)
            {
                var plane=scene.planes==null?null:Array.Find(scene.planes,p=>p.source==s);
                ranges[s*4]=points.Count;ranges[s*4+1]=plane?.samples??1;
                ranges[s*4+2]=plane?.seed??0;ranges[s*4+3]=plane==null?0:1;
                if(plane==null)points.Add(new Vector4(0,0,0,1));
                else for(int n=0;n<plane.samples;n++)points.Add(AudioSourcePlane.Sample(plane,n));
            }
            emissionRanges=new ComputeBuffer(scene.sources.Length,16);emissionRanges.SetData(ranges);
            emissionPoints=new ComputeBuffer(points.Count,16);emissionPoints.SetData(points);
            directKernel = shader.FindKernel((planar ? "EvaluateDirect2D" : "EvaluateDirect3D") +
                                             (settings.useBvh ? "Bvh" : ""));
            traceKernel =
                shader.FindKernel((planar ? "Trace2D" : "Trace3D") + (settings.useBvh ? "Bvh" : ""));
            if (!shader.IsSupported(directKernel) || !shader.IsSupported(traceKernel))
                throw new NotSupportedException("Raytraced kernels are unavailable on this GPU.");
            var materials = new Material[Math.Max(1, scene.materials.Length * bins)];
            for (int m = 0; m < scene.materials.Length; m++)
                for (int b = 0; b < bins; b++)
                {
                    var material = scene.materials[m];
                    float hz = (float)b * rate / settings.fftSize;
                    materials[m * bins + b] =
                        new Material { surface = new Vector4(Sample(material, material.absorption, hz, 0),
                                                             Sample(material, material.scattering, hz, 0),
                                                             Sample(material, material.transmission, hz, 0),
                                                             Sample(material, material.ior, hz, 1)),
                                       medium =
                                           new Vector4(Sample(material, material.damping, hz, 0),
                                               Sample(material, material.admittanceReal, hz, 0),
                                               Sample(material, material.admittanceImag, hz, 0, true), material.HasAdmittance?1:0) };
                }
            materialBuffer = new ComputeBuffer(materials.Length, 32);
            materialBuffer.SetData(materials);
            var volumes = Volumes(scene, geometry, materials, bins, planar);
            var edges = Edges(geometry, planar, materials, bins, volumes);
            edgeBuffer = new ComputeBuffer(Math.Max(1, edges.Length), 80);
            if (edges.Length > 0)
                edgeBuffer.SetData(edges);
            volumeBuffer = new ComputeBuffer(Math.Max(1, volumes.Length), 48);
            if (volumes.Length > 0)
                volumeBuffer.SetData(volumes);
            pathBuffer = new ComputeBuffer(checked(PathsPerFrame * maximumFrames), PathStride);
            enabledBuffer = new ComputeBuffer(enabledPairs.Length, 4);
            var adjacency = new List<int>[geometry.Length];
            for (int i = 0; i < adjacency.Length; i++)
                adjacency[i] = new List<int>();
            for (int i = 0; i < edges.Length; i++)
            {
                adjacency[edges[i].firstFace].Add(i);
                if (edges[i].secondFace != edges[i].firstFace)
                    adjacency[edges[i].secondFace].Add(i);
            }
            var edgeIndices = new List<int>();
            var edgeRanges = new Vector2Int[Math.Max(1, geometry.Length)];
            for (int i = 0; i < adjacency.Length; i++)
            {
                edgeRanges[i] = new Vector2Int(edgeIndices.Count, adjacency[i].Count);
                edgeIndices.AddRange(adjacency[i]);
            }
            edgeRangeBuffer = new ComputeBuffer(edgeRanges.Length, 8);
            edgeRangeBuffer.SetData(edgeRanges);
            edgeIndexBuffer = new ComputeBuffer(Math.Max(1, edgeIndices.Count), 4);
            if (edgeIndices.Count > 0)
                edgeIndexBuffer.SetData(edgeIndices);
            foreach (int kernel in new[] { directKernel, traceKernel })
            {
                shader.SetBuffer(kernel, "Primitives", primitives);
                shader.SetBuffer(kernel, "States", states);
                shader.SetBuffer(kernel, "Sources", sources);
                shader.SetBuffer(kernel, "EmissionRanges", emissionRanges);
                shader.SetBuffer(kernel, "EmissionPoints", emissionPoints);
                shader.SetBuffer(kernel, "Listeners", listeners);
                shader.SetBuffer(kernel, "RayMaterials", materialBuffer);
                shader.SetBuffer(kernel, "Edges", edgeBuffer);
                shader.SetBuffer(kernel, "EdgeRanges", edgeRangeBuffer);
                shader.SetBuffer(kernel, "EdgeIndices", edgeIndexBuffer);
                shader.SetBuffer(kernel, "Volumes", volumeBuffer);
                shader.SetBuffer(kernel, "RayPaths", pathBuffer);
                shader.SetBuffer(kernel, "RayPairEnabled", enabledBuffer);
                if (bvhNodes != null)
                {
                    shader.SetBuffer(kernel, "BvhNodes", bvhNodes);
                    shader.SetBuffer(kernel, "BvhIndices", bvhIndices);
                }
            }
            shader.SetInt("RaysPerBin", settings.RaysPerBin);
            shader.SetInt("RaySeed", settings.seed);
            shader.SetInt("FrequencyBins", bins);
            shader.SetInt("RaysPerPair", settings.RaysPerPair);
            shader.SetInt("MaxBounces", settings.maxBounces);
            shader.SetInt("SlotsPerPair", SlotsPerPair);
            shader.SetInt("DirectSamples", directSamples);
            shader.SetInt("EdgeCount", edges.Length);
            shader.SetInt("VolumeCount", volumes.Length);
            shader.SetFloat("FrequencyStep", (float)rate / settings.fftSize);
            shader.SetFloat("DiffractionFactor", settings.diffractionFactor);
            shader.SetFloat("MaximumPathDelay", settings.maximumDelaySeconds);
            shader.SetInt("UseAirAbsorption", settings.airAbsorption ? 1 : 0);
            GpuBytes = (long)edgeRangeBuffer.count * 8 + (long)edgeIndexBuffer.count * 4 +
                       (long)materials.Length * 32 + (long)Math.Max(1, edges.Length) * 80 +
                       Math.Max(1, volumes.Length) * 48L + (long)PathsPerFrame * maximumFrames * PathStride +
                       enabledPairs.Length * 4L + (long)(scene.sources.Length+points.Count)*16;
        }
        catch
        {
            Dispose();
            throw;
        }
    }
    internal void BindDebug(ComputeBuffer segments)
    {
        shader.SetBuffer(directKernel, "DebugSegments", segments);
        shader.SetBuffer(traceKernel, "DebugSegments", segments);
    }
    public void SetActivity(AcousticTake.Scene scene, AcousticTake.State[] states, int frames)
    {
        // Never-used slots and permanently muted pairs cannot contribute. Once activated,
        // keep tracing a source after it leaves: earlier emission can still be arriving.
        for (int f = 0; f < frames; f++)
            for (int s = 0; s < everActive.Length; s++)
            {
                everActive[s] |=
                    TakeRender.Enabled(states[f * scene.objects.Length + scene.sources[s].objectIndex],
                                       AcousticTake.StateFlags.Source);
                for (int l = 0; l < scene.listeners.Length; l++)
                    enabledPairs[f * pairs + s * scene.listeners.Length + l] =
                        everActive[s] && scene.sources[s].mutedListenerPlusOne != l + 1 ? 1u : 0u;
            }
        enabledBuffer.SetData(enabledPairs, 0, 0, frames * pairs);
    }
    // Closed, consistently wound meshes/loops have thickness. Open sheets only
    // apply their surface coefficients. This is derived once, not saved in takes.
    static Volume[] Volumes(AcousticTake.Scene scene, AcousticTake.Primitive[] geometry, Material[] materials,
                            int bins, bool planar)
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
                if (!planar)
                {
                    min = Vector3.Min(min, p.c);
                    max = Vector3.Max(max, p.c);
                }
            }
            // Weld floating-point seams from curved colliders for topology only.
            float tolerance = Mathf.Max(1e-7f, (max - min).magnitude * 1e-6f);
            var vertices = new Dictionary<Vector3Int, int>();
            int Vertex(Vector3 p)
            {
                p = (p - min) / tolerance;
                var key = new Vector3Int(Mathf.RoundToInt(p.x), Mathf.RoundToInt(p.y),
                                         planar ? 0 : Mathf.RoundToInt(p.z));
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
            var degree = new Dictionary<int, (int incoming, int outgoing)>();
            for (int i = start; i < end; i++)
            {
                var p = geometry[i];
                int a = Vertex(p.a), b = Vertex(p.b);
                if (planar)
                {
                    degree.TryGetValue(a, out var va);
                    degree[a] = (va.incoming, va.outgoing + 1);
                    degree.TryGetValue(b, out var vb);
                    degree[b] = (vb.incoming + 1, vb.outgoing);
                    signed += (double)p.a.x * p.b.y - (double)p.a.y * p.b.x;
                }
                else
                {
                    int c = Vertex(p.c);
                    Edge(a, b);
                    Edge(b, c);
                    Edge(c, a);
                    signed += Vector3.Dot(p.a - min, Vector3.Cross(p.b - min, p.c - min));
                }
            }
            bool closed = signed != 0;
            foreach (var edge in edges.Values)
                closed &= edge.count == 2 && edge.winding == 0;
            foreach (var vertex in degree.Values)
                closed &= vertex.incoming == 1 && vertex.outgoing == 1;
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
                minimum = new Vector4(min.x, min.y, min.z, (signed > 0 ? 1 : -1) * (planar ? -1 : 1)),
                maximum = new Vector4(max.x, max.y, max.z, (float)Math.Abs(signed)), start = (uint)start,
                count = (uint)obj.primitiveCount, owner = (uint)owner, material = (uint)material
            });
        }
        return result.ToArray();
    }
    static float Sample(AcousticTake.MaterialDefinition material, float[] values, float frequency,
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
    static Edge[] Edges(AcousticTake.Primitive[] geometry, bool planar, Material[] materials, int bins,
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
            if (planar)
            {
                var n = new Vector3(-(p.b - p.a).y, (p.b - p.a).x, 0).normalized;
                Add(i, p.a, p.a, n);
                Add(i, p.b, p.b, n);
            }
            else
            {
                var n = Vector3.Cross(p.b - p.a, p.c - p.a).normalized;
                Add(i, p.a, p.b, n);
                Add(i, p.b, p.c, n);
                Add(i, p.c, p.a, n);
            }
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
            if (planar)
                return ((p.a - a).sqrMagnitude > 1e-12f ? p.a - a : p.b - a).normalized;
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
    public void Dispatch(int frames, bool asynchronous)
    {
        if (pending)
            throw new InvalidOperationException("Ray responses are still in flight.");
        commands.BeginSample("Acoustics.GPU.Direct");
        commands.DispatchCompute(shader,directKernel,(checked(frames*pairs*frequencyBins*directSamples)+63)/64,1,1);
        commands.EndSample("Acoustics.GPU.Direct");
        commands.BeginSample("Acoustics.GPU.Trace");
        commands.DispatchCompute(shader,traceKernel,(frames*pairs*rayCount+63)/64,1,1);
        commands.EndSample("Acoustics.GPU.Trace");
        Graphics.ExecuteCommandBuffer(commands);commands.Clear();
        if (Paths.Length == 0)
            return;
        if (asynchronous)
        {
            pending = true;
            completed = false;
            failed = false;
            request =
                AsyncGPUReadback.Request(pathBuffer, checked(frames * PathsPerFrame * PathStride), 0, completion);
        }
        else
            pathBuffer.GetData(Paths, 0, 0, frames * PathsPerFrame);
    }
    void OnComplete(AsyncGPUReadbackRequest value)
    {
        failed = value.hasError;
        if (!failed)
        {
            var data = value.GetData<Path>();
            Unity.Collections.NativeArray<Path>.Copy(data, Paths, data.Length);
        }
        completed = true;
    }
    public bool TryComplete(bool wait)
    {
        if (!pending)
            return true;
        if (!completed)
        {
            if (wait)
                request.WaitForCompletion();
            else
                request.Update();
        }
        if (!completed)
            return false;
        pending = false;
        if (failed)
            throw new InvalidOperationException("Ray response readback failed.");
        return true;
    }
    public void Dispose()
    {
        if (pending && !completed)
            request.WaitForCompletion();
        pending = false;
        materialBuffer?.Dispose();
        edgeBuffer?.Dispose();
        volumeBuffer?.Dispose();
        pathBuffer?.Dispose();
        enabledBuffer?.Dispose();
        edgeRangeBuffer?.Dispose();
        edgeIndexBuffer?.Dispose();
        emissionRanges?.Dispose();
        emissionPoints?.Dispose();
        commands?.Dispose();
    }
}
