using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Runtime.InteropServices;
using UnityEngine;
using UnityEngine.Rendering;

// GPU driver of the single-loop methods (TakeRender.Method.Pressure and .Energy;
// MethodPressure.hlsl, MethodEnergy.hlsl), 3D point sources only. The hybrid
// renderer (RayTransport) is not involved.
//
// Measurement (both methods): one static frame, one source and one listener per solve,
// results read back synchronously.
//   Energy:   histogram [direction 6][band 8][bin] of 1 ms bins, energies in the
//             direct-sound convention (free field 1 m = 1), arrival directions in
//             world axes +x,-x,+y,-y,+z,-z (the direction the sound comes from); the
//             discrete arrivals are deposited into it too, and ReadArrivals lists them.
//   Pressure: per band b a complex train sampled at 2000 * 2^b Hz (16 samples per
//             centre-frequency period) starting MethodTransport.LeadSeconds before
//             emission; the band's transfer function is its Fourier transform,
//             H_b(f) = sum_n x[n] exp(-i 2 pi f (n / rate - lead)).
//   Both:     per band and interaction depth: count, sum |c|, sum |c|^2 and max |c| of
//             the listener contributions (before the 1/N average), and counters.
// Audio (energy only): every source/listener pair of up to maximumFrames frames per
// dispatch. The strongest methodArrivalSlots discrete arrivals of each pair become
// RayTransport.Path entries (one per FFT bin, gain sqrt of the band-interpolated
// energy, no phase) in the RayTransport.Layout of the energy method; the diffuse
// histogram (and arrivals beyond the slots) is a late field in the hybrid's format
// [frame][pair][head-frame direction][band][late bin]. GpuAudio and the CPU mixer
// read both exactly as they read the hybrid's.
internal sealed class MethodTransport : IDisposable, ILateField
{
    public const int Bands = 8, Directions = 6, Depths = 64, Counters = 16, CaptureVectors = 2, Magnitudes = 64, StatStride = 66;
    public const float TrainRate0 = 2000, LeadSeconds = .125f, TrainSeconds = 2.56f, BinSeconds = .001f;
    // Arrivals kept per solve before selection, and the largest candidate count of one
    // enumerated family (a larger family is left to the walk, which then holds it).
    public const int ArrivalCapacity = 4096, FamilyCandidateLimit = 1 << 18, ArrivalStride = 64;
    public static readonly string[] CounterNames = {"rays","events","escapes","roulette","horizon","eventCap","nonFinite",
        "detections","sphereHits","overflow","edgeConnections","diffuse","edgeEvents","arrivalOverflow","unused14","unused15"};
    public static readonly string[] FamilyNames = {"direct", "specular", "D", "DD", "RD", "DR"};
    [StructLayout(LayoutKind.Sequential, Pack = 4)]
    public struct Arrival
    {
        public Vector4 direction;   // xyz listener-to-arrival (world), w delay rate
        public Vector4 low, high;   // band energies
        public float delay;
        public uint identity, family;
        public float length;
        public int Family => (int)(family & 255);
        public int Depth => (int)(family >> 8);
        public float Energy(int band) => band < 4 ? low[band] : high[band - 4];
    }
    public readonly TakeRender.Method Kind;
    public readonly bool Audio;
    public readonly int Rays, Bins, TrainBase, EdgeCount, DroppedEdges, PlaneCount, ArrivalOrder, ArrivalFamilies, Frames, Pairs, ArrivalSlots, PathsPerFrame;
    public readonly long[] FamilyCandidates = new long[7]; // direct, order 1, order 2, D, DD, RD, DR
    readonly AcousticTake.Scene scene;
    readonly TakeRender.Settings settings;
    readonly RayTransport.Layout layout;
    readonly ComputeShader shader;
    readonly Action bind;
    readonly int traceKernel, clearKernel, arrivalKernel = -1, selectKernel = -1, expandKernel = -1, edgeWeightKernel = -1, edgeScanKernel = -1, outputCount, statsCount, frequencyBins, rate;
    readonly ComputeBuffer materials, edgeBuffer, edgeRanges, edgeIndices, volumes, output, stats, emptyCapture;
    readonly ComputeBuffer arrivals, arrivalCounts, selected, primitivePlanes, planePrimitives, paths, edgeCdf;
    public readonly uint[] Output, Stats;
    // CPU copies for the CPU reference audio (readback only when requested).
    public readonly RayTransport.Path[] Paths;
    readonly uint[] late;
    AsyncGPUReadbackRequest pathRequest, lateRequest;
    bool pending, failed;
    volatile bool pathsDone, lateDone;
    public double GpuSeconds { get; private set; }
    // Fraction of wall time the GPU may be busy (pacing between ray chunks, measurement).
    public double WorkFraction = 1;
    public const int ChunkRays = 1024;

    public MethodTransport(AcousticTake.Scene scene, AcousticTake.Primitive[] geometry, TakeRender.Settings settings,
                           ComputeBuffer primitives, ComputeBuffer states, ComputeBuffer sources,
                           ComputeBuffer listeners, ComputeBuffer bvhNodes, ComputeBuffer bvhIndices,
                           bool audio = false, int sampleRate = 48000, int maximumFrames = 1, bool readback = false)
    {
        if (settings.method != TakeRender.Method.Pressure && settings.method != TakeRender.Method.Energy)
            throw new ArgumentException("MethodTransport runs the Pressure or Energy method.");
        if (scene.dimension != AcousticTake.Dimension.Spatial3D)
            throw new NotSupportedException("The single-loop methods are 3D only.");
        if (scene.planes != null && scene.planes.Length > 0)
            throw new NotSupportedException("The single-loop methods take point sources only.");
        if (audio && settings.method != TakeRender.Method.Energy)
            throw new NotSupportedException("Only the energy method renders audio.");
        if (!audio && (scene.sources.Length != 1 || scene.listeners.Length != 1))
            throw new NotSupportedException("Method measurements solve one source and one listener.");
        if (!audio && settings.maximumDelaySeconds > TrainSeconds - LeadSeconds - .05f)
            throw new ArgumentOutOfRangeException(nameof(settings.maximumDelaySeconds), "The method horizon must stay inside the band trains.");
        Kind = settings.method; Audio = audio; this.scene = scene; this.settings = settings; rate = sampleRate;
        Frames = audio ? maximumFrames : 1;
        Pairs = scene.sources.Length * scene.listeners.Length;
        // One compute shader per method (Resources/Acoustics/PressureMethod.compute,
        // EnergyMethod.compute): neither the renderer's Transport.compute nor the other
        // method ever compiles these kernels.
        string name = settings.method == TakeRender.Method.Pressure ? "Acoustics/PressureMethod" : "Acoustics/EnergyMethod";
        var asset = Resources.Load<ComputeShader>(name);
        if (asset == null) throw new InvalidOperationException(name + ".compute is missing.");
        shader = asset;
        Rays = settings.methodRays;
        Bins = audio ? settings.LateBins(sampleRate) : (int)Math.Ceiling(settings.maximumDelaySeconds / BinSeconds - 1e-6);
        TrainBase = (int)Math.Round(TrainSeconds * TrainRate0);
        bool energy = Kind == TakeRender.Method.Energy;
        string suffix = settings.useBvh ? "Bvh" : "";
        traceKernel = shader.FindKernel((energy ? "TraceEnergy3D" : "TracePressure3D") + suffix);
        clearKernel = shader.FindKernel("ClearMethod");
        if (energy)
        {
            arrivalKernel = shader.FindKernel("EnergyArrivals3D" + suffix);
            selectKernel = shader.FindKernel("EnergySelectArrivals"); expandKernel = shader.FindKernel("EnergyExpandArrivals");
            edgeWeightKernel = shader.FindKernel("EnergyEdgeWeights3D" + suffix); edgeScanKernel = shader.FindKernel("EnergyEdgeScanKernel");
        }
        foreach (int kernel in new[] {traceKernel, clearKernel, arrivalKernel, selectKernel, expandKernel, edgeWeightKernel, edgeScanKernel})
            if (kernel >= 0 && !shader.IsSupported(kernel)) throw new NotSupportedException("Method kernels are unavailable on this GPU.");
        frequencyBins = settings.FrequencyBins;
        ArrivalSlots = audio ? settings.methodArrivalSlots : 0;
        if (audio) { layout = new RayTransport.Layout(scene, settings); PathsPerFrame = layout.Capacity; }
        try
        {
            // Materials at the band centres 125 Hz * 2^b.
            var bands = new RayTransport.Material[Math.Max(1, scene.materials.Length * Bands)];
            bool transmissive = false;
            for (int m = 0; m < scene.materials.Length; m++)
                for (int b = 0; b < Bands; b++)
                {
                    var material = scene.materials[m]; float hz = 125f * (1 << b);
                    bands[m * Bands + b] = new RayTransport.Material {
                        surface = new Vector4(RayTransport.Sample(material, material.absorption, hz, 0), RayTransport.Sample(material, material.scattering, hz, 0),
                                              RayTransport.Sample(material, material.transmission, hz, 0), RayTransport.Sample(material, material.ior, hz, 1)),
                        medium = new Vector4(RayTransport.Sample(material, material.damping, hz, 0), RayTransport.Sample(material, material.admittanceReal, hz, 0),
                                             RayTransport.Sample(material, material.admittanceImag, hz, 0, true), material.HasAdmittance ? 1 : 0) };
                    transmissive |= !material.HasAdmittance && bands[m * Bands + b].surface.z > 0;
                }
            materials = new ComputeBuffer(bands.Length, 32); materials.SetData(bands);
            var volumeData = RayTransport.Volumes(scene, geometry, bands, Bands, false);
            volumes = new ComputeBuffer(Math.Max(1, volumeData.Length), 48);
            if (volumeData.Length > 0) volumes.SetData(volumeData);
            // Diffracting edges only: a wedge whose index pi/angle is an integer is
            // reproduced exactly by its images (e.g. a room's concave right angles).
            var all = RayTransport.Edges(geometry, false, bands, Bands, volumeData);
            var kept = new List<RayTransport.Edge>();
            foreach (var edge in all)
            {
                double nu = Math.PI / edge.face.w;
                if (Math.Abs(nu - Math.Round(nu)) > 1e-3) kept.Add(edge);
            }
            EdgeCount = kept.Count; DroppedEdges = all.Length - kept.Count;
            edgeBuffer = new ComputeBuffer(Math.Max(1, kept.Count), 80);
            if (kept.Count > 0) edgeBuffer.SetData(kept);
            var adjacency = new List<int>[Math.Max(1, geometry.Length)];
            for (int i = 0; i < adjacency.Length; i++) adjacency[i] = new List<int>();
            for (int i = 0; i < kept.Count; i++)
            {
                adjacency[kept[i].firstFace].Add(i);
                if (kept[i].secondFace != kept[i].firstFace) adjacency[kept[i].secondFace].Add(i);
            }
            var indices = new List<int>(); var ranges = new Vector2Int[adjacency.Length];
            for (int i = 0; i < adjacency.Length; i++) { ranges[i] = new Vector2Int(indices.Count, adjacency[i].Count); indices.AddRange(adjacency[i]); }
            edgeRanges = new ComputeBuffer(ranges.Length, 8); edgeRanges.SetData(ranges);
            edgeIndices = new ComputeBuffer(Math.Max(1, indices.Count), 4);
            if (indices.Count > 0) edgeIndices.SetData(indices);
            // Mirror planes: coplanar primitives of one object share a plane (a wall split into
            // triangles is one mirror). The arrival enumeration reflects in planes and accepts a
            // hit only on the expected plane, so each image path is found once.
            var planeOf = new uint[Math.Max(1, geometry.Length)];
            var representatives = new List<uint>();
            var keys = new Dictionary<(int, long, long, long, long), uint>();
            for (int i = 0; i < geometry.Length; i++)
            {
                var g = geometry[i];
                Vector3 normal = Vector3.Cross(g.b - g.a, g.c - g.a);
                if (normal.sqrMagnitude <= 0) { planeOf[i] = uint.MaxValue; continue; }
                normal.Normalize();
                float sign = normal.x != 0 && Mathf.Abs(normal.x) > 1e-6f ? Mathf.Sign(normal.x) : Mathf.Abs(normal.y) > 1e-6f ? Mathf.Sign(normal.y) : Mathf.Sign(normal.z);
                normal *= sign;
                var key = (g.objectIndex, (long)Math.Round(normal.x * 1e5), (long)Math.Round(normal.y * 1e5), (long)Math.Round(normal.z * 1e5),
                           (long)Math.Round(Vector3.Dot(normal, g.a) * 1e4));
                if (!keys.TryGetValue(key, out uint plane)) { plane = (uint)representatives.Count; keys.Add(key, plane); representatives.Add((uint)i); }
                planeOf[i] = plane;
            }
            PlaneCount = representatives.Count;
            primitivePlanes = new ComputeBuffer(planeOf.Length, 4); primitivePlanes.SetData(planeOf);
            planePrimitives = new ComputeBuffer(Math.Max(1, PlaneCount), 4);
            if (PlaneCount > 0) planePrimitives.SetData(representatives);
            // Enumerated families (bit per MethodEnergy.hlsl EnergyFamily*). A family whose
            // candidate count exceeds FamilyCandidateLimit stays with the walk.
            bool arrivalsOn = energy && settings.methodArrivals, edges = EdgeCount > 0 && settings.diffractionFactor > 0;
            long planes = PlaneCount, e = EdgeCount;
            ArrivalOrder = !arrivalsOn ? 0 : settings.methodArrivalOrder >= 2 && planes * planes <= FamilyCandidateLimit ? 2 :
                           settings.methodArrivalOrder >= 1 && planes <= FamilyCandidateLimit ? Math.Min(1, settings.methodArrivalOrder) : 0;
            FamilyCandidates[0] = arrivalsOn ? 1 : 0;
            FamilyCandidates[1] = ArrivalOrder >= 1 ? planes : 0;
            FamilyCandidates[2] = ArrivalOrder >= 2 ? planes * planes : 0;
            FamilyCandidates[3] = arrivalsOn && edges ? e : 0;
            FamilyCandidates[4] = arrivalsOn && edges && e * e <= FamilyCandidateLimit ? e * e : 0;
            FamilyCandidates[5] = arrivalsOn && edges && planes * e <= FamilyCandidateLimit ? planes * e : 0;
            FamilyCandidates[6] = FamilyCandidates[5];
            ArrivalFamilies = (arrivalsOn ? 1 : 0) | (ArrivalOrder >= 1 ? 2 : 0) | (FamilyCandidates[3] > 0 ? 4 : 0) | (FamilyCandidates[4] > 0 ? 8 : 0) |
                              (FamilyCandidates[5] > 0 ? 16 + 32 : 0);
            if (audio)
                UnityEngine.Debug.Log($"Energy method audio: {geometry.Length} primitives, {PlaneCount} planes, {EdgeCount} diffracting edges; arrival order {ArrivalOrder}, " +
                                      $"families {string.Join(" ", Array.FindAll(FamilyNames, f => (ArrivalFamilies >> Array.IndexOf(FamilyNames, f) & 1) != 0))}, candidates {string.Join("/", FamilyCandidates)}");
            var ends = new uint[7]; long sum = 0;
            for (int f = 0; f < 7; f++) { sum += FamilyCandidates[f]; ends[f] = checked((uint)sum); }
            int solves = Frames * Pairs;
            arrivals = new ComputeBuffer(energy ? checked(solves * ArrivalCapacity) : 1, ArrivalStride);
            arrivalCounts = new ComputeBuffer(Math.Max(1, solves), 4);
            selected = new ComputeBuffer(Math.Max(1, checked(solves * Math.Max(1, ArrivalSlots))), 4);
            paths = new ComputeBuffer(audio ? Math.Max(1, checked(Frames * PathsPerFrame)) : 1, RayTransport.PathStride);
            edgeCdf = new ComputeBuffer(Math.Max(1, EdgeCount), 4); edgeCdf.SetData(new float[Math.Max(1, EdgeCount)]);
            Paths = new RayTransport.Path[audio && readback ? checked(Frames * PathsPerFrame) : 0];
            // Both outputs are (low, high) 32.32 fixed-point pairs.
            outputCount = audio ? checked(2 * Frames * Pairs * Directions * Bands * Math.Max(1, Bins))
                        : energy ? 2 * Directions * Bands * Bins : 2 * 2 * TrainBase * ((1 << Bands) - 1);
            statsCount = Bands * Depths * StatStride + Counters;
            output = new ComputeBuffer(outputCount, 4); stats = new ComputeBuffer(statsCount, 4);
            emptyCapture = new ComputeBuffer(2, 16);
            Output = new uint[audio ? 0 : outputCount]; Stats = new uint[statsCount];
            late = new uint[audio && readback ? outputCount : 0];
            // The shader asset is shared by every solver of this method (one compile per
            // process; a per-solver instance recompiled it each time and the compiler's
            // memory accumulated), so all bindings are made again at every dispatch.
            int volumeCount = volumeData.Length, keptCount = kept.Count;
            bool anyTransmission = transmissive || volumeData.Length > 0;
            var kernels = new List<int> {traceKernel, clearKernel};
            if (energy) kernels.AddRange(new[] {arrivalKernel, selectKernel, expandKernel, edgeWeightKernel, edgeScanKernel});
            bind = () =>
            {
            foreach (int kernel in kernels)
            {
                shader.SetBuffer(kernel, "Primitives", primitives); shader.SetBuffer(kernel, "States", states);
                shader.SetBuffer(kernel, "Sources", sources); shader.SetBuffer(kernel, "Listeners", listeners);
                shader.SetBuffer(kernel, "MethodMaterials", materials); shader.SetBuffer(kernel, "Edges", edgeBuffer);
                shader.SetBuffer(kernel, "EdgeRanges", edgeRanges); shader.SetBuffer(kernel, "EdgeIndices", edgeIndices);
                shader.SetBuffer(kernel, "Volumes", volumes); shader.SetBuffer(kernel, "MethodOutput", output);
                shader.SetBuffer(kernel, "MethodStats", stats); shader.SetBuffer(kernel, "MethodCapture", emptyCapture);
                if (bvhNodes != null) { shader.SetBuffer(kernel, "BvhNodes", bvhNodes); shader.SetBuffer(kernel, "BvhIndices", bvhIndices); }
                if (energy)
                {
                    shader.SetBuffer(kernel, "MethodArrivals", arrivals); shader.SetBuffer(kernel, "MethodArrivalCounts", arrivalCounts);
                    shader.SetBuffer(kernel, "MethodSelected", selected); shader.SetBuffer(kernel, "MethodPrimitivePlane", primitivePlanes);
                    shader.SetBuffer(kernel, "MethodPlanePrimitive", planePrimitives); shader.SetBuffer(kernel, "RayPaths", paths);
                    shader.SetBuffer(kernel, "MethodEdgeCdf", edgeCdf);
                }
            }
            shader.SetInt("PrimitiveCount", geometry.Length);
            shader.SetInt("PairCount", Pairs);
            shader.SetInt("ListenerCount", scene.listeners.Length);
            shader.SetInt("ObjectCount", scene.objects.Length);
            shader.SetInt("FrameCount", Frames);
            shader.SetFloat("SpeedOfSound", settings.speedOfSound);
            shader.SetFloat("ReferenceDistance", settings.referenceDistance);
            shader.SetFloat("MinimumDistance", settings.minimumDistance);
            shader.SetInt("RaySeed", settings.seed);
            shader.SetFloat("DiffractionFactor", settings.diffractionFactor);
            shader.SetInt("UseAirAbsorption", settings.airAbsorption ? 1 : 0);
            shader.SetInt("VolumeCount", volumeCount);
            shader.SetInt("EdgeCount", keptCount);
            shader.SetInt("MethodEdgeCount", keptCount);
            shader.SetInt("MethodTotalRays", Rays);
            shader.SetInt("MethodBins", Bins);
            shader.SetInt("MethodTrainBase", TrainBase);
            shader.SetInt("MethodMaxEvents", settings.methodMaxEvents);
            shader.SetInt("MethodCaptureRays", 0);
            shader.SetInt("MethodCaptureEvents", 0);
            shader.SetInt("MethodAnyTransmission", anyTransmission ? 1 : 0);
            shader.SetInt("MethodOutputCount", outputCount);
            shader.SetInt("MethodStatsCount", statsCount);
            shader.SetFloat("MethodBinSeconds", audio ? settings.LatePartition / (float)sampleRate : BinSeconds);
            shader.SetFloat("MethodHorizon", settings.maximumDelaySeconds);
            shader.SetFloat("MethodDetectRadius", settings.specularDetectionRadius);
            shader.SetFloat("MethodTrainRate0", TrainRate0);
            shader.SetFloat("MethodLeadSeconds", LeadSeconds);
            // Energy only: the pressure loop has no edge continuation (known gap, see
            // MethodPressure.hlsl), so the setting does not apply to it.
            shader.SetFloat("MethodEdgeProbability", energy ? settings.methodEdgeProbability : 0);
            if (!energy) return;
            shader.SetFloat("MethodRouletteFloor", settings.methodRouletteFloor);
            shader.SetInt("MethodHeadFrame", audio ? 1 : 0);
            shader.SetInt("MethodArrivalFamilies", ArrivalFamilies);
            shader.SetInt("MethodArrivalOrder", ArrivalOrder);
            shader.SetInt("MethodPlaneCount", PlaneCount);
            shader.SetInt("MethodArrivalCapacity", ArrivalCapacity);
            shader.SetInt("MethodArrivalSlots", ArrivalSlots);
            shader.SetInt("MethodArrivalDeposit", audio ? 0 : 1);
            shader.SetInt("MethodCountCells", solves);
            shader.SetInt("MethodFrequencyBins", frequencyBins);
            shader.SetInt("MethodFftSize", settings.fftSize);
            shader.SetFloat("MethodSampleRate", sampleRate);
            shader.SetVector("MethodFamilyEnds0", new Vector4(ends[0], ends[1], ends[2], ends[3]));
            shader.SetVector("MethodFamilyEnds1", new Vector4(ends[4], ends[5], ends[6], 0));
            };
        }
        catch { Dispose(); throw; }
    }
    public long GpuBytes => (long)(outputCount + statsCount) * 4 + (materials?.count ?? 0) * 32L + (edgeBuffer?.count ?? 0) * 80L +
                            (arrivals?.count ?? 0) * (long)ArrivalStride + (paths?.count ?? 0) * (long)RayTransport.PathStride;
    public long CpuBytes => (long)(Output.Length + Stats.Length + late.Length) * 4 + Paths.Length * (long)RayTransport.PathStride;
    internal ComputeBuffer PathBuffer => Audio ? paths : null;

    void SetSolve(int frame, int pair)
    {
        int solve = frame * Pairs + pair;
        shader.SetInt("MethodState", frame * scene.objects.Length);
        shader.SetInt("MethodSourceObject", scene.sources[pair / scene.listeners.Length].objectIndex);
        shader.SetInt("MethodListenerObject", scene.listeners[pair % scene.listeners.Length]);
        shader.SetInt("MethodCellBase", Audio ? solve * Directions * Bands * Bins : 0);
        shader.SetInt("MethodCountIndex", solve);
        shader.SetInt("MethodArrivalBase", solve * ArrivalCapacity);
        shader.SetInt("MethodSelectedBase", solve * Math.Max(1, ArrivalSlots));
        shader.SetInt("MethodPathBase", Audio ? frame * PathsPerFrame + layout.Ranges[pair].x : 0);
    }
    void Clear()
    {
        int clear = Math.Max(Math.Max(outputCount, statsCount), Frames * Pairs);
        for (int first = 0; first < clear; first += 65535 * 64)
        {
            shader.SetInt("MethodClearOffset", first);
            shader.Dispatch(clearKernel, (Math.Min(65535 * 64, clear - first) + 63) / 64, 1, 1);
        }
    }
    // The discrete arrivals of the current solve: enumerate, order (and deposit), expand.
    void DispatchArrivals()
    {
        long candidates = 0; foreach (long c in FamilyCandidates) candidates += c;
        for (long first = 0; first < candidates; first += 65535L * 64)
        {
            // Chunks above Metal's 65535 thread groups: the kernel index is offset by the chunk.
            shader.SetInt("MethodRayOffset", (int)first);
            shader.Dispatch(arrivalKernel, (int)((Math.Min(65535L * 64, candidates - first) + 63) / 64), 1, 1);
        }
        shader.Dispatch(selectKernel, (ArrivalCapacity + 63) / 64, 1, 1);
        if (Audio && ArrivalSlots > 0) shader.Dispatch(expandKernel, (ArrivalSlots * frequencyBins + 63) / 64, 1, 1);
        // The source's edge-ray distribution for this solve (read by the walk that follows).
        if (EdgeCount > 0 && settings.diffractionFactor > 0)
        {
            shader.Dispatch(edgeWeightKernel, (EdgeCount + 63) / 64, 1, 1);
            shader.Dispatch(edgeScanKernel, 1, 1, 1);
        }
    }

    // Measurement: trace every ray of one solve and read the outputs back. With a capture
    // buffer, rays below captureRays record up to captureEvents events (two float4 each).
    public void Dispatch(int bvhNodeCount, ComputeBuffer capture = null, int captureRays = 0, int captureEvents = 0)
    {
        if (Audio) throw new InvalidOperationException("Audio method transports dispatch with DispatchAudio.");
        bind();
        shader.SetInt("BvhNodeCount", bvhNodeCount);
        shader.SetBuffer(traceKernel, "MethodCapture", capture ?? emptyCapture);
        shader.SetInt("MethodCaptureRays", capture == null ? 0 : captureRays);
        shader.SetInt("MethodCaptureEvents", capture == null ? 0 : captureEvents);
        Clear();
        if (Kind == TakeRender.Method.Energy)
        {
            SetSolve(0, 0);
            long start = Stopwatch.GetTimestamp();
            DispatchArrivals();
            AsyncGPUReadback.Request(stats, 4, 0).WaitForCompletion();
            GpuSeconds += TakeRender.Elapsed(start);
        }
        for (int first = 0; first < Rays; first += ChunkRays)
        {
            int count = Math.Min(ChunkRays, Rays - first);
            shader.SetInt("MethodRayOffset", first);
            shader.SetInt("MethodRayCount", count);
            long start = Stopwatch.GetTimestamp();
            shader.Dispatch(traceKernel, (count + 63) / 64, 1, 1);
            AsyncGPUReadback.Request(stats, 4, 0).WaitForCompletion(); // fence: this chunk is done
            double busy = TakeRender.Elapsed(start);
            GpuSeconds += busy;
            if (WorkFraction < 1) System.Threading.Thread.Sleep((int)Math.Ceiling(busy * 1000 * (1 / WorkFraction - 1)));
        }
        output.GetData(Output);
        stats.GetData(Stats);
        shader.SetBuffer(traceKernel, "MethodCapture", emptyCapture);
    }
    // Audio: every pair of `frames` uploaded frames. The outputs stay on the GPU for
    // GpuAudio; with CPU copies (readback) they arrive asynchronously (TryComplete).
    public void DispatchAudio(int frames, int bvhNodeCount, bool asynchronous)
    {
        if (!Audio) throw new InvalidOperationException("Measurement method transports dispatch with Dispatch.");
        if (pending) throw new InvalidOperationException("Method responses are still in flight.");
        if (frames < 1 || frames > Frames) throw new ArgumentOutOfRangeException(nameof(frames));
        bind();
        shader.SetInt("BvhNodeCount", bvhNodeCount);
        long start = Stopwatch.GetTimestamp();
        Clear();
        for (int frame = 0; frame < frames; frame++)
            for (int pair = 0; pair < Pairs; pair++)
            {
                SetSolve(frame, pair);
                DispatchArrivals();
                shader.SetInt("MethodRayOffset", 0);
                shader.SetInt("MethodRayCount", Rays);
                shader.Dispatch(traceKernel, (Rays + 63) / 64, 1, 1);
            }
        GpuSeconds += TakeRender.Elapsed(start);
        if (Paths.Length == 0) return;
        if (asynchronous)
        {
            pending = true; failed = false; pathsDone = lateDone = false;
            pathRequest = AsyncGPUReadback.Request(paths, checked(frames * PathsPerFrame * RayTransport.PathStride), 0, value =>
            {
                failed |= value.hasError;
                if (!value.hasError) Unity.Collections.NativeArray<RayTransport.Path>.Copy(value.GetData<RayTransport.Path>(), Paths, value.GetData<RayTransport.Path>().Length);
                pathsDone = true;
            });
            lateRequest = AsyncGPUReadback.Request(output, checked(frames * LateUintsPerFrame * 4), 0, value =>
            {
                failed |= value.hasError;
                if (!value.hasError) Unity.Collections.NativeArray<uint>.Copy(value.GetData<uint>(), late, value.GetData<uint>().Length);
                lateDone = true;
            });
        }
        else
        {
            paths.GetData(Paths, 0, 0, frames * PathsPerFrame);
            output.GetData(late, 0, 0, frames * LateUintsPerFrame);
        }
    }
    public bool TryComplete(bool wait)
    {
        if (!pending) return true;
        if (!pathsDone || !lateDone)
        {
            if (wait) { pathRequest.WaitForCompletion(); lateRequest.WaitForCompletion(); }
            else { pathRequest.Update(); lateRequest.Update(); }
        }
        if (!pathsDone || !lateDone) return false;
        pending = false;
        if (failed) throw new InvalidOperationException("Method response readback failed.");
        return true;
    }
    // The discrete arrivals of one solve (measurement: frame 0, pair 0), read synchronously.
    public Arrival[] ReadArrivals(int frame = 0, int pair = 0)
    {
        if (Kind != TakeRender.Method.Energy) return new Arrival[0];
        var count = new uint[Frames * Pairs];
        arrivalCounts.GetData(count);
        int solve = frame * Pairs + pair, n = (int)Math.Min(count[solve], (uint)ArrivalCapacity);
        var result = new Arrival[n];
        if (n > 0) arrivals.GetData(result, 0, solve * ArrivalCapacity, n);
        return result;
    }
    public long ArrivalTotal(int frame = 0, int pair = 0)
    {
        var count = new uint[Frames * Pairs];
        arrivalCounts.GetData(count);
        return count[frame * Pairs + pair];
    }
    // ILateField (audio): the diffuse histogram in the hybrid's late format.
    public uint[] Late => late;
    public ComputeBuffer LateBuffer => output;
    public int LateUintsPerFrame => Audio ? 2 * Pairs * Directions * Bands * Bins : 0;
    public int LateBins => Bins;
    public int LateRays => Rays;
    public Emission.Batch Emissions => null;
    public ComputeBuffer EmissionRanges => null;
    public ComputeBuffer EmissionPoints => null;
    public ComputeBuffer EmissionClocks => null;

    // Energy of one histogram cell (Energy method measurement), direct-sound units.
    public double Energy(int direction, int band, int bin)
    {
        int cell = (direction * Bands + band) * Bins + bin;
        return (Output[2 * cell + 1] + Output[2 * cell] / 4294967296.0) / Rays;
    }
    double Fixed(int index) => (((long)(int)Output[2 * index + 1] << 32) | Output[2 * index]) / 4294967296.0;
    // Band train of the Pressure method: real and imaginary samples (already / N).
    public int TrainLength(int band) => TrainBase << band;
    public float TrainRate(int band) => TrainRate0 * (1 << band);
    public void Train(int band, double[] real, double[] imaginary)
    {
        int length = TrainLength(band), offset = 2 * TrainBase * ((1 << band) - 1);
        for (int i = 0; i < length; i++)
        {
            real[i] = Fixed(offset + i);
            imaginary[i] = Fixed(offset + length + i);
        }
    }
    // Depth statistics per band and depth: count, max |c| and the histogram of
    // floor(log2 |c|) + 40; sum |c| and sum |c|^2 are estimated at each bucket's
    // geometric centre (within a factor 2^(+-1/2) per contribution).
    public (long count, double sum, double square, double max) Depth(int band, int depth)
    {
        int cell = (band * Depths + depth) * StatStride;
        double sum = 0, square = 0;
        for (int b = 0; b < Magnitudes; b++)
        {
            double centre = Math.Pow(2, b - 40 + .5);
            sum += Stats[cell + 2 + b] * centre; square += Stats[cell + 2 + b] * centre * centre;
        }
        return (Stats[cell], sum, square, BitConverter.Int32BitsToSingle((int)Stats[cell + 1]));
    }
    public uint Histogram(int band, int depth, int bucket) => Stats[(band * Depths + depth) * StatStride + 2 + bucket];
    public long Counter(int index) => Stats[Bands * Depths * StatStride + index];
    public void Dispose()
    {
        if (pending) { if (!pathsDone) pathRequest.WaitForCompletion(); if (!lateDone) lateRequest.WaitForCompletion(); pending = false; }
        materials?.Dispose(); edgeBuffer?.Dispose(); edgeRanges?.Dispose(); edgeIndices?.Dispose(); volumes?.Dispose();
        output?.Dispose(); stats?.Dispose(); emptyCapture?.Dispose();
        arrivals?.Dispose(); arrivalCounts?.Dispose(); selected?.Dispose(); primitivePlanes?.Dispose(); planePrimitives?.Dispose(); paths?.Dispose(); edgeCdf?.Dispose();
    }
}
