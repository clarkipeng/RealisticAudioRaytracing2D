using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Runtime.InteropServices;
using UnityEngine;
using UnityEngine.Rendering;

// GPU driver of the path method (TakeRender.Method.Path, MethodPath.hlsl), 3D point sources only.
//
// One histogram per solve, [direction 6][band 8][bin] on the bins of PathBins (fine, then constant
// relative resolution), plus the contribution count per bin and the rate channel (Doppler).
// Measurement: one static frame, one source and one listener per solve, results read back
// synchronously (Energy() resamples the histogram to 1 ms bins for comparisons), with per band and
// interaction depth the count, sum |c|, sum |c|^2 and max |c| of the listener contributions, and counters.
// Audio: every source/listener pair of up to maximumFrames frames per dispatch; the histograms stay
// per frame and pair for PathReverb (GpuAudio.Path on the GPU), and the exact direct arrival becomes
// RayTransport.Path entries (one per FFT bin) in the RayTransport.Layout of the method.
internal sealed class MethodTransport : IDisposable
{
    public const int Bands = 8, Directions = 6, Depths = 64, Counters = 16, Magnitudes = 64, StatStride = 66;
    public const float BinSeconds = .001f;
    public const string ShaderResource = TransportSolver.PathShaderResource;
    public const int ArrivalStride = 64;
    public static readonly string[] CounterNames = {"rays","events","escapes","roulette","horizon","eventCap","nonFinite",
        "detections","sphereHits","overflow","edgeConnections","diffuse","edgeEvents","arrivalOverflow","branches","resumed"};
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
    public readonly bool Audio;
    // New rays per solve or update (population control: derived from settings.pathPopulation). The histogram is
    // divided by it when read; under population control the rays launched per update (NewRays) may change between
    // updates (Population), each cohort's rays then carrying Rays / NewRays of a ray's share (PathLaunchScale).
    public int Rays { get; private set; }
    public int NewRays { get; private set; }
    // Population control: the rays in flight N the next updates hold (settings.pathPopulation at first). Setting it
    // changes the new rays per update from the next update on, and every cohort's target with them; population
    // control then resamples each cohort to its new target, which is unbiased (its rho depends only on the past).
    public int Population
    {
        get => population ? (int)Math.Round(NewRays * PopulationShares) : 0;
        set { if (population) NewRays = Math.Max(1, Math.Min(maximumNewRays, (int)Math.Round(value / PopulationShares))); }
    }
    int maximumNewRays;
    public readonly int Bins, EdgeCount, DroppedEdges, Frames, Pairs, PathsPerFrame;
    readonly AcousticTake.Scene scene;
    readonly TakeRender.Settings settings;
    readonly RayTransport.Layout layout;
    readonly ComputeShader shader;
    readonly Action bind;
    readonly int clearanceKernel = -1;
    // Path: queued specular-chain connections of one chunk of rays (MethodPath.hlsl PathConnect).
    // Requests are about 0.5 per ray in a closed specular room; overflow is counted (counter 9).
    readonly ComputeBuffer clearance, mirrorCdf;
    // Persistent rays (settings.pathPersistent, MethodPath.hlsl "Persistent rays"): per pair two pools of
    // PersistentCapacity rays (read one, write the survivors to the other) and their counts; the update
    // period PathStep = responseStepSamples / sample rate. Flushes empty the pools (scene cuts).
    public const int RayStride = 160;
    // A contribution is added to the histogram of the update in which it is heard: per pair a ring of RingSlots
    // future histograms, stored after the frames' histograms in the output and count buffers.
    readonly bool persistent;
    readonly int beginKernel = -1, ringKernel = -1, edgeCopyKernel = -1, sortCountKernel = -1, sortScanKernel = -1, sortScatterKernel = -1;
    // Path: the delay rates (Doppler on reflections, MethodPath.hlsl "Delay rates") are compiled into the walk's
    // PATH_DOPPLER kernels only, dispatched while anything in the dispatched states moves and, with persistent rays, for
    // a horizon after (the rays in flight still carry the rates of what moved). A static scene runs the walk without them.
    readonly int traceStatic = -1, traceDoppler = -1;
    bool moving, doppler;
    long dopplerUntil = -1;
    public bool Doppler => doppler;
    // Population control (settings.pathPopulation, MethodPath.hlsl): N rays in flight per pair, PopulationNew of them
    // new each update; cohort counts per pair.
    readonly bool population;
    public readonly double PopulationShares; // 1 + sum over cohorts of min(1, T_f / age): N / new rays
    readonly ComputeBuffer cohortCounts;
    // Population control writes the kept rays in Morton order of their cell and direction octant (MethodPath.hlsl
    // PathSortCount/Scan/Scatter): per pair the keys' counts and first slots and per axis the fine bins' counts and
    // cells (SortFine bins over the scene's box), and per scratch ray its key and offset.
    public const int SortBits = 4, SortKeys = 1 << (3 * SortBits + 3), SortFine = 1024, SortStride = 2 * SortKeys + 6 * SortFine;
    readonly ComputeBuffer sortBuckets, sortEntries;
    Vector3 sortLow, sortScale;
    int cohorts;
    // Persistent: each pair's edge weights, refreshed EdgeSlice edges per update (MethodPath.hlsl PathEdgeWeights).
    public const int EdgeSlice = 2048;
    int edgeSkipped;
    readonly ComputeBuffer edgeWeights;
    int[] edgeCursor;
    readonly ComputeBuffer pathRays, pathLive;
    public readonly int PersistentCapacity, WarmUpdates;
    public int RingSlots { get; private set; }
    public double RingSceneDiagonal { get; private set; }
    bool ringReady;
    readonly AcousticTake.Primitive[] geometry;
    float maximumIor = 1;
    public readonly float PersistentStep;
    readonly int[] parity, ringSlot;
    int ringCells;
    ulong flushMask;
    long persistentUpdates;
    int traceKernel; readonly int clearKernel, expandKernel = -1, edgeWeightKernel = -1, edgeScanKernel = -1, outputCount, statsCount, frequencyBins, rate;
    readonly ComputeBuffer materials, edgeBuffer, edgeRanges, edgeIndices, volumes, stats;
    ComputeBuffer output; // persistent: reallocated with the ring at the first dispatch (EnsureRing)
    readonly ComputeBuffer arrivals, arrivalCounts, paths, edgeCdf;
    ComputeBuffer counts;
    // Path: the rate channel (Doppler, MethodPath.hlsl "Delay rates"): per solve [moment 3][bin] cells, the energy-weighted
    // first and second moments of the contributions' delay rates, after the histograms and the ring in `output`, then
    // its own ring [pair][slot][moment 3][bin]. RateCells: the solves' cells; RateBase: the first cell.
    readonly int rateCells;
    public const int RateMoments = 3;   // energy x rate, energy x rate^2, energy x delay within the bin
    int RateBase => (outputCount + 2 * Pairs * RingSlots * ringCells) / 2;
    int OutputUints => outputCount + 2 * Pairs * RingSlots * ringCells + 2 * rateCells + 2 * Pairs * RingSlots * RateMoments * TimeBins.Count;
    public readonly uint[] Rates;   // measurement: the solve's rate channel (read with the histogram)
    readonly uint[] lateRates;      // audio with CPU readback: the frames' rate channels
    // The histogram bins, the contribution counts (measurement readback) and solves so far (seeds).
    public readonly PathBins TimeBins;
    public readonly uint[] Counts;
    readonly int countCells;
    long pathSolves;
    public readonly uint[] Output, Stats;
    // CPU copies for the CPU reference audio (readback only when requested).
    public readonly RayTransport.Path[] Paths;
    readonly uint[] late;
    AsyncGPUReadbackRequest pathRequest, lateRequest, rateRequest, countRequest;
    bool pending, failed;
    volatile bool pathsDone, lateDone, ratesDone = true, countsDone = true;
    public double GpuSeconds { get; private set; }
    // Fraction of wall time the GPU may be busy (pacing between ray chunks, measurement).
    public double WorkFraction = 1;
    public const int ChunkRays = 1024;

    public MethodTransport(AcousticTake.Scene scene, AcousticTake.Primitive[] geometry, TakeRender.Settings settings,
                           ComputeBuffer primitives, ComputeBuffer states, ComputeBuffer bvhNodes, ComputeBuffer bvhIndices,
                           bool audio = false, int sampleRate = 48000, int maximumFrames = 1, bool readback = false)
    {
        if (settings.method != TakeRender.Method.Path)
            throw new ArgumentException("MethodTransport runs the path method.");
        if (scene.dimension != AcousticTake.Dimension.Spatial3D)
            throw new NotSupportedException("The path method is 3D only.");
        if (scene.planes != null && scene.planes.Length > 0)
            throw new NotSupportedException("The path method takes point sources only.");
        if (!audio && (scene.sources.Length != 1 || scene.listeners.Length != 1))
            throw new NotSupportedException("Method measurements solve one source and one listener.");
        Audio = audio; this.scene = scene; this.settings = settings; rate = sampleRate;
        this.geometry = geometry;
        Frames = audio ? maximumFrames : 1;
        Pairs = scene.sources.Length * scene.listeners.Length;
        var asset = Resources.Load<ComputeShader>(ShaderResource);
        if (asset == null) throw new InvalidOperationException(ShaderResource + ".compute is missing.");
        shader = asset;
        Rays = settings.methodRays;
        population = settings.pathPersistent && settings.pathPopulation > 0;
        TimeBins = new PathBins(settings);
        Bins = (int)Math.Ceiling(settings.maximumDelaySeconds / BinSeconds - 1e-6);
        string suffix = settings.useBvh ? "Bvh" : "";
        traceKernel = shader.FindKernel("TracePath3D" + suffix);
        clearKernel = shader.FindKernel("PathClear");
        edgeWeightKernel = shader.FindKernel("PathEdgeWeights3D" + suffix); edgeScanKernel = shader.FindKernel("PathEdgeScan");
        clearanceKernel = shader.FindKernel("PathClearance3D" + suffix);
        persistent = settings.pathPersistent;
        if (persistent) { beginKernel = shader.FindKernel("PathPersistentBeginKernel"); ringKernel = shader.FindKernel("PathRingKernel"); edgeCopyKernel = shader.FindKernel("PathEdgeCopyKernel"); }
        traceStatic = traceKernel; traceDoppler = shader.FindKernel("TracePath3D" + suffix + "Doppler");
        if (population) { sortCountKernel = shader.FindKernel("PathSortCountKernel"); sortScanKernel = shader.FindKernel("PathSortScanKernel"); sortScatterKernel = shader.FindKernel("PathSortScatterKernel"); }
        if (audio) expandKernel = shader.FindKernel("PathExpandDirect");
        foreach (int kernel in new[] {traceKernel, clearKernel, expandKernel, edgeWeightKernel, edgeScanKernel, beginKernel, ringKernel, edgeCopyKernel, sortCountKernel, sortScanKernel, sortScatterKernel})
            if (kernel >= 0 && !shader.IsSupported(kernel)) throw new NotSupportedException("Method kernels are unavailable on this GPU.");
        frequencyBins = settings.FrequencyBins;
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
                    if (b == 3) maximumIor = Math.Max(maximumIor, bands[m * Bands + b].surface.w);
                }
            materials = new ComputeBuffer(bands.Length, 32); materials.SetData(bands);
            var volumeData = RayTransport.Volumes(scene, geometry, bands, Bands);
            volumes = new ComputeBuffer(Math.Max(1, volumeData.Length), 48);
            if (volumeData.Length > 0) volumes.SetData(volumeData);
            // Diffracting edges only: a wedge whose index pi/angle is an integer is
            // reproduced exactly by its images (e.g. a room's concave right angles).
            var all = RayTransport.Edges(geometry, bands, Bands, volumeData);
            var kept = new List<RayTransport.Edge>();
            foreach (var edge in all)
            {
                double nu = Math.PI / edge.face.w;
                if (Math.Abs(nu - Math.Round(nu)) > 1e-3) kept.Add(edge);
            }
            EdgeCount = kept.Count; DroppedEdges = all.Length - kept.Count;
            EdgeLengths = kept.ConvertAll(e => ((Vector3)(e.b - e.a)).magnitude).ToArray();
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
            int solves = Frames * Pairs;
            arrivals = new ComputeBuffer(solves, ArrivalStride);   // one per solve: the exact direct arrival
            arrivalCounts = new ComputeBuffer(Math.Max(1, solves), 4);
            paths = new ComputeBuffer(audio ? Math.Max(1, checked(Frames * PathsPerFrame)) : 1, RayTransport.PathStride);
            edgeCdf = new ComputeBuffer(Math.Max(1, EdgeCount), 4); edgeCdf.SetData(new float[Math.Max(1, EdgeCount)]);
            // Path: mirrors of the RD edge connection are drawn in proportion to their area (local
            // coordinates: rigid motion keeps the area), so that a large wall or floor, where a mirror
            // reflection is likely to exist, is not drawn as rarely as one of thousands of small triangles.
            var areaCdf = new float[Math.Max(1, geometry.Length)]; double areaSum = 0;
            for (int i = 0; i < geometry.Length; i++)
            {
                var gm = geometry[i]; areaSum += .5 * Vector3.Cross(gm.b - gm.a, gm.c - gm.a).magnitude; areaCdf[i] = (float)areaSum;
            }
            mirrorCdf = new ComputeBuffer(areaCdf.Length, 4); mirrorCdf.SetData(areaCdf);
            Paths = new RayTransport.Path[audio && readback ? checked(Frames * PathsPerFrame) : 0];
            // (low, high) 32.32 fixed-point pairs.
            outputCount = checked(2 * Frames * Pairs * Directions * Bands * TimeBins.Count);
            countCells = checked(Frames * Pairs * TimeBins.Count);
            rateCells = checked(Frames * Pairs * RateMoments * TimeBins.Count);
            ringCells = Directions * Bands * TimeBins.Count;
            clearance = new ComputeBuffer(1, 4);
            // Persistent pools, sized for the rays expected in flight: a cohort of age t keeps at most the fraction
            // S(t) = min(1, T0 / t) of its rays (survival in time; all of them without it), plus a quarter for
            // transmitted branches and one cohort of margin. Overflow is counted (counter 9) and asserted absent by the checks.
            if (persistent)
            {
                // A cohort is (k - 1/2) periods old after its k-th update; everything heard at an update was emitted
                // within the horizon before it, so a flush is forgotten after WarmUpdates updates.
                PersistentStep = (float)(settings.responseStepSamples / (double)sampleRate);
                WarmUpdates = (int)Math.Ceiling(settings.maximumDelaySeconds / PersistentStep + .5) + 1;

                if (population)
                {
                    // Cohort k is (k - 1/2) periods old after its k-th update and kept at a share min(1, T_f / age) of the
                    // new rays; the update traces the new rays and every cohort, N in all.
                    double fineEnd = settings.pathFineBins * (double)settings.pathBinSeconds;
                    PopulationShares = 1;
                    for (int k = 1; k <= WarmUpdates; k++) PopulationShares += Math.Min(1, fineEnd / ((k - .5) * PersistentStep));
                    Rays = Math.Max(1, (int)Math.Round(settings.pathPopulation / PopulationShares));
                    maximumNewRays = Math.Max(Rays, (int)Math.Round(Math.Max(settings.pathPopulation, settings.pathPopulationMaximum) / PopulationShares));
                    cohorts = WarmUpdates + 2;
                }
                double expected = 1;
                for (int k = 1; k <= WarmUpdates; k++) expected += settings.pathSurvivalSeconds > 0 ? Math.Min(1, settings.pathSurvivalSeconds / ((k - .5) * PersistentStep)) : 1;
                long capacity = ((settings.pathRaysInFlight > 0 ? settings.pathRaysInFlight : population ? (long)Math.Ceiling(1.5 * maximumNewRays * PopulationShares) + maximumNewRays
                                  : (long)Math.Ceiling(Rays * (1.25 * expected + 1))) + 63) / 64 * 64;
                if (capacity * Pairs * 2 * RayStride > 1L << 30) throw new NotSupportedException("Persistent rays would need more than 1 GiB of ray state.");
                PersistentCapacity = (int)capacity;
            }
            NewRays = Rays;
            pathRays = new ComputeBuffer(persistent ? 2 * Pairs * PersistentCapacity : 1, RayStride);
            pathLive = new ComputeBuffer(4 * Pairs, 4); pathLive.SetData(new uint[4 * Pairs]);
            parity = new int[Pairs]; ringSlot = new int[Pairs]; edgeCursor = new int[Pairs];
            edgeWeights = new ComputeBuffer(persistent ? Math.Max(1, Pairs * EdgeCount) : 1, 4);
            cohortCounts = new ComputeBuffer(Math.Max(1, Pairs * cohorts), 4);
            sortBuckets = new ComputeBuffer(population ? Pairs * SortStride : 1, 4);
            if (population)
            {
                // Counts start empty; the first update's cells split each axis of the box evenly.
                var initial = new uint[sortBuckets.count];
                for (int pair = 0; pair < Pairs; pair++) for (int axis = 0; axis < 3; axis++) for (int b = 0; b < SortFine; b++)
                    initial[pair * SortStride + 2 * SortKeys + (2 * axis + 1) * SortFine + b] = (uint)(b * (1 << SortBits) / SortFine);
                sortBuckets.SetData(initial);
            }
            sortEntries = new ComputeBuffer(population ? Pairs * PersistentCapacity : 1, 8);
            flushMask = persistent ? 1UL : 0; // the first update empties the pools and the ring (buffers start undefined)
            counts = new ComputeBuffer(countCells, 4);
            Counts = new uint[countCells];
            statsCount = Bands * Depths * StatStride + Counters;
            output = new ComputeBuffer(OutputUints, 4); stats = new ComputeBuffer(statsCount, 4);
            Output = new uint[audio ? 0 : outputCount]; Stats = new uint[statsCount];
            late = new uint[audio ? outputCount : 0];
            Rates = new uint[!audio ? 2 * rateCells : 0];
            lateRates = new uint[audio ? 2 * rateCells : 0];
            // The shader asset is shared by every solver of this method (one compile per
            // process; a per-solver instance recompiled it each time and the compiler's
            // memory accumulated), so all bindings are made again at every dispatch.
            int volumeCount = volumeData.Length, keptCount = kept.Count;
            bool anyTransmission = transmissive || volumeData.Length > 0;
            var kernels = new List<int> {traceKernel, clearKernel, edgeWeightKernel, edgeScanKernel, clearanceKernel, traceDoppler};
            if (expandKernel >= 0) kernels.Add(expandKernel);
            if (persistent) kernels.AddRange(new[] {beginKernel, ringKernel, edgeCopyKernel});
            if (population) kernels.AddRange(new[] {sortCountKernel, sortScanKernel, sortScatterKernel});
            bind = () =>
            {
            foreach (int kernel in kernels)
            {
                shader.SetBuffer(kernel, "Primitives", primitives); shader.SetBuffer(kernel, "States", states);
                shader.SetBuffer(kernel, "MethodMaterials", materials); shader.SetBuffer(kernel, "Edges", edgeBuffer);
                shader.SetBuffer(kernel, "EdgeRanges", edgeRanges); shader.SetBuffer(kernel, "EdgeIndices", edgeIndices);
                shader.SetBuffer(kernel, "Volumes", volumes); shader.SetBuffer(kernel, "MethodOutput", output);
                shader.SetBuffer(kernel, "MethodStats", stats);
                if (bvhNodes != null) { shader.SetBuffer(kernel, "BvhNodes", bvhNodes); shader.SetBuffer(kernel, "BvhIndices", bvhIndices); }
                shader.SetBuffer(kernel, "MethodArrivals", arrivals); shader.SetBuffer(kernel, "MethodArrivalCounts", arrivalCounts);
                shader.SetBuffer(kernel, "RayPaths", paths);
                shader.SetBuffer(kernel, "MethodEdgeCdf", edgeCdf); shader.SetBuffer(kernel, "MethodEdgeCdfRead", edgeCdf); shader.SetBuffer(kernel, "PathCounts", counts);
                shader.SetBuffer(kernel, "PathClearanceOut", clearance); shader.SetBuffer(kernel, "PathClearanceRead", clearance); shader.SetBuffer(kernel, "PathMirrorCdf", mirrorCdf);
                shader.SetBuffer(kernel, "PathEdgeWeight", edgeWeights); shader.SetBuffer(kernel, "PathCohortCount", cohortCounts);
                shader.SetBuffer(kernel, "PathSortBuckets", sortBuckets); shader.SetBuffer(kernel, "PathSortEntry", sortEntries);
                shader.SetBuffer(kernel, "PathRays", pathRays); shader.SetBuffer(kernel, "PathLive", pathLive);
            }
            shader.SetInt("PrimitiveCount", geometry.Length);
            shader.SetInt("PairCount", Pairs);
            shader.SetInt("ListenerCount", scene.listeners.Length);
            shader.SetInt("ObjectCount", scene.objects.Length);
            shader.SetInt("FrameCount", Frames);
            shader.SetFloat("SpeedOfSound", settings.speedOfSound);
            shader.SetFloat("ReferenceDistance", settings.referenceDistance);
            shader.SetFloat("MinimumDistance", settings.minimumDistance);
            shader.SetInt("RaySeed", settings.seed + SeedOffset);
            shader.SetFloat("DiffractionFactor", settings.diffractionFactor);
            shader.SetInt("UseAirAbsorption", settings.airAbsorption ? 1 : 0);
            shader.SetInt("VolumeCount", volumeCount);
            shader.SetInt("EdgeCount", keptCount);
            shader.SetInt("MethodEdgeCount", keptCount);
            shader.SetInt("MethodTotalRays", Rays);
            shader.SetInt("MethodBins", Bins);
            shader.SetInt("MethodMaxEvents", settings.methodMaxEvents);
            shader.SetInt("MethodAnyTransmission", anyTransmission ? 1 : 0);
            shader.SetInt("MethodOutputCount", outputCount);
            shader.SetInt("MethodStatsCount", statsCount);
            shader.SetFloat("MethodBinSeconds", audio ? settings.LatePartition / (float)sampleRate : BinSeconds);
            shader.SetFloat("MethodHorizon", settings.maximumDelaySeconds);
            shader.SetFloat("MethodDetectRadius", settings.specularDetectionRadius);
            shader.SetFloat("MethodEdgeProbability", settings.methodEdgeProbability);
            shader.SetFloat("MethodRouletteFloor", settings.methodRouletteFloor);
            shader.SetInt("MethodHeadFrame", audio ? 1 : 0);
            shader.SetInt("MethodCountCells", solves);
            shader.SetInt("MethodFrequencyBins", frequencyBins);
            shader.SetInt("MethodFftSize", settings.fftSize);
            shader.SetFloat("MethodSampleRate", sampleRate);
            shader.SetFloat("PathBin0", (float)TimeBins.FineSeconds);
            shader.SetFloat("PathRatio", (float)(1.0 / TimeBins.FineBins));
            shader.SetInt("PathFineBins", TimeBins.FineBins);
            shader.SetInt("PathBins", TimeBins.Count);
            shader.SetInt("PathCountCells", countCells);
            shader.SetInt("PathAudio", audio ? 1 : 0);
            shader.SetInt("PathProfileFlags", 0);
            shader.SetFloat("PathSurvivalSeconds", population ? 0 : settings.pathSurvivalSeconds); // (population control applies the age profile itself)
            shader.SetInt("PathIterationCap", persistent ? Math.Max(0, settings.pathIterationCap) : 0); shader.SetFloat("PathIterationSurvival", Mathf.Clamp(settings.pathIterationSurvival, .05f, 1));
            shader.SetInt("PathPopulation", population ? 1 : 0); shader.SetInt("PathCohorts", Math.Max(1, cohorts));
            shader.SetFloat("PathCohortTarget", Rays); shader.SetFloat("PathLaunchScale", 1); shader.SetFloat("PathFineEnd", settings.pathFineBins * settings.pathBinSeconds);
            shader.SetInt("PathPersistent", persistent ? 1 : 0);
            shader.SetInt("PathBeamsOnly", settings.pathBeamsOnly ? 1 : 0);
            shader.SetInt("PathCapacity", PersistentCapacity);
            shader.SetFloat("PathStep", PersistentStep);
            shader.SetInt("PathFlush", 0);
            shader.SetInt("PathEdgeOffset", 0); shader.SetInt("PathEdgeWeightBase", 0);
            shader.SetInt("PathRateClearBase", 2 * RateBase); shader.SetInt("PathRateClearCount", 2 * rateCells);
            };
        }
        catch { Dispose(); throw; }
    }
    public long GpuBytes => (long)(outputCount + statsCount + countCells) * 4 + (materials?.count ?? 0) * 32L + (edgeBuffer?.count ?? 0) * 80L + PersistentBytes +
                            (arrivals?.count ?? 0) * (long)ArrivalStride + (paths?.count ?? 0) * (long)RayTransport.PathStride;
    public long PersistentBytes => persistent ? 2L * Pairs * PersistentCapacity * RayStride + (long)Pairs * RingSlots * (2 * ringCells + TimeBins.Count) * 4 : 0;
    public long CpuBytes => (long)(Output.Length + Stats.Length + late.Length) * 4 + Paths.Length * (long)RayTransport.PathStride;
    internal ComputeBuffer PathBuffer => Audio ? paths : null;
    // Diagnostics (MeasurePathPool): the rays pair 0's latest walk kept, before population control resampled them
    // (population control only, where the walk writes to the scratch pool), as MethodPath.hlsl's PathRay: ten float4
    // (40 floats per ray, RayStride).
    public const int PathRayFloats = 40;
    internal float[] ReadWalkPool(out int count)
    {
        count = 0;
        if (!population) return Array.Empty<float>();
        var live = new uint[4 * Pairs]; pathLive.GetData(live);
        count = (int)Math.Min(live[1], (uint)PersistentCapacity);
        var all = new float[pathRays.count * RayStride / 4];
        pathRays.GetData(all);
        var rays = new float[count * PathRayFloats];
        Array.Copy(all, PersistentCapacity * PathRayFloats, rays, 0, rays.Length);
        return rays;
    }

    void SetSolve(int frame, int pair)
    {
        int solve = frame * Pairs + pair;
        shader.SetInt("MethodState", frame * scene.objects.Length);
        shader.SetInt("MethodSourceObject", scene.sources[pair / scene.listeners.Length].objectIndex);
        shader.SetInt("MethodListenerObject", scene.listeners[pair % scene.listeners.Length]);
        shader.SetInt("MethodCellBase", solve * Directions * Bands * TimeBins.Count);
        shader.SetInt("PathCountBase", solve * TimeBins.Count);
        shader.SetInt("MethodCountIndex", solve);
        shader.SetInt("MethodArrivalBase", solve);
        shader.SetInt("MethodPathBase", Audio ? frame * PathsPerFrame + layout.Ranges[pair].x : 0);
        shader.SetInt("PathRateSolveBase", RateBase + solve * RateMoments * TimeBins.Count);
    }
    void Clear()
    {
        int clear = Math.Max(Math.Max(Math.Max(Math.Max(outputCount, statsCount), Frames * Pairs), countCells), 2 * rateCells);
        for (int first = 0; first < clear; first += 65535 * 64)
        {
            shader.SetInt("MethodClearOffset", first);
            shader.Dispatch(clearKernel, (Math.Min(65535 * 64, clear - first) + 63) / 64, 1, 1);
        }
    }
    // The source's edge-ray distribution of the current solve (read by the trace that follows).
    // Persistent: refresh EdgeSlice of the pair's edge weights (all of them after a flush), then build the
    // distribution from the kept weights.
    void DispatchEdgeDistribution(int pair = 0, bool all = true)
    {
        shader.Dispatch(clearanceKernel, 1, 1, 1); // the listener's clearance for the beam kernel (once per solve)
        if (EdgeCount == 0 || settings.diffractionFactor <= 0) return;
        if (!persistent)
        {
            shader.Dispatch(edgeWeightKernel, (EdgeCount + 63) / 64, 1, 1);
            shader.Dispatch(edgeScanKernel, 1, 1, 1);
            return;
        }
        // A refresh every pathEdgeSliceUpdates updates: in between, the distribution built last stays (one pair only:
        // the distribution's buffer is shared by the pairs). Its weights only steer the sampling, which divides by
        // the same distribution, so any refresh rate is unbiased.
        if (!all && Pairs == 1 && settings.pathEdgeSliceUpdates > 1 && ++edgeSkipped < settings.pathEdgeSliceUpdates) return;
        edgeSkipped = 0;
        int first = all ? 0 : edgeCursor[pair], count = all ? EdgeCount : Math.Min(EdgeSlice, EdgeCount - first);
        shader.SetInt("PathEdgeWeightBase", pair * EdgeCount); shader.SetInt("PathEdgeOffset", first);
        shader.Dispatch(edgeWeightKernel, (count + 63) / 64, 1, 1);
        shader.SetInt("PathEdgeOffset", 0);
        edgeCursor[pair] = all || first + count >= EdgeCount ? 0 : first + count;
        shader.Dispatch(edgeCopyKernel, (EdgeCount + 63) / 64, 1, 1);
        shader.Dispatch(edgeScanKernel, 1, 1, 1);
    }

    // One chunk of rays.
    void TraceChunk(int count)
    {
        long start = Stopwatch.GetTimestamp();
        shader.Dispatch(traceKernel, (count + 63) / 64, 1, 1);
        if (Profile) { Fence(); PhaseSeconds[1] += TakeRender.Elapsed(start); }
    }
    // Persistent: one update of one pair. The pool read holds the rays kept by the previous update; the first
    // threads continue them, the next Rays threads are the new cohort, and the survivors go to the other pool.
    // The dispatch covers the whole capacity (threads past the live count return at once): its count is on the GPU.
    void PersistentUpdate(int pair, bool flush)
    {
        long start = Stopwatch.GetTimestamp();
        int p = parity[pair]; // (population control: always 0, the control writes the rays back)
        shader.SetInt("PathCohortBase", pair * Math.Max(1, cohorts));
        shader.SetInt("PathSortBase", pair * SortStride); shader.SetInt("PathSortEntryBase", pair * PersistentCapacity);
        shader.SetInt("PathReadBase", (pair * 2 + p) * PersistentCapacity); shader.SetInt("PathWriteBase", (pair * 2 + 1 - p) * PersistentCapacity);
        shader.SetInt("PathReadCount", pair * 2 + p); shader.SetInt("PathWriteCount", pair * 2 + 1 - p); shader.SetInt("PathOverflowCount", 2 * Pairs + pair); shader.SetInt("PathRingFar", 3 * Pairs + pair);
        shader.SetInt("PathFlush", flush ? 1 : 0);
        shader.Dispatch(beginKernel, 1, 1, 1);
        shader.SetInt("PathRingSlots", RingSlots); shader.SetInt("PathRingCells", ringCells);
        shader.SetInt("PathRingBase", outputCount / 2 + pair * RingSlots * ringCells); shader.SetInt("PathRingCountBase", countCells + pair * RingSlots * TimeBins.Count);
        shader.SetInt("PathRingSlot", ringSlot[pair]);
        shader.SetInt("PathRateRingBase", RateBase + rateCells + pair * RingSlots * RateMoments * TimeBins.Count);
        if (flush) Ring(1, (long)RingSlots * ringCells);
        shader.SetInt("MethodRayCount", NewRays);
        shader.SetFloat("PathCohortTarget", NewRays); shader.SetFloat("PathLaunchScale", Rays / (float)NewRays);
        long threads = (long)PersistentCapacity + NewRays;
        for (long first = 0; first < threads; first += 65535L * 64)
        {
            shader.SetInt("MethodRayOffset", (int)first);
            shader.Dispatch(traceKernel, (int)((Math.Min(65535L * 64, threads - first) + 63) / 64), 1, 1);
        }
        shader.SetInt("PathFlush", 0);
        if (population)
        {
            shader.SetVector("PathSortLow", sortLow); shader.SetVector("PathSortScale", sortScale);
            foreach (int kernel in new[] {sortCountKernel, -1, sortScatterKernel})
            {
                if (kernel < 0) { shader.Dispatch(sortScanKernel, 1, 1, 1); continue; }
                for (long first = 0; first < PersistentCapacity; first += 65535L * 64)
                {
                    shader.SetInt("MethodRayOffset", (int)first);
                    shader.Dispatch(kernel, (int)((Math.Min(65535L * 64, PersistentCapacity - first) + 63) / 64), 1, 1);
                }
            }
        }
        Ring(0, ringCells);
        ringSlot[pair] = (ringSlot[pair] + 1) % RingSlots;
        if (!population) parity[pair] = 1 - p;
        persistentUpdates++;
        if (Profile) { Fence(); lastWalkSeconds = TakeRender.Elapsed(start); }
    }
    double lastWalkSeconds;
    void Ring(int mode, long cells)
    {
        shader.SetInt("PathRingMode", mode);
        for (long first = 0; first < cells; first += 65535L * 64)
        {
            shader.SetInt("PathRingOffset", (int)first);
            shader.Dispatch(ringKernel, (int)((Math.Min(65535L * 64, cells - first) + 63) / 64), 1, 1);
        }
    }
    // Persistent: empty every pool before the next update (a scene cut or a teleport), or before frame `frame`
    // of the next audio dispatch.
    public void Flush() => flushMask |= 1;
    public void FlushBefore(int frame) { if (frame >= 0 && frame < 64) flushMask |= 1UL << frame; }
    // Measurement, persistent: Dispatch runs one update without flushing (a sequence of moving states, one per
    // Solve) instead of a flush and WarmUpdates updates of which the last is read (a static scene's response).
    public bool PersistentSingleUpdate;
    // Check hook: every update's new rays reuse the same seed, so after the warm-up the response holds exactly
    // the full solve's rays, cut into updates.
    public bool PersistentRepeatSeed;
    public long PersistentUpdates => persistentUpdates;
    // Rays in a pair's pool after the latest update, and rays ever dropped because the pool was full (synchronous readbacks).
    public int RaysInFlight(int pair = 0)
    {
        if (!persistent) return 0;
        var live = new uint[3 * Pairs];
        pathLive.GetData(live);
        return (int)Math.Min(live[pair * 2 + parity[pair]], (uint)PersistentCapacity);
    }
    internal float[] EdgeLengths = Array.Empty<float>();   // diagnostics: each diffracting edge's length
    public long RaysOverflowed() => PersistentCount(2);
    // Live diagnostics without a stall: the latest asynchronous read of the rays in flight (pair 0), -1 before one.
    public const string TraceSample = "Acoustics.GPU.PathTrace";
    static UnityEngine.Rendering.CommandBuffer traceBegin, traceEnd;
    static UnityEngine.Rendering.CommandBuffer Marker(bool begin)
    {
        var c = new UnityEngine.Rendering.CommandBuffer {name = TraceSample};
        if (begin) c.BeginSample(TraceSample); else c.EndSample(TraceSample);
        return c;
    }
    public long AudioUpdates { get; private set; }
    public int LastRaysInFlight { get; private set; } = -1;
    bool inFlightPending;
    public void RequestRaysInFlight()
    {
        if (!persistent || inFlightPending || pathLive == null) return;
        inFlightPending = true; int p = parity[0];
        AsyncGPUReadback.Request(pathLive, 4 * 2, 0, r =>
        {
            inFlightPending = false;
            if (!r.hasError) LastRaysInFlight = (int)Math.Min(r.GetData<uint>()[p], (uint)PersistentCapacity);
        });
    }
    // Contributions heard later than the ring reaches (clamped to its last slot); none, by the ring's bound.
    public long ConnectionsBeyondRing() => PersistentCount(3);
    long PersistentCount(int block)
    {
        if (!persistent) return 0;
        var live = new uint[4 * Pairs]; long sum = 0;
        pathLive.GetData(live);
        for (int pair = 0; pair < Pairs; pair++) sum += live[block * Pairs + pair];
        return sum;
    }
    // Persistent: size the ring of future histograms before the first dispatch, from the scene's extent at
    // `states` (frame 0): a contribution is heard at most one connection's travel time after the update that
    // makes it, and a connection is at most three lines of sight (a mirror, an edge and the listener: RD; two
    // edges: DD), each no longer than the scene's bounding-box diagonal D, travelled at most maximumIor times
    // slower than in air. So RingSlots = ceil(3 D maximumIor / (c dt)) + 1 (the current slot); never more than
    // the horizon needs. Objects that later move outside the box are caught by ConnectionsBeyondRing.
    void EnsureRing(AcousticTake.State[] states)
    {
        if (!persistent || ringReady) return;
        Vector3 low = Vector3.positiveInfinity, high = Vector3.negativeInfinity;
        void Add(Vector3 p) { low = Vector3.Min(low, p); high = Vector3.Max(high, p); }
        if (states != null)
        {
            foreach (var g in geometry)
            {
                var pose = states[g.objectIndex];
                foreach (var v in new[] {g.a, g.b, g.c}) Add(pose.position + pose.rotation * v);
            }
            foreach (var source in scene.sources) Add(states[source.objectIndex].position);
            foreach (int listener in scene.listeners) Add(states[listener].position);
        }
        RingSceneDiagonal = states != null && high.x >= low.x ? (high - low).magnitude : double.PositiveInfinity;
        if (states != null && high.x >= low.x) { sortLow = low; sortScale = Vector3.one * SortFine; for (int a = 0; a < 3; a++) sortScale[a] /= Mathf.Max(high[a] - low[a], 1e-3f); }
        // (Beams only: no connections; a beam is heard at most one line of sight after its update.)
        double reach = (settings.pathBeamsOnly ? 1 : 3) * RingSceneDiagonal * maximumIor / (settings.speedOfSound * PersistentStep);
        RingSlots = (int)Math.Min(WarmUpdates + 1, Math.Ceiling(reach) + 1);
        output.Dispose(); counts.Dispose();
        output = new ComputeBuffer(checked(OutputUints), 4);
        counts = new ComputeBuffer(checked(countCells + Pairs * RingSlots * TimeBins.Count), 4);
        ringReady = true;
    }
    void Fence() => AsyncGPUReadback.Request(stats, 4, 0).WaitForCompletion();
    // Before the walk: the PATH_DOPPLER variant while anything moves, and with persistent rays for a horizon of updates
    // after. (Rays traced without it carry a zero rate, which is theirs: nothing moved.)
    void SelectDoppler(int updates)
    {
        if (moving) dopplerUntil = persistent ? persistentUpdates + updates + WarmUpdates : 0;
        bool want = moving || (persistent && persistentUpdates < dopplerUntil);
        doppler = want;
        traceKernel = doppler ? traceDoppler : traceStatic;
    }
    // Profiling (measurement only): a fence after every phase, and seconds per phase
    // [edge distribution, walk, (unused since the beams replaced the chain connections), readback].
    // ProfileFlags (MethodPath.hlsl): bit 0 skips the histogram writes; it changes the result.
    public bool Profile;
    public int ProfileFlags;
    public readonly double[] PhaseSeconds = new double[4];

    // Measurement: trace every ray of one solve and read the outputs back.
    public void Dispatch(int bvhNodeCount)
    {
        if (Audio) throw new InvalidOperationException("Audio method transports dispatch with DispatchAudio.");
        EnsureRing(null); // (no states given: a ring for the whole horizon)
        bind();
        SelectDoppler(persistent ? (PersistentSingleUpdate ? 1 : WarmUpdates) : 1);
        shader.SetInt("BvhNodeCount", bvhNodeCount);
        Clear();
        if (persistent)
        {
            DispatchPersistent();
            return;
        }
        SetSolve(0, 0);
        long edges = Stopwatch.GetTimestamp();
        shader.SetInt("PathProfileFlags", ProfileFlags); DispatchEdgeDistribution();
        AsyncGPUReadback.Request(stats, 4, 0).WaitForCompletion();
        GpuSeconds += TakeRender.Elapsed(edges);
        PhaseSeconds[0] += TakeRender.Elapsed(edges);
        for (int first = 0; first < Rays; first += ChunkRays)
        {
            int count = Math.Min(ChunkRays, Rays - first);
            shader.SetInt("MethodRayOffset", first);
            shader.SetInt("MethodRayCount", count);
            long start = Stopwatch.GetTimestamp();
            TraceChunk(count);
            AsyncGPUReadback.Request(stats, 4, 0).WaitForCompletion(); // fence: this chunk is done
            double busy = TakeRender.Elapsed(start);
            GpuSeconds += busy;
            if (WorkFraction < 1) System.Threading.Thread.Sleep((int)Math.Ceiling(busy * 1000 * (1 / WorkFraction - 1)));
        }
        long read = Stopwatch.GetTimestamp();
        output.GetData(Output, 0, 0, Output.Length);
        stats.GetData(Stats);
        counts.GetData(Counts); output.GetData(Rates, 0, 2 * RateBase, Rates.Length);
        PhaseSeconds[3] += TakeRender.Elapsed(read);
    }
    // Measurement, persistent: a flush and WarmUpdates updates of the static frame (or one update, see
    // PersistentSingleUpdate); the histogram is the last update's (each update replaces it with its ring slot),
    // the counters are totals over the updates.
    void DispatchPersistent()
    {
        int updates = PersistentSingleUpdate ? 1 : WarmUpdates;
        for (int u = 0; u < updates; u++)
        {
            bool last = u == updates - 1;
            SetSolve(0, 0);
            shader.SetInt("RaySeed", unchecked(settings.seed + SeedOffset + (PersistentRepeatSeed ? 0 : (int)persistentUpdates)));
            shader.SetInt("PathProfileFlags", ProfileFlags);
            long start = Stopwatch.GetTimestamp();
            bool flush = (flushMask & 1) != 0 || (u == 0 && !PersistentSingleUpdate);
            flushMask = 0;
            DispatchEdgeDistribution(0, flush);
            if (Profile) { Fence(); if (last) PhaseSeconds[0] += TakeRender.Elapsed(start); }
            PersistentUpdate(0, flush);
            if (Profile && last) PhaseSeconds[1] += lastWalkSeconds;
            AsyncGPUReadback.Request(stats, 4, 0).WaitForCompletion(); // fence: this update is done
            GpuSeconds += TakeRender.Elapsed(start);
            if (WorkFraction < 1) System.Threading.Thread.Sleep((int)Math.Ceiling(TakeRender.Elapsed(start) * 1000 * (1 / WorkFraction - 1)));
        }
        long read = Stopwatch.GetTimestamp();
        output.GetData(Output, 0, 0, Output.Length); // the ring follows the frames' histograms
        stats.GetData(Stats);
        counts.GetData(Counts, 0, 0, Counts.Length);
        output.GetData(Rates, 0, 2 * RateBase, Rates.Length);
        PhaseSeconds[3] += TakeRender.Elapsed(read);
    }
    // Audio: every pair of `frames` uploaded frames. The outputs stay on the GPU for
    // GpuAudio; with CPU copies (readback) they arrive asynchronously (TryComplete).
    public void DispatchAudio(int frames, int bvhNodeCount, bool asynchronous)
    {
        if (!Audio) throw new InvalidOperationException("Measurement method transports dispatch with Dispatch.");
        if (pending) throw new InvalidOperationException("Method responses are still in flight.");
        if (frames < 1 || frames > Frames) throw new ArgumentOutOfRangeException(nameof(frames));
        EnsureRing(null);
        bind();
        SelectDoppler(frames);
        shader.SetInt("PathProfileFlags", ProfileFlags); // (profiling only; zero in use)
        shader.SetInt("BvhNodeCount", bvhNodeCount);
        long start = Stopwatch.GetTimestamp();
        // A GPU profiler sample around the updates (the live HUD reads its time with a Recorder).
        traceBegin ??= Marker(true); traceEnd ??= Marker(false);
        Graphics.ExecuteCommandBuffer(traceBegin);
        Clear();
        for (int frame = 0; frame < frames; frame++)
            for (int pair = 0; pair < Pairs; pair++)
            {
                SetSolve(frame, pair);
                // A new seed for every solve: independent noise from frame to frame (no
                // temporal averaging here; that is the denoiser's job).
                shader.SetInt("RaySeed", unchecked(settings.seed + SeedOffset + (int)(pathSolves * Pairs + frame * Pairs + pair)));
                bool flush = persistent && (flushMask >> Math.Min(frame, 63) & 1) != 0;
                DispatchEdgeDistribution(pair, !persistent || flush);
                if (persistent)
                {
                    PersistentUpdate(pair, flush);
                    if (Profile) PhaseSeconds[1] += lastWalkSeconds;
                    shader.Dispatch(expandKernel, (frequencyBins + 63) / 64, 1, 1);
                    continue;
                }
                for (int first = 0; first < Rays; first += ChunkRays)
                {
                    shader.SetInt("MethodRayOffset", first);
                    shader.SetInt("MethodRayCount", Math.Min(ChunkRays, Rays - first));
                    TraceChunk(Math.Min(ChunkRays, Rays - first));
                }
                shader.Dispatch(expandKernel, (frequencyBins + 63) / 64, 1, 1);
            }
        pathSolves += frames;
        flushMask = 0;
        Graphics.ExecuteCommandBuffer(traceEnd);
        AudioUpdates += frames * Pairs;
        GpuSeconds += TakeRender.Elapsed(start);
        // GPU audio reads paths and histograms on the GPU (no CPU readback: nothing would complete it).
        if (Paths.Length == 0) return;
        if (asynchronous)
        {
            pending = true; failed = false; pathsDone = lateDone = false;
            if (Paths.Length == 0) pathsDone = true;
            else pathRequest = AsyncGPUReadback.Request(paths, checked(frames * PathsPerFrame * RayTransport.PathStride), 0, value =>
            {
                failed |= value.hasError;
                if (!value.hasError) Unity.Collections.NativeArray<RayTransport.Path>.Copy(value.GetData<RayTransport.Path>(), Paths, value.GetData<RayTransport.Path>().Length);
                pathsDone = true;
            });
            lateRequest = AsyncGPUReadback.Request(output, checked(frames * ReadbackUintsPerFrame * 4), 0, value =>
            {
                failed |= value.hasError;
                if (!value.hasError) Unity.Collections.NativeArray<uint>.Copy(value.GetData<uint>(), late, value.GetData<uint>().Length);
                lateDone = true;
            });
            ratesDone = false;
            rateRequest = AsyncGPUReadback.Request(output, checked(frames * Pairs * RateMoments * TimeBins.Count * 2 * 4), checked(2 * RateBase * 4), value =>
            {
                failed |= value.hasError;
                if (!value.hasError) Unity.Collections.NativeArray<uint>.Copy(value.GetData<uint>(), lateRates, value.GetData<uint>().Length);
                ratesDone = true;
            });
            // The contribution counts too (the histogram export, and the early part's test of a shared rate in PathReverb).
            countsDone = false;
            countRequest = AsyncGPUReadback.Request(counts, checked(frames * Pairs * TimeBins.Count * 4), 0, value =>
            {
                failed |= value.hasError;
                if (!value.hasError) Unity.Collections.NativeArray<uint>.Copy(value.GetData<uint>(), Counts, value.GetData<uint>().Length);
                countsDone = true;
            });
        }
        else
        {
            if (Paths.Length > 0) paths.GetData(Paths, 0, 0, frames * PathsPerFrame);
            output.GetData(late, 0, 0, frames * ReadbackUintsPerFrame);
            output.GetData(lateRates, 0, 2 * RateBase, frames * Pairs * RateMoments * TimeBins.Count * 2);
            counts.GetData(Counts, 0, 0, frames * Pairs * TimeBins.Count);
        }
    }
    public bool TryComplete(bool wait)
    {
        if (!pending) return true;
        if (!pathsDone || !lateDone || !ratesDone || !countsDone)
        {
            if (wait) { if (!pathsDone) pathRequest.WaitForCompletion(); lateRequest.WaitForCompletion(); if (!ratesDone) rateRequest.WaitForCompletion(); if (!countsDone) countRequest.WaitForCompletion(); }
            else { if (!pathsDone) pathRequest.Update(); lateRequest.Update(); if (!ratesDone) rateRequest.Update(); if (!countsDone) countRequest.Update(); }
        }
        if (!pathsDone || !lateDone || !ratesDone || !countsDone) return false;
        pending = false;
        if (failed) throw new InvalidOperationException("Method response readback failed.");
        return true;
    }
    // The exact direct arrival of one solve (measurement: frame 0, pair 0), read synchronously.
    public Arrival[] ReadArrivals(int frame = 0, int pair = 0)
    {
        var count = new uint[Frames * Pairs];
        arrivalCounts.GetData(count);
        int solve = frame * Pairs + pair, n = (int)Math.Min(count[solve], 1u);
        var result = new Arrival[n];
        if (n > 0) arrivals.GetData(result, 0, solve, n);
        return result;
    }
    public long ArrivalTotal(int frame = 0, int pair = 0)
    {
        var count = new uint[Frames * Pairs];
        arrivalCounts.GetData(count);
        return count[frame * Pairs + pair];
    }
    // Audio readback (CPU reference): the frames' histograms, and the histogram buffer and its layout on the GPU
    // (GpuAudio.Path): uints per frame of `output`, and the counts [frame][pair][bin].
    public uint[] Late => late;
    public ComputeBuffer LateBuffer => output;
    internal int PathUintsPerFrame => ReadbackUintsPerFrame;
    internal ComputeBuffer CountsBuffer => counts;
    int ReadbackUintsPerFrame => 2 * Pairs * Directions * Bands * TimeBins.Count;
    public int LateRays => Rays;
    // Before each dispatch (TransportSolver): the ring of future histograms, sized from the frames' states, and
    // whether anything in them moves (the walk's PATH_DOPPLER variant, SelectDoppler).
    public void PrepareFrames(AcousticTake.State[] states, int frames)
    {
        EnsureRing(states);
        moving = false;
        for (int i = 0; i < frames * scene.objects.Length && !moving; i++)
            moving = (states[i].flags & (AcousticTake.StateFlags.VelocityValid | AcousticTake.StateFlags.Cut)) == AcousticTake.StateFlags.VelocityValid &&
                     (states[i].velocity != Vector3.zero || states[i].angularVelocity != Vector3.zero);
    }
    // Added to settings.seed at every dispatch (independent realizations of the walk, e.g. averaged by a measurement).
    public int SeedOffset { get; set; }

    // Energy of one 1 ms histogram cell (measurement), direct-sound units. The histogram is resampled: each of its
    // bins spreads its energy uniformly over its time span.
    public double Energy(int direction, int band, int bin)
    {
        double sum = 0;
        for (int k = TimeBins.FirstOverlap(bin * BinSeconds); k < TimeBins.Count && TimeBins.Start(k) < (bin + 1) * BinSeconds; k++)
            sum += HistogramEnergy(0, 0, direction, band, k) * TimeBins.Overlap(k, bin * BinSeconds, (bin + 1) * BinSeconds);
        return sum;
    }
    // Path histogram cell of one solve (measurement: frame 0, pair 0, from Output; audio: from the
    // readback of the latest dispatch), direct-sound units; and its contribution count.
    public double HistogramEnergy(int frame, int pair, int direction, int band, int bin)
    {
        var source = Audio ? late : Output;
        int cell = ((frame * Pairs + pair) * Directions + direction) * Bands * TimeBins.Count + band * TimeBins.Count + bin;
        return (source[2 * cell + 1] + source[2 * cell] / 4294967296.0) / Rays;
    }
    // The rate channel of one solve (measurement: frame 0, pair 0, from Rates; audio: from the readback of the latest
    // dispatch): moment 0 is the sum of energy x delay rate, moment 1 of energy x rate^2, moment 2 of energy x delay from the
    // bin's start (seconds), over the bin's contributions, in the histogram's units (direct-sound energy). The bin's mean
    // rate is moment 0 over its energy (all directions and bands).
    public double RateMoment(int frame, int pair, int moment, int bin)
    {
        var source = Audio ? lateRates : Rates;
        int cell = ((frame * Pairs + pair) * RateMoments + moment) * TimeBins.Count + bin;
        return (((long)(int)source[2 * cell + 1] << 32) | source[2 * cell]) / 4294967296.0 / Rays;
    }
    // GpuAudio: the uint offset of a frame's rate channel in LateBuffer ([pair][moment 3][bin] cells of two uints).
    internal int PathRateOffset(int frame) => 2 * (RateBase + frame * Pairs * RateMoments * TimeBins.Count);
    public uint HistogramCount(int bin) => Counts[bin];
    public uint HistogramCount(int frame, int pair, int bin) => Counts[(frame * Pairs + pair) * TimeBins.Count + bin];
    public bool IsPath => true;
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
        if (pending) { if (!pathsDone) pathRequest.WaitForCompletion(); if (!lateDone) lateRequest.WaitForCompletion(); if (!ratesDone) rateRequest.WaitForCompletion(); if (!countsDone) countRequest.WaitForCompletion(); pending = false; }
        materials?.Dispose(); edgeBuffer?.Dispose(); edgeRanges?.Dispose(); edgeIndices?.Dispose(); volumes?.Dispose();
        output?.Dispose(); stats?.Dispose();
        counts?.Dispose();
        arrivals?.Dispose(); arrivalCounts?.Dispose(); paths?.Dispose(); edgeCdf?.Dispose(); clearance?.Dispose(); mirrorCdf?.Dispose();
        pathRays?.Dispose(); pathLive?.Dispose(); edgeWeights?.Dispose(); cohortCounts?.Dispose(); sortBuckets?.Dispose(); sortEntries?.Dispose();
    }
}
