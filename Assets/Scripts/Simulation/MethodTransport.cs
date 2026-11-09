using System;
using System.Collections.Generic;
using System.Diagnostics;
using UnityEngine;
using UnityEngine.Rendering;

// GPU driver of the single-loop measurement methods (TakeRender.Method.Pressure and
// .Energy; MethodPressure.hlsl, MethodEnergy.hlsl). Offline only: one static 3D
// frame, one point source and one listener per solve, results read back
// synchronously. The hybrid renderer (RayTransport) is not involved.
//
// Outputs of one solve:
//   Energy:   histogram [direction 6][band 8][bin] of 1 ms bins, energies in the
//             direct-sound convention (free field 1 m = 1), arrival directions in
//             world axes +x,-x,+y,-y,+z,-z (the direction the sound comes from).
//   Pressure: per band b a complex train sampled at 2000 * 2^b Hz (16 samples per
//             centre-frequency period) starting MethodTransport.LeadSeconds before
//             emission; the band's transfer function is its Fourier transform,
//             H_b(f) = sum_n x[n] exp(-i 2 pi f (n / rate - lead)).
//   Both:     per band and interaction depth: count, sum |c|, sum |c|^2 and max |c| of
//             the listener contributions (before the 1/N average), and counters.
internal sealed class MethodTransport : IDisposable
{
    public const int Bands = 8, Directions = 6, Depths = 64, Counters = 16, CaptureVectors = 2, Magnitudes = 64, StatStride = 66;
    public const float TrainRate0 = 2000, LeadSeconds = .125f, TrainSeconds = 2.56f, BinSeconds = .001f;
    public static readonly string[] CounterNames = {"rays","events","escapes","roulette","horizon","eventCap","nonFinite",
        "detections","chains","overflow","edgeConnections","diffuse","edgeEvents","unused13","unused14","unused15"};
    public readonly TakeRender.Method Kind;
    public readonly int Rays, Bins, TrainBase, EdgeCount, DroppedEdges;
    readonly ComputeShader shader;
    readonly Action bind;
    readonly int traceKernel, clearKernel, outputCount, statsCount;
    readonly ComputeBuffer materials, edgeBuffer, edgeRanges, edgeIndices, volumes, output, stats, emptyCapture;
    public readonly uint[] Output, Stats;
    public double GpuSeconds { get; private set; }
    // Fraction of wall time the GPU may be busy (pacing between ray chunks).
    public double WorkFraction = 1;
    public const int ChunkRays = 1024;

    public MethodTransport(AcousticTake.Scene scene, AcousticTake.Primitive[] geometry, TakeRender.Settings settings,
                           ComputeBuffer primitives, ComputeBuffer states, ComputeBuffer sources,
                           ComputeBuffer listeners, ComputeBuffer bvhNodes, ComputeBuffer bvhIndices)
    {
        if (settings.method != TakeRender.Method.Pressure && settings.method != TakeRender.Method.Energy)
            throw new ArgumentException("MethodTransport runs the Pressure or Energy method.");
        if (scene.dimension != AcousticTake.Dimension.Spatial3D)
            throw new NotSupportedException("The single-loop methods are 3D only.");
        if (scene.sources.Length != 1 || scene.listeners.Length != 1)
            throw new NotSupportedException("The single-loop methods solve one source and one listener.");
        if (scene.planes != null && scene.planes.Length > 0)
            throw new NotSupportedException("The single-loop methods take point sources only.");
        if (settings.maximumDelaySeconds > TrainSeconds - LeadSeconds - .05f)
            throw new ArgumentOutOfRangeException(nameof(settings.maximumDelaySeconds), "The method horizon must stay inside the band trains.");
        Kind = settings.method;
        // One compute shader per method (Resources/Acoustics/PressureMethod.compute,
        // EnergyMethod.compute): neither the renderer's Transport.compute nor the other
        // method ever compiles these kernels.
        string name = settings.method == TakeRender.Method.Pressure ? "Acoustics/PressureMethod" : "Acoustics/EnergyMethod";
        var asset = Resources.Load<ComputeShader>(name);
        if (asset == null) throw new InvalidOperationException(name + ".compute is missing.");
        shader = asset;
        Rays = settings.methodRays;
        Bins = (int)Math.Ceiling(settings.maximumDelaySeconds / BinSeconds - 1e-6);
        TrainBase = (int)Math.Round(TrainSeconds * TrainRate0);
        bool energy = Kind == TakeRender.Method.Energy;
        string suffix = settings.useBvh ? "Bvh" : "";
        traceKernel = shader.FindKernel((energy ? "TraceEnergy3D" : "TracePressure3D") + suffix);
        clearKernel = shader.FindKernel("ClearMethod");
        if (!shader.IsSupported(traceKernel) || !shader.IsSupported(clearKernel))
            throw new NotSupportedException("Method kernels are unavailable on this GPU.");
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
            // Both outputs are (low, high) 32.32 fixed-point pairs.
            outputCount = energy ? 2 * Directions * Bands * Bins : 2 * 2 * TrainBase * ((1 << Bands) - 1);
            statsCount = Bands * Depths * StatStride + Counters;
            output = new ComputeBuffer(outputCount, 4); stats = new ComputeBuffer(statsCount, 4);
            emptyCapture = new ComputeBuffer(2, 16);
            Output = new uint[outputCount]; Stats = new uint[statsCount];
            // The shader asset is shared by every solver of this method (one compile per
            // process; a per-solver instance recompiled it each time and the compiler's
            // memory accumulated), so all bindings are made again at every dispatch.
            int volumeCount = volumeData.Length, keptCount = kept.Count;
            bool anyTransmission = transmissive || volumeData.Length > 0;
            bind = () =>
            {
            foreach (int kernel in new[] { traceKernel, clearKernel })
            {
                shader.SetBuffer(kernel, "Primitives", primitives); shader.SetBuffer(kernel, "States", states);
                shader.SetBuffer(kernel, "Sources", sources); shader.SetBuffer(kernel, "Listeners", listeners);
                shader.SetBuffer(kernel, "MethodMaterials", materials); shader.SetBuffer(kernel, "Edges", edgeBuffer);
                shader.SetBuffer(kernel, "EdgeRanges", edgeRanges); shader.SetBuffer(kernel, "EdgeIndices", edgeIndices);
                shader.SetBuffer(kernel, "Volumes", volumes); shader.SetBuffer(kernel, "MethodOutput", output);
                shader.SetBuffer(kernel, "MethodStats", stats); shader.SetBuffer(kernel, "MethodCapture", emptyCapture);
                if (bvhNodes != null) { shader.SetBuffer(kernel, "BvhNodes", bvhNodes); shader.SetBuffer(kernel, "BvhIndices", bvhIndices); }
            }
            shader.SetInt("PrimitiveCount", geometry.Length);
            shader.SetInt("PairCount", 1);
            shader.SetInt("ListenerCount", 1);
            shader.SetInt("ObjectCount", scene.objects.Length);
            shader.SetInt("FrameCount", 1);
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
            shader.SetFloat("MethodBinSeconds", BinSeconds);
            shader.SetFloat("MethodHorizon", settings.maximumDelaySeconds);
            shader.SetFloat("MethodDetectRadius", settings.specularDetectionRadius);
            shader.SetFloat("MethodTrainRate0", TrainRate0);
            shader.SetFloat("MethodLeadSeconds", LeadSeconds);
            // Energy only: the pressure loop has no edge continuation (known gap, see
            // MethodPressure.hlsl), so the setting does not apply to it.
            shader.SetFloat("MethodEdgeProbability", energy ? settings.methodEdgeProbability : 0);
            };
        }
        catch { Dispose(); throw; }
    }
    public long GpuBytes => (long)(outputCount + statsCount) * 4 + (materials?.count ?? 0) * 32L + (edgeBuffer?.count ?? 0) * 80L;
    public long CpuBytes => (long)(Output.Length + Stats.Length) * 4;

    // Trace every ray of one solve and read the outputs back. With a capture buffer,
    // rays below captureRays record up to captureEvents events (two float4 each).
    public void Dispatch(int bvhNodeCount, ComputeBuffer capture = null, int captureRays = 0, int captureEvents = 0)
    {
        bind();
        shader.SetInt("BvhNodeCount", bvhNodeCount);
        shader.SetBuffer(traceKernel, "MethodCapture", capture ?? emptyCapture);
        shader.SetInt("MethodCaptureRays", capture == null ? 0 : captureRays);
        shader.SetInt("MethodCaptureEvents", capture == null ? 0 : captureEvents);
        int clear = Math.Max(outputCount, statsCount);
        for (int first = 0; first < clear; first += 65535 * 64)
        {
            shader.SetInt("MethodClearOffset", first);
            shader.Dispatch(clearKernel, (Math.Min(65535 * 64, clear - first) + 63) / 64, 1, 1);
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
    // Energy of one histogram cell (Energy method), direct-sound units.
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
        materials?.Dispose(); edgeBuffer?.Dispose(); edgeRanges?.Dispose(); edgeIndices?.Dispose(); volumes?.Dispose();
        output?.Dispose(); stats?.Dispose(); emptyCapture?.Dispose();
    }
}
