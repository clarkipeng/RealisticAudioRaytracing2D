using System;
using System.Collections.Generic;
using System.IO;
using System.Text;
using NUnit.Framework;
using UnityEngine;

// Band analysis of coherent responses, identical to tools/method_compare.py
// (energy_from_spectrum): band b uses the zero-phase filter |G_b(f)|^2 =
// cos^2(pi/2 log2(f/fc)) on [fc/2, 2fc], p_b = IFFT[G_b H], E[k] = int_bin p_b^2 /
// int |G_b|^2, so a free-field 1/d arrival integrates to 1/d^2.
static class MethodBands
{
    public static readonly double[] Centers = {125, 250, 500, 1000, 2000, 4000, 8000, 16000};
    public static double Power(double f, double fc)
    {
        if (f <= 0) return 0;
        double x = Math.Log(f / fc, 2);
        return Math.Abs(x) <= 1 ? Math.Pow(Math.Cos(Math.PI / 2 * x), 2) : 0;
    }
    // 2 * int_0^inf |G|^2 df, by the same trapezoid as method_compare.band_filter_energy.
    public static double FilterEnergy(double fc)
    {
        const int n = 200001; double sum = 0, h = 2.0 / (n - 1);
        for (int i = 0; i < n; i++)
        {
            double x = -1 + i * h, y = Math.Pow(Math.Cos(Math.PI / 2 * x), 2) * Math.Pow(2, x) * Math.Log(2) * fc;
            sum += (i == 0 || i == n - 1 ? .5 : 1) * y;
        }
        return 2 * sum * h;
    }
    public static int BandOf(double f) => Math.Clamp((int)Math.Round(Math.Log(f / 125, 2)), 0, 7);
    // In-place iterative radix-2 FFT, exp(-i...) forward; inverse includes 1/N.
    public static void Fft(double[] re, double[] im, int n, bool inverse)
    {
        for (int i = 1, j = 0; i < n; i++)
        {
            int bit = n >> 1;
            for (; (j & bit) != 0; bit >>= 1) j ^= bit;
            j ^= bit;
            if (i < j) { (re[i], re[j]) = (re[j], re[i]); (im[i], im[j]) = (im[j], im[i]); }
        }
        for (int length = 2; length <= n; length <<= 1)
        {
            double angle = 2 * Math.PI / length * (inverse ? 1 : -1), wr = Math.Cos(angle), wi = Math.Sin(angle);
            for (int i = 0; i < n; i += length)
            {
                double cr = 1, ci = 0;
                for (int k = 0; k < length / 2; k++)
                {
                    int a = i + k, b = a + length / 2;
                    double tr = re[b] * cr - im[b] * ci, ti = re[b] * ci + im[b] * cr;
                    re[b] = re[a] - tr; im[b] = im[a] - ti; re[a] += tr; im[a] += ti;
                    double next = cr * wr - ci * wi; ci = cr * wi + ci * wr; cr = next;
                }
            }
        }
        if (inverse) for (int i = 0; i < n; i++) { re[i] /= n; im[i] /= n; }
    }
}

// Offline runs of the three methods on the method-comparison suite
// (docs/benchmarks/methods-2026-11-06). Each (scene, method, budget, seed) writes raw
// arrays that tools/method_runs.py packs into docs/benchmarks/method-compare/FORMAT.md.
public partial class TakeChecks
{
    // JsonUtility-friendly copy of a method-compare scene (tools/method_runs.py writes it).
    [Serializable] sealed class MethodSceneSpec
    {
        public string sceneId, sceneSha256;
        public float[] source, points, frequencies;
        public WaveWall[] walls;
        public string[] surfaceMaterials, materialNames;
        public float[] absorption, scattering, transmission; // [material * 8 + band]
        public float speedOfSound = 343;
    }
    [Serializable] sealed class MethodJob
    {
        public string sceneFile, output;
        public string[] methods;
        public int[] budgets, seeds;
        public float maximumDelaySeconds = 2, detectionRadius = .25f, diffractionFactor = 1, edgeProbability = .5f, rouletteFloor = 1e-6f;
        public int receivers = -1;
        // Dataset runs (tools/path_dataset.py): a new seed per receiver (frames of a moving listener
        // get independent noise, as live solves do), and air absorption.
        public bool seedPerReceiver, airAbsorption;
        public float pathSurvivalSeconds = -1; // < 0: the settings' default
        public bool pathBeamsOnly, pathWalkDiffraction;
        // Method "path-modal": the path method combined with the room modes of this bank (an exported RMB1 file,
        // RoomModes.cs) as tools/room_modes.py combine does; written band-limited.
        public string roomModesBank = "";
    }
    [Serializable] sealed class MethodJobList { public MethodJob[] jobs; }
    [Serializable] sealed class MethodRunRecord
    {
        public string schema = "methods-raw-v1", sceneId, sceneSha256, method, backend, unityVersion, codeSha256;
        public int budget, seed, receivers, bins, primitives, edges, droppedEdges, lateRays, completeRouteSamples, methodRays, histogramBins;
        public float detectionRadius, rouletteFloor, maximumDelaySeconds, workFraction, binSeconds;
        public float[] frequencies, solveSeconds;
        public bool bandLimited;   // energy.f32 is band-limited (path-modal; the path histogram is not)
        public double busySeconds, gpuSeconds, analysisSeconds, wallSeconds;
        public string[] counterNames;
        public long[] counters;
        public string notes;
    }

    static string MethodArg(string name) => Environment.GetEnvironmentVariable(name);
    static AudioMaterial MethodMaterial(MethodSceneSpec spec, string name)
    {
        int m = Array.IndexOf(spec.materialNames, name);
        Assert.That(m, Is.GreaterThanOrEqualTo(0), "Unknown scene material " + name);
        var material = ScriptableObject.CreateInstance<AudioMaterial>();
        material.name = name; material.overrideBands = true;
        material.absorptionBands = new float[8]; material.scatteringBands = new float[8]; material.transmissionBands = new float[8];
        material.dampingBands = new float[8]; material.iorBands = new[] {1f, 1f, 1f, 1f, 1f, 1f, 1f, 1f};
        for (int b = 0; b < 8; b++)
        {
            material.absorptionBands[b] = spec.absorption[m * 8 + b]; material.scatteringBands[b] = spec.scattering[m * 8 + b];
            material.transmissionBands[b] = spec.transmission[m * 8 + b];
        }
        return material;
    }
    static string MethodCodeHash()
    {
        using var sha = System.Security.Cryptography.SHA256.Create();
        var all = new List<byte>();
        // The path method's code: every shader include, its compute shader and its C# driver.
        var names = new List<string>();
        foreach (var file in Directory.GetFiles(Path.Combine(Application.dataPath, "Shaders"), "*.hlsl")) names.Add("Shaders/" + Path.GetFileName(file));
        names.Sort(StringComparer.Ordinal);
        names.AddRange(new[] {"Resources/Acoustics/PathMethod.compute", "Scripts/Simulation/MethodTransport.cs", "Scripts/Simulation/RayTransport.cs"});
        foreach (var name in names)
            all.AddRange(File.ReadAllBytes(Path.Combine(Application.dataPath, name)));
        return BitConverter.ToString(sha.ComputeHash(all.ToArray())).Replace("-", "").ToLowerInvariant();
    }
    TakeRender.Settings MethodSettings(TakeRender.Method method, MethodJob job, int budget, int seed, float speed)
    {
        var settings = SpectralSettings();
        settings.method = method; settings.useBvh = true; settings.airAbsorption = job.airAbsorption; settings.speedOfSound = speed;
        settings.minimumDistance = .01f; settings.maximumDelaySeconds = job.maximumDelaySeconds; settings.seed = seed;
        settings.diffractionFactor = job.diffractionFactor;
        settings.fftSize = 256; settings.spectralHop = 32; settings.responseStepSamples = 32; settings.maximumResponsesPerBatch = 1;
        settings.maxBounces = 4096;
        settings.methodRays = budget; settings.specularDetectionRadius = job.detectionRadius;
        settings.methodEdgeProbability = job.edgeProbability; settings.methodRouletteFloor = job.rouletteFloor;
        if (job.pathSurvivalSeconds >= 0) settings.pathSurvivalSeconds = job.pathSurvivalSeconds;
        settings.pathBeamsOnly = job.pathBeamsOnly; settings.pathWalkDiffraction = job.pathWalkDiffraction;
        return settings;
    }
    static void MethodWrite(string path, float[] values)
    {
        using var writer = new BinaryWriter(File.Create(path));
        foreach (float v in values) writer.Write(v);
    }

    [Test, Category("QualityMeasurement"), Timeout(int.MaxValue)]
    public void MeasureMethods()
    {
        string file = MethodArg("ACOUSTIC_METHOD_JOBS");
        if (string.IsNullOrEmpty(file)) Assert.Ignore("Set ACOUSTIC_METHOD_JOBS to a job list (tools/method_runs.py jobs).");
        string budget = MethodArg("ACOUSTIC_TEST_WORK_FRACTION");
        double fraction = string.IsNullOrEmpty(budget) ? 1 : double.Parse(budget, System.Globalization.CultureInfo.InvariantCulture);
        var shader = RequireGpu(); string code = MethodCodeHash();
        foreach (var job in JsonUtility.FromJson<MethodJobList>(File.ReadAllText(file)).jobs)
            RunMethodJob(job, shader, fraction, code);
    }
    void RunMethodJob(MethodJob job, ComputeShader shader, double fraction, string code)
    {
        var spec = JsonUtility.FromJson<MethodSceneSpec>(File.ReadAllText(job.sceneFile));
        var materials = new Dictionary<string, AudioMaterial>();
        foreach (var name in spec.materialNames) materials[name] = MethodMaterial(spec, name);
        string scratch = Path.Combine(job.output, spec.sceneId, ".bake");
        Directory.CreateDirectory(scratch);
        var config = new WaveScene {source = spec.source, points = spec.points, walls = spec.walls, speedOfSound = spec.speedOfSound, bakeSolids = true};
        try
        {
            using var baked = new DatasetSolid(config, spec.surfaceMaterials, scratch, name => materials[name]);
            int receivers = job.receivers > 0 ? Math.Min(job.receivers, spec.points.Length / 3) : spec.points.Length / 3;
            foreach (string name in job.methods)
            {
                var method = TakeRender.Method.Path;
                Assert.That(name, Is.EqualTo("path").Or.EqualTo("path-modal"), "The path method is the only one (the hybrid, pressure and energy methods were removed); path-modal adds the room modes.");
                Assert.That(name != "path-modal" || !string.IsNullOrEmpty(job.roomModesBank), "path-modal needs the job's roomModesBank");
                for (int b = 0; b < job.budgets.Length; b++)
                    foreach (int seed in job.seeds)
                    {
                        string output = Path.Combine(job.output, spec.sceneId, name, $"r{job.budgets[b]:00000}", seed.ToString());
                        if (File.Exists(Path.Combine(output, "run.json"))) continue;
                        RenderMethod(spec, baked, method, name, job, job.budgets[b], seed, receivers, output, shader, fraction, code);
                    }
            }
        }
        finally { foreach (var m in materials.Values) UnityEngine.Object.DestroyImmediate(m); }
    }
    void RenderMethod(MethodSceneSpec spec, DatasetSolid baked, TakeRender.Method method, string name, MethodJob job, int budget, int seed,
                      int receivers, string output, ComputeShader shader, double fraction, string code)
    {
        long started = System.Diagnostics.Stopwatch.GetTimestamp();
        string partial = output + ".partial";
        if (Directory.Exists(partial)) Directory.Delete(partial, true);
        Directory.CreateDirectory(partial);
        var settings = MethodSettings(method, job, budget, seed, spec.speedOfSound);
        int rate = 8000, bins = (int)Math.Round(job.maximumDelaySeconds / MethodTransport.BinSeconds);
        var state = (AcousticTake.State[])baked.states.Clone();
        var frequencies = spec.frequencies ?? new float[0];
        var energy = new float[receivers * 8 * bins];
        var direction = new float[receivers * 8 * bins * 6];
        // The native histogram [receiver][direction][band][bin] and contributions [receiver][bin].
        float[] native = null; uint[] nativeCounts = null; PathBins pathBins = null;
        var stats = new float[receivers * 8 * MethodTransport.Depths * 4];
        var magnitudes = new float[8 * MethodTransport.Depths * MethodTransport.Magnitudes]; // summed over receivers
        var counters = new long[MethodTransport.Counters];
        var visible = new double[receivers];   // the path method renders a direct sound there
        var record = new MethodRunRecord {sceneId = spec.sceneId, sceneSha256 = spec.sceneSha256, method = name, budget = budget, seed = seed,
            backend = SystemInfo.graphicsDeviceName, unityVersion = Application.unityVersion, codeSha256 = code, receivers = receivers, bins = bins,
            primitives = baked.geometry.Length, binSeconds = MethodTransport.BinSeconds, maximumDelaySeconds = job.maximumDelaySeconds,
            workFraction = (float)fraction, frequencies = frequencies, solveSeconds = new float[receivers], detectionRadius = job.detectionRadius, rouletteFloor = job.rouletteFloor,
            methodRays = budget,
            counterNames = MethodTransport.CounterNames, counters = counters};
        double analysis = 0;
        using (var solver = new TransportSolver(baked.scene, baked.geometry, settings, shader, sampleRate: rate))
        {
            solver.Method.WorkFraction = fraction; record.edges = solver.Method.EdgeCount; record.droppedEdges = solver.Method.DroppedEdges;
            var direct = new TakeRender.DirectPath[1];
            state[1].position = WaveVector(spec.points); solver.Solve(state, direct); // exclude the cold dispatch
            double gpuBefore = solver.Method.GpuSeconds;
            for (int p = 0; p < receivers; p++)
            {
                state[1].position = WaveVector(spec.points, 3 * p);
                if (job.seedPerReceiver) solver.Method.SeedOffset = p * 1000003;
                long t0 = System.Diagnostics.Stopwatch.GetTimestamp();
                solver.Solve(state, direct);
                double work = TakeRender.Elapsed(t0);
                visible[p] = direct[0].pressureGain > 0 ? 1 : 0;
                record.solveSeconds[p] = (float)work; record.busySeconds += work;
                long a0 = System.Diagnostics.Stopwatch.GetTimestamp();
                var m = solver.Method;
                pathBins ??= m.TimeBins; int B = pathBins.Count;
                native ??= new float[receivers * 6 * 8 * B]; nativeCounts ??= new uint[receivers * B];
                for (int d = 0; d < 6; d++) for (int b = 0; b < 8; b++) for (int k = 0; k < B; k++)
                    native[((p * 6 + d) * 8 + b) * B + k] = (float)m.HistogramEnergy(0, 0, d, b, k);
                Array.Copy(m.Counts, 0, nativeCounts, p * B, B);
                for (int b = 0; b < 8; b++) for (int k = 0; k < bins; k++)
                {
                    double sum = 0;
                    for (int d = 0; d < 6; d++) { double e = m.Energy(d, b, k); sum += e; direction[((p * 8 + b) * bins + k) * 6 + d] = (float)e; }
                    energy[(p * 8 + b) * bins + k] = (float)sum;
                }
                for (int b = 0; b < 8; b++) for (int d = 0; d < MethodTransport.Depths; d++)
                {
                    var s = m.Depth(b, d); int at = ((p * 8 + b) * MethodTransport.Depths + d) * 4;
                    stats[at] = s.count; stats[at + 1] = (float)s.sum; stats[at + 2] = (float)s.square; stats[at + 3] = (float)s.max;
                    for (int h = 0; h < MethodTransport.Magnitudes; h++) magnitudes[(b * MethodTransport.Depths + d) * MethodTransport.Magnitudes + h] += m.Histogram(b, d, h);
                }
                for (int c = 0; c < counters.Length; c++) counters[c] += m.Counter(c);
                analysis += TakeRender.Elapsed(a0);
            }
            record.gpuSeconds = solver.Method.GpuSeconds - gpuBefore;
        }
        if (name == "path-modal")
        {
            // The path method's energies (direct sound included) combined with the bank's modes (room_modes.combine).
            long m0 = System.Diagnostics.Stopwatch.GetTimestamp();
            var modes = RoomModesFor(spec, job.roomModesBank);
            var path = new double[energy.Length]; for (int i = 0; i < path.Length; i++) path[i] = energy[i];
            var combined = RoomModeOffline.Combine(modes, WaveVector(spec.source), Points(spec.points), visible, path, out var touched);
            for (int i = 0; i < energy.Length; i++) energy[i] = (float)combined[i];
            direction = null; record.bandLimited = true;
            analysis += TakeRender.Elapsed(m0);
            record.notes = $"path method combined with the room modes of {Path.GetFileName(job.roomModesBank)} ({modes.Bank.Modes} modes, Schroeder {modes.SchroederHz:F1} Hz, bands {string.Join(",", touched)}): " +
                           "the coherent part (W_m H_modal + (1 - W_m) H_direct) band-limited, plus g_b times the histogram less its direct sound, smoothed by each band's energy kernel (tools/room_modes.py combine)";
        }
        record.analysisSeconds = analysis;
        MethodWrite(Path.Combine(partial, "energy.f32"), energy);
        if (direction != null) MethodWrite(Path.Combine(partial, "direction.f32"), direction);
        MethodWrite(Path.Combine(partial, "stats.f32"), stats);
        MethodWrite(Path.Combine(partial, "magnitudes.f32"), magnitudes); // [band, depth, floor(log2|c|)+40]
        if (native != null)
        {
            MethodWrite(Path.Combine(partial, "histogram.f32"), native);
            using (var writer = new BinaryWriter(File.Create(Path.Combine(partial, "counts.u32")))) foreach (uint c in nativeCounts) writer.Write(c);
            var edges = new float[pathBins.Count + 1]; for (int k = 0; k <= pathBins.Count; k++) edges[k] = (float)pathBins.Start(k);
            MethodWrite(Path.Combine(partial, "bin-edges.f32"), edges);
            record.histogramBins = pathBins.Count;
        }
        record.wallSeconds = TakeRender.Elapsed(started);
        record.notes ??= "arrival energies (not band-limited), resampled to 1 ms from the native histogram (histogram.f32, counts.u32, bin-edges.f32)";
        File.WriteAllText(Path.Combine(partial, "run.json"), JsonUtility.ToJson(record, true));
        if (Directory.Exists(output)) Directory.Delete(output, true);
        Directory.Move(partial, output);
        Debug.Log($"Method {name} {spec.sceneId} r{budget} s{seed}: {record.busySeconds:F2} s busy, {record.wallSeconds:F1} s wall");
    }
    // A suite scene's room modes: the bank's materials from the scene's (admittance from absorption + transmission).
    static RoomModes RoomModesFor(MethodSceneSpec spec, string bankPath)
    {
        var bank = RoomModeBank.Load(bankPath); var admittance = new double[bank.Materials.Length][];
        for (int m = 0; m < admittance.Length; m++)
        {
            int at = Array.IndexOf(spec.materialNames, bank.Materials[m]);
            Assert.That(at, Is.GreaterThanOrEqualTo(0), "Bank material " + bank.Materials[m] + " is not in the scene.");
            admittance[m] = new double[8];
            for (int b = 0; b < 8; b++) admittance[m][b] = RoomModes.AdmittanceFromAbsorption(spec.absorption[at * 8 + b] + spec.transmission[at * 8 + b]);
        }
        return new RoomModes(bank, admittance, spec.speedOfSound, 1, 8000, 16000, 16);
    }
}
