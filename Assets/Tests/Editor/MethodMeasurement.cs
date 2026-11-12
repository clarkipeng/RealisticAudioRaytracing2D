using System;
using System.Collections.Generic;
using System.IO;
using System.Text;
using NUnit.Framework;
using UnityEngine;
using Complex = System.Numerics.Complex;

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
    // Scratch for the band analysis, allocated once at the largest size: per-call
    // arrays of a million doubles grew the editor heap by about a gigabyte per run.
    static double[] scratchRe = new double[0], scratchIm = new double[0];
    public static double[] TrainRe = new double[0], TrainIm = new double[0];
    // Shared train buffers, at least `length` long and cleared.
    public static void TrainBuffers(int length)
    {
        if (TrainRe.Length < length) { TrainRe = new double[length]; TrainIm = new double[length]; }
        Array.Clear(TrainRe, 0, length); Array.Clear(TrainIm, 0, length);
    }
    // Band-limited energy per bin of one band train (`length` samples at rate, sample n
    // at time n / rate - lead; values are arrival gains, complex for impedance phases).
    public static double[] Energy(double[] trainRe, double[] trainIm, int length, int band, double rate, double lead, int bins, double binSeconds)
    {
        var energy = new double[bins];
        bool any = false;
        for (int i = 0; i < length && !any; i++) any = trainRe[i] != 0 || trainIm[i] != 0;
        if (!any) return energy;
        // The train already holds the lead (pre-ringing) and 0.43 s past the 2 s horizon,
        // so one power of two at or above its length cannot wrap into the output bins.
        int n = 1;
        while (n < length) n <<= 1;
        if (scratchRe.Length < n) { scratchRe = new double[n]; scratchIm = new double[n]; } // grow only
        var re = scratchRe; var im = scratchIm;
        Array.Clear(re, 0, n); Array.Clear(im, 0, n);
        Array.Copy(trainRe, re, length); Array.Copy(trainIm, im, length);
        Fft(re, im, n, false);
        double fc = Centers[band];
        for (int k = 0; k < n; k++)
        {
            double g = k <= n / 2 ? 2 * Math.Sqrt(Power(k * rate / n, fc)) : 0;
            re[k] *= g; im[k] *= g;
        }
        Fft(re, im, n, true);
        double norm = FilterEnergy(fc);
        for (int i = 0; i < n; i++)
        {
            // Negative times (pre-ringing inside the lead) count in bin 0, as in method_compare.
            double p = rate * re[i], t = i / rate - lead;
            int k = Math.Max(0, (int)Math.Floor(t / binSeconds));
            if (k < bins) energy[k] += p * p / rate / norm;
        }
        return energy;
    }
    public static Complex Response(double[] re, double[] im, int length, double rate, double lead, double f)
    {
        Complex sum = Complex.Zero;
        double step = -2 * Math.PI * f / rate;
        for (int i = 0; i < length; i++)
            if (re[i] != 0 || im[i] != 0) sum += new Complex(re[i], im[i]) * Complex.FromPolarCoordinates(1, step * i);
        return sum * Complex.FromPolarCoordinates(1, 2 * Math.PI * f * lead);
    }
    // 4-tap cubic Lagrange deposit, identical to MethodDepositPressure on the GPU.
    public static void Deposit(double[] re, double[] im, int length, double rate, double lead, double delay, Complex gain)
    {
        double u = (delay + lead) * rate;
        int i = (int)Math.Floor(u);
        if (i < 1 || i + 2 >= length) return;
        double w = u - i;
        double[] c = {-w * (w - 1) * (w - 2) / 6, (w + 1) * (w - 1) * (w - 2) / 2, -(w + 1) * w * (w - 2) / 2, (w + 1) * w * (w - 1) / 6};
        for (int t = 0; t < 4; t++) { re[i - 1 + t] += gain.Real * c[t]; im[i - 1 + t] += gain.Imaginary * c[t]; }
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
        public int receivers = -1, captureReceiver = -1, captureRays = 48, captureEvents = 192, hybridRouteFactor = 1;
        // Dataset runs (tools/path_dataset.py): a new seed per receiver (frames of a moving listener
        // get independent noise, as live solves do), and air absorption.
        public bool seedPerReceiver, airAbsorption;
    }
    [Serializable] sealed class MethodJobList { public MethodJob[] jobs; }
    [Serializable] sealed class MethodRunRecord
    {
        public string schema = "methods-raw-v1", sceneId, sceneSha256, method, backend, unityVersion, codeSha256;
        public int budget, seed, receivers, bins, primitives, edges, droppedEdges, lateRays, completeRouteSamples, methodRays, histogramBins;
        public float detectionRadius, rouletteFloor, maximumDelaySeconds, workFraction, binSeconds;
        public float[] frequencies, solveSeconds;
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
        foreach (var name in new[] {"Shaders/Geometry.hlsl", "Shaders/Transport.hlsl", "Shaders/Transport.compute", "Shaders/CompleteDiffraction.hlsl",
                                   "Shaders/DiffractionIntegral.hlsl", "Shaders/LateEnergy.hlsl", "Shaders/MethodCommon.hlsl", "Shaders/MethodPressure.hlsl",
                                   "Shaders/MethodEnergy.hlsl", "Shaders/MethodPath.hlsl", "Scripts/Simulation/MethodTransport.cs", "Scripts/Simulation/RayTransport.cs"})
            all.AddRange(File.ReadAllBytes(Path.Combine(Application.dataPath, name)));
        return BitConverter.ToString(sha.ComputeHash(all.ToArray())).Replace("-", "").ToLowerInvariant();
    }
    TakeRender.Settings MethodSettings(TakeRender.Method method, MethodJob job, int budget, int seed, float speed)
    {
        var settings = SpectralSettings();
        settings.method = method; settings.useBvh = true; settings.airAbsorption = job.airAbsorption; settings.speedOfSound = speed;
        settings.minimumDistance = .01f; settings.maximumDelaySeconds = job.maximumDelaySeconds; settings.seed = seed;
        settings.diffractionFactor = job.diffractionFactor;
        // Hybrid: 32 kHz / 256-point grid puts an FFT bin on every band centre (125 Hz
        // steps); 32-sample steps make the late histogram's bins 1 ms.
        settings.fftSize = 256; settings.spectralHop = 32; settings.responseStepSamples = 32; settings.maximumResponsesPerBatch = 1;
        settings.raysPerBin = 1; settings.reflectionCandidates = 1 << 20; settings.maxBounces = 4096;
        settings.lateRays = budget; settings.completeRouteSamples = job.hybridRouteFactor * budget;
        settings.methodRays = budget; settings.specularDetectionRadius = job.detectionRadius;
        settings.methodEdgeProbability = job.edgeProbability; settings.methodRouletteFloor = job.rouletteFloor;
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
                var method = name == "pressure" ? TakeRender.Method.Pressure : name == "energy" ? TakeRender.Method.Energy : name == "path" ? TakeRender.Method.Path : TakeRender.Method.Hybrid;
                Assert.That(name, Is.EqualTo("hybrid").Or.EqualTo("pressure").Or.EqualTo("energy").Or.EqualTo("path"));
                for (int b = 0; b < job.budgets.Length; b++)
                    foreach (int seed in job.seeds)
                    {
                        string output = Path.Combine(job.output, spec.sceneId, name, $"r{job.budgets[b]:00000}", seed.ToString());
                        if (File.Exists(Path.Combine(output, "run.json"))) continue;
                        bool capture = job.captureReceiver >= 0 && b == 0 && seed == job.seeds[0];
                        RenderMethod(spec, baked, method, name, job, job.budgets[b], seed, receivers, output, shader, fraction, code, capture);
                    }
            }
        }
        finally { foreach (var m in materials.Values) UnityEngine.Object.DestroyImmediate(m); }
    }
    void RenderMethod(MethodSceneSpec spec, DatasetSolid baked, TakeRender.Method method, string name, MethodJob job, int budget, int seed,
                      int receivers, string output, ComputeShader shader, double fraction, string code, bool capture)
    {
        long started = System.Diagnostics.Stopwatch.GetTimestamp();
        string partial = output + ".partial";
        if (Directory.Exists(partial)) Directory.Delete(partial, true);
        Directory.CreateDirectory(partial);
        var settings = MethodSettings(method, job, budget, seed, spec.speedOfSound);
        bool hybrid = method == TakeRender.Method.Hybrid;
        int rate = hybrid ? 32000 : 8000, bins = (int)Math.Round(job.maximumDelaySeconds / MethodTransport.BinSeconds);
        var state = (AcousticTake.State[])baked.states.Clone();
        var frequencies = spec.frequencies ?? new float[0];
        var energy = new float[receivers * 8 * bins]; var late = hybrid ? new float[energy.Length] : null;
        bool histogramMethod = method == TakeRender.Method.Energy || method == TakeRender.Method.Path;
        var direction = histogramMethod ? new float[receivers * 8 * bins * 6] : null;
        var complex = histogramMethod ? null : new float[receivers * frequencies.Length * 2];
        // Path: the native histogram [receiver][direction][band][bin] and contributions [receiver][bin].
        float[] native = null; uint[] nativeCounts = null; PathBins pathBins = null;
        var stats = hybrid ? null : new float[receivers * 8 * MethodTransport.Depths * 4];
        var magnitudes = hybrid ? null : new float[8 * MethodTransport.Depths * MethodTransport.Magnitudes]; // summed over receivers
        var counters = new long[MethodTransport.Counters];
        var record = new MethodRunRecord {sceneId = spec.sceneId, sceneSha256 = spec.sceneSha256, method = name, budget = budget, seed = seed,
            backend = SystemInfo.graphicsDeviceName, unityVersion = Application.unityVersion, codeSha256 = code, receivers = receivers, bins = bins,
            primitives = baked.geometry.Length, binSeconds = MethodTransport.BinSeconds, maximumDelaySeconds = job.maximumDelaySeconds,
            workFraction = (float)fraction, frequencies = frequencies, solveSeconds = new float[receivers], detectionRadius = job.detectionRadius, rouletteFloor = job.rouletteFloor,
            lateRays = hybrid ? settings.lateRays : 0, completeRouteSamples = hybrid ? settings.completeRouteSamples : 0, methodRays = hybrid ? 0 : budget,
            counterNames = MethodTransport.CounterNames, counters = counters};
        double analysis = 0;
        using (var solver = new TransportSolver(baked.scene, baked.geometry, settings, shader, sampleRate: rate))
        {
            if (solver.Method != null) { solver.Method.WorkFraction = fraction; record.edges = solver.Method.EdgeCount; record.droppedEdges = solver.Method.DroppedEdges; }
            var direct = new TakeRender.DirectPath[1];
            state[1].position = WaveVector(spec.points); solver.Solve(state, direct); // exclude the cold dispatch
            double gpuBefore = solver.Method?.GpuSeconds ?? 0;
            for (int p = 0; p < receivers; p++)
            {
                state[1].position = WaveVector(spec.points, 3 * p);
                if (job.seedPerReceiver && solver.Method != null) solver.Method.SeedOffset = p * 1000003;
                long t0 = System.Diagnostics.Stopwatch.GetTimestamp();
                solver.Solve(state, direct);
                double work = TakeRender.Elapsed(t0);
                record.solveSeconds[p] = (float)work; record.busySeconds += work;
                long a0 = System.Diagnostics.Stopwatch.GetTimestamp();
                if (hybrid) HybridReceiver(solver, settings, rate, p, bins, frequencies, energy, late, complex);
                else
                {
                    var m = solver.Method;
                    if (method == TakeRender.Method.Path)
                    {
                        pathBins ??= m.TimeBins; int B = pathBins.Count;
                        native ??= new float[receivers * 6 * 8 * B]; nativeCounts ??= new uint[receivers * B];
                        for (int d = 0; d < 6; d++) for (int b = 0; b < 8; b++) for (int k = 0; k < B; k++)
                            native[((p * 6 + d) * 8 + b) * B + k] = (float)m.HistogramEnergy(0, 0, d, b, k);
                        Array.Copy(m.Counts, 0, nativeCounts, p * B, B);
                    }
                    if (histogramMethod)
                    {
                        for (int b = 0; b < 8; b++) for (int k = 0; k < bins; k++)
                        {
                            double sum = 0;
                            for (int d = 0; d < 6; d++) { double e = m.Energy(d, b, k); sum += e; direction[((p * 8 + b) * bins + k) * 6 + d] = (float)e; }
                            energy[(p * 8 + b) * bins + k] = (float)sum;
                        }
                    }
                    else PressureReceiver(m, p, bins, frequencies, energy, complex);
                    for (int b = 0; b < 8; b++) for (int d = 0; d < MethodTransport.Depths; d++)
                    {
                        var s = m.Depth(b, d); int at = ((p * 8 + b) * MethodTransport.Depths + d) * 4;
                        stats[at] = s.count; stats[at + 1] = (float)s.sum; stats[at + 2] = (float)s.square; stats[at + 3] = (float)s.max;
                        for (int h = 0; h < MethodTransport.Magnitudes; h++) magnitudes[(b * MethodTransport.Depths + d) * MethodTransport.Magnitudes + h] += m.Histogram(b, d, h);
                    }
                    for (int c = 0; c < counters.Length; c++) counters[c] += m.Counter(c);
                }
                analysis += TakeRender.Elapsed(a0);
                if (hybrid && fraction < 1) System.Threading.Thread.Sleep((int)Math.Ceiling(work * 1000 * (1 / fraction - 1)));
            }
            record.gpuSeconds = (solver.Method?.GpuSeconds ?? 0) - gpuBefore;
            if (capture && job.captureReceiver < receivers)
            {
                state[1].position = WaveVector(spec.points, 3 * job.captureReceiver);
                File.WriteAllText(Path.Combine(partial, "capture.json"), hybrid ? HybridCapture(solver, state, job.captureReceiver)
                    : MethodCaptureJson(solver.CaptureMethod(state, job.captureRays, job.captureEvents), job.captureRays, job.captureEvents,
                                        job.captureReceiver, state[0].position, state[1].position));
            }
        }
        record.analysisSeconds = analysis;
        MethodWrite(Path.Combine(partial, "energy.f32"), energy);
        if (late != null) MethodWrite(Path.Combine(partial, "late.f32"), late);
        if (direction != null) MethodWrite(Path.Combine(partial, "direction.f32"), direction);
        if (complex != null) MethodWrite(Path.Combine(partial, "complex.f32"), complex);
        if (stats != null) MethodWrite(Path.Combine(partial, "stats.f32"), stats);
        if (magnitudes != null) MethodWrite(Path.Combine(partial, "magnitudes.f32"), magnitudes); // [band, depth, floor(log2|c|)+40]
        if (native != null)
        {
            MethodWrite(Path.Combine(partial, "histogram.f32"), native);
            using (var writer = new BinaryWriter(File.Create(Path.Combine(partial, "counts.u32")))) foreach (uint c in nativeCounts) writer.Write(c);
            var edges = new float[pathBins.Count + 1]; for (int k = 0; k <= pathBins.Count; k++) edges[k] = (float)pathBins.Start(k);
            MethodWrite(Path.Combine(partial, "bin-edges.f32"), edges);
            record.histogramBins = pathBins.Count;
        }
        record.wallSeconds = TakeRender.Elapsed(started);
        record.notes = hybrid ? "early (direct, first images, complete routes) band-limited in energy.f32; late histogram (not band-limited) in late.f32"
                              : method == TakeRender.Method.Pressure ? "band-limited energies from the coherent band trains"
                              : method == TakeRender.Method.Path ? "arrival energies (not band-limited), resampled to 1 ms from the native histogram (histogram.f32, counts.u32, bin-edges.f32)"
                              : "arrival energies (not band-limited)";
        File.WriteAllText(Path.Combine(partial, "run.json"), JsonUtility.ToJson(record, true));
        if (Directory.Exists(output)) Directory.Delete(output, true);
        Directory.Move(partial, output);
        Debug.Log($"Method {name} {spec.sceneId} r{budget} s{seed}: {record.busySeconds:F2} s busy, {record.wallSeconds:F1} s wall");
    }
    // Hybrid arrivals: every RayPath of the band-centre FFT bins (125 Hz * 2^b at
    // 32 kHz / 256 points) is a delayed complex gain; the late histogram adds energy.
    static void HybridReceiver(TransportSolver solver, TakeRender.Settings settings, int rate, int p, int bins, float[] frequencies,
                               float[] energy, float[] late, float[] complex)
    {
        double step = (double)rate / settings.fftSize;
        var arrivals = new List<(int band, double delay, Complex gain)>();
        foreach (var path in solver.RayPaths)
        {
            if (path.pressureGain == 0 && path.pressureImag == 0) continue;
            double hz = path.frequencyBin * step; int band = MethodBands.BandOf(hz);
            if (Math.Abs(hz - MethodBands.Centers[band]) > 1e-3) continue;
            Assert.That(float.IsFinite(path.pressureGain) && float.IsFinite(path.delaySeconds), Is.True, "Non-finite hybrid arrival.");
            var gain = new Complex(path.pressureGain, path.pressureImag);
            arrivals.Add((band, path.delaySeconds, gain));
        }
        for (int b = 0; b < 8; b++)
        {
            int length = MethodTransportTrainLength(b); double trainRate = MethodTransport.TrainRate0 * (1 << b);
            MethodBands.TrainBuffers(length);
            foreach (var a in arrivals)
                if (a.band == b) MethodBands.Deposit(MethodBands.TrainRe, MethodBands.TrainIm, length, trainRate, MethodTransport.LeadSeconds, a.delay, a.gain);
            var e = MethodBands.Energy(MethodBands.TrainRe, MethodBands.TrainIm, length, b, trainRate, MethodTransport.LeadSeconds, bins, MethodTransport.BinSeconds);
            for (int k = 0; k < bins; k++) energy[(p * 8 + b) * bins + k] = (float)e[k];
        }
        var frame = solver.Late(0); int lateBins = frame.Transport.LateBins;
        Assert.That(settings.LatePartition, Is.EqualTo((int)Math.Round(rate * MethodTransport.BinSeconds)), "Late bins must be 1 ms.");
        for (int b = 0; b < 8; b++) for (int k = 0; k < Math.Min(bins, lateBins); k++)
        {
            double sum = 0; for (int d = 0; d < 6; d++) sum += frame.Energy(0, d, b, k);
            late[(p * 8 + b) * bins + k] = (float)sum;
        }
        for (int q = 0; q < frequencies.Length; q++)
        {
            int band = MethodBands.BandOf(frequencies[q]); Complex sum = Complex.Zero;
            foreach (var a in arrivals) if (a.band == band) sum += a.gain * Complex.FromPolarCoordinates(1, -2 * Math.PI * frequencies[q] * a.delay);
            complex[2 * (p * frequencies.Length + q)] = (float)sum.Real; complex[2 * (p * frequencies.Length + q) + 1] = (float)sum.Imaginary;
        }
    }
    static int MethodTransportTrainLength(int band) => (int)Math.Round(MethodTransport.TrainSeconds * MethodTransport.TrainRate0) << band;
    static void PressureReceiver(MethodTransport m, int p, int bins, float[] frequencies, float[] energy, float[] complex)
    {
        for (int b = 0; b < 8; b++)
        {
            int length = m.TrainLength(b);
            MethodBands.TrainBuffers(length);
            m.Train(b, MethodBands.TrainRe, MethodBands.TrainIm);
            var e = MethodBands.Energy(MethodBands.TrainRe, MethodBands.TrainIm, length, b, m.TrainRate(b), MethodTransport.LeadSeconds, bins, MethodTransport.BinSeconds);
            for (int k = 0; k < bins; k++) energy[(p * 8 + b) * bins + k] = (float)e[k];
            for (int q = 0; q < frequencies.Length; q++)
            {
                if (MethodBands.BandOf(frequencies[q]) != b) continue;
                var h = MethodBands.Response(MethodBands.TrainRe, MethodBands.TrainIm, length, m.TrainRate(b), MethodTransport.LeadSeconds, frequencies[q]);
                complex[2 * (p * frequencies.Length + q)] = (float)h.Real; complex[2 * (p * frequencies.Length + q) + 1] = (float)h.Imaginary;
            }
        }
    }
    static string Json(Vector3 v) => FormattableString.Invariant($"[{v.x:R},{v.y:R},{v.z:R}]");
    // Captured events: per ray, kind / position / weight / contribution / time / extra.
    static string MethodCaptureJson(Vector4[] data, int rays, int events, int receiver, Vector3 source, Vector3 listener)
    {
        var text = new StringBuilder();
        text.Append("{\"schema\":\"methods-capture-v1\",\"receiver\":").Append(receiver).Append(",\"source\":").Append(Json(source))
            .Append(",\"listener\":").Append(Json(listener)).Append(",\"kinds\":{\"1\":\"source\",\"2\":\"specular\",\"3\":\"diffuse\",\"4\":\"transmission\",\"5\":\"edge\",")
            .Append("\"6\":\"connection\",\"7\":\"chain\",\"8\":\"escape\",\"9\":\"end\",\"10\":\"edgeConnection\"},\"rays\":[");
        for (int r = 0; r < rays; r++)
        {
            text.Append(r == 0 ? "[" : ",[");
            bool first = true;
            for (int e = 0; e < events; e++)
            {
                var a = data[2 * (r * events + e)]; var b = data[2 * (r * events + e) + 1];
                if (a.w == 0) break;
                text.Append(first ? "" : ",").Append(FormattableString.Invariant($"[{(int)a.w},{a.x:R},{a.y:R},{a.z:R},{b.x:R},{b.y:R},{b.z:R},{b.w:R}]"));
                first = false;
            }
            text.Append("]");
        }
        return text.Append("]}").ToString().Replace("NaN", "null").Replace("Infinity", "1e39");
    }
    static string HybridCapture(TransportSolver solver, AcousticTake.State[] state, int receiver)
    {
        var segments = solver.CaptureRays(state, 0, 4); // 500 Hz bin
        var text = new StringBuilder();
        text.Append("{\"schema\":\"methods-capture-hybrid-v1\",\"receiver\":").Append(receiver).Append(",\"source\":").Append(Json(state[0].position))
            .Append(",\"listener\":").Append(Json(state[1].position)).Append(",\"segments\":[");
        bool first = true;
        foreach (var s in segments)
        {
            if (s.kind == TransportSolver.RayEvent.None) continue;
            text.Append(first ? "" : ",").Append(FormattableString.Invariant(
                $"[{(int)s.kind},{s.path},{s.start.x:R},{s.start.y:R},{s.start.z:R},{s.end.x:R},{s.end.y:R},{s.end.z:R},{s.end.w:R},{s.pressure:R}]"));
            first = false;
        }
        return text.Append("]}").ToString().Replace("NaN", "null").Replace("Infinity", "1e39");
    }
}
