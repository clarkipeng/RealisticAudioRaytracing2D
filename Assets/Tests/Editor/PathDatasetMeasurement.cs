using System;
using System.IO;
using NUnit.Framework;
using UnityEngine;

// Opt-in measurement: Path-method histograms of a recorded take at chosen frames (the Church part
// of the denoiser pilot, tools/path_dataset.py). Every frame is one measurement solve with the
// take's poses at that sample (moving listener and doors as recorded; velocities kept).
//   ACOUSTIC_PATH_TAKE=/abs/job.json     TakeBatch job (take, output, settings; method Path)
//   ACOUSTIC_PATH_SAMPLES=0,1024,...     take samples to solve
//   ACOUSTIC_PATH_RUNS=1024:0:4,16384:100:2   rays:first seed:seeds, one output per (rays, seed)
// Writes <output>/r<rays>/<seed>/histogram.f32 [frame][direction 6][band 8][bin], counts.u32
// [frame][bin], bin-edges.f32, poses.f32 [frame][listener xyz, source xyz] and run.json. Frame f
// uses the seed seed + f * 1000003 (independent noise per frame, as live solves).
public partial class TakeChecks
{
    [Serializable] sealed class PathTakeRecord
    {
        public string schema = "path-take-raw-v1", take, codeSha256, unityVersion, backend;
        public int rays, seed, frames, bins, primitives, edges;
        public long[] samples;
        public float horizon, detectionRadius, binSeconds; public int fineBins;
        public double gpuSeconds, wallSeconds;
        public float[] solveSeconds;
        public string[] counterNames; public long[] counters;
    }

    [Test, Category("QualityMeasurement"), Timeout(int.MaxValue)]
    public void MeasurePathTake()
    {
        string file = Environment.GetEnvironmentVariable("ACOUSTIC_PATH_TAKE");
        if (string.IsNullOrEmpty(file)) Assert.Ignore("Set ACOUSTIC_PATH_TAKE to a TakeBatch job (tools/path_dataset.py).");
        var job = TakeBatch.ParseJob(File.ReadAllText(file));
        var settings = job.settings;
        Assert.That(settings.method, Is.EqualTo(TakeRender.Method.Path));
        var samples = Array.ConvertAll(Environment.GetEnvironmentVariable("ACOUSTIC_PATH_SAMPLES").Split(','), long.Parse);
        var runs = Environment.GetEnvironmentVariable("ACOUSTIC_PATH_RUNS").Split(',');
        string budget = Environment.GetEnvironmentVariable("ACOUSTIC_TEST_WORK_FRACTION");
        double fraction = string.IsNullOrEmpty(budget) ? 1 : double.Parse(budget, System.Globalization.CultureInfo.InvariantCulture);
        string code = MethodCodeHash();
        using var take = AcousticTake.Open(job.take);
        var scene = take.Metadata.scene;
        int listener = scene.listeners[0], source = scene.sources[0].objectIndex;
        var state = new AcousticTake.State[scene.objects.Length];
        foreach (string run in runs)
        {
            var parts = Array.ConvertAll(run.Split(':'), int.Parse);
            settings.methodRays = parts[0];
            using var solver = new TransportSolver(scene, take.Geometry, settings, RequireGpu(), sampleRate: take.Metadata.sampleRate);
            var m = solver.Method; m.WorkFraction = fraction;
            int B = m.TimeBins.Count;
            for (int seed = parts[1]; seed < parts[1] + parts[2]; seed++)
            {
                string output = Path.Combine(job.output, $"r{parts[0]:00000}", seed.ToString());
                if (File.Exists(Path.Combine(output, "run.json"))) continue;
                string partial = output + ".partial";
                if (Directory.Exists(partial)) Directory.Delete(partial, true);
                Directory.CreateDirectory(partial);
                settings.seed = seed;
                var histogram = new float[samples.Length * 6 * 8 * B]; var counts = new uint[samples.Length * B]; var poses = new float[samples.Length * 6];
                var record = new PathTakeRecord {take = job.take, codeSha256 = code, unityVersion = Application.unityVersion, backend = SystemInfo.graphicsDeviceName,
                    rays = parts[0], seed = seed, frames = samples.Length, bins = B, primitives = take.Geometry.Length, edges = m.EdgeCount, samples = samples,
                    horizon = settings.maximumDelaySeconds, detectionRadius = settings.specularDetectionRadius, binSeconds = settings.pathBinSeconds,
                    fineBins = settings.pathFineBins, solveSeconds = new float[samples.Length], counterNames = MethodTransport.CounterNames,
                    counters = new long[MethodTransport.Counters]};
                long started = System.Diagnostics.Stopwatch.GetTimestamp(); double gpuBefore = m.GpuSeconds;
                for (int f = 0; f < samples.Length; f++)
                {
                    take.Evaluate(Math.Min(samples[f], take.Metadata.durationSamples), state);
                    m.SeedOffset = f * 1000003;
                    long t0 = System.Diagnostics.Stopwatch.GetTimestamp();
                    solver.Solve(state, new TakeRender.DirectPath[1]);
                    record.solveSeconds[f] = (float)TakeRender.Elapsed(t0);
                    for (int d = 0; d < 6; d++) for (int b = 0; b < 8; b++) for (int k = 0; k < B; k++)
                        histogram[((f * 6 + d) * 8 + b) * B + k] = (float)m.HistogramEnergy(0, 0, d, b, k);
                    Array.Copy(m.Counts, 0, counts, f * B, B);
                    Vector3 l = state[listener].position, s = state[source].position;
                    poses[6 * f] = l.x; poses[6 * f + 1] = l.y; poses[6 * f + 2] = l.z; poses[6 * f + 3] = s.x; poses[6 * f + 4] = s.y; poses[6 * f + 5] = s.z;
                    for (int c = 0; c < MethodTransport.Counters; c++) record.counters[c] += m.Counter(c);
                }
                record.gpuSeconds = m.GpuSeconds - gpuBefore; record.wallSeconds = TakeRender.Elapsed(started);
                MethodWrite(Path.Combine(partial, "histogram.f32"), histogram);
                using (var writer = new BinaryWriter(File.Create(Path.Combine(partial, "counts.u32")))) foreach (uint c in counts) writer.Write(c);
                var edges = new float[B + 1]; for (int k = 0; k <= B; k++) edges[k] = (float)m.TimeBins.Start(k);
                MethodWrite(Path.Combine(partial, "bin-edges.f32"), edges);
                MethodWrite(Path.Combine(partial, "poses.f32"), poses);
                File.WriteAllText(Path.Combine(partial, "run.json"), JsonUtility.ToJson(record, true));
                if (Directory.Exists(output)) Directory.Delete(output, true);
                Directory.Move(partial, output);
                Debug.Log($"Path take r{parts[0]} s{seed}: {samples.Length} frames, {record.gpuSeconds:F1} s GPU, {record.wallSeconds:F1} s wall");
            }
        }
    }
}
