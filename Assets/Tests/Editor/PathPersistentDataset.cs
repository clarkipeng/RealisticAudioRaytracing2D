using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using NUnit.Framework;
using UnityEngine;
using UnityEngine.TestTools;

// Opt-in measurement: persistent-ray histograms along listener tracks, for the persistent denoiser
// (tools/path_persistent_dataset.py; docs/benchmarks/path-denoise-2026-11-12, part 25).
//   ACOUSTIC_PERSISTENT_DATASET=/abs/items.json   items (PersistentItem): a generated scene (spec) or a take (TakeBatch job)
// Every item is replayed update by update (1,024 samples at 44.1 kHz) in persistent mode under population
// control: a flush, `warm` updates, then `record` recorded updates, the listener following the item's track
// (or the take's own frames). The input population is calibrated to 23 ms of GPU time per update unless the
// item gives one; the targets use `highFactor` times it with independent seeds, along the same updates, so a
// moving listener's target is the same retarded-time quantity. Writes <output>/<run>/histogram.f32
// [update][direction 6][band 8][bin], counts.u32 [update][bin], bin-edges.f32 and run.json.
public partial class TakeChecks
{
    [Serializable] sealed class PersistentItem
    {
        public string kind, spec, job, output;
        public float[] track;     // scene items: listener xyz per update (warm + record)
        public long[] samples;    // take items: take sample per update
        public int warm, record, population;
        public float highFactor = 8;
        public int[] seeds, highSeeds;
    }
    [Serializable] sealed class PersistentItems { public PersistentItem[] items; }
    [Serializable] sealed class PersistentRunRecord
    {
        public string schema = "path-persistent-raw-v1", codeSha256, kind;
        public int population, newRays, seed, updates, bins, warm;
        public float stepSeconds, horizon, updateMs;
        public long overflowed;
    }

    // A UnityTest: the editor ticks between solvers, which is when Unity frees the released GPU buffers
    // (without the ticks every solver's buffers stayed allocated).
    [UnityTest, Category("QualityMeasurement"), Timeout(int.MaxValue)]
    public IEnumerator MeasurePathPersistentSequences()
    {
        string file = Environment.GetEnvironmentVariable("ACOUSTIC_PERSISTENT_DATASET");
        if (string.IsNullOrEmpty(file)) Assert.Ignore("Set ACOUSTIC_PERSISTENT_DATASET (tools/path_persistent_dataset.py).");
        var list = JsonUtility.FromJson<PersistentItems>(File.ReadAllText(file));
        string code = MethodCodeHash();
        var shader = RequireGpu();
        const int rate = 44100, step = 1024;
        foreach (var item in list.items)
        {
            if (File.Exists(Path.Combine(item.output, "done"))) continue;
            AcousticTake take = null; DatasetSolid baked = null; var materials = new Dictionary<string, AudioMaterial>();
            try
            {
                AcousticTake.Scene scene; AcousticTake.Primitive[] geometry; AcousticTake.State[] state; TakeRender.Settings template; int listener;
                if (item.kind == "take")
                {
                    var job = TakeBatch.ParseJob(File.ReadAllText(item.job));
                    take = AcousticTake.Open(job.take);
                    scene = take.Metadata.scene; geometry = take.Geometry; template = job.settings; listener = scene.listeners[0];
                    state = new AcousticTake.State[scene.objects.Length];
                }
                else
                {
                    var spec = JsonUtility.FromJson<MethodSceneSpec>(File.ReadAllText(item.spec));
                    foreach (var name in spec.materialNames) materials[name] = MethodMaterial(spec, name);
                    string scratch = Path.Combine(item.output, ".bake");
                    Directory.CreateDirectory(scratch);
                    var config = new WaveScene {source = spec.source, points = spec.points, walls = spec.walls, speedOfSound = spec.speedOfSound, bakeSolids = true};
                    baked = new DatasetSolid(config, spec.surfaceMaterials, scratch, name => materials[name]);
                    scene = baked.scene; geometry = baked.geometry; state = (AcousticTake.State[])baked.states.Clone(); listener = 1;
                    template = MethodSettings(TakeRender.Method.Path, new MethodJob {airAbsorption = true}, 1024, 0, spec.speedOfSound);
                }
                int updates = item.warm + item.record, paths = scene.sources.Length * scene.listeners.Length;
                void Place(int u)
                {
                    if (take != null) take.Evaluate(Math.Min(item.samples[u], take.Metadata.durationSamples), state);
                    else state[listener].position = new Vector3(item.track[3 * u], item.track[3 * u + 1], item.track[3 * u + 2]);
                }
                TransportSolver Solver(int population, int seed)
                {
                    var settings = JsonUtility.FromJson<TakeRender.Settings>(JsonUtility.ToJson(template));
                    settings.method = TakeRender.Method.Path; settings.pathPersistent = true; settings.pathPopulation = population;
                    settings.responseStepSamples = step; settings.seed = seed; settings.pathHistogramExport = settings.pathHistogramImport = "";
                    return new TransportSolver(scene, geometry, settings, shader, sampleRate: rate);
                }
                int population = item.population;
                if (population <= 0)
                {
                    // Calibrate: GPU time per update after a horizon of updates, first at 8,192 rays in flight, then
                    // at that estimate (the time is close to, not exactly, proportional to the population).
                    population = 8192;
                    for (int pass = 0; pass < 3; pass++)
                    {
                        GC.Collect(); yield return null;
                        using var probe = Solver(population, 999);
                        probe.Method.PersistentSingleUpdate = true;
                        int warm = Math.Min(item.warm, updates - 20);
                        for (int u = 0; u < warm; u++) { Place(u); probe.Solve(state, new TakeRender.DirectPath[paths]); }
                        double before = probe.Method.GpuSeconds;
                        for (int u = warm; u < warm + 20; u++) { Place(u); probe.Solve(state, new TakeRender.DirectPath[paths]); }
                        double perUpdate = (probe.Method.GpuSeconds - before) / 20;
                        int next = (int)Math.Max(4096, Math.Min(262144, population * .023 / Math.Max(perUpdate, 1e-4)));
                        Debug.Log($"Persistent dataset {item.output}: {perUpdate * 1000:F1} ms per update at {population}, population {next}");
                        population = next;
                    }
                }
                var runs = new List<(int population, int seed, string name)>();
                foreach (int s in item.seeds) runs.Add((population, s, $"low-{s}"));
                foreach (int s in item.highSeeds) runs.Add(((int)Math.Min(1 << 19, population * item.highFactor), s, $"high-{s}"));
                foreach (var (n, seed, name) in runs)
                {
                    string output = Path.Combine(item.output, name);
                    if (File.Exists(Path.Combine(output, "run.json"))) continue;
                    Directory.CreateDirectory(output);
                    GC.Collect(); yield return null;
                    using var solver = Solver(n, seed);
                    var m = solver.Method; m.PersistentSingleUpdate = true;
                    int B = m.TimeBins.Count;
                    var histogram = new float[item.record * 6 * 8 * B]; var counts = new uint[item.record * B];
                    double gpu = 0;
                    for (int u = 0; u < updates; u++)
                    {
                        Place(u);
                        double before = m.GpuSeconds;
                        solver.Solve(state, new TakeRender.DirectPath[paths]);
                        if (u < item.warm) continue;
                        gpu += m.GpuSeconds - before;
                        int r = u - item.warm;
                        for (int d = 0; d < 6; d++) for (int b = 0; b < 8; b++) for (int k = 0; k < B; k++)
                            histogram[((r * 6 + d) * 8 + b) * B + k] = (float)m.HistogramEnergy(0, 0, d, b, k);
                        Array.Copy(m.Counts, 0, counts, r * B, B);
                    }
                    MethodWrite(Path.Combine(output, "histogram.f32"), histogram);
                    using (var writer = new BinaryWriter(File.Create(Path.Combine(output, "counts.u32")))) foreach (uint c in counts) writer.Write(c);
                    var edges = new float[B + 1]; for (int k = 0; k <= B; k++) edges[k] = (float)m.TimeBins.Start(k);
                    MethodWrite(Path.Combine(output, "bin-edges.f32"), edges);
                    var record = new PersistentRunRecord {codeSha256 = code, kind = item.kind, population = n, newRays = m.Rays, seed = seed, updates = item.record,
                        bins = B, warm = item.warm, stepSeconds = m.PersistentStep, horizon = template.maximumDelaySeconds,
                        updateMs = (float)(1000 * gpu / item.record), overflowed = m.RaysOverflowed()};
                    File.WriteAllText(Path.Combine(output, "run.json"), JsonUtility.ToJson(record, true));
                    Debug.Log($"Persistent dataset {output}: population {n}, {record.updateMs:F1} ms per update; resident {System.Diagnostics.Process.GetCurrentProcess().WorkingSet64 >> 20} MiB, graphics driver {UnityEngine.Profiling.Profiler.GetAllocatedMemoryForGraphicsDriver() >> 20} MiB, managed {GC.GetTotalMemory(false) >> 20} MiB");
                }
                File.WriteAllText(Path.Combine(item.output, "done"), population.ToString());
                GC.Collect(); yield return null;
                Debug.Log($"Persistent dataset {item.output} released: resident {System.Diagnostics.Process.GetCurrentProcess().WorkingSet64 >> 20} MiB, graphics driver {UnityEngine.Profiling.Profiler.GetAllocatedMemoryForGraphicsDriver() >> 20} MiB, managed {GC.GetTotalMemory(false) >> 20} MiB");
            }
            finally
            {
                take?.Dispose(); baked?.Dispose();
                foreach (var material in materials.Values) UnityEngine.Object.DestroyImmediate(material);
            }
        }
    }
}
