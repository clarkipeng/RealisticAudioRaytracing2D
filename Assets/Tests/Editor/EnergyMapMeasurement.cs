using System;
using System.IO;
using NUnit.Framework;
using UnityEngine;

// Opt-in measurement: energy-method band energies on a grid of receivers in a take's scene
// (docs/benchmarks/energy-audio-2026-11-09/church_map.py). Every receiver is one measurement
// solve with the take's source and geometry at a chosen frame (doors closed or open).
//   ACOUSTIC_ENERGY_MAP=/abs/job.json            TakeBatch job (take, output, settings; method Energy)
//   ACOUSTIC_ENERGY_MAP_RECEIVERS=/abs/xyz.f32   receiver positions, float32 x, y, z
//   ACOUSTIC_ENERGY_MAP_FRAMES=0,650             take frames (state ticks) to solve
//   ACOUSTIC_ENERGY_MAP_RAYS=2048,8192           optional ray counts (default: the job's); one map each
//   ACOUSTIC_ENERGY_MAP_SEEDS=4                  optional: average this many walk seeds per receiver
// Writes <output>/map-<frame>[-r<rays>][-s<seeds>].f32 [receiver][quantity: direct, early, late, total][band 8]
// (energies in the direct-sound convention, free field at 1 m = 1; early = up to 80 ms after
// the direct sound's delay, or after the geometric delay where the direct sound is blocked) and
// map-<frame>.json with the timing.
public partial class TakeChecks
{
    [Serializable] sealed class EnergyMapMeta { public int frame, receivers, rays, seeds; public float horizon, earlySeconds, radius; public double gpuSeconds, wallSeconds; }

    [Test, Category("QualityMeasurement"), Timeout(int.MaxValue)]
    public void MeasureEnergyMap()
    {
        string file = Environment.GetEnvironmentVariable("ACOUSTIC_ENERGY_MAP");
        if (string.IsNullOrEmpty(file)) Assert.Ignore("Set ACOUSTIC_ENERGY_MAP to a TakeBatch job.");
        var job = TakeBatch.ParseJob(File.ReadAllText(file));
        var settings = job.settings;
        Assert.That(settings.method, Is.EqualTo(TakeRender.Method.Energy));
        var bytes = File.ReadAllBytes(Environment.GetEnvironmentVariable("ACOUSTIC_ENERGY_MAP_RECEIVERS"));
        var xyz = new float[bytes.Length / 4]; Buffer.BlockCopy(bytes, 0, xyz, 0, bytes.Length);
        int receivers = xyz.Length / 3;
        var frames = Array.ConvertAll((Environment.GetEnvironmentVariable("ACOUSTIC_ENERGY_MAP_FRAMES") ?? "0").Split(','), int.Parse);
        string rayList = Environment.GetEnvironmentVariable("ACOUSTIC_ENERGY_MAP_RAYS");
        var rayCounts = string.IsNullOrEmpty(rayList) ? new[] {settings.methodRays} : Array.ConvertAll(rayList.Split(','), int.Parse);
        string seedText = Environment.GetEnvironmentVariable("ACOUSTIC_ENERGY_MAP_SEEDS");
        int seeds = string.IsNullOrEmpty(seedText) ? 1 : int.Parse(seedText);
        using var take = AcousticTake.Open(job.take);
        var scene = take.Metadata.scene;
        Directory.CreateDirectory(job.output);
        var state = new AcousticTake.State[scene.objects.Length];
        const float early = .08f;
        int listener = scene.listeners[0], source = scene.sources[0].objectIndex;
        foreach (int rays in rayCounts)
        foreach (int frame in frames)
        {
            settings.methodRays = rays;
            using var solver = new TransportSolver(scene, take.Geometry, settings, RequireGpu(), sampleRate: take.Metadata.sampleRate);
            string budget = Environment.GetEnvironmentVariable("ACOUSTIC_TEST_WORK_FRACTION");
            solver.Method.WorkFraction = string.IsNullOrEmpty(budget) ? 1 : double.Parse(budget, System.Globalization.CultureInfo.InvariantCulture);
            string name = $"map-{frame}" + (rayCounts.Length > 1 ? $"-r{rays}" : "") + (seeds > 1 ? $"-s{seeds}" : "");
            long started = System.Diagnostics.Stopwatch.GetTimestamp(); double gpuBefore = solver.Method.GpuSeconds;
            take.Evaluate(Math.Min(frame * 882L, take.Metadata.durationSamples), state);
            var values = new float[receivers * 4 * 8];
            for (int r = 0; r < receivers; r++)
            {
                state[listener].position = new Vector3(xyz[3 * r], xyz[3 * r + 1], xyz[3 * r + 2]);
                state[listener].velocity = Vector3.zero; state[listener].flags &= ~AcousticTake.StateFlags.VelocityValid;
                for (int seed = 0; seed < seeds; seed++)
                {
                    solver.Method.SeedOffset = seed;
                    solver.Solve(state, new TakeRender.DirectPath[1]);
                    var m = solver.Method;
                    var arrivals = m.ReadArrivals();
                    double delay = (state[listener].position - state[source].position).magnitude / settings.speedOfSound;
                    var direct = new double[8];
                    foreach (var a in arrivals)
                        if (a.Family == 0) { delay = a.delay; for (int b = 0; b < 8; b++) direct[b] += a.Energy(b); }
                    int split = (int)Math.Floor((delay + early) / MethodTransport.BinSeconds);
                    for (int b = 0; b < 8; b++)
                    {
                        double before = 0, after = 0;
                        for (int k = 0; k < m.Bins; k++)
                        {
                            double e = 0;
                            for (int d = 0; d < 6; d++) e += m.Energy(d, b, k);
                            if (k <= split) before += e; else after += e;
                        }
                        int at = (r * 4) * 8 + b;
                        values[at] += (float)(direct[b] / seeds); values[at + 8] += (float)(before / seeds);
                        values[at + 16] += (float)(after / seeds); values[at + 24] += (float)((before + after) / seeds);
                    }
                }
            }
            using (var writer = new BinaryWriter(File.Create(Path.Combine(job.output, name + ".f32"))))
                foreach (float v in values) writer.Write(v);
            File.WriteAllText(Path.Combine(job.output, name + ".json"), JsonUtility.ToJson(new EnergyMapMeta {frame = frame, receivers = receivers, rays = rays, seeds = seeds,
                horizon = settings.maximumDelaySeconds, earlySeconds = early, radius = settings.specularDetectionRadius,
                gpuSeconds = solver.Method.GpuSeconds - gpuBefore, wallSeconds = TakeRender.Elapsed(started)}, true));
            TestContext.WriteLine($"{name}: {receivers} receivers in {TakeRender.Elapsed(started):F0} s, GPU {solver.Method.GpuSeconds - gpuBefore:F1} s");
        }
    }
}
