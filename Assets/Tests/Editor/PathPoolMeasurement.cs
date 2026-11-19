using System;
using System.Collections.Generic;
using System.IO;
using NUnit.Framework;
using UnityEngine;

// Opt-in measurement: the persistent rays the walk keeps, before population control resamples them
// (docs/benchmarks/path-denoise-2026-11-12, part 31 onward: the sort's cells and the resampling's weights).
//   ACOUSTIC_PATH_POOL=/abs/dir                  output: <scene>-<snapshot>.f32, [ray][16] = position xyz, age,
//                                                direction xyz, flags (1 in a branch, + 2 * chain kind), flux per band (8)
//   ACOUSTIC_PATH_POOL_TAKE=/abs/job.json        optional: the Church take (TakeBatch job), static at 7 s, N = 11,000,
//                                                the playable Church's settings (cap 16, edge slice every 4th update)
//   ACOUSTIC_PATH_POOL_SCENES=a.json,b.json      optional: method-compare scene specs, N = 16,384, the first receiver
// Each scene: a horizon of updates, then 8 snapshots one update apart. pool.json lists the snapshots and boxes.
public partial class TakeChecks
{
    [Test, Category("QualityMeasurement"), Timeout(int.MaxValue)]
    public void MeasurePathPool()
    {
        string output = Environment.GetEnvironmentVariable("ACOUSTIC_PATH_POOL");
        if (string.IsNullOrEmpty(output)) Assert.Ignore("Set ACOUSTIC_PATH_POOL.");
        Directory.CreateDirectory(output);
        var shader = RequireGpu();
        var log = new List<string>();
        void Snapshots(string name, TransportSolver solver, AcousticTake.State[] state, AcousticTake.Primitive[] geometry, int listener)
        {
            var m = solver.Method;
            solver.Solve(state, new TakeRender.DirectPath[1]); // a flush and a horizon of updates
            m.PersistentSingleUpdate = true;
            Vector3 low = Vector3.positiveInfinity, high = Vector3.negativeInfinity;
            foreach (var g in geometry)
            {
                var pose = state[g.objectIndex];
                foreach (var v in new[] {g.a, g.b, g.c}) { var w = pose.position + pose.rotation * v; low = Vector3.Min(low, w); high = Vector3.Max(high, w); }
            }
            for (int s = 0; s < 8; s++)
            {
                solver.Solve(state, new TakeRender.DirectPath[1]);
                var rays = m.ReadWalkPool(out int count);
                var data = new float[count * 16];
                for (int i = 0; i < count; i++)
                {
                    int o = 16 * i, r = MethodTransport.PathRayFloats * i; // position, direction, phiLow, phiHigh, wLow, wHigh, plane, scalars, meta
                    for (int k = 0; k < 7; k++) data[o + k] = rays[r + k];
                    uint meta = BitConverter.ToUInt32(BitConverter.GetBytes(rays[r + 32]), 0);
                    data[o + 7] = ((meta >> 24) & 1) + 2 * ((meta >> 27) & 3);
                    for (int k = 0; k < 8; k++) data[o + 8 + k] = rays[r + 8 + k] * rays[r + 16 + k];
                }
                MethodWrite(Path.Combine(output, $"{name}-{s}.f32"), data);
                log.Add(FormattableString.Invariant($"{{\"scene\":\"{name}\",\"snapshot\":{s},\"rays\":{count},\"newRays\":{m.Rays},\"step\":{m.PersistentStep:R},") +
                        FormattableString.Invariant($"\"box\":[{low.x},{low.y},{low.z},{high.x},{high.y},{high.z}],\"listener\":[{state[listener].position.x},{state[listener].position.y},{state[listener].position.z}]}}"));
            }
            Debug.Log($"PATH_POOL {name}: {m.Rays} new rays per update, overflow {m.RaysOverflowed()}");
        }
        string takeJob = Environment.GetEnvironmentVariable("ACOUSTIC_PATH_POOL_TAKE");
        if (!string.IsNullOrEmpty(takeJob))
        {
            var job = TakeBatch.ParseJob(File.ReadAllText(takeJob));
            using var take = AcousticTake.Open(job.take);
            var settings = job.settings;
            settings.method = TakeRender.Method.Path; settings.pathPersistent = true; settings.pathPopulation = 11000;
            settings.pathIterationCap = 16; settings.pathEdgeSliceUpdates = 4; settings.pathHistogramExport = settings.pathHistogramImport = "";
            var state = new AcousticTake.State[take.Metadata.scene.objects.Length];
            take.Evaluate(7 * 44100, state);
            using var solver = new TransportSolver(take.Metadata.scene, take.Geometry, settings, shader, sampleRate: take.Metadata.sampleRate);
            Snapshots("church", solver, state, take.Geometry, take.Metadata.scene.listeners[0]);
        }
        string scenes = Environment.GetEnvironmentVariable("ACOUSTIC_PATH_POOL_SCENES");
        if (!string.IsNullOrEmpty(scenes))
            foreach (string file in scenes.Split(','))
            {
                var spec = JsonUtility.FromJson<MethodSceneSpec>(File.ReadAllText(file));
                var materials = new Dictionary<string, AudioMaterial>();
                foreach (var name in spec.materialNames) materials[name] = MethodMaterial(spec, name);
                string scratch = Path.Combine(output, ".bake-" + spec.sceneId);
                Directory.CreateDirectory(scratch);
                var config = new WaveScene {source = spec.source, points = spec.points, walls = spec.walls, speedOfSound = spec.speedOfSound, bakeSolids = true};
                try
                {
                    using var baked = new DatasetSolid(config, spec.surfaceMaterials, scratch, name => materials[name]);
                    var settings = MethodSettings(TakeRender.Method.Path, new MethodJob {airAbsorption = true}, 1024, 0, spec.speedOfSound);
                    settings.pathPersistent = true; settings.pathPopulation = 16384; settings.responseStepSamples = 1024;
                    var state = (AcousticTake.State[])baked.states.Clone();
                    state[1].position = WaveVector(spec.points, 0);
                    using var solver = new TransportSolver(baked.scene, baked.geometry, settings, shader, sampleRate: 44100);
                    Snapshots(spec.sceneId, solver, state, baked.geometry, 1);
                }
                finally { foreach (var m in materials.Values) UnityEngine.Object.DestroyImmediate(m); }
            }
        File.WriteAllText(Path.Combine(output, "pool.json"), "[" + string.Join(",\n", log) + "]\n");
    }
}
