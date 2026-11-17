using System;
using System.Collections.Generic;
using System.IO;
using System.Text;
using NUnit.Framework;
using UnityEngine;

// Opt-in measurement: where a Path-method solve spends its time (docs/benchmarks/path-denoise-2026-11-12).
//   ACOUSTIC_PATH_PROFILE=/abs/out.json       results
//   ACOUSTIC_PATH_PROFILE_SCENES=a.json,b.json method-compare scene specs (tools/method_runs.py), first 4 receivers
//   ACOUSTIC_PATH_PROFILE_TAKE=/abs/job.json  optional TakeBatch job (method Path): the take at a few samples
//   ACOUSTIC_PATH_PROFILE_RAYS=2048           rays per solve
//   ACOUSTIC_PATH_PROFILE_PERSISTENT=1        persistent rays (settings.pathPersistent): updates of about 23.2 ms
//                                             (1,024 samples at 44.1 kHz), the time of one update after a horizon of
//                                             them; also the rays in flight and the ray state's memory
//   ACOUSTIC_PATH_PROFILE_BEAMS_ONLY=1        with persistent rays: the beams-only mode (settings.pathBeamsOnly)
//   ACOUSTIC_PATH_PROFILE_POPULATION=1          with persistent rays: the RAYS list is the rays in flight N (settings.pathPopulation)
// For each scene, variants (ACOUSTIC_PATH_PROFILE_FLAGS, ProfileFlags) with a fence after every phase:
// edge distribution and clearance, walk, readback. The "connect" field is zero since the beams replaced
// the chain connections. "candidates" is the most loop iterations of one thread in one dispatch (the walk's
// serial chain; persistent: the largest over the warm-up's updates), "chains" all iterations (persistent: of
// the whole warm-up), per pose.
public partial class TakeChecks
{
    [Test, Category("QualityMeasurement"), Timeout(int.MaxValue)]
    public void MeasurePathProfile()
    {
        string output = Environment.GetEnvironmentVariable("ACOUSTIC_PATH_PROFILE");
        if (string.IsNullOrEmpty(output)) Assert.Ignore("Set ACOUSTIC_PATH_PROFILE.");
        var rayList = Array.ConvertAll((Environment.GetEnvironmentVariable("ACOUSTIC_PATH_PROFILE_RAYS") ?? "2048").Split(','), int.Parse);
        int rays = rayList[0];
        var json = new StringBuilder("{\"rays\":" + rays + ",\"supportsInlineRayTracing\":" + (SystemInfo.supportsInlineRayTracing ? "true" : "false") +
                                     ",\"supportsRayTracing\":" + (SystemInfo.supportsRayTracing ? "true" : "false") + ",\"scenes\":[");
        var shader = RequireGpu(); bool first = true;
        var survivals = Array.ConvertAll((Environment.GetEnvironmentVariable("ACOUSTIC_PATH_PROFILE_SURVIVAL") ?? "-1").Split(','),
                                         v => float.Parse(v, System.Globalization.CultureInfo.InvariantCulture));
        float survival = -1;
        bool persistent = Environment.GetEnvironmentVariable("ACOUSTIC_PATH_PROFILE_PERSISTENT") == "1";
        void Persist(TakeRender.Settings settings, int rate)
        {
            if (!persistent) return;
            settings.pathPersistent = true;
            if (Environment.GetEnvironmentVariable("ACOUSTIC_PATH_PROFILE_POPULATION") == "1") settings.pathPopulation = rays;
            settings.pathBeamsOnly = Environment.GetEnvironmentVariable("ACOUSTIC_PATH_PROFILE_BEAMS_ONLY") == "1";
            settings.responseStepSamples = Math.Max(1, (int)Math.Round(1024.0 / 44100 * rate / settings.spectralHop)) * settings.spectralHop;
        }
        void Record(string name, TransportSolver solver, AcousticTake.State[] state, Action<int> place, int poses)
        {
            var m = solver.Method; m.Profile = true;
            name += FormattableString.Invariant($"|survival {survival}|rays {rays}");
            place(0); solver.Solve(state, new TakeRender.DirectPath[1]); // warm-up
            json.Append((first ? "" : ",") + "{\"scene\":\"" + name + "\",\"rays\":" + rays + ",\"bins\":" + m.TimeBins.Count +
                        FormattableString.Invariant($",\"persistent\":{(persistent ? 1 : 0)},\"newRays\":{m.Rays},\"population\":{solver.Method.RaysInFlight()},\"step\":{m.PersistentStep:R},\"capacity\":{m.PersistentCapacity},\"stateBytes\":{m.PersistentBytes}") + ",\"variants\":[");
            first = false;
            var flagList = Array.ConvertAll((Environment.GetEnvironmentVariable("ACOUSTIC_PATH_PROFILE_FLAGS") ?? "0,1").Split(','), int.Parse);
            foreach (int flags in flagList)
            {
                m.ProfileFlags = flags; Array.Clear(m.PhaseSeconds, 0, 4);
                long events = 0, chains = 0, trials = 0, detections = 0, inFlight = 0;
                for (int p = 0; p < poses; p++)
                {
                    place(p); solver.Solve(state, new TakeRender.DirectPath[1]);
                    events += m.Counter(1); detections += m.Counter(7); chains += m.Counter(8); trials += m.Counter(15); inFlight += m.RaysInFlight();
                    if (m.Counter(9) > 0) Debug.LogWarning($"Path profile {name}: {m.Counter(9)} persistent rays overflowed");
                }
                json.Append((flags == flagList[0] ? "" : ",") + FormattableString.Invariant($"{{\"flags\":{flags},\"inFlight\":{inFlight / poses},\"edge\":{m.PhaseSeconds[0] / poses:R},\"walk\":{m.PhaseSeconds[1] / poses:R},\"connect\":{m.PhaseSeconds[2] / poses:R},\"readback\":{m.PhaseSeconds[3] / poses:R},\"events\":{events / poses},\"candidates\":{detections / poses},\"chains\":{chains / poses},\"trials\":{trials / poses}}}"));
                Debug.Log($"Path profile {name} flags {flags}: walk {m.PhaseSeconds[1] / poses * 1000:F1} ms, connect {m.PhaseSeconds[2] / poses * 1000:F1} ms, edge {m.PhaseSeconds[0] / poses * 1000:F1} ms, readback {m.PhaseSeconds[3] / poses * 1000:F1} ms");
            }
            m.ProfileFlags = 0; m.Profile = false;
            json.Append("]}");
        }
        string scenes = Environment.GetEnvironmentVariable("ACOUSTIC_PATH_PROFILE_SCENES");
        if (!string.IsNullOrEmpty(scenes))
            foreach (string file in scenes.Split(','))
            {
                var spec = JsonUtility.FromJson<MethodSceneSpec>(File.ReadAllText(file));
                var materials = new Dictionary<string, AudioMaterial>();
                foreach (var name in spec.materialNames) materials[name] = MethodMaterial(spec, name);
                string scratch = Path.Combine(Path.GetDirectoryName(output), ".bake-" + spec.sceneId);
                Directory.CreateDirectory(scratch);
                var config = new WaveScene {source = spec.source, points = spec.points, walls = spec.walls, speedOfSound = spec.speedOfSound, bakeSolids = true};
                try
                {
                    using var baked = new DatasetSolid(config, spec.surfaceMaterials, scratch, name => materials[name]);
                    var job = new MethodJob();
                    foreach (int rayCount in rayList)
                    foreach (float value in survivals)
                    {
                        rays = rayCount;
                        survival = value;
                        var settings = MethodSettings(TakeRender.Method.Path, job, rays, 0, spec.speedOfSound);
                        if (value >= 0) settings.pathSurvivalSeconds = value;
                        Persist(settings, 8000);
                        var state = (AcousticTake.State[])baked.states.Clone();
                        using var solver = new TransportSolver(baked.scene, baked.geometry, settings, shader, sampleRate: 8000);
                        Record(spec.sceneId, solver, state, p => state[1].position = WaveVector(spec.points, 3 * p), Math.Min(4, spec.points.Length / 3));
                    }
                }
                finally { foreach (var m in materials.Values) UnityEngine.Object.DestroyImmediate(m); }
            }
        string takeJob = Environment.GetEnvironmentVariable("ACOUSTIC_PATH_PROFILE_TAKE");
        if (!string.IsNullOrEmpty(takeJob))
        {
            var job = TakeBatch.ParseJob(File.ReadAllText(takeJob));
            using var take = AcousticTake.Open(job.take);
            foreach (int rayCount in rayList)
            foreach (float value in survivals)
            {
                survival = value; rays = rayCount;
                var settings = job.settings; settings.methodRays = rays;
                if (value >= 0) settings.pathSurvivalSeconds = value;
                Persist(settings, take.Metadata.sampleRate);
                var state = new AcousticTake.State[take.Metadata.scene.objects.Length];
                using var solver = new TransportSolver(take.Metadata.scene, take.Geometry, settings, shader, sampleRate: take.Metadata.sampleRate);
                long[] samples = {44100, 4 * 44100, 7 * 44100, 9 * 44100};
                Record("church", solver, state, p => take.Evaluate(samples[p], state), samples.Length);
            }
        }
        json.Append("]}");
        File.WriteAllText(output, json.ToString());
    }
}
