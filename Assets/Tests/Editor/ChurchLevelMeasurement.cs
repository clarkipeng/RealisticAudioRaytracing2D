using System;
using System.IO;
using NUnit.Framework;
using UnityEngine;

// Opt-in measurement: band energies of the energy and path methods at fixed Church points, same
// horizon, so their level gap can be split by time (docs/benchmarks/path-denoise-2026-11-12).
//   ACOUSTIC_CHURCH_LEVELS=/abs/out.json       results
//   ACOUSTIC_CHURCH_LEVELS_TAKE=/abs/take      a take (its first frame: poses, doors)
//   ACOUSTIC_CHURCH_LEVELS_POINTS=x,y,z;x,y,z  listener positions
// Each method solves with 8,192 rays and a 2 s horizon, air absorption on; the energy method with its
// listening settings (2 m receiver sphere, arrivals, 96 near planes), the path method with its defaults.
// Output per method, point and band: energy in [0, 50 ms), [50, 200 ms), [200 ms, 2 s) after emission.
public partial class TakeChecks
{
    [Test, Category("QualityMeasurement"), Timeout(int.MaxValue)]
    public void MeasureChurchLevels()
    {
        string output = Environment.GetEnvironmentVariable("ACOUSTIC_CHURCH_LEVELS");
        if (string.IsNullOrEmpty(output)) Assert.Ignore("Set ACOUSTIC_CHURCH_LEVELS.");
        using var take = AcousticTake.Open(Environment.GetEnvironmentVariable("ACOUSTIC_CHURCH_LEVELS_TAKE"));
        var scene = take.Metadata.scene;
        var points = Environment.GetEnvironmentVariable("ACOUSTIC_CHURCH_LEVELS_POINTS").Split(';');
        var state = new AcousticTake.State[scene.objects.Length];
        int listener = scene.listeners[0];
        var json = new System.Text.StringBuilder("{");
        // Path variants: ACOUSTIC_CHURCH_LEVELS_PATH = radius:seed;radius:seed (default 0.25:0).
        var variants = new System.Collections.Generic.List<(TakeRender.Method method, float radius, int seed, string key)>();
        foreach (string v in (Environment.GetEnvironmentVariable("ACOUSTIC_CHURCH_LEVELS_ENERGY") ?? "2").Split(';'))
        {
            var rs = v.Split(':'); float radius = float.Parse(rs[0], System.Globalization.CultureInfo.InvariantCulture); int seed = rs.Length > 1 ? int.Parse(rs[1]) : 0;
            variants.Add((TakeRender.Method.Energy, radius, seed, v == "2" ? "Energy" : FormattableString.Invariant($"Energy r{radius} s{seed}")));
        }
        int rays = int.Parse(Environment.GetEnvironmentVariable("ACOUSTIC_CHURCH_LEVELS_RAYS") ?? "8192");
        foreach (string v in (Environment.GetEnvironmentVariable("ACOUSTIC_CHURCH_LEVELS_PATH") ?? "0.25:0").Split(';'))
        {
            var rs = v.Split(':'); float radius = float.Parse(rs[0], System.Globalization.CultureInfo.InvariantCulture); int seed = int.Parse(rs[1]);
            variants.Add((TakeRender.Method.Path, radius, seed, v == "0.25:0" ? "Path" : FormattableString.Invariant($"Path r{radius} s{seed}")));
        }
        foreach (var (method, radius, seed, key) in variants)
        {
            var settings = SpectralSettings();
            settings.method = method; settings.useBvh = true; settings.airAbsorption = true; settings.speedOfSound = 343; settings.minimumDistance = .1f;
            settings.maximumDelaySeconds = 2; settings.methodRays = rays; settings.diffractionFactor = 1; settings.maxBounces = 64;
            settings.specularDetectionRadius = radius; settings.seed = seed;
            string survival = Environment.GetEnvironmentVariable("ACOUSTIC_CHURCH_LEVELS_SURVIVAL");
            if (!string.IsNullOrEmpty(survival)) settings.pathSurvivalSeconds = float.Parse(survival, System.Globalization.CultureInfo.InvariantCulture);
            if (method == TakeRender.Method.Energy) { settings.methodArrivalSlots = 96; settings.methodNearPlanes = 96; }
            using var solver = new TransportSolver(scene, take.Geometry, settings, RequireGpu(), sampleRate: take.Metadata.sampleRate);
            json.Append((key == variants[0].key ? "" : ",") + "\"" + key + "\":[");
            for (int i = 0; i < points.Length; i++)
            {
                take.Evaluate(0, state);
                var xyz = Array.ConvertAll(points[i].Split(','), v => float.Parse(v, System.Globalization.CultureInfo.InvariantCulture));
                state[listener].position = new Vector3(xyz[0], xyz[1], xyz[2]);
                state[listener].velocity = Vector3.zero; state[listener].flags &= ~AcousticTake.StateFlags.VelocityValid;
                solver.Solve(state, new TakeRender.DirectPath[1]);
                json.Append((i == 0 ? "" : ",") + "[");
                for (int b = 0; b < 8; b++)
                {
                    var e = ControlEnergy(solver, b);
                    json.Append(FormattableString.Invariant($"{(b == 0 ? "" : ",")}[{Window(e, 0, 50):R},{Window(e, 50, 200):R},{Window(e, 200, e.Length):R}]"));
                }
                var m = solver.Method;
                json.Append("]");
                var names = MethodTransport.CounterNames; var parts = new System.Text.StringBuilder();
                for (int c = 0; c < MethodTransport.Counters; c++) parts.Append($"{names[c]} {m.Counter(c)}, ");
                Debug.Log($"Church levels {key} point {i}: {parts}");
            }
            json.Append("]");
        }
        json.Append("}");
        File.WriteAllText(output, json.ToString());
    }
}
