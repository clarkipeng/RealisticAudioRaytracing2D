using System;
using NUnit.Framework;

// Opt-in (ACOUSTIC_PATH_HASH=1): SHA-256 of the Path method's raw outputs (the histogram and the contribution counts,
// fixed-point words) for a static scene with every kind of contribution, full solve and persistent rays, fixed seeds.
// Run on two versions of the tracer, it shows whether they are bit-identical in a static scene (README part 40).
public partial class TakeChecks
{
    [Test, Category("QualityMeasurement")]
    public void MeasurePathHistogramHash()
    {
        if (Environment.GetEnvironmentVariable("ACOUSTIC_PATH_HASH") != "1") Assert.Ignore("Set ACOUSTIC_PATH_HASH=1.");
        ApertureRoom(true, false, ApertureOutside, out var scene, out var geometry, out var state, transmission: .05f, roomScattering: .5f);
        foreach (bool persistent in Environment.GetEnvironmentVariable("ACOUSTIC_PATH_HASH_REPEAT") == "1" ? new[] {false, false, true, true} : new[] {false, true})
        {
            var settings = ControlSettings(TakeRender.Method.Path, 4096, .3f); settings.diffractionFactor = 1;
            if (persistent) { settings.pathPersistent = true; settings.responseStepSamples = 256; settings.pathPopulation = 13000; }
            using var solver = new TransportSolver(scene, geometry, settings, RequireGpu(), sampleRate: 8000);
            solver.Solve(state, new TakeRender.DirectPath[1]);
            var m = solver.Method;
            var bytes = new byte[(m.Output.Length + m.Counts.Length) * 4];
            Buffer.BlockCopy(m.Output, 0, bytes, 0, m.Output.Length * 4); Buffer.BlockCopy(m.Counts, 0, bytes, m.Output.Length * 4, m.Counts.Length * 4);
            using var sha = System.Security.Cryptography.SHA256.Create();
            var dump = Environment.GetEnvironmentVariable("ACOUSTIC_PATH_HASH_OUT");   // optional: the words, for a closer comparison
            if (!string.IsNullOrEmpty(dump)) System.IO.File.WriteAllBytes(System.IO.Path.Combine(dump, persistent ? "persistent.u32" : "full.u32"), bytes);
            UnityEngine.Debug.Log($"PATH_HASH {(persistent ? "persistent" : "full")} {BitConverter.ToString(sha.ComputeHash(bytes)).Replace("-", "")} ({m.Output.Length} + {m.Counts.Length} words)");
        }
    }
}
