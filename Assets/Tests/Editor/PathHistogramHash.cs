using System;
using System.Linq;
using NUnit.Framework;

// Opt-in (ACOUSTIC_PATH_HASH=1): SHA-256 of the Path method's raw outputs (the histogram and the contribution counts,
// fixed-point words) for a scene with every kind of contribution, full solve and persistent rays, fixed seeds, with the source
// at rest (the static walk) and moving (the Doppler walk, with its rate channel).
// Run on two versions of the tracer, it shows whether they are bit-identical (README parts 40 and 54).
public partial class TakeChecks
{
    [Test, Category("QualityMeasurement")]
    public void MeasurePathHistogramHash()
    {
        if (Environment.GetEnvironmentVariable("ACOUSTIC_PATH_HASH") != "1") Assert.Ignore("Set ACOUSTIC_PATH_HASH=1.");
        ApertureRoom(true, false, ApertureOutside, out var scene, out var geometry, out var state, transmission: .05f, roomScattering: .5f);
        foreach (bool moving in new[] {false, true})
        foreach (bool persistent in Environment.GetEnvironmentVariable("ACOUSTIC_PATH_HASH_REPEAT") == "1" ? new[] {false, false, true, true} : new[] {false, true})
        {
            var posed = (AcousticTake.State[])state.Clone();
            if (moving) { posed[0].velocity = new UnityEngine.Vector3(3, 0, 1); posed[0].flags |= AcousticTake.StateFlags.VelocityValid; }
            var settings = ControlSettings(TakeRender.Method.Path, 4096, .3f); settings.diffractionFactor = 1;
            if (int.TryParse(Environment.GetEnvironmentVariable("ACOUSTIC_PATH_HASH_EVENTS"), out int events)) settings.methodMaxEvents = events;
            if (persistent) { settings.pathPersistent = true; settings.responseStepSamples = 256; settings.pathPopulation = int.TryParse(Environment.GetEnvironmentVariable("ACOUSTIC_PATH_HASH_POPULATION"), out int n) ? n : 13000; }
            using var solver = new TransportSolver(scene, geometry, settings, RequireGpu(), sampleRate: 8000);
            if (int.TryParse(Environment.GetEnvironmentVariable("ACOUSTIC_PATH_HASH_FLAGS"), out int flags)) solver.Method.ProfileFlags = flags;
            solver.Solve(posed, new TakeRender.DirectPath[1]);
            if (moving) Assert.That(solver.Method.Doppler, "the moving source runs the Doppler walk");
            var m = solver.Method;
            var bytes = new byte[(m.Output.Length + m.Counts.Length) * 4];
            Buffer.BlockCopy(m.Output, 0, bytes, 0, m.Output.Length * 4); Buffer.BlockCopy(m.Counts, 0, bytes, m.Output.Length * 4, m.Counts.Length * 4);
            using var sha = System.Security.Cryptography.SHA256.Create();
            var dump = Environment.GetEnvironmentVariable("ACOUSTIC_PATH_HASH_OUT");   // optional: the words, for a closer comparison
            if (!string.IsNullOrEmpty(dump)) System.IO.File.WriteAllBytes(System.IO.Path.Combine(dump, (moving ? "moving-" : "") + (persistent ? "persistent.u32" : "full.u32")), bytes);
            int counters = MethodTransport.Bands * MethodTransport.Depths * MethodTransport.StatStride;
            UnityEngine.Debug.Log($"PATH_COUNTERS {(moving ? "moving " : "")}{(persistent ? "persistent" : "full")} " +
                                  string.Join(", ", System.Linq.Enumerable.Range(0, MethodTransport.Counters).Select(c => $"{MethodTransport.CounterNames[c]} {m.Stats[counters + c]}")));
            UnityEngine.Debug.Log($"PATH_HASH {(moving ? "moving " : "")}{(persistent ? "persistent" : "full")} {BitConverter.ToString(sha.ComputeHash(bytes)).Replace("-", "")} ({m.Output.Length} + {m.Counts.Length} words)");
        }
    }
}
