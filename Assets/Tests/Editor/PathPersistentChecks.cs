using System;
using System.Collections.Generic;
using NUnit.Framework;
using UnityEngine;

// Checks of the Path method's persistent rays (settings.pathPersistent, MethodPath.hlsl "Persistent rays";
// docs/benchmarks/path-denoise-2026-11-12, part 17). The analytic controls run in persistent mode too
// (their Path cases with persistent = true, see PersistentControl).
public partial class TakeChecks
{
    // Persistent variant of a check: updates of 32 ms (256 samples at the checks' 8 kHz) and room for
    // `inFlight` times the new rays of an update (0: the automatic bound); a measurement reads the last update
    // after a horizon of them. In a static scene every update costs about one full solve (each cohort covers
    // one update of its life), so a check's warm-up costs as many full solves as it has updates.
    static void PersistentControl(TakeRender.Settings settings, bool persistent, int inFlight)
    {
        if (!persistent) return;
        settings.pathPersistent = true; settings.responseStepSamples = 256; settings.pathRaysInFlight = inFlight * settings.methodRays;
    }
    static void AssertNoOverflow(TransportSolver solver)
    {
        if (!solver.Method.IsPath || solver.Method.PersistentCapacity == 0) return;
        TestContext.WriteLine($"persistent: {solver.Method.RaysInFlight()} rays in flight of {solver.Method.PersistentCapacity}, {solver.Method.Counter(15)} continued, {solver.Method.Counter(14)} branches");
        Assert.That(solver.Method.RaysOverflowed(), Is.EqualTo(0), "persistent rays overflowed their pool");
    }

    // A static scene with every kind of contribution (the aperture room with a ground outside, half-specular
    // walls, diffraction: beams, scattering points, edge rays, edge connections, the direct sound), no
    // transmission. With the same seed for every update's new rays, each update launches the full solve's rays,
    // so after the warm-up the response holds each of their contributions exactly once, made in the update in
    // which the ray gets there and added to the update in which it is heard: it must equal the full solve's to
    // rounding (rays stopped and resumed in mid-air). Updates of 16 ms, 5.5 m of travel (the room is 6 m long).
    [Test, Category("GPU"), Category("Methods"), Category("Slow"), Timeout(1800000)]
    public void PersistentRaysAreTheFullSolveCutIntoUpdates()
    {
        ApertureRoom(true, false, ApertureOutside, out var scene, out var geometry, out var state, roomScattering: .5f);
        double[][] Solve(bool persistent)
        {
            var settings = ControlSettings(TakeRender.Method.Path, 2048, .3f); settings.diffractionFactor = 1;
            if (persistent) { settings.pathPersistent = true; settings.responseStepSamples = 128; }
            using var solver = new TransportSolver(scene, geometry, settings, RequireGpu(), sampleRate: 8000);
            solver.Method.PersistentRepeatSeed = true;
            solver.Solve(state, new TakeRender.DirectPath[1]);
            if (persistent) { AssertNoOverflow(solver); Assert.That(solver.Method.Counter(15), Is.GreaterThan(0)); }
            var bands = new double[8][];
            for (int b = 0; b < 8; b++) bands[b] = ControlEnergy(solver, b);
            return bands;
        }
        var full = Solve(false); var cut = Solve(true);
        foreach (int b in new[] {0, 2, 5})
        {
            double total = Window(full[b], 0, int.MaxValue), difference = 0;
            for (int k = 0; k < full[b].Length; k++) difference += Math.Abs(cut[b][k] - full[b][k]);
            TestContext.WriteLine($"band {b}: full {total:G6}, persistent {Window(cut[b], 0, int.MaxValue):G6}, sum of |differences| {difference / total:E2} of the total; " +
                                  $"windows 0-20 / 20-100 / 100-300 ms: {Window(cut[b], 0, 20) / Window(full[b], 0, 20):F5} / {Window(cut[b], 20, 100) / Window(full[b], 20, 100):F5} / {Window(cut[b], 100, 300) / Window(full[b], 100, 300):F5}");
            Assert.That(difference / total, Is.LessThan(2e-3));
        }
    }

    // The same room with transmitting walls (t = 0.05: branch reservoirs, queued as rays of their own when a ray
    // stops) and independent seeds: the persistent response equals the full solve's in expectation, window by window.
    [Test, Category("GPU"), Category("Methods"), Category("Slow"), Timeout(1800000)]
    public void PersistentRaysMatchTheFullSolveInExpectation()
    {
        ApertureRoom(true, false, ApertureOutside, out var scene, out var geometry, out var state, transmission: .05f, roomScattering: .5f);
        int[][] windows = {new[] {0, 20}, new[] {20, 60}, new[] {60, 500}};
        int seeds = int.Parse(Environment.GetEnvironmentVariable("ACOUSTIC_PERSISTENT_SEEDS") ?? "6"); // more: a closer look
        var sums = new double[2, 2, 3]; var squares = new double[2, 2, 3];
        long branches = 0;
        for (int mode = 0; mode < 2; mode++)
            for (int seed = 0; seed < seeds; seed++)
            {
                var settings = ControlSettings(TakeRender.Method.Path, 4096, .5f); settings.diffractionFactor = 1; settings.seed = 100 + seed;
                if (mode == 1) { settings.pathPersistent = true; settings.responseStepSamples = 256; }
                using var solver = new TransportSolver(scene, geometry, settings, RequireGpu(), sampleRate: 8000);
                solver.Solve(state, new TakeRender.DirectPath[1]);
                if (mode == 1) { AssertNoOverflow(solver); branches += solver.Method.Counter(14); }
                for (int band = 0; band < 2; band++)
                {
                    var e = ControlEnergy(solver, band == 0 ? 2 : 5);
                    for (int w = 0; w < 3; w++) { double x = Window(e, windows[w][0], windows[w][1]); sums[mode, band, w] += x; squares[mode, band, w] += x * x; }
                }
            }
        Assert.That(branches, Is.GreaterThan(0), "the walls' transmission makes branches");
        for (int band = 0; band < 2; band++)
            for (int w = 0; w < 3; w++)
            {
                double[] mean = new double[2], error = new double[2];
                for (int mode = 0; mode < 2; mode++)
                {
                    mean[mode] = sums[mode, band, w] / seeds;
                    error[mode] = Math.Sqrt(Math.Max(0, squares[mode, band, w] / seeds - mean[mode] * mean[mode]) / (seeds - 1));
                }
                double tolerance = 3 * Math.Sqrt(error[0] * error[0] + error[1] * error[1]) + .02 * mean[0];
                TestContext.WriteLine($"{(band == 0 ? 500 : 4000)} Hz, {windows[w][0]}-{windows[w][1]} ms: full {mean[0]:G5} +- {error[0]:G3}, persistent {mean[1]:G5} +- {error[1]:G3}, ratio {mean[1] / mean[0]:F3}");
                Assert.That(Math.Abs(mean[1] - mean[0]), Is.LessThan(tolerance));
            }
    }

    // A source moving at 100 m/s past a rigid plane, the listener still (audio solver, updates of 4 ms). The
    // response of an update must hold the sound the listener hears then, emitted earlier: the direct sound (the
    // exact delayed path) and the plane's image at their retarded delays, the image (traced by the persistent
    // rays) with the level of its retarded distance, 1 / (c tau)^2; a frozen source would put it 10 ms earlier.
    [Test, Category("GPU"), Category("Methods"), Category("Slow"), Timeout(1800000)]
    public void PersistentRaysHearAMovingSourceAtTheRetardedTime()
    {
        var scene = Scene(2); scene.listeners = new[] {1}; scene.materials = new[] {RigidControl()};
        var a = new Vector3(-200, 0, -200); var b = new Vector3(200, 0, -200); var c = new Vector3(200, 0, 200); var d = new Vector3(-200, 0, 200);
        var geometry = new[] {new AcousticTake.Primitive {objectIndex = 2, a = a, b = c, c = b}, new AcousticTake.Primitive {objectIndex = 2, a = a, b = d, c = c}};
        scene.objects[2].primitiveCount = geometry.Length;
        var state = States(2); var L = new Vector3(12, 1.5f, 3); var v = new Vector3(100, 0, 0);
        state[1].position = L; state[2].position = Vector3.zero; state[2].flags = AcousticTake.StateFlags.Active | AcousticTake.StateFlags.Geometry;
        state[0].velocity = v; state[0].flags |= AcousticTake.StateFlags.VelocityValid;
        var settings = ControlSettings(TakeRender.Method.Path, 1 << 17, .06f); settings.useBvh = false;
        settings.pathPersistent = true; settings.responseStepSamples = 32; settings.fftSize = 64; settings.spectralHop = 16; settings.pathRaysInFlight = 3 << 17;
        using var solver = new TransportSolver(scene, geometry, settings, RequireGpu(), sampleRate: 8000, methodAudio: true);
        var m = solver.Method; float dt = m.PersistentStep; int updates = m.WarmUpdates + 4;
        var S0 = new Vector3(2.8f, 2, 0) - v * ((updates - 1) * dt); // the source passes x = 2.8 at the last update
        Vector3 S = S0;
        for (int n = 0; n < updates; n++)
        {
            S = S0 + v * (n * dt); state[0].position = S;
            solver.Solve(state, new TakeRender.DirectPath[1]);
        }
        AssertNoOverflow(solver);
        const float speed = 343;
        Doppler.Direct(L - S, v, Vector3.zero, speed, out float direct, out _, out _);
        Vector3 image = new Vector3(S.x, -S.y, S.z), mirrored = new Vector3(v.x, -v.y, v.z);
        Doppler.Direct(L - image, mirrored, Vector3.zero, speed, out float retarded, out _, out _);
        double frozen = (L - image).magnitude / speed;
        double heard = 0, energy = 0;
        for (int k = 0; k < m.TimeBins.Count; k++)
        {
            double e = 0;
            for (int dir = 0; dir < 6; dir++) e += m.HistogramEnergy(0, 0, dir, 1, k);
            energy += e; heard += e * (m.TimeBins.Start(k) + .5 * m.TimeBins.Width(k));
        }
        heard /= energy;
        double level = energy * (speed * retarded) * (speed * retarded);
        float pathDelay = float.NaN;
        foreach (var path in solver.RayPaths) if (path.pressureGain > 0) { pathDelay = path.delaySeconds; break; }
        TestContext.WriteLine($"direct path {pathDelay * 1000:F4} ms vs retarded {direct * 1000:F4} ms; image heard at {heard * 1000:F3} ms vs retarded {retarded * 1000:F3} ms " +
                              $"(frozen {frozen * 1000:F3} ms), level {level:F3} x 1/(c tau)^2");
        Assert.That(Math.Abs(frozen - retarded), Is.GreaterThan(.005), "the geometry separates retarded from frozen");
        Assert.That(pathDelay, Is.EqualTo(direct).Within(2e-6));
        Assert.That(heard, Is.EqualTo(retarded).Within(.0005));
        Assert.That(level, Is.EqualTo(1).Within(.15));
    }

    // A door filling the aperture room's hole (like the walls, a = 0.1, s = 1; diffraction off) opens at once
    // after the closed room has rung for a horizon; the listener stands outside on the hole's axis, 10 m away.
    // Closed, nothing reaches it. After the opening nothing is heard before the first sound through the hole can
    // arrive (the direct sound, 42 ms: the room's surfaces it sees through the hole are on the far wall, 16 m
    // away), something is from then on, and once the opening is a horizon old the level is the open room's.
    [Test, Category("GPU"), Category("Methods"), Category("Slow"), Timeout(1800000)]
    public void PersistentRaysHearADoorOpenAfterTheTravelTime()
    {
        var listener = new Vector3(13, ApertureY, ApertureZ);
        ApertureRoom(false, false, listener, out var scene, out var room, out var state);
        // Exactly the hole, in the wall's plane: no crack between door and frame (a 1 cm crack leaked 0.2%).
        float x = ApertureSize.x / 2, y0 = ApertureY - ApertureHalfY, y1 = ApertureY + ApertureHalfY, z0 = ApertureZ - ApertureHalfZ, z1 = ApertureZ + ApertureHalfZ;
        int door = scene.objects.Length;
        var geometry = new List<AcousticTake.Primitive>(room)
        {
            new AcousticTake.Primitive {objectIndex = door, a = new Vector3(x, y0, z0), b = new Vector3(x, y1, z0), c = new Vector3(x, y1, z1)},
            new AcousticTake.Primitive {objectIndex = door, a = new Vector3(x, y0, z0), b = new Vector3(x, y1, z1), c = new Vector3(x, y0, z1)}
        };
        var objects = new List<AcousticTake.ObjectDefinition>(scene.objects)
            {new AcousticTake.ObjectDefinition {id = "door", name = "door", scale = Vector3.one, primitiveStart = room.Length, primitiveCount = 2}};
        scene.objects = objects.ToArray();
        var states = new List<AcousticTake.State>(state) {new AcousticTake.State {rotation = Quaternion.identity, flags = AcousticTake.StateFlags.Active | AcousticTake.StateFlags.Geometry}};
        state = states.ToArray();
        double open;
        {
            var reference = ControlSettings(TakeRender.Method.Path, 1 << 15, .4f); reference.pathSurvivalSeconds = .02f;
            state[door].position = new Vector3(0, 0, 100);
            open = 0;
            for (int seed = 0; seed < 2; seed++)
            {
                reference.seed = 200 + seed;
                using var full = new TransportSolver(scene, geometry.ToArray(), reference, RequireGpu(), sampleRate: 8000);
                full.Solve(state, new TakeRender.DirectPath[1]);
                open += Window(ControlEnergy(full, 2), 0, int.MaxValue) / 2;
            }
        }
        var settings = ControlSettings(TakeRender.Method.Path, 8192, .4f); settings.pathPersistent = true; settings.responseStepSamples = 64;
        settings.pathSurvivalSeconds = .02f; // fewer rays in flight (unbiased; the reference uses the same)
        using var solver = new TransportSolver(scene, geometry.ToArray(), settings, RequireGpu(), sampleRate: 8000);
        var m = solver.Method; float dt = m.PersistentStep;
        state[door].position = Vector3.zero;
        solver.Solve(state, new TakeRender.DirectPath[1]); // closed, a horizon of updates
        double closed = Window(ControlEnergy(solver, 2), 0, int.MaxValue);
        m.PersistentSingleUpdate = true;
        state[door].position = new Vector3(0, 0, 100);
        int updates = m.WarmUpdates + 10; var outside = new double[updates];
        for (int n = 0; n < updates; n++)
        {
            solver.Solve(state, new TakeRender.DirectPath[1]);
            outside[n] = Window(ControlEnergy(solver, 2), 0, int.MaxValue);
        }
        AssertNoOverflow(solver);
        double first = (listener - ApertureSource).magnitude / 343.0, late = 0;
        for (int n = updates - 10; n < updates; n++) late += outside[n] / 10;
        var curve = new System.Text.StringBuilder();
        for (int n = 0; n < Math.Min(updates, 16); n++) curve.Append($"{outside[n] / open:F3} ");
        TestContext.WriteLine($"closed {closed:G4}, open (full solve) {open:G4}; after the opening, per {dt * 1000:F0} ms update / open: {curve}...; last 10 updates {late / open:F3}; direct sound {first * 1000:F1} ms");
        // (Not exactly zero: a connection grazing a seam between the wall's triangles can slip through, 1e-4 of the open level.)
        Assert.That(closed, Is.LessThan(1e-3 * open));
        int silent = (int)Math.Ceiling(first / dt - 1); // updates whose span ends before first - dt / 2
        for (int n = 0; n < silent; n++) Assert.That(outside[n], Is.LessThan(1e-3 * open), $"update {n} hears the opening before the sound could arrive");
        for (int n = silent; n < silent + 6; n++) Assert.That(outside[n], Is.GreaterThan(.3 * open), $"update {n} hears the opening");
        Assert.That(late / open, Is.EqualTo(1).Within(.1));
    }
}
