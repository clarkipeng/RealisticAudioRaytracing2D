using System;
using System.Collections.Generic;
using NUnit.Framework;
using UnityEngine;

// Checks of the wave walk and its hardware queries (settings.pathQueries, README part 52; MethodPath.hlsl "Waves",
// Native/AcousticRT). The waves run the megakernel's walk query for query, so with the software tree answering (WavesSoftware)
// the response must be the megakernel's to rounding; Metal's intersector must give the software tree's hits.
public partial class TakeChecks
{
    double[][] WaveSolve(AcousticTake.Scene scene, AcousticTake.Primitive[] geometry, AcousticTake.State[] state, TakeRender.PathQueries queries,
                                int population, int waves, out uint[] header)
    {
        var settings = ControlSettings(TakeRender.Method.Path, 2048, .3f); settings.diffractionFactor = 1;
        settings.pathPersistent = true; settings.responseStepSamples = 128; settings.pathQueries = queries; settings.pathPopulation = population;
        using var solver = new TransportSolver(scene, geometry, settings, RequireGpu(), sampleRate: 8000);
        Assert.That(solver.Method.Waves, Is.EqualTo(queries != TakeRender.PathQueries.Software), "the wave walk runs when asked for");
        solver.Method.PersistentRepeatSeed = true;
        if (waves > 0) solver.Method.WaveCount = waves;
        solver.Solve(state, new TakeRender.DirectPath[1]);
        AssertNoOverflow(solver);
        header = solver.Method.ReadWaveHeader();
        var bands = new double[8][];
        for (int b = 0; b < 8; b++) bands[b] = ControlEnergy(solver, b);
        return bands;
    }
    static double WaveDifference(double[][] a, double[][] b, string name)
    {
        double worst = 0;
        foreach (int band in new[] {0, 2, 5})
        {
            double total = Window(a[band], 0, int.MaxValue), difference = 0;
            for (int k = 0; k < a[band].Length; k++) difference += Math.Abs(b[band][k] - a[band][k]);
            worst = Math.Max(worst, difference / total);
            TestContext.WriteLine($"{name}, band {band}: megakernel {total:G7}, waves {Window(b[band], 0, int.MaxValue):G7}, sum of |differences| {difference / total:E2} of the total");
        }
        return worst;
    }

    // The aperture room with a ground outside, half-specular transmitting walls (branches, partition crossings) and diffraction
    // (edge rays, edge connections), persistent rays with the same seed every update, with and without population control: the
    // wave walk with software answers is the megakernel to rounding, also with few waves (the finishing walk takes most rays), and
    // with Metal's answers it differs only where a hit does (a grazing ray on a triangle's edge).
    [Test, Category("GPU"), Category("Methods"), Category("Slow"), Timeout(1800000)]
    public void WaveWalkIsTheMegakernelQueryForQuery()
    {
        ApertureRoom(true, false, ApertureOutside, out var scene, out var geometry, out var state, transmission: .05f, roomScattering: .5f);
        foreach (int population in new[] {0, 6000})
        {
            var reference = WaveSolve(scene, geometry, state, TakeRender.PathQueries.Software, population, 0, out _);
            var waves = WaveSolve(scene, geometry, state, TakeRender.PathQueries.WavesSoftware, population, 0, out var header);
            TestContext.WriteLine($"population {population}: waves {header[15]}, rays stopped {header[16]}, queries {header[17]}, to the finish list {header[18]}");
            var one = WaveSolve(scene, geometry, state, TakeRender.PathQueries.WavesSoftware, population, 1, out _);
            var few = WaveSolve(scene, geometry, state, TakeRender.PathQueries.WavesSoftware, population, 3, out _);
            double dWaves = WaveDifference(reference, waves, "software answers"), dOne = WaveDifference(reference, one, "software answers, 1 wave"),
                   dFew = WaveDifference(reference, few, "software answers, 3 waves");
            Assert.That(header[15], Is.GreaterThan(1)); Assert.That(header[17], Is.GreaterThan(0));
            Assert.That(dWaves, Is.LessThan(1e-5)); Assert.That(dOne, Is.LessThan(1e-5)); Assert.That(dFew, Is.LessThan(1e-5));
            if (!HardwareQueries.Available) { TestContext.WriteLine("no Metal ray tracing here: hardware answers not checked"); continue; }
            var hardware = WaveSolve(scene, geometry, state, TakeRender.PathQueries.Hardware, population, 0, out header);
            TestContext.WriteLine($"hardware: waves {header[15]}, rays stopped {header[16]}, queries {header[17]}");
            Assert.That(WaveDifference(reference, hardware, "hardware answers"), Is.LessThan(2e-3));
        }
    }

    // Opt-in (ACOUSTIC_WARM_PATH=1): every walk kernel compiled in a light process (static and moving, megakernel and waves), so
    // that a heavy measurement after a shader change does not compile them on top of its own memory.
    [Test, Category("GPU"), Timeout(3600000)]
    public void WarmPathKernels()
    {
        if (Environment.GetEnvironmentVariable("ACOUSTIC_WARM_PATH") != "1") Assert.Ignore("Set ACOUSTIC_WARM_PATH=1.");
        ApertureRoom(true, false, ApertureOutside, out var scene, out var geometry, out var state, transmission: .05f);
        foreach (var queries in new[] {TakeRender.PathQueries.Software, TakeRender.PathQueries.Hardware})
        {
            var settings = ControlSettings(TakeRender.Method.Path, 256, .3f);
            settings.pathPersistent = true; settings.responseStepSamples = 128; settings.pathQueries = queries;
            using var solver = new TransportSolver(scene, geometry, settings, RequireGpu(), sampleRate: 8000);
            solver.Method.PersistentSingleUpdate = true;
            solver.Solve(state, new TakeRender.DirectPath[1]);
            var moving = (AcousticTake.State[])state.Clone();
            moving[0].velocity = Vector3.right; moving[0].flags |= AcousticTake.StateFlags.VelocityValid; moving[0].position += Vector3.right * .01f;
            solver.Solve(moving, new TakeRender.DirectPath[1]);
            TestContext.WriteLine($"{queries}: waves {solver.Method.Waves}, Doppler {solver.Method.Doppler}");
        }
    }

    // Random rays in each scene (uniform origins in its box, uniform directions; closest hits to 1 km, any-hit segments, and the
    // single-object probes of point location on every closed object): Metal's intersector agrees with the software tree, the
    // same primitive and the distance within 1e-4 m + 1e-5 of it, except at a tie (another primitive at the same distance) and on
    // rays that graze an edge, a set of measure zero that must stay below 1e-3 of the rays.
    [Test, Category("GPU"), Category("Methods"), Timeout(1800000)]
    public void HardwareQueriesAgreeWithTheSoftwareTree()
    {
        if (!HardwareQueries.Available) Assert.Ignore("No Metal ray tracing here.");
        var scenes = new List<(string name, AcousticTake.Scene scene, AcousticTake.Primitive[] geometry, AcousticTake.State[] state)>();
        {
            ApertureRoom(true, false, ApertureOutside, out var s, out var g, out var st, transmission: .05f); scenes.Add(("aperture room", s, g, st));
            ApertureRoom(true, true, ApertureOutside, out s, out g, out st, porch: true); scenes.Add(("porch and screen", s, g, st));
            ApertureRoom(false, false, ApertureInside, out s, out g, out st);
            // a pose that is neither identity nor axis-aligned, for the instance transforms
            for (int o = 0; o < st.Length; o++) if (s.objects[o].primitiveCount > 0) { st[o].rotation = Quaternion.Euler(17, 31, -9); st[o].position += new Vector3(.3f, -.2f, .7f); break; }
            scenes.Add(("rotated room", s, g, st));
            KnifeEdgeFixture(false, out s, out _, out st);
            g = TransportBox(s.dimension, 2, 0, 1); s.objects[2].primitiveCount = g.Length;
            st[0].position = Vector3.left * 3; st[1].position = Vector3.right * 3;
            st[2].rotation = Quaternion.Euler(-23, 41, 7); st[2].position += new Vector3(.2f, .1f, -.3f);
            scenes.Add(("solid box, rotated", s, g, st));
        }
        foreach (var (name, scene, geometry, state) in scenes)
        {
            var settings = ControlSettings(TakeRender.Method.Path, 256, .3f);
            settings.pathPersistent = true; settings.responseStepSamples = 128; settings.pathQueries = TakeRender.PathQueries.Hardware;
            using var solver = new TransportSolver(scene, geometry, settings, RequireGpu(), sampleRate: 8000);
            solver.Method.PersistentSingleUpdate = true;
            solver.Solve(state, new TakeRender.DirectPath[1]);
            Vector3 low = Vector3.positiveInfinity, high = Vector3.negativeInfinity;
            var world = new Vector3[3 * geometry.Length];
            for (int i = 0; i < geometry.Length; i++)
            {
                var pose = state[geometry[i].objectIndex];
                world[3 * i] = pose.position + pose.rotation * geometry[i].a; world[3 * i + 1] = pose.position + pose.rotation * geometry[i].b; world[3 * i + 2] = pose.position + pose.rotation * geometry[i].c;
                for (int k = 0; k < 3; k++) { low = Vector3.Min(low, world[3 * i + k]); high = Vector3.Max(high, world[3 * i + k]); }
            }
            var meshes = new List<int>();
            foreach (int o in solver.Method.VolumeOwners) if (solver.MeshRoot(o) >= 0) meshes.Add(o);
            var random = new System.Random(7);
            float U() => (float)random.NextDouble();
            Vector3 Point() => new Vector3(Mathf.Lerp(low.x, high.x, U()), Mathf.Lerp(low.y, high.y, U()), Mathf.Lerp(low.z, high.z, U()));
            Vector3 Direction() { float z = 2 * U() - 1, a = 2 * Mathf.PI * U(), r = Mathf.Sqrt(1 - z * z); return new Vector3(r * Mathf.Cos(a), r * Mathf.Sin(a), z); }
            const int count = 1 << 16;
            var records = new uint[12 * count];
            for (int i = 0; i < count; i++)
            {
                Vector3 o = Point(), d = Direction(); float max = 1000; uint any = 0, obj = 0, root = 0;
                int kind = i % 4;
                if (kind == 1) { var e = Point(); d = e - o; max = d.magnitude; d /= Mathf.Max(max, 1e-6f); any = 1; }
                if (kind == 3 && meshes.Count > 0) { int c = meshes[i / 4 % meshes.Count]; obj = (uint)c + 1; root = (uint)solver.MeshRoot(c); }
                uint F(float x) => BitConverter.ToUInt32(BitConverter.GetBytes(x), 0);
                int at = 12 * i;
                records[at] = F(o.x); records[at + 1] = F(o.y); records[at + 2] = F(o.z); records[at + 3] = F(i % 8 == 0 ? 0 : 1e-4f);
                records[at + 4] = F(d.x); records[at + 5] = F(d.y); records[at + 6] = F(d.z); records[at + 7] = F(max);
                records[at + 8] = any; records[at + 9] = obj; records[at + 10] = root;
            }
            var software = solver.Method.AnswerQueries(records, false);
            var hardware = solver.Method.AnswerQueries(records, true);
            int hits = 0, differ = 0, ties = 0; double worst = 0;
            for (int i = 0; i < count; i++)
            {
                uint hs = software[2 * i], hh = hardware[2 * i];
                float ts = BitConverter.ToSingle(BitConverter.GetBytes(software[2 * i + 1]), 0), th = BitConverter.ToSingle(BitConverter.GetBytes(hardware[2 * i + 1]), 0);
                bool any = records[12 * i + 8] != 0;
                if (hs != uint.MaxValue) hits++;
                void Report(string why)
                {
                    if (differ > 8) return;
                    int at = 12 * i;
                    float G(int k) => BitConverter.ToSingle(BitConverter.GetBytes(records[at + k]), 0);
                    string Tri(uint h) => h >= geometry.Length ? "-" : $"{world[3 * h]:F2} {world[3 * h + 1]:F2} {world[3 * h + 2]:F2}";
                    TestContext.WriteLine($"  {why}: o ({G(0):F3}, {G(1):F3}, {G(2):F3}) d ({G(4):F3}, {G(5):F3}, {G(6):F3}) [{G(3)}, {G(7)}] any {records[at + 8]} object {records[at + 9]}: " +
                                          $"software {hs} at {ts:F4} [{Tri(hs)}], hardware {hh} at {th:F4} [{Tri(hh)}]");
                }
                if ((hs == uint.MaxValue) != (hh == uint.MaxValue)) { Report("hit or not"); differ++; continue; }
                if (hs == uint.MaxValue || any) continue; // (any-hit: whichever hit, only that there is one)
                float tolerance = 1e-4f + 1e-5f * ts;
                worst = Math.Max(worst, Math.Abs(ts - th));
                if (Math.Abs(ts - th) > tolerance) { Report("distance"); differ++; continue; }
                if (hs != hh) ties++;
            }
            TestContext.WriteLine($"{name}: {geometry.Length} triangles, {meshes.Count} meshes, {count} rays, {hits} hits; disagreeing {differ}, other primitive at the same distance {ties}, largest distance difference {worst:E2} m");
            Assert.That(hits, Is.GreaterThan(count / 8));
            Assert.That(differ, Is.LessThanOrEqualTo(count / 1000));
        }
    }
}
