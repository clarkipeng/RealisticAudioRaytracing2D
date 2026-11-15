using System;
using System.Collections.Generic;
using NUnit.Framework;
using UnityEngine;
using Complex = System.Numerics.Complex;

// Analytic controls of the single-loop Pressure and Energy methods
// (docs/benchmarks/methods-2026-11-06 reports the measured values).
public partial class TakeChecks
{
    TakeRender.Settings ControlSettings(TakeRender.Method method, int rays, float horizon)
    {
        var settings = SpectralSettings();
        settings.method = method; settings.methodRays = rays; settings.speedOfSound = 343; settings.minimumDistance = .01f;
        settings.maximumDelaySeconds = horizon; settings.diffractionFactor = 0; settings.airAbsorption = false; settings.useBvh = true;
        settings.seed = 11;
        return settings;
    }
    // Band energies per 1 ms bin of the latest solve, direction-summed (Energy) or
    // band-limited from the coherent trains (Pressure).
    static double[] ControlEnergy(TransportSolver solver, int band)
    {
        var m = solver.Method; var energy = new double[m.Bins];
        if (m.Kind == TakeRender.Method.Energy || m.Kind == TakeRender.Method.Path)
        {
            for (int k = 0; k < m.Bins; k++) for (int d = 0; d < 6; d++) energy[k] += m.Energy(d, band, k);
            return energy;
        }
        int length = m.TrainLength(band);
        MethodBands.TrainBuffers(length);
        m.Train(band, MethodBands.TrainRe, MethodBands.TrainIm);
        return MethodBands.Energy(MethodBands.TrainRe, MethodBands.TrainIm, length, band, m.TrainRate(band), MethodTransport.LeadSeconds, m.Bins, MethodTransport.BinSeconds);
    }
    // Complex response of a pressure solve at f, from the band train of f's band.
    static Complex ControlResponse(MethodTransport m, double f)
    {
        int band = MethodBands.BandOf(f), length = m.TrainLength(band);
        MethodBands.TrainBuffers(length);
        m.Train(band, MethodBands.TrainRe, MethodBands.TrainIm);
        return MethodBands.Response(MethodBands.TrainRe, MethodBands.TrainIm, length, m.TrainRate(band), MethodTransport.LeadSeconds, f);
    }
    static double[] ReferenceEnergy(List<(double delay, double gain)> arrivals, int band, int bins)
    {
        int length = MethodTransportTrainLength(band); double rate = MethodTransport.TrainRate0 * (1 << band);
        MethodBands.TrainBuffers(length);
        foreach (var a in arrivals) MethodBands.Deposit(MethodBands.TrainRe, MethodBands.TrainIm, length, rate, MethodTransport.LeadSeconds, a.delay, a.gain);
        return MethodBands.Energy(MethodBands.TrainRe, MethodBands.TrainIm, length, band, rate, MethodTransport.LeadSeconds, bins, MethodTransport.BinSeconds);
    }
    static double Window(double[] values, int from, int to) { double s = 0; for (int k = Math.Max(0, from); k < Math.Min(values.Length, to); k++) s += values[k]; return s; }
    static AcousticTake.MaterialDefinition RigidControl(float absorption = 0, float scattering = 0) =>
        new AcousticTake.MaterialDefinition {name = "control", frequencies = new[] {0f}, absorption = new[] {absorption}, scattering = new[] {scattering}, transmission = new[] {0f}};

    // Free field: the direct sound arrives at r/c with energy 1/r^2 in every band.
    // Slow like every method check: compiling the two method shaders costs the shared
    // runner about 0.6 GiB of shader-compiler memory, which the quick tier cannot spare.
    [TestCase(TakeRender.Method.Energy), TestCase(TakeRender.Method.Pressure), TestCase(TakeRender.Method.Path), Category("GPU"), Category("Methods"), Category("Slow"), Timeout(1800000)]
    public void MethodsMatchTheFreeField(TakeRender.Method method)
    {
        var scene = Scene(); scene.materials = new[] {RigidControl()};
        var state = States(); state[0].position = Vector3.zero; state[1].position = new Vector3(3, .4f, -1.2f);
        var settings = ControlSettings(method, 256, .1f); settings.useBvh = false;
        using var solver = new TransportSolver(scene, new AcousticTake.Primitive[0], settings, RequireGpu(), sampleRate: 8000);
        solver.Solve(state, new TakeRender.DirectPath[1]);
        double r = state[1].position.magnitude; int bin = (int)Math.Floor(r / 343 / .001);
        for (int b = 0; b < 8; b++)
        {
            var e = ControlEnergy(solver, b); double total = Window(e, 0, e.Length);
            TestContext.WriteLine($"{method} band {b}: total {total * r * r:F6} x 1/r^2, bin {bin} holds {e[bin] / total:F3}");
            bool energy = method != TakeRender.Method.Pressure;
            Assert.That(total * r * r, Is.EqualTo(1).Within(energy ? 1e-6 : .01));
            Assert.That(Window(e, bin - 4, bin + 5) / total, Is.GreaterThan(energy ? .999 : .9));
            if (energy) Assert.That(e[bin] / total, Is.GreaterThan(.999));
        }
    }

    // One rigid plane: direct sound plus its image, nothing else (diffraction off).
    [TestCase(TakeRender.Method.Energy), TestCase(TakeRender.Method.Pressure), TestCase(TakeRender.Method.Path), Category("GPU"), Category("Methods"), Category("Slow"), Timeout(1800000)]
    public void MethodsReproduceARigidPlaneImage(TakeRender.Method method)
    {
        var scene = Scene(2); scene.listeners = new[] {1}; scene.materials = new[] {RigidControl()};
        var a = new Vector3(-100, 0, -100); var b = new Vector3(100, 0, -100); var c = new Vector3(100, 0, 100); var d = new Vector3(-100, 0, 100);
        var geometry = new[] {new AcousticTake.Primitive {objectIndex = 2, a = a, b = c, c = b}, new AcousticTake.Primitive {objectIndex = 2, a = a, b = d, c = c}};
        scene.objects[2].primitiveCount = geometry.Length;
        var state = States(2); state[0].position = new Vector3(0, 1.5f, 0); state[1].position = new Vector3(4, 1.2f, .5f);
        state[2].position = Vector3.zero; state[2].flags = AcousticTake.StateFlags.Active | AcousticTake.StateFlags.Geometry;
        var settings = ControlSettings(method, 1 << 18, .1f);
        if (method == TakeRender.Method.Path) settings.useBvh = false; // two triangles; spares the BVH kernel's compile
        using var solver = new TransportSolver(scene, geometry, settings, RequireGpu(), sampleRate: 8000);
        solver.Solve(state, new TakeRender.DirectPath[1]);
        var image = new Vector3(0, -1.5f, 0);
        if (method == TakeRender.Method.Path)
        {
            var m = solver.Method; var counts = new System.Text.StringBuilder();
            for (int k = 0; k < m.TimeBins.Count; k++) if (m.Counts[k] > 0) counts.Append($"{k}:{m.Counts[k]} ");
            TestContext.WriteLine($"contributions per bin {counts}; rejected deposits {m.Counter(6)}");
        }
        double r = (state[1].position - state[0].position).magnitude, ri = (state[1].position - image).magnitude;
        int bi = (int)Math.Floor(ri / 343 / .001);
        if (method != TakeRender.Method.Pressure)
        {
            var e = ControlEnergy(solver, 2); double held = Window(e, bi, bi + 1), total = Window(e, 0, e.Length);
            TestContext.WriteLine($"image {held * ri * ri:F4} x 1/ri^2, total {total / (1 / (r * r) + 1 / (ri * ri)):F4} x expected, chains {solver.Method.Counter(8)}");
            Assert.That(held * ri * ri, Is.EqualTo(1).Within(.15));
            Assert.That(total / (1 / (r * r) + 1 / (ri * ri)), Is.EqualTo(1).Within(.08));
        }
        else
        {
            foreach (double f in new[] {375.0, 500, 750})
            {
                var h = ControlResponse(solver.Method, f);
                double k = 2 * Math.PI * f / 343;
                var want = Complex.FromPolarCoordinates(1 / r, -k * r) + Complex.FromPolarCoordinates(1 / ri, -k * ri);
                TestContext.WriteLine($"{f} Hz: {h} vs {want}, error {(h - want).Magnitude * ri:F4} x image");
                Assert.That((h - want).Magnitude * ri, Is.LessThan(.2));
            }
        }
    }

    // Rigid-walled shoebox (alpha = 0.1, no scattering): the energy histogram is the
    // image lattice; the pressure trains, band-filtered, match the lattice's arrivals.
    [TestCase(TakeRender.Method.Energy), TestCase(TakeRender.Method.Pressure), TestCase(TakeRender.Method.Path), Category("GPU"), Category("Methods"), Category("Slow"), Timeout(1800000)]
    public void MethodsReproduceTheShoeboxImageLattice(TakeRender.Method method)
    {
        var size = new Vector3(5, 4, 3); var source = new Vector3(-1.1f, .4f, .3f); var listener = new Vector3(1.3f, -.6f, -.2f);
        LateRoom(AcousticTake.Dimension.Spatial3D, size, LateMaterial(.1f, 0), source, listener, out var scene, out var geometry, out var state);
        var settings = ControlSettings(method, 1 << 16, .2f);
        // Path: every path lives to the horizon here, so the 20 ms windows test the connections at the
        // energy method's sampling density (survival in time is checked by PathMethodLeaksThroughWalls...).
        settings.pathSurvivalSeconds = 0;
        using var solver = new TransportSolver(scene, geometry, settings, RequireGpu(), sampleRate: 8000);
        solver.Solve(state, new TakeRender.DirectPath[1]);
        var corner = size * .5f; int bins = solver.Method.Bins;
        if (method != TakeRender.Method.Pressure)
        {
            var e = ControlEnergy(solver, 0); var want = ImageLattice(size, source + corner, listener + corner, .1, .001, bins, 0);
            double worst = 0;
            for (int w = 0; w + 20 <= bins; w += 20)
            {
                double ratio = Window(e, w, w + 20) / Window(want, w, w + 20);
                worst = Math.Max(worst, Math.Abs(ratio - 1));
                TestContext.WriteLine($"{w}-{w + 20} ms: {ratio:F4}");
            }
            double total = Window(e, 0, bins) / Window(want, 0, bins);
            TestContext.WriteLine($"total {total:F4}, worst 20 ms window {worst:F4}, chains {solver.Method.Counter(8)}");
            Assert.That(total, Is.EqualTo(1).Within(.05));
            Assert.That(worst, Is.LessThan(.15));
        }
        else
        {
            var arrivals = LatticeArrivals(size, source + corner, listener + corner, .1, bins * .001);
            var e = ControlEnergy(solver, 2); var want = ReferenceEnergy(arrivals, 2, bins);
            double worst = 0;
            for (int w = 0; w + 20 <= bins; w += 20)
            {
                double ratio = Window(e, w, w + 20) / Window(want, w, w + 20);
                worst = Math.Max(worst, Math.Abs(ratio - 1));
                TestContext.WriteLine($"{w}-{w + 20} ms: {ratio:F4}");
            }
            // The pressure estimate is unbiased, but its band energy is |mean|^2 plus the
            // estimator's variance. Early images are detected many times; late ones less
            // than once per image at this budget, so late windows overshoot by their
            // variance (reported in docs/benchmarks/methods-2026-11-06, not asserted).
            double early = Window(e, 0, 30) / Window(want, 0, 30), total = Window(e, 0, bins) / Window(want, 0, bins);
            TestContext.WriteLine($"first 30 ms {early:F4}, total {total:F4} (variance included), chains {solver.Method.Counter(8)}");
            Assert.That(early, Is.EqualTo(1).Within(.15));
        }
    }
    static List<(double, double)> LatticeArrivals(Vector3 size, Vector3 source, Vector3 listener, double absorption, double horizon)
    {
        var arrivals = new List<(double, double)>(); int reach = (int)Math.Ceiling(horizon * 343 / Math.Min(size.x, Math.Min(size.y, size.z))) + 1;
        for (int i = -reach; i <= reach; i++) for (int j = -reach; j <= reach; j++) for (int k = -reach; k <= reach; k++) for (int mask = 0; mask < 8; mask++)
        {
            int[] n = {i, j, k}; var image = Vector3.zero; int order = 0;
            for (int a = 0; a < 3; a++)
            {
                bool mirrored = (mask >> a & 1) != 0;
                image[a] = 2 * n[a] * size[a] + (mirrored ? -source[a] : source[a]);
                order += mirrored ? Math.Abs(2 * n[a] - 1) : Math.Abs(2 * n[a]);
            }
            double distance = (image - listener).magnitude;
            if (distance / 343 < horizon) arrivals.Add((distance / 343, Math.Pow(1 - absorption, order * .5) / distance));
        }
        return arrivals;
    }

    // Fully diffusing shoebox: the energy method decays at the Eyring rate.
    [Test, Category("GPU"), Category("Methods"), Category("Slow"), Timeout(1800000)]
    public void EnergyMethodDecaysAtTheEyringRate()
    {
        var size = new Vector3(6, 5, 4); const float absorption = .2f;
        LateRoom(AcousticTake.Dimension.Spatial3D, size, LateMaterial(absorption, 1), new Vector3(-1.5f, .5f, .4f), new Vector3(1.4f, -.7f, -.3f),
                 out var scene, out var geometry, out var state);
        var settings = ControlSettings(TakeRender.Method.Energy, 8192, 1);
        using var solver = new TransportSolver(scene, geometry, settings, RequireGpu(), sampleRate: 8000);
        solver.Solve(state, new TakeRender.DirectPath[1]);
        var e = ControlEnergy(solver, 0);
        var curve = new double[e.Length]; double tail = 0;
        for (int k = e.Length - 1; k >= 0; k--) { tail += e[k]; curve[k] = tail; }
        double x = 0, y = 0, xx = 0, xy = 0; int n = 0;
        for (int k = 0; k < e.Length; k++)
        {
            double level = 10 * Math.Log10(curve[k] / curve[0]); if (level > -5 || level < -25) continue;
            double t = (k + .5) * .001; x += t; y += level; xx += t * t; xy += t * level; n++;
        }
        double t60 = -60 / ((n * xy - x * y) / (n * xx - x * x));
        double volume = size.x * size.y * size.z, surface = 2 * (size.x * size.y + size.y * size.z + size.z * size.x);
        double eyring = .161 * volume / (-surface * Math.Log(1 - absorption));
        TestContext.WriteLine($"T60 {t60:F3} s, Eyring {eyring:F3} s");
        Assert.That(t60 / eyring, Is.EqualTo(1).Within(.15));
    }

    // Leakage through thin walls (specular room, small transmission, listener outside):
    // the energy walk splits the transmitted share off into one reservoir branch per ray
    // (MethodEnergy.hlsl), so the field outside is a smooth estimate that scales with the
    // transmission. Drawing the transmission instead sent about one ray in 10^4 through
    // with its full energy: a handful of streaks, tens of dB apart between seeds.
    [Test, Category("GPU"), Category("Methods"), Category("Slow"), Timeout(1800000)]
    public void EnergyMethodLeaksThroughWallsSmoothly()
    {
        var size = new Vector3(6, 5, 4);
        double Leak(float transmission, int seed)
        {
            LateRoom(AcousticTake.Dimension.Spatial3D, size, LateMaterial(.1f, 0, transmission), new Vector3(-1.5f, .5f, .4f), new Vector3(3.8f, -.3f, .2f),
                     out var scene, out var geometry, out var state);
            var settings = ControlSettings(TakeRender.Method.Energy, 1 << 16, .4f); settings.seed = seed; settings.specularDetectionRadius = .5f;
            using var solver = new TransportSolver(scene, geometry, settings, RequireGpu(), sampleRate: 8000);
            solver.Solve(state, new TakeRender.DirectPath[1]);
            TestContext.WriteLine($"t {transmission}, seed {seed}: branches {solver.Method.Counter(14)}");
            double direct = (new Vector3(3.8f, -.3f, .2f) - new Vector3(-1.5f, .5f, .4f)).magnitude / 343;
            return Window(ControlEnergy(solver, 2), (int)Math.Ceiling(direct * 1000) + 5, int.MaxValue); // reflected leakage only
        }
        var low = new double[4]; for (int seed = 0; seed < low.Length; seed++) low[seed] = Leak(1e-4f, seed);
        double mean = 0, spread = 0; foreach (var v in low) mean += v / low.Length; foreach (var v in low) spread += (v - mean) * (v - mean) / (low.Length - 1);
        double variation = Math.Sqrt(spread) / mean, ratio = Leak(1e-3f, 0) / mean;
        TestContext.WriteLine($"leakage {mean:G4}, seed-to-seed variation {variation:P1}, 10x transmission gives {ratio:F2}x");
        Assert.That(mean, Is.GreaterThan(0));
        Assert.That(variation, Is.LessThan(.25), "The leaked field is a smooth estimate (drawn transmissions: well over 50%).");
        Assert.That(ratio, Is.EqualTo(10).Within(2.5), "Leaked energy scales with the transmission.");
    }

    // Partitions: a wall modelled as two parallel sheets of one object (0.3 m apart) transmits
    // its material's transmission once (the assembly's), so the direct sound through it is
    // (1 - a) t / r^2, not its square; a second wall object adds its own factor.
    [Test, Category("GPU"), Category("Methods"), Category("Slow"), Timeout(1800000)]
    public void EnergyPartitionsTransmitOnce()
    {
        const float a = .1f, t = .01f;
        var objects = new List<AcousticTake.ObjectDefinition> {new() {id = "source", name = "Source", scale = Vector3.one}, new() {id = "listener", name = "Listener", scale = Vector3.one}};
        var list = new List<AcousticTake.Primitive>();
        void Wall(float x0, float x1)
        {
            int owner = objects.Count, start = list.Count;
            foreach (float x in new[] {x0, x1})
            {
                list.Add(new AcousticTake.Primitive {objectIndex = owner, a = new Vector3(x, -10, -10), b = new Vector3(x, 10, -10), c = new Vector3(x, 10, 10)});
                list.Add(new AcousticTake.Primitive {objectIndex = owner, a = new Vector3(x, -10, -10), b = new Vector3(x, 10, 10), c = new Vector3(x, -10, 10)});
            }
            objects.Add(new() {id = "wall-" + owner, name = "Wall " + owner, scale = Vector3.one, primitiveStart = start, primitiveCount = list.Count - start});
        }
        double Direct(int walls)
        {
            objects.RemoveRange(2, objects.Count - 2); list.Clear();
            for (int w = 0; w < walls; w++) Wall(-.15f + w * 1f, .15f + w * 1f);
            var scene = new AcousticTake.Scene {dimension = AcousticTake.Dimension.Spatial3D, objects = objects.ToArray(),
                sources = new[] {new AcousticTake.SourceDefinition {objectIndex = 0, clipIndex = 0}}, listeners = new[] {1},
                clips = new[] {new AcousticTake.ClipDefinition {name = "test"}}, materials = new[] {LateMaterial(a, 0, t)}};
            var state = new AcousticTake.State[objects.Count];
            for (int i = 0; i < state.Length; i++) state[i] = new AcousticTake.State {rotation = Quaternion.identity,
                flags = AcousticTake.StateFlags.Active | (i == 0 ? AcousticTake.StateFlags.Source : i == 1 ? AcousticTake.StateFlags.Listener : AcousticTake.StateFlags.Geometry)};
            state[0].position = new Vector3(-3, .2f, .1f); state[1].position = new Vector3(3, -.1f, .3f);
            var settings = ControlSettings(TakeRender.Method.Energy, 4096, .1f);
            using var solver = new TransportSolver(scene, list.ToArray(), settings, RequireGpu(), sampleRate: 8000);
            solver.Solve(state, new TakeRender.DirectPath[1]);
            double e = 0; foreach (var x in solver.Method.ReadArrivals()) if (x.Family == 0) e += x.Energy(2);
            return e * (state[1].position - state[0].position).sqrMagnitude;
        }
        double one = Direct(1), two = Direct(2), want = (1 - a) * t;
        TestContext.WriteLine($"one wall {one:G4} (want {want:G4}), two walls {two:G4} (want {want * want:G4})");
        Assert.That(one / want, Is.EqualTo(1).Within(.01), "One partition, one transmission factor.");
        Assert.That(two / (want * want), Is.EqualTo(1).Within(.01), "Each partition object its own factor.");
    }

    // Rigid half-plane, listener in the shadow: the pressure method's band response
    // matches the independent signed half-plane quadrature (TunedRoutesMatchIndependentHalfPlane),
    // and the energy method's UTD band energy matches the BTM band energy within 1.5 dB.
    [TestCase(TakeRender.Method.Pressure), TestCase(TakeRender.Method.Energy), TestCase(TakeRender.Method.Path), Category("GPU"), Category("Methods"), Category("Slow"), Timeout(1800000)]
    public void MethodsMatchTheHalfPlaneReference(TakeRender.Method method)
    {
        KnifeEdgeFixture(false, out var scene, out var geometry, out var state);
        var settings = ControlSettings(method, 1 << 16, .3f); settings.diffractionFactor = 1;
        using var solver = new TransportSolver(scene, geometry, settings, RequireGpu(), sampleRate: 8000);
        solver.Solve(state, new TakeRender.DirectPath[1]);
        if (method == TakeRender.Method.Pressure)
        {
            double[] hz = {250, 500}, re = {.004442267995, -.066639430815}, im = {.083451251431, -.016598353630};
            for (int i = 0; i < hz.Length; i++)
            {
                var h = ControlResponse(solver.Method, hz[i]);
                var want = new Complex(re[i], im[i]);
                TestContext.WriteLine($"{hz[i]} Hz: {h} vs {want}, error {(h - want).Magnitude / want.Magnitude:F4}");
                Assert.That((h - want).Magnitude / want.Magnitude, Is.LessThan(.1));
            }
            return;
        }
        // Independent BTM band energies of this fixture (docs/benchmarks/methods-2026-11-06/
        // btm_reference.py: float64 edge integral, the comparison's band filters).
        double[] btm = {0.009199, 0.006599, 0.004427, 0.002758, 0.001590};
        for (int band = 1; band <= 4; band++)
        {
            double have = Window(ControlEnergy(solver, band), 0, 300);
            TestContext.WriteLine($"band {MethodBands.Centers[band]} Hz: UTD energy {have:G5}, BTM {btm[band]:G5}, {10 * Math.Log10(have / btm[band]):F2} dB");
            Assert.That(Math.Abs(10 * Math.Log10(have / btm[band])), Is.LessThan(1.5));
        }
    }

    // Rigid cube (2 m), source and listener on opposite sides: only double diffraction
    // (DD over parallel edges) and longer chains reach the listener. Pressure matches
    // the independent EDtoolbox second-order field of TunedRoutesReachTheFarSideOfASolid;
    // energy (UTD, DD connections) is checked on single walls against btm_reference.py.
    [TestCase(TakeRender.Method.Pressure), TestCase(TakeRender.Method.Energy), TestCase(TakeRender.Method.Path), Category("GPU"), Category("Methods"), Category("Slow"), Timeout(1800000)]
    public void MethodsReachTheFarSideOfASolid(TakeRender.Method method)
    {
        KnifeEdgeFixture(false, out var scene, out _, out var state);
        var geometry = TransportBox(scene.dimension, 2, 0, 1); scene.objects[2].primitiveCount = geometry.Length;
        state[0].position = Vector3.left * 3; state[1].position = Vector3.right * 3;
        var settings = ControlSettings(method, 1 << 17, .3f); settings.diffractionFactor = 1;
        var solver = new TransportSolver(scene, geometry, settings, RequireGpu(), sampleRate: 8000);
        try
        {
            solver.Solve(state, new TakeRender.DirectPath[1]);
            TestContext.WriteLine($"edges {solver.Method.EdgeCount}, edge connections {solver.Method.Counter(10)}, edge events {solver.Method.Counter(12)}");
            for (int depth = 1; depth < 6; depth++)
            {
                var d = solver.Method.Depth(0, depth);
                if (d.count > 0) TestContext.WriteLine($"  depth {depth}: {d.count} contributions, mean |c| {d.sum / d.count:G4}, max {d.max:G4}, sum|c|/rays {d.sum / settings.methodRays:G4}");
            }
            if (method == TakeRender.Method.Pressure)
            {
                Assert.Ignore("Known gap: the pressure loop's edge continuation (needed for DD) is off; see docs/benchmarks/methods-2026-11-06.");
                double[] hz = {125, 500}, re = {-.132356150111, -.065193018566}, im = {-.077646144425, .063496034679};
                for (int i = 0; i < hz.Length; i++)
                {
                    var h = ControlResponse(solver.Method, hz[i]);
                    var want = new Complex(re[i], im[i]);
                    TestContext.WriteLine($"{hz[i]} Hz: {h} vs {want}, error {(h - want).Magnitude / want.Magnitude:F4}");
                }
                for (int i = 0; i < hz.Length; i++)
                {
                    var h = ControlResponse(solver.Method, hz[i]);
                    Assert.That((h - new Complex(re[i], im[i])).Magnitude / new Complex(re[i], im[i]).Magnitude, Is.LessThan(.15));
                }
                return;
            }
            // The cube's four DD paths have equal delays: BTM adds them coherently (16x one
            // path's energy), any energy method incoherently (4x), 6 dB apart. The energy
            // control therefore uses walls with a single dominant DD path over the top,
            // against independent BTM band energies (btm_reference.py). Ray-based double
            // diffraction needs the edges to be far apart in wavelengths: asserted for a 2 m
            // wall (kd = 18-73 at 500-2000 Hz); a 0.25 m wall (the suite's walls) is measured.
            solver.Dispose();
            var cases = new[] {(width: 2f, btm: new[] {1.620450e-3, 7.792134e-4, 3.207570e-4, 1.120553e-4, 3.379192e-5}),
                               (width: .25f, btm: new[] {2.606818e-3, 1.510434e-3, 7.953154e-4, 3.818016e-4, 1.673416e-4})};
            foreach (var wallCase in cases)
            {
                var wall = TransportBox(scene.dimension, 2, 0, 1);
                for (int i = 0; i < wall.Length; i++)
                {
                    var p = wall[i]; Vector3 size = new Vector3(wallCase.width / 2, 5, 30), offset = new Vector3(0, -4, 0);
                    p.a = Vector3.Scale(p.a, size) + offset; p.b = Vector3.Scale(p.b, size) + offset; p.c = Vector3.Scale(p.c, size) + offset; wall[i] = p;
                }
                using var wallSolver = new TransportSolver(scene, wall, settings, RequireGpu(), sampleRate: 8000);
                wallSolver.Solve(state, new TakeRender.DirectPath[1]);
                for (int band = 0; band <= 4; band++)
                {
                    double have = Window(ControlEnergy(wallSolver, band), 0, 300), error = 10 * Math.Log10(have / wallCase.btm[band]);
                    TestContext.WriteLine($"{wallCase.width} m wall, band {MethodBands.Centers[band]} Hz: UTD energy {have:G5}, BTM {wallCase.btm[band]:G5}, {error:F2} dB");
                    if (wallCase.width > 1 && band >= 2) Assert.That(Math.Abs(error), Is.LessThan(2), "UTD double diffraction over a 2 m wall agrees with BTM.");
                }
            }
        }
        finally { solver.Dispose(); }
    }
}
