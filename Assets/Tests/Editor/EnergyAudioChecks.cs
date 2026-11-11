using System;
using System.Collections.Generic;
using NUnit.Framework;
using UnityEngine;

// The energy method's discrete arrivals, its split from the diffuse field, and its audio
// (docs/benchmarks/energy-audio-2026-11-09). All compile EnergyMethod.compute, so all
// run with the method checks (categories Methods and Slow).
public partial class TakeChecks
{
    static readonly Vector3 ArrivalRoom = new Vector3(5, 4, 3), ArrivalSource = new Vector3(-1.1f, .4f, .3f), ArrivalListener = new Vector3(1.3f, -.6f, -.2f);

    // Rigid shoebox (alpha 0.1, no scattering): the arrival list is the image lattice up
    // to second order, every image exactly once with its exact delay, energy and
    // direction; the diffuse histogram holds nothing before the first third-order image.
    [Test, Category("GPU"), Category("Methods"), Category("Slow"), Timeout(1800000)]
    public void EnergyArrivalsAreTheExactImagePaths()
    {
        const double alpha = .1;
        LateRoom(AcousticTake.Dimension.Spatial3D, ArrivalRoom, LateMaterial((float)alpha, 0), ArrivalSource, ArrivalListener, out var scene, out var geometry, out var state);
        var settings = ControlSettings(TakeRender.Method.Energy, 4096, .2f);
        using var solver = new TransportSolver(scene, geometry, settings, RequireGpu(), sampleRate: 8000);
        solver.Solve(state, new TakeRender.DirectPath[1]);
        var m = solver.Method; var arrivals = m.ReadArrivals();
        Assert.That(m.ArrivalOrder, Is.EqualTo(2));
        var corner = ArrivalRoom * .5f; Vector3 s = ArrivalSource + corner, l = ArrivalListener + corner;
        var images = new List<(int order, Vector3 image)>(); double third = double.MaxValue;
        for (int i = -2; i <= 2; i++) for (int j = -2; j <= 2; j++) for (int k = -2; k <= 2; k++) for (int mask = 0; mask < 8; mask++)
        {
            int[] n = {i, j, k}; var image = Vector3.zero; int order = 0;
            for (int a = 0; a < 3; a++)
            {
                bool mirrored = (mask >> a & 1) != 0;
                image[a] = 2 * n[a] * ArrivalRoom[a] + (mirrored ? -s[a] : s[a]);
                order += mirrored ? Math.Abs(2 * n[a] - 1) : Math.Abs(2 * n[a]);
            }
            if (order <= 2) images.Add((order, image));
            else if (order == 3) third = Math.Min(third, (image - l).magnitude);
        }
        TestContext.WriteLine($"{arrivals.Length} arrivals, {images.Count} images of order <= 2, planes {m.PlaneCount}");
        Assert.That(arrivals.Length, Is.EqualTo(images.Count), "Each image path is one arrival.");
        var used = new bool[arrivals.Length];
        foreach (var (order, image) in images)
        {
            double distance = (image - l).magnitude, want = Math.Pow(1 - alpha, order) / (distance * distance);
            int found = -1;
            for (int a = 0; a < arrivals.Length; a++)
                if (!used[a] && Math.Abs(arrivals[a].delay - distance / 343) < 2e-6) { found = a; break; }
            Assert.That(found, Is.GreaterThanOrEqualTo(0), $"order {order} image at {distance:F4} m has an arrival");
            used[found] = true;
            var arrival = arrivals[found];
            Assert.That(arrival.Depth, Is.EqualTo(order));
            Vector3 from = arrival.direction, toward = (image - l).normalized;
            Assert.That((from - toward).magnitude, Is.LessThan(1e-4), "arrival direction points at the image");
            Assert.That(arrival.direction.w, Is.EqualTo(0).Within(1e-7), "static scene: no delay rate");
            for (int b = 0; b < 8; b++) Assert.That(arrival.Energy(b) / want, Is.EqualTo(1).Within(1e-4), $"order {order} band {b}");
        }
        // Diffuse field = histogram minus the binned arrivals: empty before the first
        // third-order image (less the receiver sphere's radius).
        int before = (int)Math.Floor((third - settings.specularDetectionRadius) / 343 / .001);
        for (int b = 0; b < 8; b += 3)
        {
            var e = ControlEnergy(solver, b); double held = 0, diffuse = 0;
            foreach (var a in arrivals) { int bin = (int)Math.Floor(a.delay / .001); if (bin < before) held += a.Energy(b); }
            diffuse = Window(e, 0, before) - held;
            TestContext.WriteLine($"band {b}: before {before} ms arrivals {held:G6}, diffuse {diffuse:G3}");
            Assert.That(Math.Abs(diffuse), Is.LessThan(1e-6 * held));
        }
    }

    // The discrete arrivals and the walk partition the paths: with the arrivals off the
    // walk estimates every path itself, and both band totals agree (a scattering room;
    // a rigid cube where only edge routes reach the far side).
    // near: second order for only the 3 most important of the room's 6 planes (methodNearPlanes);
    // the walk then keeps the other second-order paths, and the totals still agree.
    [TestCase(false), TestCase(true), TestCase(false, 3), Category("GPU"), Category("Methods"), Category("Slow"), Timeout(1800000)]
    public void EnergyArrivalsAndDiffuseFieldSumToTheWalk(bool cube, int near = 0)
    {
        AcousticTake.Scene scene; AcousticTake.Primitive[] geometry; AcousticTake.State[] state;
        var settings = ControlSettings(TakeRender.Method.Energy, 1 << 16, cube ? .3f : .4f);
        settings.methodNearPlanes = near;
        if (cube)
        {
            KnifeEdgeFixture(false, out scene, out _, out state);
            geometry = TransportBox(scene.dimension, 2, 0, 1); scene.objects[2].primitiveCount = geometry.Length;
            state[0].position = new Vector3(-3, .3f, .2f); state[1].position = new Vector3(3, -.2f, .1f);
            settings.diffractionFactor = 1;
        }
        else LateRoom(AcousticTake.Dimension.Spatial3D, new Vector3(6, 5, 4), LateMaterial(.15f, .3f), new Vector3(-1.5f, .5f, .4f), new Vector3(1.4f, -.7f, -.3f),
                      out scene, out geometry, out state);
        var totals = new double[2, 8];
        for (int run = 0; run < 2; run++)
        {
            settings.methodArrivals = run == 0;
            using var solver = new TransportSolver(scene, geometry, settings, RequireGpu(), sampleRate: 8000);
            solver.Solve(state, new TakeRender.DirectPath[1]);
            if (run == 0) TestContext.WriteLine($"families {solver.Method.ArrivalFamilies}, arrivals {solver.Method.ArrivalTotal()}, edges {solver.Method.EdgeCount}, near planes {solver.Method.NearPlanes} of {solver.Method.PlaneCount}");
            if (run == 0 && near > 0) Assert.That(solver.Method.NearPlanes, Is.EqualTo(near));
            for (int b = 0; b < 8; b++) { var e = ControlEnergy(solver, b); totals[run, b] = Window(e, 0, e.Length); }
        }
        for (int b = 0; b < (cube ? 5 : 8); b++)
        {
            double ratio = totals[0, b] / totals[1, b];
            TestContext.WriteLine($"band {b}: arrivals + diffuse {totals[0, b]:G6}, walk only {totals[1, b]:G6}, ratio {ratio:F4}");
            Assert.That(ratio, Is.EqualTo(1).Within(cube ? .06 : .02));
        }
    }

    // A moving source or listener: the direct and the reflected arrival carry the exact
    // retarded delay and Doppler delay rate of their (image) sources, and the audio
    // paths carry them with a stable identity.
    [TestCase(15f, 0f), TestCase(0f, 5f), TestCase(15f, 5f), Category("GPU"), Category("Methods"), Category("Slow"), Timeout(1800000)]
    public void EnergyArrivalsCarryTheDopplerShift(float sourceSpeed, float listenerSpeed)
    {
        var scene = Scene(2); scene.listeners = new[] {1}; scene.materials = new[] {RigidControl()};
        var a = new Vector3(-100, 0, -100); var b = new Vector3(100, 0, -100); var c = new Vector3(100, 0, 100); var d = new Vector3(-100, 0, 100);
        var geometry = new[] {new AcousticTake.Primitive {objectIndex = 2, a = a, b = c, c = b}, new AcousticTake.Primitive {objectIndex = 2, a = a, b = d, c = c}};
        scene.objects[2].primitiveCount = geometry.Length;
        var state = States(2); state[0].position = new Vector3(0, 1.5f, 0); state[1].position = new Vector3(4, 1.2f, .5f);
        state[2].position = Vector3.zero; state[2].flags = AcousticTake.StateFlags.Active | AcousticTake.StateFlags.Geometry;
        Vector3 vs = new Vector3(1, .3f, 0).normalized * sourceSpeed, vl = new Vector3(0, -.2f, 1).normalized * listenerSpeed;
        state[0].velocity = vs; state[1].velocity = vl;
        state[0].flags |= AcousticTake.StateFlags.VelocityValid; state[1].flags |= AcousticTake.StateFlags.VelocityValid;
        var settings = ControlSettings(TakeRender.Method.Energy, 256, .1f);
        using var solver = new TransportSolver(scene, geometry, settings, RequireGpu(), sampleRate: 8000);
        solver.Solve(state, new TakeRender.DirectPath[1]);
        var arrivals = solver.Method.ReadArrivals();
        Assert.That(arrivals.Length, Is.EqualTo(2), "direct sound and one image");
        Vector3 S = state[0].position, image = new Vector3(S.x, -S.y, S.z), mirrored = new Vector3(vs.x, -vs.y, vs.z);
        bool both = sourceSpeed > 0 && listenerSpeed > 0;
        foreach (var (origin, velocity, family) in new[] {(S, vs, 0), (image, mirrored, 1)})
        {
            Doppler.Direct(state[1].position - origin, velocity, vl, 343, out float delay, out float rate, out _);
            var arrival = Array.Find(arrivals, x => x.Family == family);
            TestContext.WriteLine($"family {family}: delay {arrival.delay:F7} vs {delay:F7}, rate {arrival.direction.w:F7} vs {rate:F7}");
            Assert.That(arrival.delay, Is.EqualTo(delay).Within(2e-6));
            // With both ends moving the arrival direction is not retarded: O(v^2/c^2).
            Assert.That(arrival.direction.w, Is.EqualTo(rate).Within(both ? 2e-4 : 2e-6));
        }
        // Audio paths: one entry per FFT bin with the arrival's delay, rate and identity,
        // and the identity survives a moved source (continuous path clocks).
        settings.fftSize = 64; settings.spectralHop = 16; settings.responseStepSamples = 64;
        using var audio = new TransportSolver(scene, geometry, settings, RequireGpu(), sampleRate: 8000, methodAudio: true);
        var identities = new List<uint>();
        for (int step = 0; step < 2; step++)
        {
            state[0].position = S + vs * (step * .01f);
            audio.Solve(state, new TakeRender.DirectPath[1]);
            var paths = audio.RayPaths; var seen = new HashSet<uint>();
            foreach (var path in paths)
            {
                if (path.pressureGain == 0) continue;
                seen.Add(path.identity);
                if (step == 0) Assert.That(Array.Exists(arrivals, x => x.identity == path.identity && Math.Abs(x.direction.w - path.direction.w) < 1e-6 &&
                                                       Math.Abs(x.delay - path.delaySeconds) < 1e-6), "path matches an arrival");
            }
            Assert.That(seen.Count, Is.EqualTo(2));
            if (step == 0) identities.AddRange(seen); else foreach (uint id in seen) Assert.That(identities, Does.Contain(id));
        }
    }

    // Audio: a source receding from the listener at 30 m/s towards a far wall. The direct
    // sound is heard at f c / (c + v) and the wall's reflection, whose image source
    // approaches, at f c / (c - v): both pitches within 0.3% (2 Hz bins, interpolated).
    [Test, Category("GPU"), Category("Methods"), Category("Slow"), Timeout(1800000)]
    public void EnergyAudioPitchFollowsTheDopplerShift()
    {
        const int rate = 8192, size = 256, frames = 32; const float speed = 30, tone = 1000, wall = 40;
        var scene = Scene(2); scene.listeners = new[] {1}; scene.materials = new[] {RigidControl()};
        var a = new Vector3(wall, -100, -100); var b = new Vector3(wall, 100, -100); var c = new Vector3(wall, 100, 100); var d = new Vector3(wall, -100, 100);
        var geometry = new[] {new AcousticTake.Primitive {objectIndex = 2, a = a, b = b, c = c}, new AcousticTake.Primitive {objectIndex = 2, a = a, b = c, c = d}};
        scene.objects[2].primitiveCount = geometry.Length;
        var state = States(2); state[1].position = new Vector3(-5, 0, 0);
        state[2].position = Vector3.zero; state[2].flags = AcousticTake.StateFlags.Active | AcousticTake.StateFlags.Geometry;
        var settings = EnergyAudioSettings(size, .5f); settings.methodRays = 256;
        var clip = new float[rate]; for (int n = 0; n < clip.Length; n++) clip[n] = Mathf.Sin(2 * Mathf.PI * tone * n / rate); // 1000 whole cycles
        var signals = new[] {new TakeRender.Signal(clip)};
        signals[0].SetActive(0, true); signals[0].Append(new AcousticTake.VoiceEvent {action = AcousticTake.VoiceAction.Play, gain = 1, loop = true});
        using var gpu = new GpuAudio(scene, rate, settings, signals, transported: true);
        using var solver = new TransportSolver(scene, geometry, settings, RequireGpu(), sampleRate: rate, readbackPaths: false, methodAudio: true);
        var output = new float[frames * size]; var block = new float[size];
        for (int frame = 0; frame < frames; frame++)
        {
            long sample = frame * (long)size;
            state[0].position = new Vector3(speed * sample / rate, 0, 0); state[0].velocity = new Vector3(speed, 0, 0);
            state[0].flags |= AcousticTake.StateFlags.VelocityValid;
            solver.Solve(state, new TakeRender.DirectPath[1]);
            gpu.SetResponse(sample, frame == 0, solver.RayBuffer, 0, scene, state, late: solver.Late(0));
            gpu.Render(sample, size, block); Array.Copy(block, 0, output, sample, size);
        }
        const int window = 4096; var spectrum = new double[window / 2];
        var x = new Vector2[window];
        for (int n = 0; n < window; n++) x[n] = new Vector2(output[output.Length - window + n] * (.5f - .5f * Mathf.Cos(2 * Mathf.PI * n / window)), 0);
        ImpulseResponseAudio.Fft(x, false);
        for (int k = 0; k < spectrum.Length; k++) spectrum[k] = x[k].sqrMagnitude;
        double Peak(double near)
        {
            int centre = (int)Math.Round(near * window / rate), best = centre;
            for (int k = centre - 15; k <= centre + 15; k++) if (spectrum[k] > spectrum[best]) best = k;
            double l = Math.Log(spectrum[best - 1]), m = Math.Log(spectrum[best]), r = Math.Log(spectrum[best + 1]);
            return (best + .5 * (l - r) / (l - 2 * m + r)) * rate / (double)window;
        }
        double receding = tone * 343 / (343 + speed), approaching = tone * 343 / (343 - speed);
        double direct = Peak(receding), reflected = Peak(approaching);
        TestContext.WriteLine($"direct {direct:F2} Hz (theory {receding:F2}), reflection {reflected:F2} Hz (theory {approaching:F2})");
        Assert.That(direct / receding, Is.EqualTo(1).Within(.003));
        Assert.That(reflected / approaching, Is.EqualTo(1).Within(.003));
    }
    TakeRender.Settings EnergyAudioSettings(int size, float horizon) => new TakeRender.Settings {
        transport = TakeRender.Transport.Raytraced, synthesis = TakeRender.Synthesis.Spectral, listenerOutput = TakeRender.ListenerOutput.Mono,
        delayFilter = TakeRender.DelayFilter.WindowedSinc, fftSize = size, spectralHop = size / 4, responseStepSamples = size, transitionSamples = size / 4,
        maximumDelaySeconds = horizon, airAbsorption = false, maxBounces = 64, speedOfSound = 343, minimumDistance = .1f, useBvh = true, seed = 5,
        method = TakeRender.Method.Energy, methodRays = 1024, diffractionFactor = 0};

    // Late accumulation (lateAccumulationSeconds): a static listener's late field is the
    // moving average of solves with new walk seeds. Its level is the seeds' mean (unbiased)
    // and it lies much closer to that mean than one fixed-seed solve does.
    [Test, Category("GPU"), Category("Methods"), Category("Slow"), Timeout(1800000)]
    public void EnergyLateAccumulationAveragesTheWalk()
    {
        const int rate = 8192, size = 256, solves = 40;
        LateRoom(AcousticTake.Dimension.Spatial3D, new Vector3(6, 4, 5), LateMaterial(.25f, .4f), new Vector3(-1.2f, .3f, .5f), new Vector3(1.5f, -.4f, -.6f),
                 out var scene, out var geometry, out var state);
        var settings = EnergyAudioSettings(size, .4f); settings.methodRays = 128;
        double Late(TransportSolver solver) { var e = LateEnergy(solver, 2); double sum = 0; for (int k = e.Length / 4; k < e.Length; k++) sum += e[k]; return sum; }
        var single = new double[solves];
        using (var solver = new TransportSolver(scene, geometry, settings, RequireGpu(), sampleRate: rate, methodAudio: true))
            for (int i = 0; i < solves; i++) { solver.Method.SeedOffset = i; solver.Solve(state, new TakeRender.DirectPath[1]); single[i] = Late(solver); }
        double mean = 0, spread = 0; foreach (var v in single) mean += v / solves; foreach (var v in single) spread += (v - mean) * (v - mean) / (solves - 1); spread = Math.Sqrt(spread);
        settings.lateAccumulationSeconds = 10f * size / rate; // alpha 0.1: about 19 solves' worth
        double averaged;
        using (var solver = new TransportSolver(scene, geometry, settings, RequireGpu(), sampleRate: rate, methodAudio: true))
        {
            for (int i = 0; i < solves; i++) solver.Solve(state, new TakeRender.DirectPath[1]);
            averaged = Late(solver);
        }
        TestContext.WriteLine($"single solves {mean:G4} +- {spread:G3}, accumulated {averaged:G4} ({(averaged - mean) / spread:F2} single-solve deviations)");
        Check(spread > 0, "Seeds change the walk.");
        Check(Math.Abs(averaged - mean) < .75 * spread + .01 * mean, "The accumulated late field is the seeds' mean, closer than one solve.");
    }

    // CPU reference and GPU audio render the same energy-method response (arrival paths
    // and late field) of a scattering room with a walking, turning listener.
    [TestCase(TakeRender.ListenerOutput.Mono), TestCase(TakeRender.ListenerOutput.Binaural), TestCase(TakeRender.ListenerOutput.Binaural, true),
     Category("GPU"), Category("Methods"), Category("Slow"), Timeout(1800000)]
    public void EnergyAudioMatchesTheCpuReference(TakeRender.ListenerOutput ears, bool tail = false)
    {
        const int rate = 8192, size = 256;
        LateRoom(AcousticTake.Dimension.Spatial3D, new Vector3(6, 4, 5), LateMaterial(.25f, .4f), new Vector3(-1.2f, .3f, .5f), new Vector3(1.5f, -.4f, -.6f),
                 out var scene, out var geometry, out var state);
        var settings = EnergyAudioSettings(size, .4f); settings.listenerOutput = ears; settings.methodArrivalSlots = 24;
        if (tail) settings.lateTailSeconds = 1.2f; // continued from the room energy past 0.4 s (LateExtend)
        var clip = new float[509]; for (int n = 0; n < clip.Length; n++) clip[n] = .3f * Mathf.Sin(n * .37f) + .1f * Mathf.Cos(n * 1.93f);
        var signals = new[] {new TakeRender.Signal(clip)};
        signals[0].SetActive(0, true); signals[0].Append(new AcousticTake.VoiceEvent {action = AcousticTake.VoiceAction.Play, gain = 1, loop = true});
        using var gpu = new GpuAudio(scene, rate, settings, signals, transported: true); var cpu = new TakeRender.Mixer(scene, rate, settings, signals);
        using var solver = new TransportSolver(scene, geometry, settings, RequireGpu(), sampleRate: rate, methodAudio: true);
        var direct = new TakeRender.DirectPath[1]; int channels = settings.ChannelsPerListener;
        var x = new float[size * channels]; var y = new float[x.Length]; double peak = 0, error = 0, energy = 0;
        for (int frame = 0; frame < (tail ? 40 : 10); frame++)
        {
            long sample = frame * (long)size;
            state[1].position = new Vector3(1.5f - frame * (tail ? .0125f : .05f), -.4f, -.6f); state[1].rotation = Quaternion.Euler(0, frame * 11, 0);
            solver.Solve(state, direct);
            cpu.SetResponse(sample, direct, state, frame == 0, solver.RayPaths, 0, late: solver.Late(0));
            gpu.SetResponse(sample, frame == 0, solver.RayBuffer, 0, scene, state, late: solver.Late(0));
            cpu.Render(sample, size, x); gpu.Render(sample, size, y);
            for (int n = 0; n < x.Length; n++) { double e = x[n] - y[n]; peak = Math.Max(peak, Math.Abs(e)); error += e * e; energy += x[n] * x[n]; }
        }
        double relative = Math.Sqrt(error / energy);
        TestContext.WriteLine($"{ears}: peak {peak:G3}, relative {relative:G3}, energy {energy:G4}");
        Check(energy > 1e-4, "The fixture is audible.");
        Check(relative < 1e-3, $"GPU/CPU energy-method audio relative error {relative}");
    }

    // No clicks when solves change: a 440 Hz tone in a scattering room heard by a walking,
    // turning listener (new arrivals, delays and late envelopes at every step). A click is
    // broadband, so the second difference (high-pass) folded by phase within the response
    // step would peak at the update phase; every phase stays within 3 dB of the median.
    [Test, Category("GPU"), Category("Methods"), Category("Slow"), Timeout(1800000)]
    public void EnergyAudioHasNoClicksAcrossSolves()
    {
        const int rate = 16384, size = 512, frames = 96, phases = 16;
        LateRoom(AcousticTake.Dimension.Spatial3D, new Vector3(8, 4, 6), LateMaterial(.2f, .3f), new Vector3(-2, .2f, 1), new Vector3(2.5f, -.3f, -1.5f),
                 out var scene, out var geometry, out var state);
        var settings = EnergyAudioSettings(size, .6f); settings.methodArrivalSlots = 32;
        var clip = new float[rate]; for (int n = 0; n < clip.Length; n++) clip[n] = .5f * Mathf.Sin(2 * Mathf.PI * 440 * n / rate) * (1 + .3f * Mathf.Sin(2 * Mathf.PI * 3 * n / rate));
        var signals = new[] {new TakeRender.Signal(clip)};
        signals[0].SetActive(0, true); signals[0].Append(new AcousticTake.VoiceEvent {action = AcousticTake.VoiceAction.Play, gain = 1, loop = true});
        using var gpu = new GpuAudio(scene, rate, settings, signals, transported: true);
        using var solver = new TransportSolver(scene, geometry, settings, RequireGpu(), sampleRate: rate, readbackPaths: false, methodAudio: true);
        var output = new float[frames * size]; var block = new float[size];
        for (int frame = 0; frame < frames; frame++)
        {
            long sample = frame * (long)size; float t = sample / (float)rate;
            state[1].position = new Vector3(2.5f - 1.5f * t, -.3f, -1.5f + .8f * t); state[1].velocity = new Vector3(-1.5f, 0, .8f);
            state[1].rotation = Quaternion.Euler(0, 40 * t, 0); state[1].flags |= AcousticTake.StateFlags.VelocityValid;
            solver.Solve(state, new TakeRender.DirectPath[1]);
            gpu.SetResponse(sample, frame == 0, solver.RayBuffer, 0, scene, state, late: solver.Late(0));
            gpu.Render(sample, size, block); Array.Copy(block, 0, output, sample, size);
        }
        var folded = new double[phases];
        for (int n = 8 * size; n < output.Length; n++)
        {
            double second = output[n] - 2 * output[n - 1] + output[n - 2];
            folded[n % size * phases / size] += second * second;
        }
        var sorted = (double[])folded.Clone(); Array.Sort(sorted); double median = sorted[phases / 2];
        for (int p = 0; p < phases; p++) TestContext.WriteLine($"phase {p}: {10 * Math.Log10(folded[p] / median):F2} dB");
        foreach (float v in output) Assert.That(float.IsFinite(v), Is.True);
        Assert.That(sorted[phases - 1] / median, Is.LessThan(2), "no phase of the response step carries extra high-frequency energy");
    }

    // Live and offline share the code path: one recorded walk rendered offline (CPU
    // reference, GPU one response per batch, GPU three per batch) and live gives the
    // same audio, so batching, readback and the late field agree.
    [Test, Category("GPU"), Category("Methods"), Category("Slow"), Timeout(1800000)]
    public void EnergyLiveMatchesOfflineRender()
    {
        const int rate = 8192;
        LateRoom(AcousticTake.Dimension.Spatial3D, new Vector3(6, 4, 5), LateMaterial(.25f, .4f), new Vector3(-1.2f, .3f, .5f), new Vector3(1.5f, -.4f, -.6f),
                 out var scene, out var geometry, out var state);
        var pcm = new float[301]; for (int i = 0; i < pcm.Length; i++) pcm[i] = .4f * Mathf.Sin(i * .31f) + .1f * Mathf.Sin(i * 1.7f);
        var events = new[] {new AcousticTake.VoiceEvent {gain = .5f, loop = true}};
        long[] times = {0, 97, 230, 411, 600, 777, 1024};
        string path = System.IO.Path.Combine(directory, "energy-walk");
        using (var writer = new AcousticTake.Writer(path, rate, scene, geometry, new[] {pcm}))
        {
            foreach (long time in times)
            {
                state[1].position = new Vector3(1.5f - time * .0004f, -.4f, -.6f); state[1].velocity = new Vector3(-3.3f, 0, 0);
                state[1].flags |= AcousticTake.StateFlags.VelocityValid; writer.Append(time, state);
            }
            foreach (var e in events) writer.Append(e);
            writer.Complete();
        }
        using var take = AcousticTake.Open(path);
        var settings = EnergyAudioSettings(64, .2f); settings.methodRays = 512; settings.methodArrivalSlots = 16; settings.tailSamples = 512;
        var shader = RequireGpu(); float[] expected = null;
        foreach (var mode in new[] {(TakeRender.AudioBackend.CpuReference, 1), (TakeRender.AudioBackend.Gpu, 1), (TakeRender.AudioBackend.Gpu, 3)})
        {
            settings.audioBackend = mode.Item1; settings.maximumResponsesPerBatch = mode.Item2;
            string output = System.IO.Path.Combine(directory, "energy-walk-" + mode.Item1 + mode.Item2);
            var report = TakeRender.RenderDirect(take, settings, output, null, shader);
            Assert.That(report.method, Is.EqualTo("energy-arrivals-diffuse-v1"));
            var actual = ReadWave(System.IO.Path.Combine(output, "listener-0.wav"));
            if (expected == null) { expected = actual; double energy = 0; foreach (float v in actual) energy += v * v; Assert.That(energy, Is.GreaterThan(1e-4)); }
            else AssertBurstAudio(expected, actual);
        }
        settings.audioBackend = TakeRender.AudioBackend.Gpu; settings.maximumResponsesPerBatch = 2;
        var rendered = new List<float>(); Action<long, float[], int> collect = (_, block, count) => { for (int i = 0; i < count; i++) rendered.Add(block[i]); };
        var left = (AcousticTake.State[])state.Clone(); var right = (AcousticTake.State[])state.Clone(); take.ReadStoredFrame(0, left);
        using (var live = new TakeRender.Live(scene, take.Geometry, new[] {pcm}, rate, settings, null, 0, 0, shader))
        {
            live.Prepare(left); live.Append(events[0]); live.Observe(0, 0, left, left, collect);
            for (int f = 1; f < times.Length; f++) { take.ReadStoredFrame(f, right); live.Observe(times[f - 1], times[f], left, right, collect); Array.Copy(right, left, left.Length); }
            live.Finish(times[times.Length - 1], left, collect); Assert.That(live.Completed, Is.True);
        }
        AssertBurstAudio(expected, rendered.ToArray());
    }
}
