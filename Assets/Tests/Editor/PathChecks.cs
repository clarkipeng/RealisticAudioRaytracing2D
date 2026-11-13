using System;
using System.Collections.Generic;
using NUnit.Framework;
using UnityEngine;

// Checks of the Path method (one energy path tracer, MethodPath.hlsl) and of its audio
// synthesis (PathReverb); docs/benchmarks/path-tracer-2026-11-12 reports the values. The analytic
// controls shared with the other methods (free field, rigid plane, image lattice, half-plane,
// far side of a solid) are MethodChecks cases with TakeRender.Method.Path.
public partial class TakeChecks
{
    // Fully diffusing shoebox: the path tracer decays at the Eyring rate.
    [Test, Category("GPU"), Category("Methods"), Category("Slow"), Timeout(1800000)]
    public void PathMethodDecaysAtTheEyringRate()
    {
        var size = new Vector3(6, 5, 4); const float absorption = .2f;
        LateRoom(AcousticTake.Dimension.Spatial3D, size, LateMaterial(absorption, 1), new Vector3(-1.5f, .5f, .4f), new Vector3(1.4f, -.7f, -.3f),
                 out var scene, out var geometry, out var state);
        var settings = ControlSettings(TakeRender.Method.Path, 8192, 1);
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
        TestContext.WriteLine($"T60 {t60:F3} s, Eyring {eyring:F3} s, counted contributions {Sum(solver.Method.Counts)}");
        Assert.That(t60 / eyring, Is.EqualTo(1).Within(.15));
    }
    // Sound leaking out of a scattering room through its walls (t = 1e-3), listener outside: the path
    // method's connections through walls (traced with probability = the first wall's transmission,
    // weighted back) give the energy method's leaked energy. Both are estimates; the tolerance covers
    // their seed noise.
    [Test, Category("GPU"), Category("Methods"), Category("Slow"), Timeout(1800000)]
    public void PathMethodLeaksThroughWallsLikeTheEnergyMethod()
    {
        var size = new Vector3(6, 5, 4);
        double Leak(TakeRender.Method method, int seed, float survival = -1, int flags = 0)
        {
            LateRoom(AcousticTake.Dimension.Spatial3D, size, LateMaterial(.1f, .5f, 1e-3f), new Vector3(-1.5f, .5f, .4f), new Vector3(3.8f, -.3f, .2f),
                     out var scene, out var geometry, out var state);
            var settings = ControlSettings(method, 1 << 16, .4f); settings.seed = seed; settings.specularDetectionRadius = .5f;
            if (survival >= 0) settings.pathSurvivalSeconds = survival;
            using var solver = new TransportSolver(scene, geometry, settings, RequireGpu(), sampleRate: 8000);
            if (method == TakeRender.Method.Path) solver.Method.ProfileFlags = flags;
            solver.Solve(state, new TakeRender.DirectPath[1]);
            double direct = (new Vector3(3.8f, -.3f, .2f) - new Vector3(-1.5f, .5f, .4f)).magnitude / 343;
            return Window(ControlEnergy(solver, 2), (int)Math.Ceiling(direct * 1000) + 5, int.MaxValue);
        }
        double path = 0, all = 0, lifelong = 0, energy = 0;
        for (int seed = 0; seed < 2; seed++)
        {
            path += Leak(TakeRender.Method.Path, seed) / 2; all += Leak(TakeRender.Method.Path, seed, -1, 16) / 2;
            lifelong += Leak(TakeRender.Method.Path, seed, 0, 16) / 2; energy += Leak(TakeRender.Method.Energy, seed) / 2;
        }
        TestContext.WriteLine($"leaked energy: path {path:G4}; every blocked connection traced {all:G4}; and no survival in time {lifelong:G4}; energy method {energy:G4}");
        Assert.That(path / all, Is.EqualTo(1).Within(.2), "Tracing blocked connections with probability t (weighted 1/t) keeps the mean.");
        Assert.That(all / lifelong, Is.EqualTo(1).Within(.2), "Survival in time keeps the mean.");
    }
    // Diffuse-field leakage, an analytic reference for sound through walls. In a closed room whose field
    // is diffuse (isotropic, uniform), the radiance L at a wall is E_in / (4 pi), E_in the reverberant
    // energy measured just inside it; a wall transmitting tau = (1 - a) t (undeviated) shows the
    // listener outside radiance tau L over the solid angle Omega it subtends, so the reverberant energy
    // outside is tau E_in Omega / (4 pi). The direct sound through the wall, (1 - a) t / r^2, is
    // subtracted from the outside total, and the direct sound from the inside one. s = 1 makes the
    // field diffuse; s = 0.5 leaves half of every reflection specular (the field is then only
    // approximately diffuse, and its specular part leaks through chains of reflections).
    [TestCase(TakeRender.Method.Path, 1f), TestCase(TakeRender.Method.Path, .5f), TestCase(TakeRender.Method.Energy, 1f), TestCase(TakeRender.Method.Energy, .5f),
     Category("GPU"), Category("Methods"), Category("Slow"), Timeout(1800000)]
    public void MethodsLeakAsTheDiffuseFieldPredicts(TakeRender.Method method, float scattering)
    {
        var size = new Vector3(6, 5, 4); const float a = .1f, t = 1e-3f;
        var source = new Vector3(-1.5f, .5f, .4f); var inside = new Vector3(2.2f, -.3f, .2f); var outside = new Vector3(3.8f, -.3f, .2f);
        double Total(Vector3 listener, int seed)
        {
            LateRoom(AcousticTake.Dimension.Spatial3D, size, LateMaterial(a, scattering, t), source, listener, out var scene, out var geometry, out var state);
            var settings = ControlSettings(method, 1 << 16, 1.5f); settings.seed = seed; settings.specularDetectionRadius = .5f;
            using var solver = new TransportSolver(scene, geometry, settings, RequireGpu(), sampleRate: 8000);
            solver.Solve(state, new TakeRender.DirectPath[1]);
            return Window(ControlEnergy(solver, 2), 0, int.MaxValue);
        }
        double tau = (1 - a) * t;
        double inRev = (Total(inside, 0) + Total(inside, 1)) / 2 - 1 / (inside - source).sqrMagnitude;
        double outRev = (Total(outside, 0) + Total(outside, 1)) / 2 - tau / (outside - source).sqrMagnitude;
        double h = outside.x - size.x / 2, omega = 0;
        foreach (double ay in new[] {size.y / 2 - outside.y, size.y / 2 + outside.y})
            foreach (double bz in new[] {size.z / 2 - outside.z, size.z / 2 + outside.z})
                omega += Math.Atan(ay * bz / (h * Math.Sqrt(h * h + ay * ay + bz * bz)));
        double predicted = tau * inRev * omega / (4 * Math.PI);
        TestContext.WriteLine($"{method}, s = {scattering}: reverberant energy inside {inRev:G4}, outside {outRev:G4}, diffuse-field prediction {predicted:G4} (wall solid angle {omega:F3} sr), ratio {outRev / predicted:F3}");
        Assert.That(outRev / predicted, Is.EqualTo(1).Within(scattering >= 1 ? .2 : .35));
    }
    static long Sum(uint[] values) { long s = 0; foreach (var v in values) s += v; return s; }

    // A specular image whose detection region straddles a visibility boundary: a rigid floor and
    // an absorbing panel that clips about half of the region but not the exact image path. The
    // retrace-and-divide-by-the-region estimator is low by the clipped share; dividing by the
    // share that follows the chain, measured by geometric trials, makes it unbiased. With one
    // trial (pathMaxTrials = 1) the trials degenerate to the plain estimator, which is reported.
    [Test, Category("GPU"), Category("Methods"), Category("Slow"), Timeout(1800000)]
    public void PathDetectionIsUnbiasedAtVisibilityBoundaries()
    {
        var scene = Scene(2); scene.listeners = new[] {1};
        scene.materials = new[] {RigidControl(), new AcousticTake.MaterialDefinition {name = "absorber", frequencies = new[] {0f}, absorption = new[] {1f}, scattering = new[] {0f}, transmission = new[] {0f}}};
        var list = new List<AcousticTake.Primitive>();
        void Quad(Vector3 a, Vector3 b, Vector3 c, Vector3 d, int material)
        {
            list.Add(new AcousticTake.Primitive {objectIndex = 2, materialIndex = material, a = a, b = c, c = b});
            list.Add(new AcousticTake.Primitive {objectIndex = 2, materialIndex = material, a = a, b = d, c = c});
        }
        Quad(new Vector3(-50, 0, -50), new Vector3(50, 0, -50), new Vector3(50, 0, 50), new Vector3(-50, 0, 50), 0);
        // Downward leg of the image path: (0, 1.5, 0) -> (2.222, 0, 0.278); at x = 1.1 it passes z = 0.138. The
        // panel's edge is 4 cm beside it: rays of the region that hit the panel first never reflect off
        // the floor, so they cannot detect this image (blocking the last leg would not matter: a ray
        // detects at its reflection, and the exact last leg is retraced).
        Quad(new Vector3(1.1f, .01f, .178f), new Vector3(1.1f, 3, .178f), new Vector3(1.1f, 3, 3), new Vector3(1.1f, .01f, 3), 1);
        var geometry = list.ToArray(); scene.objects[2].primitiveCount = geometry.Length;
        var state = States(2); state[0].position = new Vector3(0, 1.5f, 0); state[1].position = new Vector3(4, 1.2f, .5f);
        state[2].position = Vector3.zero; state[2].flags = AcousticTake.StateFlags.Active | AcousticTake.StateFlags.Geometry;
        double ri = (state[1].position - new Vector3(0, -1.5f, 0)).magnitude;
        int bin = (int)Math.Floor(ri / 343 / .001);
        double Image(int trials)
        {
            var settings = ControlSettings(TakeRender.Method.Path, 1 << 18, .05f); settings.specularDetectionRadius = .5f; settings.pathMaxTrials = trials;
            settings.useBvh = false; // four triangles
            using var solver = new TransportSolver(scene, geometry, settings, RequireGpu(), sampleRate: 8000);
            solver.Solve(state, new TakeRender.DirectPath[1]);
            double held = Window(ControlEnergy(solver, 2), bin, bin + 1) * ri * ri;
            TestContext.WriteLine($"trial cap {trials}: image {held:F4} x 1/ri^2; detections {solver.Method.Counter(7)}, scored {solver.Method.Counter(8)}, " +
                                  $"trials {solver.Method.Counter(15)}, capped {solver.Method.Counter(13)}");
            return held;
        }
        double plain = Image(1), corrected = Image(64);
        Assert.That(corrected, Is.EqualTo(1).Within(.1), "Detection with the measured share is unbiased.");
        Assert.That(plain, Is.LessThan(.9), "The plain region estimator is low where the region is clipped (this fixture's purpose).");
    }

    // The audio synthesis reproduces the histogram: one fine-bin arrival becomes one band-shaped
    // click at its delay, and a late exponential decay keeps its band energies (measured through
    // the comparison's band filters). Each impulse's spectrum interpolates the band amplitudes
    // between band centres (amplitude-complementary weights), so a band measured through the
    // comparison's filter also sees its neighbours: the expectation is that interpolated
    // spectrum's band energy (equal to the histogram's for a flat spectrum, a little below the
    // band value where the spectrum is steep; both are printed).
    [Test, Category("Core")]
    public void PathReverbReproducesTheHistogram()
    {
        const int rate = 32000;
        var settings = SpectralSettings(); settings.method = TakeRender.Method.Path; settings.maximumDelaySeconds = 1.5f;
        settings.responseStepSamples = 1024; settings.fftSize = 1024; settings.listenerOutput = TakeRender.ListenerOutput.Mono;
        var bins = new PathBins(settings);
        var scene = Scene(); var state = States();
        double[] click = {1, .7, .5, .35, .25, .18, .12, .09};
        double Histogram(int pair, int d, int b, int k)
        {
            double t = bins.Start(k), e = 0;
            if (k == bins.Of(.01)) e += d == 0 ? click[b] : 0;                                   // one arrival at 10 ms, from +x
            if (t >= .05) e += d == 2 ? .002 * Math.Pow(.8, b) * bins.Width(k) / .001 : 0;  // stationary late field, -1 dB per octave
            return e;
        }
        var pcm = new float[1]; pcm[0] = 1;
        var signal = new TakeRender.Signal(pcm); signal.SetActive(0, true);
        signal.Append(new AcousticTake.VoiceEvent {action = AcousticTake.VoiceAction.Play, gain = 1});
        var reverb = new PathReverb(scene, rate, settings, new[] {signal});
        reverb.SetResponse(0, true, Histogram, state);
        int length = (int)(1.6 * rate), latency = settings.DspLatencySamples;
        var output = new float[length];
        var block = new float[1024];
        for (int at = 0; at < length; at += 1024)
        {
            int count = Math.Min(1024, length - at);
            Array.Clear(block, 0, block.Length); reverb.Render(at, count, block); Array.Copy(block, 0, output, at, count);
        }
        // Response (latency removed), in 1 ms band energies.
        var h = new double[length - latency];
        for (int n = 0; n < h.Length; n++) h[n] = output[n + latency];
        for (int b = 0; b < 6; b++)
        {
            var e = ImpulseBandEnergy(h, rate, b, 1600);
            double clickEnergy = Window(e, 5, 16), want = Interpolated(click, b);
            TestContext.WriteLine($"band {MethodBands.Centers[b]} Hz: click {clickEnergy:G4} (interpolated spectrum {want:G4}, histogram {click[b]:G4}), peak bin {Array.IndexOf(e, Max(e, 0, 30))} ms");
            Assert.That(clickEnergy / want, Is.EqualTo(1).Within(.15), "A single arrival keeps its band energy.");
            // Late field (random signs): band energy over 0.1-1.4 s; its relative spread is about one over
            // sqrt(bandwidth x duration), about 10% at 125 Hz.
            double late = Window(e, 100, 1400), wantLate = 0, bandLate = 0;
            var bandEnergies = new double[8];
            for (int k = 0; k < bins.Count; k++)
            {
                double s0 = bins.Start(k); if (s0 < .1 || s0 >= 1.4) continue;
                for (int q = 0; q < 8; q++) bandEnergies[q] = Histogram(0, 2, q, k);
                wantLate += Interpolated(bandEnergies, b); bandLate += bandEnergies[b];
            }
            TestContext.WriteLine($"   late 0.1-1.4 s {late:G4} (interpolated spectrum {wantLate:G4}, histogram {bandLate:G4})");
            Assert.That(late / wantLate, Is.EqualTo(1).Within(b == 0 ? .3 : .15), "The late field keeps its band energy.");
        }
    }
    // Updates every step between two histograms that split a late field over disjoint bins (the
    // extreme of two independent noisy estimates) must sound as loud as their mean, held fixed.
    [Test, Category("Core")]
    public void PathReverbCrossfadeKeepsThePower()
    {
        const int rate = 32000;
        var settings = SpectralSettings(); settings.method = TakeRender.Method.Path; settings.maximumDelaySeconds = .5f;
        settings.responseStepSamples = 512; settings.fftSize = 512; settings.listenerOutput = TakeRender.ListenerOutput.Mono;
        var bins = new PathBins(settings);
        var scene = Scene(); var state = States();
        double Late(int k, int b) => bins.Start(k) >= .02 && bins.Start(k) < .4 ? .002 * Math.Pow(.8, b) * bins.Width(k) / .001 : 0;
        Func<int, int, int, int, double> mean = (p, d, b, k) => d == 2 ? Late(k, b) : 0;
        Func<int, int, int, int, double> even = (p, d, b, k) => d == 2 && k % 2 == 0 ? 2 * Late(k, b) : 0;
        Func<int, int, int, int, double> odd = (p, d, b, k) => d == 2 && k % 2 == 1 ? 2 * Late(k, b) : 0;
        int length = 3 * rate, step = settings.responseStepSamples;
        var noise = new float[length]; var rng = new System.Random(1);
        for (int n = 0; n < length; n++) noise[n] = (float)(rng.NextDouble() * 2 - 1);
        double Energy(bool alternate)
        {
            var signal = new TakeRender.Signal(noise); signal.SetActive(0, true);
            signal.Append(new AcousticTake.VoiceEvent {action = AcousticTake.VoiceAction.Play, gain = 1});
            var reverb = new PathReverb(scene, rate, settings, new[] {signal});
            var block = new float[step]; double energy = 0;
            for (int at = 0, i = 0; at < length; at += step, i++)
            {
                reverb.SetResponse(at, false, alternate ? (i % 2 == 0 ? even : odd) : mean, state);
                Array.Clear(block, 0, step); reverb.Render(at, step, block);
                if (at >= rate / 2) foreach (var x in block) energy += (double)x * x;
            }
            return energy;
        }
        double fixedEnergy = Energy(false), alternating = Energy(true);
        TestContext.WriteLine($"alternating / fixed output energy {alternating / fixedEnergy:F3} ({10 * Math.Log10(alternating / fixedEnergy):+0.00;-0.00} dB; a plain linear crossfade gives 0.70, -1.55 dB)");
        Assert.That(alternating / fixedEnergy, Is.EqualTo(1).Within(.05), "Crossfading between updates keeps the power.");
    }
    // Band energies in 1 ms bins of a sampled impulse response (unit sample = 1 m free field), with
    // the comparison's zero-phase band filters (tools/method_compare.py energy_from_impulse).
    static double[] ImpulseBandEnergy(double[] h, int rate, int band, int bins)
    {
        int n = 1; while (n < 2 * h.Length) n <<= 1;
        var re = new double[n]; var im = new double[n];
        Array.Copy(h, re, h.Length);
        MethodBands.Fft(re, im, n, false);
        double fc = MethodBands.Centers[band];
        for (int k = 0; k < n; k++)
        {
            double f = (k <= n / 2 ? k : n - k) * (double)rate / n, g = Math.Sqrt(MethodBands.Power(f, fc));
            re[k] *= g; im[k] *= g;
        }
        MethodBands.Fft(re, im, n, true);
        // A unit sample has energy 1 / rate per sample squared; the filter's energy normalises the band.
        double norm = rate / MethodBands.FilterEnergy(fc);
        var e = new double[bins];
        for (int i = 0; i < n; i++)
        {
            int t = i < n / 2 ? i : i - n; int k = Math.Max(0, (int)Math.Floor(t * 1000.0 / rate));
            if (k < bins) e[k] += re[i] * re[i] * norm;
        }
        return e;
    }
    // Band energy, through the comparison's filter of `band`, of one impulse whose amplitude
    // spectrum interpolates sqrt(energies) with PathReverb's band weights.
    static double Interpolated(double[] energies, int band)
    {
        double fc = MethodBands.Centers[band], sum = 0, norm = 0;
        for (int i = 0; i < 4000; i++)
        {
            double f = fc / 2 * Math.Pow(4, (i + .5) / 4000), df = f * Math.Log(4) / 4000, a = 0;
            for (int q = 0; q < 8; q++) a += Math.Sqrt(energies[q]) * PathReverb.BandWeight(q, f);
            double g = MethodBands.Power(f, fc);
            sum += a * a * g * df; norm += g * df;
        }
        return sum / norm;
    }
    static double Max(double[] values, int from, int to) { double m = double.MinValue; for (int i = from; i < to; i++) m = Math.Max(m, values[i]); return m; }
}
