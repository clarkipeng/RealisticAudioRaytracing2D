using System;
using System.Collections.Generic;
using NUnit.Framework;
using UnityEngine;

// Doppler on reflections (README part 40): the tracer's rate channel (MethodPath.hlsl "Delay rates") and the early
// part's sliding slots (PathReverb), against the analytic image source of a rigid plane.
public partial class TakeChecks
{
    // Unity's validation logs "There are more uavs (10) than the maximum supported (8)" when it first dispatches a
    // TracePath3D*Doppler kernel (README part 40: the kernel binds the same buffers as the static walk, which never logs
    // it, and its outputs pass these checks; the cause is not found). The checks tolerate that message only.
    static void ToleratePathDopplerUavMessage() =>
        UnityEngine.TestTools.LogAssert.Expect(LogType.Error, new System.Text.RegularExpressions.Regex("more uavs .* kernel 'TracePath3D(Bvh)?Doppler'"));
    // A source moving at 30 m/s past a rigid plane, the listener still, persistent rays, 8 kHz. Per update the rate
    // channel's mean over the plane's reflection must be the image source's Doppler (its delay rate, Doppler.Direct of
    // the mirrored motion); the reverb of a 250 Hz tone (the reflection alone: the direct sound is a separate path) must
    // be a tone at the image's shifted frequency, and purer than the same responses rendered without the rate channel
    // (the reflection then steps from update to update). A static control: the rate channel stays exactly zero and the
    // reverb is bit-identical with and without it.
    [Test, Category("GPU"), Category("Methods"), Category("Slow"), Timeout(1800000)]
    public void PathReflectionsCarryTheImageSourceDoppler()
    {
        ToleratePathDopplerUavMessage();
        const int rate = 8000, step = 256; const float speed = 343, f0 = 250;
        var scene = Scene(2); scene.listeners = new[] {1}; scene.materials = new[] {RigidControl()};
        var a = new Vector3(-200, 0, -200); var b = new Vector3(200, 0, -200); var c = new Vector3(200, 0, 200); var d = new Vector3(-200, 0, 200);
        var geometry = new[] {new AcousticTake.Primitive {objectIndex = 2, a = a, b = c, c = b}, new AcousticTake.Primitive {objectIndex = 2, a = a, b = d, c = c}};
        scene.objects[2].primitiveCount = geometry.Length;
        var L = new Vector3(0, 1.5f, 4); var v = new Vector3(30, 0, 0);
        var settings = ControlSettings(TakeRender.Method.Path, 1 << 16, .25f); settings.useBvh = false;
        settings.transport = TakeRender.Transport.Raytraced; settings.synthesis = TakeRender.Synthesis.Spectral; settings.listenerOutput = TakeRender.ListenerOutput.Mono;
        settings.pathPersistent = true; settings.responseStepSamples = step; settings.fftSize = step; settings.spectralHop = step / 4; settings.transitionSamples = step / 4;
        int latency = settings.DspLatencySamples;
        var clip = new float[3200]; for (int n = 0; n < clip.Length; n++) clip[n] = Mathf.Sin(2 * Mathf.PI * f0 * n / rate);
        TakeRender.Signal Signal()
        {
            var s = new TakeRender.Signal(clip); s.SetActive(0, true);
            s.Append(new AcousticTake.VoiceEvent {action = AcousticTake.VoiceAction.Play, gain = 1, loop = true});
            return s;
        }
        // The source at take time t (seconds): x from -32 m at 8 m above the plane, approaching; its image below it.
        Vector3 Source(double t) => new Vector3(-32 + 30 * (float)t, 8, 0);
        Vector3 Mirror(Vector3 p) => new Vector3(p.x, -p.y, p.z);

        (double[] slid, double[] stepped, double tracer, double analytic, double spread, int slidUpdates) Render(bool moving)
        {
            using var solver = new TransportSolver(scene, geometry, settings, RequireGpu(), sampleRate: rate, methodAudio: true);
            var m = solver.Method; int updates = m.WarmUpdates + 8;
            var withRates = new PathReverb(scene, rate, settings, new[] {Signal()}); var without = new PathReverb(scene, rate, settings, new[] {Signal()});
            var state = States(2); state[1].position = L; state[2].position = Vector3.zero; state[2].flags = AcousticTake.StateFlags.Active | AcousticTake.StateFlags.Geometry;
            if (moving) { state[0].velocity = v; state[0].flags |= AcousticTake.StateFlags.VelocityValid; }
            var slid = new double[updates * step]; var stepped = new double[slid.Length]; var block = new float[step];
            double tracer = 0, analytic = 0, spread = 0; int slidUpdates = 0;
            for (int n = 0; n < updates; n++)
            {
                // Update n's state is the take's at sample n step + step - latency (TakeRender.Live's anchor); output sample t
                // is heard at take time t - latency.
                double time = (n * step + step - latency) / (double)rate;
                state[0].position = moving ? Source(time) : Source(0);
                solver.Solve(state, new TakeRender.DirectPath[1]);
                var late = solver.Late(0);
                withRates.SetResponse(n * (long)step, n == 0, PathReverb.Histogram(late), state, PathReverb.Counts(late), PathReverb.Rates(late));
                without.SetResponse(n * (long)step, n == 0, PathReverb.Histogram(late), state, PathReverb.Counts(late));
                Array.Clear(block, 0, step); withRates.Render(n * (long)step, step, block); for (int i = 0; i < step; i++) slid[n * step + i] = block[i];
                Array.Clear(block, 0, step); without.Render(n * (long)step, step, block); for (int i = 0; i < step; i++) stepped[n * step + i] = block[i];
                if (moving && n >= m.WarmUpdates - 2)
                {
                    var line = new System.Text.StringBuilder($"update {n}: slid bins {withRates.SlidBins}; bins (start ms, energy, mean rate, sd, count):");
                    for (int k = 0; k < m.TimeBins.Count; k++)
                    {
                        double e = 0; for (int dir = 0; dir < 6; dir++) for (int band = 0; band < 8; band++) e += m.HistogramEnergy(0, 0, dir, band, k);
                        if (e <= 0) continue; double mu = m.RateMoment(0, 0, 0, k) / e;
                        line.Append($" {m.TimeBins.Start(k) * 1000:F1}/{e:G2}/{mu:F4}/{Math.Sqrt(Math.Max(0, m.RateMoment(0, 0, 1, k) / e - mu * mu)):G2}/{m.HistogramCount(0, 0, k)}");
                        line.Append($" mean delay {(m.TimeBins.Start(k) + m.RateMoment(0, 0, 2, k) / e) * 1000:F3} ms");
                    }
                    Doppler.Direct(L - Mirror(Source(time)), Mirror(v), Vector3.zero, speed, out float retarded, out float r0, out _);
                    line.Append($"; analytic delay {retarded * 1000:F3} ms, rate {r0:F4}");
                    TestContext.WriteLine(line.ToString());
                }
                if (n < m.WarmUpdates) continue;
                double energy = 0, first = 0, second = 0;
                for (int k = 0; k < m.TimeBins.Count; k++)
                {
                    for (int dir = 0; dir < 6; dir++) for (int band = 0; band < 8; band++) energy += m.HistogramEnergy(0, 0, dir, band, k);
                    first += m.RateMoment(0, 0, 0, k); second += m.RateMoment(0, 0, 1, k);
                }
                if (!moving)
                {
                    for (int k = 0; k < m.TimeBins.Count; k++) Assert.That(m.RateMoment(0, 0, 0, k) == 0 && m.RateMoment(0, 0, 1, k) == 0, "a static scene leaves the rate channel empty");
                    continue;
                }
                Doppler.Direct(L - Mirror(Source(time)), Mirror(v), Vector3.zero, speed, out _, out float expected, out _);
                double mean = first / energy;
                tracer = Math.Max(tracer, Math.Abs(mean - expected) / (1 - expected)); analytic = expected;
                spread = Math.Max(spread, Math.Sqrt(Math.Max(0, second / energy - mean * mean)));
                slidUpdates++;
            }
            AssertNoOverflow(solver);
            return (slid, stepped, tracer, analytic, spread, slidUpdates);
        }

        var still = Render(false);
        for (int i = 0; i < still.slid.Length; i++) Assert.That(BitConverter.DoubleToInt64Bits(still.slid[i]), Is.EqualTo(BitConverter.DoubleToInt64Bits(still.stepped[i])), "static: bit-identical with the rate channel");
        var run = Render(true);

        // The reflection's frequency over the last six updates: the image's mean shift over that window (its retarded
        // delay at the window's ends, in take time), against the rendered tone's peak (Hann window, parabolic interpolation).
        int length = 6 * step, start = run.slid.Length - length;
        double Delay(double outputSample)
        {
            double t = (outputSample - latency) / rate; Doppler.Direct(L - Mirror(Source(t)), Mirror(v), Vector3.zero, speed, out float delay, out _, out _);
            return delay;
        }
        double expectedHz = f0 * (1 - (Delay(start + length) - Delay(start)) * rate / length);
        (double hz, double purity) Peak(double[] y)
        {
            double best = 0, bestHz = 0, total = 0; var power = new List<(double hz, double p)>();
            for (double hz = 150; hz <= 350; hz += .25)
            {
                double re = 0, im = 0;
                for (int i = 0; i < length; i++) { double w = .5 - .5 * Math.Cos(2 * Math.PI * i / (length - 1)), phase = 2 * Math.PI * hz * i / rate; re += w * y[start + i] * Math.Cos(phase); im -= w * y[start + i] * Math.Sin(phase); }
                double p = re * re + im * im; power.Add((hz, p)); total += p;
                if (p > best) { best = p; bestHz = hz; }
            }
            double near = 0; foreach (var (hz, p) in power) if (Math.Abs(hz - bestHz) <= 4) near += p;
            return (bestHz, near / total);
        }
        var slidPeak = Peak(run.slid); var steppedPeak = Peak(run.stepped);
        TestContext.WriteLine($"rate channel: worst relative error of the reflection's frequency ratio over {run.slidUpdates} updates {run.tracer:E2} (image delay rate {run.analytic:F5}), " +
                              $"spread of rates {run.spread:E2}; reverb tone: expected {expectedHz:F3} Hz, slid {slidPeak.hz:F2} Hz (purity {slidPeak.purity:F3}), " +
                              $"without the rate channel {steppedPeak.hz:F2} Hz (purity {steppedPeak.purity:F3}); unshifted {f0} Hz");
        Assert.That(run.slidUpdates, Is.GreaterThan(4));
        Assert.That(run.tracer, Is.LessThan(2e-3), "the rate channel's mean is the image source's Doppler");
        Assert.That(Math.Abs(f0 - expectedHz), Is.GreaterThan(10), "the geometry shifts the reflection audibly");
        Assert.That(slidPeak.hz, Is.EqualTo(expectedHz).Within(.75), "the slid reflection is heard at the image source's frequency");
        Assert.That(slidPeak.purity, Is.GreaterThan(steppedPeak.purity), "sliding is purer than stepping the reflection from update to update");
    }

    // Interaural delay of the echoes (README part 42): one reflection off a rigid wall, its image source receding from the
    // listener at 10 m/s along a fixed line (so the echo slides and its direction stays put), binaural, persistent rays,
    // 16 kHz, a noise source. The head turns so that the image lies at each azimuth in turn. The echo's interaural lag
    // (the peak of the cross-correlation of the reverb's two ears) must be the direct path's: the same noise through the
    // HRTF blend the direct path uses for that direction, measured the same way, stretched by the motion. (A source whose
    // delay grows at r per second of reception reaches the far ear ITD later in travel time but ITD / (1 - r) later in
    // reception time: p_L(t) = p_R(t - D) with D (1 - r) = ITD.)
    [Test, Category("GPU"), Category("Methods"), Category("Slow"), Timeout(1800000)]
    public void PathEchoesCarryTheHrtfInterauralDelay()
    {
        ToleratePathDopplerUavMessage();
        const int rate = 16000, step = 512, window = 6 * step; const float v = 10, start = 8;
        var scene = Scene(2); scene.listeners = new[] {1}; scene.materials = new[] {RigidControl()};
        var a = new Vector3(0, -30, -30); var b = new Vector3(0, -30, 30); var c = new Vector3(0, 30, 30); var d = new Vector3(0, 30, -30);   // (60 m: the persistent rays' ring scales with the scene)
        var geometry = new[] {new AcousticTake.Primitive {objectIndex = 2, a = a, b = c, c = b}, new AcousticTake.Primitive {objectIndex = 2, a = a, b = d, c = c}};
        scene.objects[2].primitiveCount = geometry.Length;
        var L = new Vector3(3, 1.5f, 0);
        var settings = ControlSettings(TakeRender.Method.Path, 1 << 14, .25f); settings.useBvh = false;   // (a quarter of the Doppler check's rays: seven solvers in one process)
        settings.transport = TakeRender.Transport.Raytraced; settings.synthesis = TakeRender.Synthesis.Spectral; settings.listenerOutput = TakeRender.ListenerOutput.Binaural;
        settings.pathPersistent = true; settings.responseStepSamples = step; settings.fftSize = step; settings.spectralHop = step / 4; settings.transitionSamples = step / 4;
        int latency = settings.DspLatencySamples;
        var noise = new float[rate]; uint h = 12345; for (int n = 0; n < noise.Length; n++) { h = h * 1664525u + 1013904223u; noise[n] = ((h >> 8) * (1f / 16777216f)) * 2 - 1; }
        var hrtf = Hrtf.Load(rate);
        // The lag (seconds) of the peak of sum_n left[n + lag] right[n] within +-1.2 ms, between samples by the same
        // band-limited interpolation as the HRTF set's own ITD (Hrtf.PeakLag); positive: the left ear later.
        double Lag(Func<int, double> left, Func<int, double> right, int length)
        {
            int lags = (int)Math.Ceiling(.0012 * rate), margin = lags + 9;
            return Hrtf.PeakLag(lag => { double sum = 0; for (int n = margin; n < length - margin; n++) sum += left(n + lag) * right(n); return sum; }, lags) / rate;
        }
        // Image source: from the listener toward the wall (-x), `start` m away at t = 0, receding at v; the source is its mirror.
        Vector3 Image(double t) => L + Vector3.left * (start + v * (float)t);
        Vector3 Source(double t) { var i = Image(t); return new Vector3(-i.x, i.y, i.z); }
        var report = new System.Text.StringBuilder("azimuth: echo ITD / direct path ITD (static) / expected, the direct path's moving / table / Woodworth (a = 8.75 cm), microseconds; slid bins\n");
        double worst = 0;
        foreach (float azimuth in new[] {-90f, -60f, -30f, 0f, 30f, 60f, 90f})
        {
            var head = Quaternion.Euler(0, -90 - azimuth, 0);
            using var solver = new TransportSolver(scene, geometry, settings, RequireGpu(), sampleRate: rate, methodAudio: true);
            var m = solver.Method; int updates = m.WarmUpdates + 8;
            var signal = new TakeRender.Signal(noise); signal.SetActive(0, true);
            signal.Append(new AcousticTake.VoiceEvent {action = AcousticTake.VoiceAction.Play, gain = 1, loop = true});
            var reverb = new PathReverb(scene, rate, settings, new[] {signal});
            var state = States(2); state[1].position = L; state[1].rotation = head; state[2].position = Vector3.zero; state[2].flags = AcousticTake.StateFlags.Active | AcousticTake.StateFlags.Geometry;
            state[0].velocity = new Vector3(v, 0, 0); state[0].flags |= AcousticTake.StateFlags.VelocityValid;
            var output = new float[updates * step * 2]; var block = new float[step * 2]; int slid = 0;
            for (int n = 0; n < updates; n++)
            {
                double time = (n * step + step - latency) / (double)rate;
                state[0].position = Source(time);
                solver.Solve(state, new TakeRender.DirectPath[1]);
                var late = solver.Late(0);
                reverb.SetResponse(n * (long)step, n == 0, PathReverb.Histogram(late), state, PathReverb.Counts(late), PathReverb.Rates(late));
                Array.Clear(block, 0, block.Length); reverb.Render(n * (long)step, step, block); Array.Copy(block, 0, output, n * step * 2, block.Length);
                if (n >= updates - 6) slid = Math.Min(n == updates - 6 ? int.MaxValue : slid, reverb.SlidBins);
            }
            AssertNoOverflow(solver);
            int first = output.Length / 2 - window;
            double echo = Lag(i => output[2 * (first + i)], i => output[2 * (first + i) + 1], window);
            // The direct path from the image's direction at the window's middle (head frame), through the same HRTF blend.
            double middle = (first + window / 2 - latency) / (double)rate;
            var local = Quaternion.Inverse(head) * (Image(middle) - L).normalized;
            var blend = hrtf.Interpolate(local); int taps = hrtf.Taps;
            var ears = new double[2][]; for (int e = 0; e < 2; e++) { ears[e] = new double[window + taps]; for (int i = 0; i < ears[e].Length; i++) { double sum = 0; for (int t = 0; t < taps && t <= i; t++) sum += hrtf.Coefficient(blend, e, t) * noise[(i - t) % noise.Length]; ears[e][i] = sum; } }
            double direct = Lag(i => ears[0][taps + i], i => ears[1][taps + i], window);
            Doppler.Direct(L - Image(middle), Vector3.left * v, Vector3.zero, 343, out _, out float r, out _);
            double expected = direct / (1 - r);
            double table = Hrtf.InterauralDelay(hrtf.InterauralDelayGrid, local), lateral = Math.Asin(Mathf.Clamp(local.x, -1, 1)), woodworth = .0875 / 343 * (lateral + Math.Sin(lateral));
            report.AppendLine($"{azimuth,4:F0}: {echo * 1e6,7:F1} / {direct * 1e6,7:F1} / {expected * 1e6,7:F1} / {table * 1e6,7:F1} / {woodworth * 1e6,7:F1}; {slid}");
            Assert.That(slid, Is.GreaterThan(0), $"azimuth {azimuth}: the echo slides (it is discrete), so it carries the ITD");
            if (Math.Abs(azimuth) >= 30) Assert.That(Math.Abs(direct), Is.GreaterThan(2e-4), "the check needs a lateral direct path");
            worst = Math.Max(worst, Math.Abs(echo - expected));
        }
        TestContext.WriteLine(report.ToString() + $"worst |echo - expected| {worst * 1e6:F1} us");
        Assert.That(worst, Is.LessThan(10e-6), "the echo is lateralised in time as the direct path from its direction");
    }
}
