using System;
using System.Diagnostics;
using System.IO;
using UnityEngine.Rendering;
using NUnit.Framework;
using UnityEngine;

// The path method's GPU audio (GpuAudio.Path: the band filters folded into one kernel per channel, built on the
// GPU) against the CPU reference (TakeRender.Mixer with PathReverb), on the same persistent-ray solves.
public partial class TakeChecks
{
    [TestCase(TakeRender.ListenerOutput.Binaural)]
    [TestCase(TakeRender.ListenerOutput.Mono)]
    [Category("GPU"), Category("Methods")]
    public void GpuPathReverbMatchesTheCpuReference(TakeRender.ListenerOutput ears)
    {
        var shader = RequireGpu();
        const int rate = 8000, size = 256;
        var scene = Scene(2); scene.listeners = new[] {1}; scene.materials = new[] {RigidControl()};
        var a = new Vector3(-20, 0, -20); var b = new Vector3(20, 0, -20); var c = new Vector3(20, 0, 20); var d = new Vector3(-20, 0, 20);
        var geometry = new[] {new AcousticTake.Primitive {objectIndex = 2, a = a, b = c, c = b}, new AcousticTake.Primitive {objectIndex = 2, a = a, b = d, c = c}};
        scene.objects[2].primitiveCount = geometry.Length;
        var state = States(2);
        state[0].position = new Vector3(0, 2, 0); state[1].position = new Vector3(3, 1.5f, 1);
        state[2].position = Vector3.zero; state[2].flags = AcousticTake.StateFlags.Active | AcousticTake.StateFlags.Geometry;
        var settings = ControlSettings(TakeRender.Method.Path, 4096, .25f); settings.useBvh = false;
        settings.transport = TakeRender.Transport.Raytraced; settings.synthesis = TakeRender.Synthesis.Spectral; settings.listenerOutput = ears;
        settings.pathPersistent = true; settings.responseStepSamples = size; settings.fftSize = size; settings.spectralHop = size / 4; settings.transitionSamples = size / 4;
        var clip = new float[331]; for (int n = 0; n < clip.Length; n++) clip[n] = .2f * Mathf.Sin(n * .37f) + .1f * Mathf.Cos(n * 1.93f);
        TakeRender.Signal Signal()
        {
            var s = new TakeRender.Signal(clip); s.SetActive(0, true);
            s.Append(new AcousticTake.VoiceEvent {action = AcousticTake.VoiceAction.Play, gain = 1, loop = true});
            return s;
        }
        var cpuSignal = new[] {Signal()}; var gpuSignal = new[] {Signal()};
        using var gpu = new GpuAudio(scene, rate, settings, gpuSignal);
        var cpu = new TakeRender.Mixer(scene, rate, settings, cpuSignal);
        using var solver = new TransportSolver(scene, geometry, settings, shader, sampleRate: rate, methodAudio: true);
        var direct = new TakeRender.DirectPath[1];
        int channels = settings.ChannelsPerListener;
        var x = new float[size * channels]; var y = new float[x.Length];
        double peak = 0, error = 0, energy = 0, late = 0;
        for (int frame = 0; frame < 14; frame++)
        {
            long sample = frame * (long)size; bool cut = frame == 9;
            state[1].position.x = 3 + frame * .05f; state[1].rotation = Quaternion.Euler(0, frame * 11, 0);
            solver.Solve(state, direct);
            cpu.SetResponse(sample, direct, state, cut, solver.RayPaths, 0, late: solver.Late(0));
            gpu.SetResponse(sample, cut, solver.RayBuffer, 0, scene, state, late: solver.Late(0));
            int at = 0;
            foreach (int count in new[] {37, size - 37})
            {
                Array.Clear(x, 0, x.Length); Array.Clear(y, 0, y.Length);
                cpu.Render(sample + at, count, x); gpu.Render(sample + at, count, y);
                for (int n = 0; n < count * channels; n++)
                {
                    double e = x[n] - y[n]; peak = Math.Max(peak, Math.Abs(e)); error += e * e; energy += x[n] * x[n];
                    if (frame >= 4) late += x[n] * x[n];
                }
                at += count;
            }
        }
        double relative = Math.Sqrt(error / energy);
        TestContext.WriteLine($"{ears}: GPU against CPU peak {peak:G3}, relative {relative:G3}, output energy {energy:G3}");
        Assert.That(late, Is.GreaterThan(1e-6), "the fixture is audible after the direct sound's onset");
        Assert.That(relative, Is.LessThan(2e-3), "the GPU path reverb renders what the CPU reference does");
    }

    // The live renderer with the path method on GPU audio, as the playable Church runs it: persistent rays, the
    // denoiser, observations that end inside a partition (the reverb reads its input half a partition ahead),
    // GPU-only batches (no CPU readback of the method), and the denoiser switched mid-way.
    [Test, Category("GPU"), Category("Methods")]
    public void LivePathRendersOnTheGpuWithTheDenoiser()
    {
        var shader = RequireGpu();
        const int rate = 8000, size = 256, lead = 1024;
        var scene = Scene(2); scene.listeners = new[] {1}; scene.materials = new[] {RigidControl()};
        var a = new Vector3(-20, 0, -20); var b = new Vector3(20, 0, -20); var c = new Vector3(20, 0, 20); var d = new Vector3(-20, 0, 20);
        var geometry = new[] {new AcousticTake.Primitive {objectIndex = 2, a = a, b = c, c = b}, new AcousticTake.Primitive {objectIndex = 2, a = a, b = d, c = c}};
        scene.objects[2].primitiveCount = geometry.Length;
        var state = States(2);
        state[0].position = new Vector3(0, 2, 0); state[1].position = new Vector3(3, 1.5f, 1);
        state[2].position = Vector3.zero; state[2].flags = AcousticTake.StateFlags.Active | AcousticTake.StateFlags.Geometry;
        var settings = ControlSettings(TakeRender.Method.Path, 2048, .25f); settings.useBvh = false;
        settings.transport = TakeRender.Transport.Raytraced; settings.synthesis = TakeRender.Synthesis.Spectral; settings.listenerOutput = TakeRender.ListenerOutput.Binaural;
        settings.audioBackend = TakeRender.AudioBackend.Gpu; settings.pathPersistent = true; settings.pathPopulation = 8192; settings.pathDenoise = true;
        settings.responseStepSamples = size; settings.fftSize = size; settings.spectralHop = size / 4; settings.transitionSamples = size / 4; settings.maximumResponsesPerBatch = 2;
        var queue = new AudioManager.PcmQueue(16384, rate, null, 2); queue.Publish(0, null, lead);
        var clip = new float[331]; for (int n = 0; n < clip.Length; n++) clip[n] = .2f * Mathf.Sin(n * .37f);
        using var live = new TakeRender.Live(scene, geometry, new[] {clip}, rate, settings, queue, lead, 0, shader);
        Assert.That(live.HasPathDenoiser, "pathDenoise loads the bundled denoiser");
        live.Prepare(state); live.Append(new AcousticTake.VoiceEvent {gain = 1, loop = true});
        var output = new float[2 * 8192]; long consumed = 0;
        for (long to = 300; to <= 6000; to += 300)
        {
            state[1].position.x = 3 + to * 1e-4f;
            live.Observe(to - 300, to, state, state);
            live.Pump(true);
            if (to == 3000) live.PathDenoising = false;
            int frames = (int)Math.Min(live.Cursor + lead - consumed, 1024);
            if (frames > 0) { var block = new float[2 * frames]; queue.Consume(block, 2); Array.Copy(block, 0, output, 2 * (int)consumed % output.Length, Math.Min(block.Length, output.Length - 2 * (int)consumed % output.Length)); consumed += frames; }
        }
        double energy = 0; foreach (float v in output) { Assert.That(float.IsFinite(v)); energy += v * v; }
        TestContext.WriteLine($"live path on GPU audio: cursor {live.Cursor}, {live.GpuDiagnostics.readbacks} readbacks, output energy {energy:G3}");
        Assert.That(live.Cursor, Is.GreaterThanOrEqualTo(5000), "the renderer keeps up with the observations");
        Assert.That(energy, Is.GreaterThan(1e-4), "audible");
    }

    // Opt-in (ACOUSTIC_PATH_REVERB_COST=1): the CPU reference's cost at the live Church's size, 48 kHz, two ears,
    // eight bands, a 6 s horizon, updates of 1,024 samples: the median time of one update's kernel build
    // (SetResponse) and of one block's convolution (Render), against the 21.3 ms an update lasts.
    [Test, Category("QualityMeasurement")]
    public void MeasurePathReverbCpuCost()
    {
        if (Environment.GetEnvironmentVariable("ACOUSTIC_PATH_REVERB_COST") != "1") Assert.Ignore("Set ACOUSTIC_PATH_REVERB_COST=1.");
        const int rate = 48000, step = 1024;
        var settings = SpectralSettings(); settings.method = TakeRender.Method.Path; settings.maximumDelaySeconds = 6;
        settings.responseStepSamples = step; settings.fftSize = 1024; settings.listenerOutput = TakeRender.ListenerOutput.Binaural;
        var bins = new PathBins(settings);
        var scene = Scene(); var state = States();
        double Histogram(int pair, int d, int b, int k) => Math.Exp(-bins.Start(k) / .5) * bins.Width(k) * (1 + .1 * ((k * 7 + d * 3 + b) % 5));
        var pcm = new float[4096]; for (int n = 0; n < pcm.Length; n++) pcm[n] = (float)Math.Sin(n * .05);
        var signal = new TakeRender.Signal(pcm); signal.SetActive(0, true);
        signal.Append(new AcousticTake.VoiceEvent {action = AcousticTake.VoiceAction.Play, gain = 1, loop = true});
        var reverb = new PathReverb(scene, rate, settings, new[] {signal});
        var block = new float[2 * step];
        var build = new double[12]; var render = new double[12];
        for (int u = 0; u < build.Length; u++)
        {
            var w = Stopwatch.StartNew(); reverb.SetResponse((long)u * step, u == 0, Histogram, state); build[u] = w.Elapsed.TotalMilliseconds;
            w.Restart(); reverb.Render((long)u * step, step, block); render[u] = w.Elapsed.TotalMilliseconds;
        }
        Array.Sort(build, 2, build.Length - 2); Array.Sort(render, 2, render.Length - 2);
        double b0 = build[2 + (build.Length - 2) / 2], r0 = render[2 + (render.Length - 2) / 2];
        UnityEngine.Debug.Log($"PathReverb CPU at 48 kHz, binaural, 8 bands, 6 s ({bins.Count} bins): kernel build {b0:F1} ms per update, convolution {r0:F1} ms per block of {step} (an update lasts {1000.0 * step / rate:F1} ms)");
    }

    // Opt-in (ACOUSTIC_PATH_BUDGET=/abs/job.json, a TakeBatch job of the Church): the live path method's GPU time per
    // update in the Church, as the playable Church runs it (persistent rays under population control, GPU audio,
    // binaural), with a fence after each phase: the tracer's update, then the audio (the denoiser if on, the path
    // kernels, the convolution, the readback of 1,024 samples), and the CPU time of the audio calls.
    // ACOUSTIC_PATH_BUDGET_VARIANTS: variants separated by ';', each "name:key=value,..." with keys population,
    // denoise (0/1), flags (MethodTransport.ProfileFlags), iterationCap, kernels (0: the path kernels built once and
    // then left, a profiling bound), spectra (0: every partition through the band FFTs); default "base:". All variants live in one
    // process and are measured interleaved, in rounds of 8 updates each (ACOUSTIC_PATH_BUDGET_ROUNDS, default 25),
    // after a horizon of warm-up updates each, so drift from other work on the machine falls on every variant
    // alike. Reported: median and the 10th-90th percentile range per phase, and the load average before and after.
    [Test, Category("QualityMeasurement"), Timeout(int.MaxValue)]
    public void MeasurePathLiveBudget()
    {
        string file = Environment.GetEnvironmentVariable("ACOUSTIC_PATH_BUDGET");
        if (string.IsNullOrEmpty(file)) Assert.Ignore("Set ACOUSTIC_PATH_BUDGET.");
        string spec = Environment.GetEnvironmentVariable("ACOUSTIC_PATH_BUDGET_VARIANTS") ?? "base:";
        int rounds = int.Parse(Environment.GetEnvironmentVariable("ACOUSTIC_PATH_BUDGET_ROUNDS") ?? "25"), perRound = 8;
        var shader = RequireGpu();
        var job = TakeBatch.ParseJob(File.ReadAllText(file));
        using var take = AcousticTake.Open(job.take);
        var scene = take.Metadata.scene; int rate = take.Metadata.sampleRate;
        var clips = new float[scene.clips.Length][]; for (int c = 0; c < clips.Length; c++) clips[c] = take.ReadClip(c);
        using var probe = new ComputeBuffer(1, 4);
        void Fence() => AsyncGPUReadback.Request(probe).WaitForCompletion();
        string Load() { try { var p = System.Diagnostics.Process.Start(new ProcessStartInfo("/usr/sbin/sysctl", "-n vm.loadavg") {RedirectStandardOutput = true, UseShellExecute = false}); string o = p.StandardOutput.ReadToEnd().Trim(); p.WaitForExit(); return o; } catch { return "?"; } }
        string loadBefore = Load();
        var variants = new System.Collections.Generic.List<(string name, GpuAudio gpu, TransportSolver solver, AcousticTake.State[] state, long cursor, System.Collections.Generic.List<double>[] times)>();
        try
        {
            foreach (string entry in spec.Split(';'))
            {
                string name = entry.Split(':')[0];
                var settings = JsonUtility.FromJson<TakeRender.Settings>(JsonUtility.ToJson(job.settings));
                settings.method = TakeRender.Method.Path; settings.audioBackend = TakeRender.AudioBackend.Gpu; settings.pathPersistent = true;
                settings.pathPopulation = 11000; settings.pathDenoise = false; settings.maximumResponsesPerBatch = 1;
                settings.pathHistogramExport = settings.pathHistogramImport = "";
                int flags = 0, kernels = 1, spectra = 1, convolution = 1;
                string options = entry.Contains(":") ? entry.Substring(entry.IndexOf(':') + 1) : "";
                foreach (string kv in options.Split(new[] {','}, StringSplitOptions.RemoveEmptyEntries))
                {
                    var p = kv.Split('='); int v = int.Parse(p[1]);
                    switch (p[0])
                    {
                        case "population": settings.pathPopulation = v; break;
                        case "denoise": settings.pathDenoise = v != 0; break;
                        case "flags": flags = v; break;
                        case "kernels": kernels = v; break;
                        case "spectra": spectra = v; break;
                        case "convolution": convolution = v; break;
                        case "edgeSlice": settings.pathEdgeSliceUpdates = v; break;
                        case "iterationCap": settings.pathIterationCap = v; break;
                        default: throw new ArgumentException("Unknown budget option " + p[0]);
                    }
                }
                var gpu = new GpuAudio(scene, rate, settings, TakeRender.Signal.FromTake(take, clips), 1, transported: true);
                var solver = new TransportSolver(scene, take.Geometry, settings, shader, 1, rate, false, methodAudio: true);
                solver.Method.ProfileFlags = flags;
                gpu.SkipPathKernels = kernels == 0; gpu.PathGroupSpectraOff = spectra == 0; gpu.SkipLateConvolution = convolution == 0;
                variants.Add((name, gpu, solver, new AcousticTake.State[scene.objects.Length], 0, new[] {new System.Collections.Generic.List<double>(), new System.Collections.Generic.List<double>(), new System.Collections.Generic.List<double>()}));
            }
            var pcm = new float[2048];
            void Update(int i, bool record)
            {
                var v = variants[i]; int step = 1024; long sample = v.cursor;
                take.Evaluate(Math.Min(7 * 44100L + sample % (8 * 44100L), take.Metadata.durationSamples), v.state);
                Fence();
                var w = Stopwatch.StartNew();
                v.solver.BeginSolve(v.state, 1, false); Fence();
                double t = w.Elapsed.TotalMilliseconds; w.Restart();
                v.gpu.SetResponse(sample, sample == 0, v.solver.RayBuffer, 0, scene, v.state, late: v.solver.Late(0));
                v.gpu.QueueRender(sample, step); double c = w.Elapsed.TotalMilliseconds;
                v.gpu.BeginReadback(); v.gpu.TryComplete(pcm, true); v.solver.CompleteAfterGpuAudio();
                double a = w.Elapsed.TotalMilliseconds;
                if (record) { v.times[0].Add(t); v.times[1].Add(a); v.times[2].Add(c); }
                variants[i] = (v.name, v.gpu, v.solver, v.state, sample + step, v.times);
            }
            for (int i = 0; i < variants.Count; i++) for (int u = 0; u < variants[i].solver.Method.WarmUpdates; u++) Update(i, false);
            for (int r = 0; r < rounds; r++) for (int i = 0; i < variants.Count; i++) for (int u = 0; u < perRound; u++) Update(i, true);
            string Stat(System.Collections.Generic.List<double> v) { var x = v.ToArray(); Array.Sort(x); return $"{x[x.Length / 2]:F2} [{x[x.Length / 10]:F2}-{x[x.Length * 9 / 10]:F2}]"; }
            string loadAfter = Load();
            var lengths = (float[])variants[0].solver.Method.EdgeLengths.Clone(); Array.Sort(lengths);
            if (lengths.Length > 0)
            {
                string Share(double l) { int n = 0; foreach (float x in lengths) if (x < l) n++; return $"{100.0 * n / lengths.Length:F1}%"; }
                UnityEngine.Debug.Log($"PATH_BUDGET edges: {lengths.Length}, length quantiles 10/50/90% {lengths[lengths.Length / 10]:F3}/{lengths[lengths.Length / 2]:F3}/{lengths[lengths.Length * 9 / 10]:F3} m; " +
                                      $"shorter than the wavelength at 16 kHz {Share(343.0 / 16000)}, 4 kHz {Share(343.0 / 4000)}, 1 kHz {Share(343.0 / 1000)}, 250 Hz {Share(343.0 / 250)}, 125 Hz {Share(343.0 / 125)}");
            }
            foreach (var v in variants)
                UnityEngine.Debug.Log($"PATH_BUDGET {v.name}: tracer {Stat(v.times[0])} ms, audio {Stat(v.times[1])} ms, CPU calls {Stat(v.times[2])} ms; " +
                                      $"{v.times[0].Count} updates, population {v.solver.Method.LastRaysInFlight}, load {loadBefore} -> {loadAfter}");
        }
        finally { foreach (var v in variants) { v.gpu.Dispose(); v.solver.Dispose(); } }
    }
}
