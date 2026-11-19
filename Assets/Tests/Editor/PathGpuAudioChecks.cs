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

    // Opt-in (ACOUSTIC_PATH_BUDGET=/abs/job.json, a TakeBatch job of the Church; ACOUSTIC_PATH_BUDGET_POPULATION, default
    // 11000): the live path method's GPU time per update in the Church, as the playable Church runs it (persistent
    // rays under population control, GPU audio, binaural), with a fence after each phase: the tracer's update, then
    // the audio (the denoiser if on, the path kernels, the convolution, the readback of 1,024 samples), and the CPU
    // time of the audio calls. Medians over 40 updates after a horizon of them, the listener on the take's walk.
    [Test, Category("QualityMeasurement"), Timeout(int.MaxValue)]
    public void MeasurePathLiveBudget()
    {
        string file = Environment.GetEnvironmentVariable("ACOUSTIC_PATH_BUDGET");
        if (string.IsNullOrEmpty(file)) Assert.Ignore("Set ACOUSTIC_PATH_BUDGET.");
        int population = int.Parse(Environment.GetEnvironmentVariable("ACOUSTIC_PATH_BUDGET_POPULATION") ?? "11000");
        var shader = RequireGpu();
        var job = TakeBatch.ParseJob(File.ReadAllText(file));
        using var take = AcousticTake.Open(job.take);
        var scene = take.Metadata.scene; int rate = take.Metadata.sampleRate;
        var clips = new float[scene.clips.Length][]; for (int c = 0; c < clips.Length; c++) clips[c] = take.ReadClip(c);
        using var probe = new ComputeBuffer(1, 4);
        void Fence() => AsyncGPUReadback.Request(probe).WaitForCompletion();
        foreach (bool denoise in new[] {false, true})
        {
            var settings = JsonUtility.FromJson<TakeRender.Settings>(JsonUtility.ToJson(job.settings));
            settings.method = TakeRender.Method.Path; settings.audioBackend = TakeRender.AudioBackend.Gpu; settings.pathPersistent = true;
            settings.pathPopulation = population; settings.pathDenoise = denoise; settings.maximumResponsesPerBatch = 1;
            settings.pathHistogramExport = settings.pathHistogramImport = "";
            var signals = TakeRender.Signal.FromTake(take, clips);
            using var gpu = new GpuAudio(scene, rate, settings, signals, 1, transported: true);
            using var solver = new TransportSolver(scene, take.Geometry, settings, shader, 1, rate, false, methodAudio: true);
            var state = new AcousticTake.State[scene.objects.Length]; var pcm = new float[settings.responseStepSamples * 2];
            int warm = solver.Method.WarmUpdates, measured = 40, step = settings.responseStepSamples;
            var trace = new double[measured]; var audio = new double[measured]; var cpu = new double[measured];
            for (int u = 0; u < warm + measured; u++)
            {
                long sample = (long)u * step;
                take.Evaluate(Math.Min(7 * 44100L + sample, take.Metadata.durationSamples), state);
                Fence();
                var w = Stopwatch.StartNew();
                solver.BeginSolve(state, 1, false); Fence();
                double t = w.Elapsed.TotalMilliseconds; w.Restart();
                gpu.SetResponse(sample, u == 0, solver.RayBuffer, 0, scene, state, late: solver.Late(0));
                gpu.QueueRender(sample, step); double c = w.Elapsed.TotalMilliseconds;
                gpu.BeginReadback(); gpu.TryComplete(pcm, true); solver.CompleteAfterGpuAudio();
                double a = w.Elapsed.TotalMilliseconds;
                if (u >= warm) { trace[u - warm] = t; audio[u - warm] = a; cpu[u - warm] = c; }
            }
            double Median(double[] v) { var x = (double[])v.Clone(); Array.Sort(x); return x[x.Length / 2]; }
            UnityEngine.Debug.Log($"PATH_BUDGET population {population}, denoiser {(denoise ? "on" : "off")}: tracer {Median(trace):F2} ms, audio {Median(audio):F2} ms " +
                                  $"(CPU calls {Median(cpu):F2} ms), {solver.Method.Rays} new rays per update, {take.Geometry.Length} triangles");
        }
    }
}
