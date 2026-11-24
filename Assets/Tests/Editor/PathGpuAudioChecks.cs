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
    // moving: the source passes at 20 m/s, so the reflections' echoes slide at their delay rates (README part 40).
    [TestCase(TakeRender.ListenerOutput.Binaural, false)]
    [TestCase(TakeRender.ListenerOutput.Mono, false)]
    [TestCase(TakeRender.ListenerOutput.Binaural, true)]
    [Category("GPU"), Category("Methods")]
    public void GpuPathReverbMatchesTheCpuReference(TakeRender.ListenerOutput ears, bool moving)
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
            if (moving) { state[0].velocity = new Vector3(20, 0, 0); state[0].flags |= AcousticTake.StateFlags.VelocityValid; state[0].position.x = -2 + 20 * frame * size / (float)rate; }
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
        TestContext.WriteLine($"{ears}{(moving ? ", moving source" : "")}: GPU against CPU peak {peak:G3}, relative {relative:G3}, output energy {energy:G3}");
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
    // denoise (0/1), flags (MethodTransport.ProfileFlags), iterationCap, edgeSlice, kernels (0: the path
    // kernels built once and then left, a profiling bound), spectra (0: every partition through the band FFTs), kernelSkip (GpuAudio.PathKernelSkip, a profiling bound), otherDenoiser (1: the weights in ACOUSTIC_PATH_BUDGET_DENOISER), spectralLimit (GpuAudio.PathSpectralGroupLimit), fftSingle (1: one FFT stage per pass),
    // budget (ms: the auto-budget, PathPopulationBudget, on the update's tracer and audio time, N within 4,000 and maximum, default 23,000),
    // load (a round: from it on every walk segment is traced twice, ProfileFlags bit 15, a slower GPU), profile (1: after the
    // rounds, the work profile of 16 updates, MethodTransport.WorkProfile), merged (0: the GPU queries descend the instance tree,
    // not StaticTraceTree's), inline (0: the merged tree's leaves index the primitives instead of storing their triangles), stages (1: a fence after each
    // stage of the tracer's update, PATH_STAGE lines with each stage's time), mergedWalk (0: the pairs walked one by one, MethodTransport.MergedWalk),
    // denoisePerPair (1: one denoiser per pair, each dispatched alone, GpuAudio.EnablePathDenoiser); default "base:".
    // All variants live in one process and are measured interleaved, in rounds of 8 updates each
    // (ACOUSTIC_PATH_BUDGET_ROUNDS, default 25), after a horizon of warm-up updates each, so drift from other work on
    // the machine falls on every variant alike. Reported: median and the 10th-90th percentile range per phase, and the
    // load average before and after; with a budget, also the population per round (ACOUSTIC_PATH_BUDGET_TRACE, a file:
    // per update the variant, round, population, tracer and audio ms).
    sealed class BudgetVariant
    {
        public string name; public GpuAudio gpu; public TransportSolver solver; public AcousticTake.State[] state; public long cursor;
        public readonly System.Collections.Generic.List<double>[] times = {new(), new(), new()};
        public readonly System.Collections.Generic.List<int> populations = new();
        public PathPopulationBudget budget; public int flags, loadRound = -1, kernelSkip, spectralLimit = -1; public bool fftSingle, profile, merged = true, inline = true, stages, mergedWalk = true;
        public readonly System.Collections.Generic.SortedDictionary<string, System.Collections.Generic.List<double>> stageTimes = new();
    }
    [Test, Category("QualityMeasurement"), Timeout(int.MaxValue)]
    public void MeasurePathLiveBudget()
    {
        string file = Environment.GetEnvironmentVariable("ACOUSTIC_PATH_BUDGET");
        if (string.IsNullOrEmpty(file)) Assert.Ignore("Set ACOUSTIC_PATH_BUDGET.");
        string spec = Environment.GetEnvironmentVariable("ACOUSTIC_PATH_BUDGET_VARIANTS") ?? "base:";
        string tracePath = Environment.GetEnvironmentVariable("ACOUSTIC_PATH_BUDGET_TRACE");
        int rounds = int.Parse(Environment.GetEnvironmentVariable("ACOUSTIC_PATH_BUDGET_ROUNDS") ?? "25"), perRound = 8;
        // The take's window the updates cycle through (ACOUSTIC_PATH_BUDGET_WINDOW: "start,length" in seconds, default 7,8).
        var window = (Environment.GetEnvironmentVariable("ACOUSTIC_PATH_BUDGET_WINDOW") ?? "7,8").Split(',');
        double windowStart = double.Parse(window[0], System.Globalization.CultureInfo.InvariantCulture), windowLength = double.Parse(window[1], System.Globalization.CultureInfo.InvariantCulture);
        var shader = RequireGpu();
        var job = TakeBatch.ParseJob(File.ReadAllText(file));
        using var take = AcousticTake.Open(job.take);
        var scene = take.Metadata.scene; int rate = take.Metadata.sampleRate;
        var clips = new float[scene.clips.Length][]; for (int c = 0; c < clips.Length; c++) clips[c] = take.ReadClip(c);
        using var probe = new ComputeBuffer(1, 4);
        void Fence() => AsyncGPUReadback.Request(probe).WaitForCompletion();
        string Load() { try { var p = System.Diagnostics.Process.Start(new ProcessStartInfo("/usr/sbin/sysctl", "-n vm.loadavg") {RedirectStandardOutput = true, UseShellExecute = false}); string o = p.StandardOutput.ReadToEnd().Trim(); p.WaitForExit(); return o; } catch { return "?"; } }
        string loadBefore = Load();
        var variants = new System.Collections.Generic.List<BudgetVariant>();
        var trace = new System.Text.StringBuilder("variant,round,population,tracerMs,audioMs\n");
        try
        {
            foreach (string entry in spec.Split(';'))
            {
                var v = new BudgetVariant {name = entry.Split(':')[0]};
                var settings = JsonUtility.FromJson<TakeRender.Settings>(JsonUtility.ToJson(job.settings));
                settings.method = TakeRender.Method.Path; settings.audioBackend = TakeRender.AudioBackend.Gpu; settings.pathPersistent = true;
                settings.pathPopulation = 11000; settings.pathDenoise = false; settings.maximumResponsesPerBatch = 1;
                settings.pathHistogramExport = settings.pathHistogramImport = "";
                int kernels = 1, spectra = 1, convolution = 1, maximum = 23000; double budget = 0; bool other = false, perPair = false;
                string options = entry.Contains(":") ? entry.Substring(entry.IndexOf(':') + 1) : "";
                foreach (string kv in options.Split(new[] {','}, StringSplitOptions.RemoveEmptyEntries))
                {
                    var p = kv.Split('=');
                    if (p[0] == "budget") { budget = double.Parse(p[1], System.Globalization.CultureInfo.InvariantCulture); continue; }
                    int value = int.Parse(p[1]);
                    switch (p[0])
                    {
                        case "population": settings.pathPopulation = value; break;
                        case "denoise": settings.pathDenoise = value != 0; break;
                        case "flags": v.flags = value; break;
                        case "kernels": kernels = value; break;
                        case "spectra": spectra = value; break;
                        case "convolution": convolution = value; break;
                        case "edgeSlice": settings.pathEdgeSliceUpdates = value; break;
                        case "iterationCap": settings.pathIterationCap = value; break;
                        case "load": v.loadRound = value; break;
                        case "kernelSkip": v.kernelSkip = value; break;
                        case "otherDenoiser": other = value != 0; break;
                        case "spectralLimit": v.spectralLimit = value; break;
                        case "fftSingle": v.fftSingle = value != 0; break;
                        case "profile": v.profile = value != 0; break;
                        case "maximum": maximum = value; break;
                        case "merged": v.merged = value != 0; break;
                        case "inline": v.inline = value != 0; break;
                        case "stages": v.stages = value != 0; break;
                        case "mergedWalk": v.mergedWalk = value != 0; break;
                        case "denoisePerPair": perPair = value != 0; break;
                        default: throw new ArgumentException("Unknown budget option " + p[0]);
                    }
                }
                if (budget > 0) { settings.pathPopulationMaximum = maximum; v.budget = new PathPopulationBudget(budget, settings.pathPopulation, 4000, maximum); }
                v.gpu = new GpuAudio(scene, rate, settings, TakeRender.Signal.FromTake(take, clips), 1, transported: true);
                v.solver = new TransportSolver(scene, take.Geometry, settings, shader, 1, rate, false, methodAudio: true);
                v.solver.Method.ProfileFlags = v.flags; v.solver.Method.MergedWalk = v.mergedWalk;
                v.solver.TraceTreeEnabled = v.merged; v.solver.TraceTreeInline = v.inline;
                if (perPair && settings.pathDenoise)
                {
                    var (bundled, header) = PathDenoiser.LoadBundled();
                    v.gpu.EnablePathDenoiser(Resources.Load<ComputeShader>(PathDenoiser.ShaderResource), bundled, header, perPair: true);
                }
                if (other)
                {
                    // Another denoiser's weights (ACOUSTIC_PATH_BUDGET_DENOISER: a folder with weights.bytes and header.json).
                    string folder = Environment.GetEnvironmentVariable("ACOUSTIC_PATH_BUDGET_DENOISER");
                    v.gpu.EnablePathDenoiser(Resources.Load<ComputeShader>(PathDenoiser.ShaderResource), File.ReadAllBytes(Path.Combine(folder, "weights.bytes")),
                                             File.ReadAllText(Path.Combine(folder, "header.json")));
                }
                v.gpu.PathKernelSkip = v.kernelSkip; v.gpu.PathSpectralGroupLimit = v.spectralLimit; v.gpu.FftSingleStages = v.fftSingle; v.gpu.SkipPathKernels = kernels == 0; v.gpu.PathGroupSpectraOff = spectra == 0; v.gpu.SkipLateConvolution = convolution == 0;
                v.state = new AcousticTake.State[scene.objects.Length];
                variants.Add(v);
            }
            var pcm = new float[2048];
            void Update(BudgetVariant v, int round)
            {
                int step = 1024; long sample = v.cursor;
                if (v.loadRound >= 0) v.solver.Method.ProfileFlags = round >= v.loadRound ? v.flags | 32768 : v.flags;
                take.Evaluate(Math.Min((long)(windowStart * rate) + sample % (long)(windowLength * rate), take.Metadata.durationSamples), v.state);
                Fence();
                var w = Stopwatch.StartNew();
                if (v.stages)
                {
                    // stages=1: a fence after each stage of the tracer's update (MethodTransport.StageHook), the time since the
                    // previous one attributed to it; "fence" is an empty fence's round trip, which every stage's time includes.
                    double mark = 0;
                    void Add(string name, double ms) { if (round < 0) return; if (!v.stageTimes.TryGetValue(name, out var list)) v.stageTimes[name] = list = new(); list.Add(ms); }
                    v.solver.Method.StageHook = name => { Fence(); double now = w.Elapsed.TotalMilliseconds; Add(name, now - mark); mark = now; };
                    v.solver.BeginSolve(v.state, 1, false); Fence();
                    Add("tail", w.Elapsed.TotalMilliseconds - mark);
                    var f = Stopwatch.StartNew(); Fence(); Add("fence", f.Elapsed.TotalMilliseconds);
                }
                else { v.solver.BeginSolve(v.state, 1, false); Fence(); }
                double t = w.Elapsed.TotalMilliseconds; w.Restart();
                v.gpu.SetResponse(sample, sample == 0, v.solver.RayBuffer, 0, scene, v.state, late: v.solver.Late(0));
                v.gpu.QueueRender(sample, step); double c = w.Elapsed.TotalMilliseconds;
                v.gpu.BeginReadback(); v.gpu.TryComplete(pcm, true); v.solver.CompleteAfterGpuAudio();
                double a = w.Elapsed.TotalMilliseconds;
                int population = v.solver.Method.Population;
                if (round >= 0)
                {
                    v.times[0].Add(t); v.times[1].Add(a); v.times[2].Add(c); v.populations.Add(population);
                    trace.Append(FormattableString.Invariant($"{v.name},{round},{population},{t:F3},{a:F3}\n"));
                }
                if (v.budget != null) v.solver.Method.Population = v.budget.Observe(t + a);
                v.cursor = sample + step;
            }
            foreach (var v in variants) for (int u = 0; u < v.solver.Method.WarmUpdates; u++) Update(v, -1);
            for (int r = 0; r < rounds; r++) foreach (var v in variants) for (int u = 0; u < perRound; u++) Update(v, r);
            // The work profile (profile=1, README part 46): after the timed rounds, 16 updates of the walk's counting kernel.
            foreach (var v in variants)
            {
                if (!v.profile) continue;
                var totals = new MethodTransport.WorkProfileTotals();
                v.solver.Method.WorkProfile = true;
                for (int u = 0; u < 16; u++) { Update(v, -1); v.solver.Method.AddWorkProfile(totals); }
                v.solver.Method.WorkProfile = false;
                UnityEngine.Debug.Log($"PATH_WORK {v.name}: " + MethodTransport.WorkProfileReport(totals));
                var m = v.solver.Method;
                string dump = Environment.GetEnvironmentVariable("ACOUSTIC_PATH_BUDGET_DUMP");
                if (!string.IsNullOrEmpty(dump))
                {
                    // For an external ceiling benchmark (README part 46): world-space triangles at the current state, then the rays the
                    // latest walk kept (position, direction), as little-endian float32 with counts first.
                    var rays = m.ReadWalkPool(out int rayCount);
                    using var w = new BinaryWriter(File.Create(dump));
                    w.Write(take.Geometry.Length);
                    foreach (var g in take.Geometry) { var pose = v.state[g.objectIndex]; foreach (var x in new[] {g.a, g.b, g.c}) { var y = pose.position + pose.rotation * x; w.Write(y.x); w.Write(y.y); w.Write(y.z); } }
                    w.Write(rayCount);
                    for (int r = 0; r < rayCount; r++) for (int k = 0; k < 7; k++) if (k != 3) w.Write(rays[r * MethodTransport.PathRayFloats + k]);
                }
                UnityEngine.Debug.Log(FormattableString.Invariant($"PATH_CEILING {v.name}: closest hit, fresh directions {m.TraceBench(1 << 20, 0) / 1e6:F0} M/s; closest hit, own directions {m.TraceBench(1 << 20, 1) / 1e6:F0} M/s; any hit to the listener {m.TraceBench(1 << 20, 2) / 1e6:F0} M/s"));
            }
            string Stat(System.Collections.Generic.List<double> v) { var x = v.ToArray(); Array.Sort(x); return $"{x[x.Length / 2]:F2} [{x[x.Length / 10]:F2}-{x[x.Length * 9 / 10]:F2}]"; }
            string loadAfter = Load();
            foreach (var v in variants)
            {
                var n = v.populations.ToArray(); Array.Sort(n);
                UnityEngine.Debug.Log($"PATH_BUDGET {v.name}: tracer {Stat(v.times[0])} ms, audio {Stat(v.times[1])} ms, CPU calls {Stat(v.times[2])} ms; " +
                                      $"{v.times[0].Count} updates, population {n[n.Length / 2]} [{n[n.Length / 10]}-{n[n.Length * 9 / 10]}], in flight {v.solver.Method.RaysInFlight()}, trace-tree rebuilds {v.solver.TraceTreeRebuilds}, overflowed {v.solver.Method.RaysOverflowed()}, load {loadBefore} -> {loadAfter}");
            }
            foreach (var v in variants)
                foreach (var stage in v.stageTimes)
                {
                    var x = stage.Value.ToArray(); Array.Sort(x); double sum = 0; foreach (double y in x) sum += y;
                    UnityEngine.Debug.Log(FormattableString.Invariant($"PATH_STAGE {v.name} {stage.Key}: {x.Length / v.times[0].Count} per update, median {x[x.Length / 2]:F3} ms, mean {sum / x.Length:F3} ms, per update {sum / v.times[0].Count:F3} ms"));
                }
            if (!string.IsNullOrEmpty(tracePath)) File.WriteAllText(tracePath, trace.ToString());
            UnityEngine.Debug.Log($"PATH_BUDGET impulse groups per partition (count:partitions): {variants[0].gpu.PathPartitionGroups}");
        }
        finally { foreach (var v in variants) { v.gpu?.Dispose(); v.solver?.Dispose(); } }
    }
}
