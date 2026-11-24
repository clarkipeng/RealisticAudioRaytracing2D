using System;
using System.Diagnostics;
using System.IO;
using System.Linq;
using NUnit.Framework;
using UnityEditor;
using UnityEngine;
using UnityEngine.Rendering;

// The persistent-ray denoiser's compute kernels (PathDenoiser, PathDenoise.compute) against the trained model
// in Python (tools/path_persistent_denoise.py export writes the weights and this fixture), and its GPU time.
public partial class TakeChecks
{
    static (ComputeShader shader, byte[] weights, string header) RequireDenoiser()
    {
        RequireGpu();
        var shader = AssetDatabase.LoadAssetAtPath<ComputeShader>("Assets/Resources/Acoustics/PathDenoise.compute");
        Assert.That(shader, Is.Not.Null, "PathDenoise.compute must import successfully.");
        var (weights, header) = PathDenoiser.LoadBundled();
        Assert.That(weights, Is.Not.Null, "Resources/Acoustics/PathDenoise holds the exported weights.");
        return (shader, weights, header);
    }

    [Test]
    public void PathDenoiserKernelMatchesTheModel()
    {
        var (shader, weights, header) = RequireDenoiser();
        var blob = File.ReadAllBytes("Assets/Tests/Editor/PathDenoiseFixture.bytes");
        var f = new float[blob.Length / 4]; Buffer.BlockCopy(blob, 0, f, 0, blob.Length);
        int updates = (int)f[0], bins = (int)f[1], at = 2, cells = PathDenoiser.Cells * bins;
        var starts = f.Skip(at).Take(bins).Select(v => (double)v).ToArray(); at += bins;
        int histograms = at; at += updates * cells;
        int counts = at; at += updates * bins;
        using var denoiser = new PathDenoiser(shader, weights, header, starts);
        Assert.That(f.Length, Is.EqualTo(at + (updates - denoiser.K + 1) * cells), "fixture layout");
        var output = new float[cells];
        // outputHash: FNV-1a of every output's bits, logged; a kernel change meant to be exact must keep it.
        double worst = 0; ulong outputHash = 14695981039346656037UL;
        for (int u = 0; u < updates; u++)
        {
            denoiser.Push(f.Skip(histograms + u * cells).Take(cells).ToArray(), f.Skip(counts + u * bins).Take(bins).ToArray());
            if (u < denoiser.K - 1) continue;
            denoiser.Output.GetData(output);
            foreach (var v in output) { outputHash ^= (ulong)BitConverter.SingleToInt32Bits(v) & 0xffffffffUL; outputHash *= 1099511628211UL; }
            int expected = at + (u - denoiser.K + 1) * cells;
            double peak = 0;
            for (int i = 0; i < cells; i++) peak = Math.Max(peak, f[expected + i]);
            Assert.That(peak, Is.GreaterThan(0), "the fixture has energy");
            for (int i = 0; i < cells; i++)
            {
                double e = f[expected + i], error = Math.Abs(output[i] - e) / (1e-4 * peak + Math.Abs(e));
                worst = Math.Max(worst, error);
                Assert.That(error, Is.LessThan(2e-3), $"update {u}, cell {i / bins} bin {i % bins}: kernel {output[i]:G6}, model {e:G6}");
            }
        }
        UnityEngine.Debug.Log($"Denoiser kernel against the model: worst relative error {worst:G3} over {updates - denoiser.K + 1} updates of {bins} bins; output bits hash {outputHash:X16}");
    }

    // GPU time per update at the Church's bin count (6 s horizon) and a generated scene's (2 s): the update
    // dispatched and fenced, minus an empty fence; the median of 50 after 10 warm-up updates.
    [Test]
    public void PathDenoiserFitsTheUpdateBudget()
    {
        var (shader, weights, header) = RequireDenoiser();
        foreach (float horizon in new[] {2f, 6f})
        {
            var settings = new TakeRender.Settings {method = TakeRender.Method.Path, maximumDelaySeconds = horizon};
            var bins = new PathBins(settings);
            var starts = Enumerable.Range(0, bins.Count).Select(k => bins.Start(k)).ToArray();
            using var denoiser = new PathDenoiser(shader, weights, header, starts);
            var rng = new System.Random(1);
            var h = Enumerable.Range(0, PathDenoiser.Cells * bins.Count).Select(_ => (float)rng.NextDouble()).ToArray();
            var c = Enumerable.Range(0, bins.Count).Select(_ => (float)rng.Next(100)).ToArray();
            using var probe = new ComputeBuffer(1, 4);
            double Fence() { var w = Stopwatch.StartNew(); AsyncGPUReadback.Request(probe).WaitForCompletion(); return w.Elapsed.TotalMilliseconds; }
            var times = new double[60]; var empty = new double[60];
            for (int i = 0; i < times.Length; i++)
            {
                empty[i] = Fence();
                var w = Stopwatch.StartNew();
                denoiser.Push(h, c);
                AsyncGPUReadback.Request(denoiser.Output, 4, 0).WaitForCompletion();
                times[i] = w.Elapsed.TotalMilliseconds;
            }
            double Median(double[] v) { var s = v.Skip(10).OrderBy(x => x).ToArray(); return s[s.Length / 2]; }
            double ms = Median(times) - Median(empty);
            UnityEngine.Debug.Log($"Denoiser at {bins.Count} bins ({horizon} s horizon): {ms:F2} ms per update (dispatch and GPU; empty fence {Median(empty):F2} ms)");
            Assert.That(ms, Is.LessThan(5), "the denoiser must fit beside the tracer in a 23 ms update");
        }
    }

    // Opt-in (ACOUSTIC_DENOISER_STEPS=1): GPU time of each denoiser step at the Church's 447 bins, a fence after each
    // (medians of 60 updates after 10).
    [Test, Category("QualityMeasurement")]
    public void MeasurePathDenoiserSteps()
    {
        if (Environment.GetEnvironmentVariable("ACOUSTIC_DENOISER_STEPS") != "1") Assert.Ignore("Set ACOUSTIC_DENOISER_STEPS=1.");
        var (shader, weights, header) = RequireDenoiser();
        var bins = new PathBins(new TakeRender.Settings {method = TakeRender.Method.Path, maximumDelaySeconds = 6});
        var starts = Enumerable.Range(0, bins.Count).Select(k => bins.Start(k)).ToArray();
        using var denoiser = new PathDenoiser(shader, weights, header, starts);
        var rng = new System.Random(1);
        var h = Enumerable.Range(0, PathDenoiser.Cells * bins.Count).Select(_ => (float)rng.NextDouble()).ToArray();
        var c = Enumerable.Range(0, bins.Count).Select(_ => (float)rng.Next(100)).ToArray();
        using var probe = new ComputeBuffer(1, 4);
        var times = new System.Collections.Generic.Dictionary<string, System.Collections.Generic.List<double>>();
        var order = new System.Collections.Generic.List<string>();
        var w = new Stopwatch();
        int iteration = 0;
        denoiser.StepFence = step =>
        {
            AsyncGPUReadback.Request(probe).WaitForCompletion();
            if (iteration >= 10) { if (!times.ContainsKey(step)) { times[step] = new System.Collections.Generic.List<double>(); order.Add(step); } times[step].Add(w.Elapsed.TotalMilliseconds); }
            w.Restart();
        };
        for (; iteration < 70; iteration++) { AsyncGPUReadback.Request(probe).WaitForCompletion(); w.Restart(); denoiser.Push(h, c); }
        double total = 0; var line = new System.Text.StringBuilder();
        foreach (var step in order) { var x = times[step].OrderBy(v => v).ToArray(); double m = x[x.Length / 2]; total += m; line.Append($"{step} {m:F3}, "); }
        UnityEngine.Debug.Log($"Denoiser steps (ms, fenced medians; each includes one fence's round trip): {line}total {total:F2}");
    }

    // One denoiser serving three pairs (every dispatch running all three, README part 57) against three denoisers of one
    // pair each, on the fixture's updates (pair 1 scaled by 3.7, pair 2 fed the updates in reverse order): every output
    // bit for bit the same, after a reset too.
    [Test]
    public void PathDenoiserPairsMatchOnePairEach()
    {
        var (shader, weights, header) = RequireDenoiser();
        var blob = File.ReadAllBytes("Assets/Tests/Editor/PathDenoiseFixture.bytes");
        var f = new float[blob.Length / 4]; Buffer.BlockCopy(blob, 0, f, 0, blob.Length);
        int updates = (int)f[0], bins = (int)f[1], at = 2, cells = PathDenoiser.Cells * bins;
        var starts = f.Skip(at).Take(bins).Select(v => (double)v).ToArray(); at += bins;
        int histograms = at; at += updates * cells;
        int counts = at;
        const int pairs = 3;
        using var joint = new PathDenoiser(shader, weights, header, starts, pairs: pairs);
        var single = new PathDenoiser[pairs];
        try
        {
            for (int p = 0; p < pairs; p++) single[p] = new PathDenoiser(shader, weights, header, starts);
            float[] Histogram(int p, int u)
            {
                int v = p == 2 ? updates - 1 - u : u;
                return f.Skip(histograms + v * cells).Take(cells).Select(x => p == 1 ? x * 3.7f : x).ToArray();
            }
            float[] Counts(int p, int u) => f.Skip(counts + (p == 2 ? updates - 1 - u : u) * bins).Take(bins).ToArray();
            var all = new float[pairs * cells]; var one = new float[cells];
            int compared = 0;
            for (int pass = 0; pass < 2; pass++)
            {
                if (pass == 1) { joint.Reset(); foreach (var d in single) d.Reset(); }
                for (int u = 0; u < updates; u++)
                {
                    var h = new float[pairs * cells]; var c = new float[pairs * bins];
                    for (int p = 0; p < pairs; p++)
                    {
                        Array.Copy(Histogram(p, u), 0, h, p * cells, cells); Array.Copy(Counts(p, u), 0, c, p * bins, bins);
                        single[p].Push(Histogram(p, u), Counts(p, u));
                    }
                    joint.Push(h, c);
                    joint.Output.GetData(all);
                    for (int p = 0; p < pairs; p++)
                    {
                        single[p].Output.GetData(one);
                        for (int i = 0; i < cells; i++)
                            Assert.That(BitConverter.SingleToInt32Bits(all[p * cells + i]), Is.EqualTo(BitConverter.SingleToInt32Bits(one[i])), $"pass {pass}, update {u}, pair {p}, cell {i}");
                        compared += cells;
                    }
                }
            }
            Assert.That(all.Any(v => v > 0), "the outputs have energy");
            UnityEngine.Debug.Log($"Denoiser of {pairs} pairs against one denoiser per pair: {compared} outputs, all bit-identical");
        }
        finally { foreach (var d in single) d?.Dispose(); }
    }

    // The live path audio of two sources and two listeners (four pairs) in the aperture room, binaural, a source moving and
    // a head turning, rendered update by update through GpuAudio as the live renderer does (the method's output read on the
    // GPU, no CPU readback): a hash of every PCM sample. configure adjusts the GpuAudio before the first update.
    string LivePathAudioHash(Action<GpuAudio> configure, bool denoise = true, int updates = 16)
    {
        var shader = RequireGpu();
        const int rate = 8000, size = 256;
        ApertureRoom(true, false, ApertureOutside, out var scene, out var geometry, out var state, transmission: .05f, roomScattering: .5f);
        int source2 = scene.objects.Length, listener2 = source2 + 1;
        var objects = new System.Collections.Generic.List<AcousticTake.ObjectDefinition>(scene.objects)
        {
            new AcousticTake.ObjectDefinition {id = "source-2", name = "Source 2", scale = Vector3.one},
            new AcousticTake.ObjectDefinition {id = "listener-2", name = "Listener 2", scale = Vector3.one}
        };
        scene.objects = objects.ToArray();
        scene.sources = new[] {scene.sources[0], new AcousticTake.SourceDefinition {objectIndex = source2, clipIndex = 0}};
        scene.listeners = new[] {1, listener2};
        var states = new System.Collections.Generic.List<AcousticTake.State>(state)
        {
            new AcousticTake.State {position = new Vector3(1.5f, 1, -1), rotation = Quaternion.identity, flags = AcousticTake.StateFlags.Active | AcousticTake.StateFlags.Source},
            new AcousticTake.State {position = new Vector3(-.5f, -1, 1), rotation = Quaternion.identity, flags = AcousticTake.StateFlags.Active | AcousticTake.StateFlags.Listener}
        };
        var posed = states.ToArray();
        var settings = ControlSettings(TakeRender.Method.Path, 4096, .3f); settings.diffractionFactor = 1;
        settings.transport = TakeRender.Transport.Raytraced; settings.synthesis = TakeRender.Synthesis.Spectral; settings.listenerOutput = TakeRender.ListenerOutput.Binaural;
        settings.audioBackend = TakeRender.AudioBackend.Gpu; settings.pathPersistent = true; settings.pathPopulation = 8000; settings.pathDenoise = denoise;
        settings.responseStepSamples = 4 * size; settings.fftSize = size; settings.spectralHop = size / 4; settings.transitionSamples = size / 4; settings.maximumResponsesPerBatch = 1;
        var clip = new float[331]; for (int n = 0; n < clip.Length; n++) clip[n] = .2f * Mathf.Sin(n * .37f);
        var signals = new TakeRender.Signal[2];
        for (int s = 0; s < 2; s++) { signals[s] = new TakeRender.Signal(clip); signals[s].SetActive(0, true); signals[s].Append(new AcousticTake.VoiceEvent {gain = 1, loop = true}); }
        using var solver = new TransportSolver(scene, geometry, settings, shader, 1, rate, false, methodAudio: true);
        using var gpu = new GpuAudio(scene, rate, settings, signals, 1, transported: true);
        configure?.Invoke(gpu);
        using var sha = System.Security.Cryptography.SHA256.Create();
        int step = settings.responseStepSamples; var pcm = new float[step * 2 * 2]; double energy = 0;
        for (int u = 0; u < updates; u++)
        {
            long sample = (long)u * step; float t = sample / (float)rate;
            posed[0].velocity = new Vector3(20, 0, 5); posed[0].flags |= AcousticTake.StateFlags.VelocityValid;
            posed[0].position = states[0].position + posed[0].velocity * t;
            posed[1].rotation = Quaternion.Euler(0, u * 19, 0); posed[1].angularVelocity = Vector3.up; posed[1].flags |= AcousticTake.StateFlags.VelocityValid;
            solver.BeginSolve(posed, 1, false);
            gpu.SetResponse(sample, sample == 0, solver.RayBuffer, 0, scene, posed, late: solver.Late(0));
            gpu.QueueRender(sample, step); gpu.BeginReadback(); gpu.TryComplete(pcm, true); solver.CompleteAfterGpuAudio();
            foreach (float v in pcm) { Assert.That(float.IsFinite(v)); energy += v * v; }
            var bytes = new byte[pcm.Length * 4]; Buffer.BlockCopy(pcm, 0, bytes, 0, bytes.Length);
            sha.TransformBlock(bytes, 0, bytes.Length, null, 0);
        }
        Assert.That(energy, Is.GreaterThan(1e-6), "audible");
        sha.TransformFinalBlock(Array.Empty<byte>(), 0, 0);
        return BitConverter.ToString(sha.Hash).Replace("-", "");
    }

    // The pair-batched denoiser in the live audio (README part 57): four pairs rendered with one denoiser of all of them and
    // with one denoiser per pair, reading the method's fixed-point output on the GPU, every PCM sample bit for bit the same.
    [Test, Category("GPU"), Category("Methods"), Timeout(900000)]
    public void PathDenoiserPairsRenderTheSameAudio()
    {
        string joint = LivePathAudioHash(null), apart = LivePathAudioHash(gpu =>
        {
            var (weights, header) = PathDenoiser.LoadBundled();
            gpu.EnablePathDenoiser(Resources.Load<ComputeShader>(PathDenoiser.ShaderResource), weights, header, perPair: true);
        });
        string plain = LivePathAudioHash(null, denoise: false);
        TestContext.WriteLine($"one denoiser {joint}, one per pair {apart}, no denoiser {plain}");
        Assert.That(apart, Is.EqualTo(joint));
        Assert.That(plain, Is.Not.EqualTo(joint), "the denoiser changes the audio");
    }
}
