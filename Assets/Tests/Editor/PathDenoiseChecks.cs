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
}
