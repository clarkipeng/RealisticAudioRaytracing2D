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
        double worst = 0;
        for (int u = 0; u < updates; u++)
        {
            denoiser.Push(f.Skip(histograms + u * cells).Take(cells).ToArray(), f.Skip(counts + u * bins).Take(bins).ToArray());
            if (u < denoiser.K - 1) continue;
            denoiser.Output.GetData(output);
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
        UnityEngine.Debug.Log($"Denoiser kernel against the model: worst relative error {worst:G3} over {updates - denoiser.K + 1} updates of {bins} bins");
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
}
