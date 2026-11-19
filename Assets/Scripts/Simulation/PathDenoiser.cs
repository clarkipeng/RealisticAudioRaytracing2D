using System;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Rendering;

// The persistent-ray denoiser on the GPU (Assets/Resources/Acoustics/PathDenoise.compute; trained by
// tools/path_persistent_denoise.py, docs/benchmarks/path-denoise-2026-11-12 part 25). One instance per
// source-listener pair: every update pushes that pair's histogram [direction 6][band 8][bin] (straight from
// the method's fixed-point output, or from floats in tests), and Output holds the denoised histogram in
// the same layout, energy units. It keeps the last K updates and two moving averages, so it must see every
// update of its pair, in order (Reset after a flush). Work is recorded into a command buffer, so it stays
// ordered with the tracer before it and the audio after it (GpuAudio.Path).
public sealed class PathDenoiser : IDisposable
{
    [Serializable] sealed class Layer { public int input, output, taps, dilation; }
    [Serializable] sealed class Header { public int K, channels, floats; public float[] decays; public Layer[] layers; }

    public const int Cells = 48;
    public const string Resource = "Acoustics/PathDenoise", ShaderResource = "Acoustics/PathDenoise", ProfilerSample = "Acoustics.GPU.PathDenoise";
    public readonly int Bins, K;
    public int Updates { get; private set; }
    public ComputeBuffer Output => outBuffer;
    public int OutputBase => outBase;

    readonly ComputeShader shader;
    readonly Header header;
    readonly ComputeBuffer weights, segment, current, currentCounts, history, countHistory, ema, sums, x, ms, norm, smoothed, binMean, den, outBuffer, scratchA, scratchB, raw;
    readonly int[] weightBase, biasBase;
    readonly int outBase;
    readonly bool ownsOutput;
    CommandBuffer cb;
    // Profiling only (MeasurePathDenoiserSteps): called after each step with the work so far executed.
    internal Action<string> StepFence;
    void Mark(string step) { if (StepFence == null) return; Graphics.ExecuteCommandBuffer(cb); cb.Clear(); StepFence(step); }
    readonly int convert, ingest, scale, features, smooth, cellMean, normaliser, convolve, predict, totals, conserve;
    readonly float[] decays;
    static readonly double[] SegmentStarts = {.05, .2};   // tools/path_denoise.py SEGMENTS

    // The bundled weights (Resources/Acoustics/PathDenoise), or null when they are missing.
    public static (byte[] weights, string header) LoadBundled()
    {
        var w = Resources.Load<TextAsset>(Resource + "/weights"); var h = Resources.Load<TextAsset>(Resource + "/header");
        return w == null || h == null ? (null, null) : (w.bytes, h.text);
    }

    // output: an external buffer to write into at `outputBase` (GpuAudio's histogram of all pairs), or null.
    public PathDenoiser(ComputeShader shader, byte[] weightBytes, string headerJson, IReadOnlyList<double> binStarts, ComputeBuffer output = null, int outputBase = 0)
    {
        this.shader = shader;
        header = JsonUtility.FromJson<Header>(headerJson);
        if (header.channels != Cells || weightBytes.Length != 4 * header.floats) throw new ArgumentException("Denoiser weights do not match their header.");
        K = header.K; Bins = binStarts.Count; decays = header.decays;
        if (decays.Length != 2) throw new ArgumentException("The denoiser kernel expects two moving averages.");
        var w = new float[header.floats]; Buffer.BlockCopy(weightBytes, 0, w, 0, weightBytes.Length);
        weights = new ComputeBuffer(w.Length, 4); weights.SetData(w);
        weightBase = new int[header.layers.Length]; biasBase = new int[header.layers.Length];
        int at = 0, widest = 0;
        for (int l = 0; l < header.layers.Length; l++)
        {
            var layer = header.layers[l];
            weightBase[l] = at; at += layer.output * layer.input * layer.taps;
            biasBase[l] = at; at += layer.output;
            widest = Math.Max(widest, layer.output);
        }
        if (at != header.floats || header.layers[^1].output != 4 * Cells + 8 * 3) throw new ArgumentException("Denoiser layers do not match the kernel.");
        var seg = new uint[Bins];
        for (int k = 0; k < Bins; k++) { uint s = 0; foreach (double t in SegmentStarts) if (binStarts[k] >= t) s++; seg[k] = s; }
        segment = new ComputeBuffer(Bins, 4); segment.SetData(seg);
        ComputeBuffer Floats(long n) => new ComputeBuffer((int)Math.Max(1, n), 4);
        current = Floats(Cells * Bins); currentCounts = Floats(Bins);
        history = Floats(K * Cells * Bins); countHistory = Floats(K * Bins); ema = Floats(2 * Cells * Bins);
        sums = Floats(2 + 2 * 8 * 3);
        x = Floats(header.layers[0].input * Bins); ms = Floats(3 * Cells * Bins);
        norm = Floats(Cells * Bins); smoothed = Floats(Cells * Bins); binMean = Floats(Bins);
        den = Floats(Cells * Bins); outBuffer = output ?? Floats(Cells * Bins); ownsOutput = output == null; outBase = outputBase;
        scratchA = Floats(widest * Bins); scratchB = Floats(widest * Bins); raw = Floats((4 * Cells + 8 * 3) * Bins);
        convert = shader.FindKernel("Convert"); ingest = shader.FindKernel("Ingest"); scale = shader.FindKernel("Scale");
        features = shader.FindKernel("Features"); smooth = shader.FindKernel("Smooth"); cellMean = shader.FindKernel("CellMean");
        normaliser = shader.FindKernel("Normaliser"); convolve = shader.FindKernel("Convolve"); predict = shader.FindKernel("Predict");
        totals = shader.FindKernel("Totals"); conserve = shader.FindKernel("Conserve");
        Reset();
    }

    public void Reset()
    {
        ema.SetData(new float[2 * Cells * Bins]);
        Updates = 0;
    }

    // Records one update from the method's output buffer: cells fixedBase.. (two uints each, / rays) and counts countBase..
    public void Record(CommandBuffer commands, ComputeBuffer fixedOutput, int fixedBase, ComputeBuffer fixedCounts, int countBase, float rays)
    {
        cb = commands;
        Common();
        Bind(convert, ("Fixed", fixedOutput), ("FixedCounts", fixedCounts), ("Current", current), ("CurrentCounts", currentCounts));
        Int("FixedBase", fixedBase); Int("CountBase", countBase); cb.SetComputeFloatParam(shader, "Rays", rays);
        cb.DispatchCompute(shader, convert, Groups(Cells * Bins), 1, 1);
        Run();
    }

    // One update from the CPU (tests): histogram [48][B] in energy units and counts [B], executed now.
    public void Push(float[] histogram, float[] counts)
    {
        current.SetData(histogram); currentCounts.SetData(counts);
        using var commands = new CommandBuffer {name = "Path denoiser"};
        cb = commands;
        Common();
        Run();
        Graphics.ExecuteCommandBuffer(commands);
    }

    void Common()
    {
        Int("Bins", Bins); Int("K", K); Int("OutBase", outBase);
        cb.SetComputeFloatParam(shader, "Decay0", decays[0]); cb.SetComputeFloatParam(shader, "Decay1", decays[1]);
    }

    void Int(string name, int value) => cb.SetComputeIntParam(shader, name, value);

    static int Groups(int n) => (n + 63) / 64;

    void Bind(int kernel, params (string name, ComputeBuffer buffer)[] buffers)
    {
        foreach (var (name, buffer) in buffers) cb.SetComputeBufferParam(shader, kernel, name, buffer);
    }

    void Run()
    {
        if (StepFence == null) cb.BeginSample(ProfilerSample);
        int cells = Cells * Bins;
        int slot = Updates % K;
        Updates++;
        cb.SetComputeFloatParam(shader, "Weight0", 1 - Mathf.Pow(decays[0], Updates)); cb.SetComputeFloatParam(shader, "Weight1", 1 - Mathf.Pow(decays[1], Updates));
        Bind(ingest, ("Current", current), ("CurrentCounts", currentCounts), ("History", history), ("CountHistory", countHistory), ("Ema", ema));
        // The first update also fills the other slots (its history is "the same update K times"), averaging it once.
        for (int s = 0; s < K; s++)
        {
            if (s != slot && Updates > 1) continue;
            Int("Slot", s); Int("UpdateAverages", s == slot ? 1 : 0);
            cb.DispatchCompute(shader, ingest, Groups(cells), 1, 1);
        }
        Mark("ingest");
        Int("Slot", slot);
        Bind(scale, ("History", history), ("Ema", ema), ("Sums", sums));
        cb.DispatchCompute(shader, scale, 1, 1, 1);
        Mark("scale");
        Bind(features, ("History", history), ("CountHistory", countHistory), ("Ema", ema), ("Sums", sums), ("X", x), ("Ms", ms), ("Norm", norm));
        cb.DispatchCompute(shader, features, Groups(cells), 1, 1);
        Mark("features");
        Bind(smooth, ("Norm", norm), ("Smoothed", smoothed));
        cb.DispatchCompute(shader, smooth, Groups(cells), 1, 1);
        Mark("smooth");
        Bind(cellMean, ("Smoothed", smoothed), ("Norm", norm), ("BinMean", binMean), ("Sums", sums));
        cb.DispatchCompute(shader, cellMean, Math.Max(1, (Bins + 255) / 256), 1, 1);
        Mark("cellMean");
        Bind(normaliser, ("Smoothed", smoothed), ("BinMean", binMean), ("Sums", sums), ("Norm", norm));
        cb.DispatchCompute(shader, normaliser, Groups(cells), 1, 1);
        Mark("normaliser");
        var input = x;
        for (int l = 0; l < header.layers.Length; l++)
        {
            var layer = header.layers[l];
            bool last = l == header.layers.Length - 1;
            var output = last ? raw : (l % 2 == 0 ? scratchA : scratchB);
            Int("InChannels", layer.input); Int("OutChannels", layer.output); Int("Taps", layer.taps);
            Int("Dilation", layer.dilation); Int("WeightBase", weightBase[l]); Int("BiasBase", biasBase[l]);
            Int("Gelu", last ? 0 : 1);
            Bind(convolve, ("Weights", weights), ("Input", input), ("Output", output));
            cb.DispatchCompute(shader, convolve, Groups(Bins), layer.output, 1);
            input = output;
            Mark("conv" + l);
        }
        Bind(predict, ("Raw", raw), ("Ms", ms), ("Norm", norm), ("Den", den));
        cb.DispatchCompute(shader, predict, Groups(cells), 1, 1);
        Mark("predict");
        Bind(totals, ("Den", den), ("Ms", ms), ("Raw", raw), ("Segment", segment), ("Sums", sums));
        cb.DispatchCompute(shader, totals, 8 * 3, 1, 1);
        Mark("totals");
        Bind(conserve, ("Den", den), ("Segment", segment), ("Sums", sums), ("Out", outBuffer));
        cb.DispatchCompute(shader, conserve, Groups(cells), 1, 1);
        Mark("conserve");
        if (StepFence == null) cb.EndSample(ProfilerSample);
    }

    public void Dispose()
    {
        foreach (var b in new[] {weights, segment, current, currentCounts, history, countHistory, ema, sums, x, ms, norm, smoothed, binMean, den, scratchA, scratchB, raw})
            b?.Dispose();
        if (ownsOutput) outBuffer?.Dispose();
    }
}
