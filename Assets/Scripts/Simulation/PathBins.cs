using System;

// Time bins of the Path method's histogram (MethodPath.hlsl PathBinOf mirrors them): FineBins bins
// of FineSeconds (0.25 ms by default) up to T0 = FineBins * FineSeconds, then constant relative
// resolution, each bin 1 / FineBins of its start time (so the two regions join without a step).
// Early on, where single reflections are heard as echoes, bins are fine; later, where the
// reflection density makes the field diffuse, a bin spans a fixed fraction of its arrival time.
public sealed class PathBins
{
    public readonly double FineSeconds, Horizon, T0, Growth;
    public readonly int FineBins, Count;
    public PathBins(TakeRender.Settings settings) : this(settings.pathBinSeconds, settings.pathFineBins, settings.maximumDelaySeconds) {}
    public PathBins(double fineSeconds, int fineBins, double horizon)
    {
        if (!(fineSeconds > 0) || fineBins < 1 || !(horizon > 0)) throw new ArgumentException("Invalid path histogram bins.");
        FineSeconds = fineSeconds; FineBins = fineBins; Horizon = horizon;
        T0 = fineSeconds * fineBins; Growth = 1 + 1.0 / fineBins;
        Count = horizon <= T0 ? (int)Math.Ceiling(horizon / fineSeconds - 1e-9)
                              : fineBins + (int)Math.Ceiling(Math.Log(horizon / T0) / Math.Log(Growth) - 1e-9);
    }
    public double Start(int bin) => bin <= FineBins ? bin * FineSeconds : T0 * Math.Pow(Growth, bin - FineBins);
    public double End(int bin) => Start(bin + 1);
    public double Width(int bin) => End(bin) - Start(bin);
    public int Of(double time) => time < T0 ? (int)Math.Floor(time / FineSeconds) : FineBins + (int)Math.Floor(Math.Log(time / T0) / Math.Log(Growth));
    // First bin that ends after `time`.
    public int FirstOverlap(double time) => Math.Max(0, Math.Min(Count, Of(Math.Max(0, time))));
    // Share of bin k's time span inside [from, to).
    public double Overlap(int bin, double from, double to)
    {
        double s = Start(bin), e = End(bin);
        return Math.Max(0, Math.Min(e, to) - Math.Max(s, from)) / (e - s);
    }
}
