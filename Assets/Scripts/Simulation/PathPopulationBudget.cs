using System;

// The auto-budget of the path method's persistent rays (docs/benchmarks/path-denoise-2026-11-12, part 34): it
// chooses the rays in flight N so that the GPU time per update stays near a budget.
//
// The rule. Times are gathered in blocks of BlockUpdates updates at a fixed N; the first update after a change of N
// is left out (it still traces the old cohorts' rays). If the budget lies outside the block median's confidence
// interval (order statistics 4 and 13 of 16: 1 - 2 P(Binomial(16, 1/2) <= 3) = 97.9%), N becomes N B / T, with T the
// block's median and B the budget, within [Minimum, Maximum]; otherwise N stays. The update's time is T = T0 + c N
// with fixed costs T0 >= 0 (the audio, the denoiser, the dispatches), so after the step the time is
// T0 + c N B / T and its error T0 (1 - B / T) = (T0 / T) (T - B): the same sign and a fraction T0 / T < 1 of the error
// before. N therefore approaches its target from one side, without overshoot or oscillation, and stops inside the
// noise of the median. No gain is chosen: the step assumes all time scales with N, which T0 >= 0 makes conservative.
//
// Unbiasedness is unaffected: N only sets the new rays of the next update (fixed before that cohort launches) and the
// cohorts' targets, which population control meets by splitting or thinning with weights divided by rho.
public sealed class PathPopulationBudget
{
    public const int BlockUpdates = 16;
    public readonly double BudgetMs;
    public readonly int Minimum, Maximum;
    public int Population { get; private set; }
    public double LastMedianMs { get; private set; } = double.NaN;
    readonly double[] block = new double[BlockUpdates];
    int filled;
    bool settle = true;

    public PathPopulationBudget(double budgetMs, int population, int minimum, int maximum)
    {
        if (!(budgetMs > 0)) throw new ArgumentOutOfRangeException(nameof(budgetMs));
        if (minimum < 1 || maximum < minimum) throw new ArgumentOutOfRangeException(nameof(minimum));
        BudgetMs = budgetMs; Minimum = minimum; Maximum = maximum;
        Population = Math.Max(minimum, Math.Min(maximum, population));
    }

    // After an update at Population: its GPU time in milliseconds. Returns the population for the next update.
    public int Observe(double milliseconds)
    {
        if (settle) { settle = false; return Population; }
        if (!(milliseconds >= 0) || double.IsInfinity(milliseconds)) return Population;
        block[filled++] = milliseconds;
        if (filled < BlockUpdates) return Population;
        filled = 0;
        Array.Sort(block);
        double median = .5 * (block[BlockUpdates / 2 - 1] + block[BlockUpdates / 2]);
        LastMedianMs = median;
        if (BudgetMs >= block[3] && BudgetMs <= block[12]) return Population;
        int next = (int)Math.Max(Minimum, Math.Min(Maximum, Math.Round(Population * BudgetMs / median)));
        if (next != Population) { Population = next; settle = true; }
        return Population;
    }
}
