"""Error metrics and non-learned baselines for the audio-DLSS denoiser dataset.

All inputs are complex arrays from dlss_dataset.load_bundle: low runs
[S, P, Q, F] (seed, receiver, frequency, family), label mean [P, Q].
Errors are complex relative L2 over receivers and frequencies, the metric used
by the wave comparisons; a network must beat these baselines on held-out
families at matched cost.
"""
import numpy as np

BANDS = dict(low=(0, 500), mid=(500, 2000), high=(2000, 4000))


def relative_error_squared(estimate, target, axes=None):
    """sum|x - t|^2 / sum|t|^2 (estimate may carry a leading seed axis)."""
    diff = np.abs(estimate - target) ** 2
    if diff.ndim > target.ndim:
        diff = diff.mean(0)
    return float(diff.sum() / np.sum(np.abs(target) ** 2))


def spike_scores(estimate, target):
    """|x - t| over the receiver-RMS label magnitude at each frequency."""
    scale = np.sqrt(np.mean(np.abs(target) ** 2, axis=0, keepdims=True))
    return np.abs(estimate - target) / np.maximum(scale, 1e-30)


def seed_mean(runs, n):
    """Disjoint n-seed averages of [S, ...] runs -> [S // n, ...]."""
    s = runs.shape[0] // n * n
    return runs[:s].reshape(s // n, n, *runs.shape[1:]).mean(1)


def neighbour_mean(totals, neighbours):
    """Average each receiver with its 4-connected grid neighbours: [S, P, Q]."""
    out = np.empty_like(totals)
    for p, row in enumerate(neighbours):
        out[:, p] = totals[:, [p] + list(row)].mean(1)
    return out


def family_clamp(runs, factor=3.0):
    """Robust per-family clamp: limit |value| to factor x the receiver median.

    Applied per seed, frequency and family before summing families. Biased, but
    removes isolated high-weight path samples; costs nothing at run time.
    """
    magnitude = np.abs(runs)
    limit = factor * np.median(magnitude, axis=1, keepdims=True) + 1e-30
    scale = np.minimum(1.0, limit / np.maximum(magnitude, 1e-30))
    return (runs * scale).sum(-1)


def band_masks(frequencies):
    f = np.asarray(frequencies)
    return {name: (f > lo) & (f <= hi) for name, (lo, hi) in BANDS.items()}
