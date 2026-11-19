"""README part 31 onward. The rays the walk keeps, before population control (MeasurePathPool), per scene:
where they are (the sort's cells) and how their weights spread within a cohort (the resampling's variance).

    python pool.py /abs/.context/path-optimise/r31/pool

For each cohort (rays of one age, A of them, target n), the variance that resampling adds to the cohort's
band total, summed over cohorts and bands relative to today's rule (keep or split every ray with rho = n / A):
  energy-aware   keep with p_i = min(1, c s_i), sum p_i = n, survivor weight w_i / p_i (s: band-max or band-sum)
  systematic     today's rho, one uniform per cohort, copies floor((r + 1) rho + u) - floor(r rho + u) by rank r
  both           energy-aware probabilities, systematic in rank order
The systematic variants are simulated (400 draws per cohort, rays in random order).
"""
import json
import sys
from pathlib import Path

import numpy as np

FINE_END = 64 * .25e-3   # the histogram's fine bins end (T_f)
SPLIT_LIMIT = 16


def load(folder):
    meta = json.loads((folder / 'pool.json').read_text())
    for m in meta:
        x = np.fromfile(folder / f"{m['scene']}-{m['snapshot']}.f32", '<f4').reshape(-1, 16).astype(np.float64)
        yield m, x


def energy_aware(s, n):
    """p_i = min(1, c s_i) with sum p_i = n (water-filling); all ones if n >= the number of positive weights."""
    s = np.where(s > 0, s, 0)
    if n >= np.count_nonzero(s):
        return (s > 0).astype(float)
    order = np.sort(s)[::-1]
    for capped in range(len(order)):
        c = (n - capped) / order[capped:].sum()
        if c * order[capped] <= 1:
            return np.minimum(1, c * s)
    return (s > 0).astype(float)


def systematic(expected, rng, draws):
    """Copies per ray for `draws` uniforms, in a random order of the rays: [draws, rays]."""
    order = rng.permutation(len(expected))
    edges = np.concatenate([[0], np.cumsum(expected[order])])
    u = rng.random((draws, 1))
    copies = np.diff(np.floor(edges[None] + u), axis=1)
    out = np.empty_like(copies); out[:, order] = copies
    return out


def cohort_variances(w, n, rng, draws=400):
    """Added variance of the band totals [8] for each rule, for one cohort's weights w [A, 8] and target n."""
    A = len(w)
    rho = min(n / A, SPLIT_LIMIT)
    f = rho - np.floor(rho)
    out = dict(today=(w ** 2).sum(0) * f * (1 - f) / rho ** 2)
    sys_copies = systematic(np.full(A, rho), rng, draws)
    out['systematic'] = ((sys_copies / rho) @ w).var(0)
    # (l2: the minimiser of the summed relative variances, sum_k sum_i w_ik^2 (1 / p_i - 1) / W_k^2 at sum p_i = n, is
    # p_i proportional to sqrt(sum_k (w_ik / W_k)^2), by Lagrange)
    total = np.where(w.sum(0) > 0, w.sum(0), 1)
    for name, s in (('max', w.max(1)), ('sum', w.sum(1)), ('l2', np.sqrt(((w / total) ** 2).sum(1)))):
        if rho >= 1:
            out['aware-' + name] = out['today']; out['both-' + name] = out['systematic']; continue
        p = energy_aware(s, n)
        keep = p > 0
        out['aware-' + name] = ((w[keep] ** 2) * (1 / p[keep, None] - 1)).sum(0) + 0 * w[0]
        copies = systematic(p, rng, draws)
        scale = np.where(p > 0, 1 / np.where(p > 0, p, 1), 0)
        out['both-' + name] = ((copies * scale) @ w).var(0)
        # (rays with s = 0 carry no energy in any band and are dropped by every rule)
    return out


def main(folder):
    folder = Path(folder)
    rng = np.random.default_rng(0)
    scenes = {}
    for m, x in load(folder):
        scenes.setdefault(m['scene'], []).append((m, x))
    for scene, snaps in scenes.items():
        m0 = snaps[0][0]
        step, fresh = m0['step'], m0['newRays']
        box = np.array(m0['box']).reshape(2, 3)
        pos = np.concatenate([x[:, :3] for _, x in snaps])
        print(f"\n== {scene}: {np.mean([len(x) for _, x in snaps]):.0f} rays kept per update, {fresh} new; scene box {box[0].round(1)} .. {box[1].round(1)}")
        for q in (0, .005, .01, .05, .5, .95, .99, .995, 1):
            print(f"   position quantile {q:5.3f}: {np.quantile(pos, q, axis=0).round(1)}")
        inside = np.all((pos >= np.quantile(pos, .01, 0)) & (pos <= np.quantile(pos, .99, 0)), 1).mean()
        print(f"   share inside the 1-99% box: {inside:.3f}")
        totals, cv2 = {}, {'max': [], 'sum': [], 'band': []}
        branch = []
        for m, x in snaps:
            age, w, flags = x[:, 3], x[:, 8:], x[:, 7]
            k = np.minimum(np.round(age / step + .5), 1e9).astype(int)
            for c in np.unique(k):
                sel = k == c
                A, a = sel.sum(), (c - .5) * step
                n = fresh * min(1, FINE_END / a)
                wc = w[sel]
                for name, s in (('max', wc.max(1)), ('sum', wc.sum(1))):
                    if A > 1 and s.mean() > 0:
                        cv2[name].append((A, n, s.var() / s.mean() ** 2))
                band = [wc[:, b].var() / wc[:, b].mean() ** 2 for b in range(8) if wc[:, b].mean() > 0]
                if A > 1 and band:
                    cv2['band'].append((A, n, float(np.mean(band))))
                branch.append((A, (flags[sel] % 2 == 1).mean()))
                v = cohort_variances(wc, n, rng)
                for rule, val in v.items():
                    totals.setdefault(rule, np.zeros(8))
                    totals[rule] += val / np.maximum(wc.sum(0) ** 2, 1e-300)   # relative variance per cohort and band, summed
        for name, rows in cv2.items():
            r = np.array(rows)
            thin = r[r[:, 1] < r[:, 0]]
            print(f"   CV^2 of the {name:4s} weight per cohort: median {np.median(r[:, 2]):.3g}, ray-weighted mean {np.average(r[:, 2], weights=r[:, 0]):.3g}, "
                  f"thinned cohorts' ray-weighted mean {np.average(thin[:, 2], weights=thin[:, 0]) if len(thin) else float('nan'):.3g}, 90th pct {np.quantile(r[:, 2], .9):.3g}")
        b = np.array(branch)
        print(f"   rays in a transmitted branch: {np.average(b[:, 1], weights=b[:, 0]):.3f}")
        base = totals['today']
        for rule, val in totals.items():
            print(f"   added relative variance, {rule:10s}: per band / today's {np.round(val / base, 3)}")


if __name__ == '__main__':
    main(sys.argv[1])
