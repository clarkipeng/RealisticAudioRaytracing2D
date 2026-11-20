"""README part 34. The auto-budget's trace (MeasurePathLiveBudget with ACOUSTIC_PATH_BUDGET_TRACE): per variant the
population per round, where it settles, the update's time against the budget, and the noise after the 64-wide
denoiser at the settled population, interpolated in log N from populations.py's Church table (part 33).

    python autobudget.py /abs/autobudget.csv
"""
import csv
import sys
from collections import defaultdict

import numpy as np

# populations.py, the 64-wide denoiser (part 33): N -> window / band level / decay curve, dB
NOISE = {4000: (8.09, 3.69, 4.89), 6000: (7.48, 2.50, 3.66), 11000: (6.34, 2.09, 2.75), 16000: (6.24, 2.11, 2.71), 23000: (6.12, 2.30, 2.54)}


def noise_at(n):
    keys = np.log(sorted(NOISE))
    return [float(np.interp(np.log(n), keys, [NOISE[k][i] for k in sorted(NOISE)])) for i in range(3)]


def main(path):
    rows = defaultdict(list)
    for r in csv.DictReader(open(path)):
        rows[r['variant']].append((int(r['round']), int(r['population']), float(r['tracerMs']) + float(r['audioMs'])))
    for name, v in rows.items():
        rounds = max(r for r, _, _ in v) + 1
        per = defaultdict(list)
        for r, n, t in v:
            per[r].append((n, t))
        load = name.endswith('load')
        halves = [range(rounds // 4, rounds // 2), range(3 * rounds // 4, rounds)] if load else [range(rounds // 2, rounds)]
        trail = ' '.join(f"{int(np.median([n for n, _ in per[r]]))}" for r in range(0, rounds, max(1, rounds // 12)))
        print(f"{name}: population every {max(1, rounds // 12)} rounds: {trail}")
        for h in halves:
            ns = [n for r in h for n, _ in per[r]]; ts = [t for r in h for _, t in per[r]]
            n = int(np.median(ns))
            changes = sum(1 for r in h if r - 1 in per and np.median([x for x, _ in per[r]]) != np.median([x for x, _ in per[r - 1]]))
            print(f"   rounds {h.start}-{h.stop - 1}: N {n} [{int(np.percentile(ns, 10))}-{int(np.percentile(ns, 90))}], "
                  f"update {np.median(ts):.2f} [{np.percentile(ts, 10):.2f}-{np.percentile(ts, 90):.2f}] ms, rounds with a change {changes}; "
                  f"noise after the denoiser at that N {'/'.join(f'{x:.2f}' for x in noise_at(n))} dB")


if __name__ == '__main__':
    main(sys.argv[1])
