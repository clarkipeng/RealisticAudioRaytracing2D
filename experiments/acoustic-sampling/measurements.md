# Finite-edge sampling measurements

Apple M5; Unity 6000.4.1f1. Silent EditMode compute, one low-priority process.

All rows compare prototype samplers for a discretized two-edge contribution. The source-only control is **not the production tracer's Cauchy proposal**, and these are not whole-renderer gains.

Expected error is calculated from **every candidate path**, not just observed trial error. Timings are medians of three batches after warmup. Each batch evaluates 32 independent estimates with 4,096 samples each. They include dispatch and blocking readback; they exclude table construction and the CPU SetData call, reported separately below. Queued upload work can still affect the dispatch wait. These are batch costs, not single-listener frame times or hardware GPU timestamps.

## 62.5 Hz

| Scene | Sampler | Guided | Expected relative RMSE | Observed relative RMSE | Batch ms | Upload ms | Relative MSE × time |
|---|---|---|---:|---:|---:|---:|---:|
| box | source | no | 12.82% | 11.84% | 0.775 | 0.178 | 1.00 |
| box | listener | no | 12.59% | 12.28% | 0.864 | 0.156 | 1.08 |
| box | join | no | 4.36% | 3.95% | 0.859 | 0.176 | 0.13 |
| box | bidirectional_mis | no | 6.93% | 6.32% | 0.916 | 0.191 | 0.35 |
| box | source | yes | 12.04% | 11.31% | 0.779 | 0.187 | 0.89 |
| box | listener | yes | 11.99% | 11.26% | 0.861 | 0.184 | 0.97 |
| box | join | yes | 4.30% | 4.27% | 0.839 | 0.173 | 0.12 |
| box | bidirectional_mis | yes | 6.75% | 6.04% | 0.980 | 0.175 | 0.35 |
| staggered | source | no | 11.57% | 10.71% | 1.382 | 0.734 | 1.00 |
| staggered | listener | no | 10.63% | 10.59% | 1.612 | 0.699 | 0.98 |
| staggered | join | no | 12.15% | 12.46% | 1.406 | 0.709 | 1.12 |
| staggered | bidirectional_mis | no | 10.22% | 10.50% | 1.500 | 0.704 | 0.85 |
| staggered | source | yes | 11.72% | 11.08% | 1.309 | 0.733 | 0.97 |
| staggered | listener | yes | 11.05% | 10.22% | 1.454 | 0.696 | 0.96 |
| staggered | join | yes | 15.23% | 15.64% | 1.435 | 0.713 | 1.80 |
| staggered | bidirectional_mis | yes | 11.07% | 11.94% | 1.545 | 0.716 | 1.02 |

Last column is `expected MSE × measured batch time`, normalized to source-only for that scene and frequency; lower is better. It is a throughput efficiency indicator, **not a measured speedup** at a separately rerun equal-time budget. Small differences are not meaningful given timing variation.

## 250 Hz

| Scene | Sampler | Guided | Expected relative RMSE | Observed relative RMSE | Batch ms | Upload ms | Relative MSE × time |
|---|---|---|---:|---:|---:|---:|---:|
| box | source | no | 35.03% | 31.05% | 0.904 | 0.192 | 1.00 |
| box | listener | no | 34.42% | 33.91% | 0.861 | 0.183 | 0.92 |
| box | join | no | 12.47% | 12.29% | 0.876 | 0.173 | 0.12 |
| box | bidirectional_mis | no | 19.22% | 19.84% | 0.909 | 0.177 | 0.30 |
| box | source | yes | 32.92% | 31.62% | 0.770 | 0.199 | 0.75 |
| box | listener | yes | 32.79% | 30.46% | 0.950 | 0.181 | 0.92 |
| box | join | yes | 12.32% | 12.15% | 0.873 | 0.168 | 0.12 |
| box | bidirectional_mis | yes | 18.74% | 19.07% | 0.903 | 0.167 | 0.29 |
| staggered | source | no | 24.42% | 24.06% | 1.336 | 0.660 | 1.00 |
| staggered | listener | no | 22.47% | 20.76% | 1.585 | 0.643 | 1.00 |
| staggered | join | no | 25.64% | 25.31% | 1.440 | 0.685 | 1.19 |
| staggered | bidirectional_mis | no | 21.61% | 22.26% | 1.724 | 0.611 | 1.01 |
| staggered | source | yes | 24.73% | 24.38% | 1.292 | 0.687 | 0.99 |
| staggered | listener | yes | 23.34% | 23.25% | 1.450 | 0.676 | 0.99 |
| staggered | join | yes | 32.05% | 33.70% | 1.356 | 0.649 | 1.75 |
| staggered | bidirectional_mis | yes | 23.37% | 22.38% | 1.436 | 0.708 | 0.98 |

Last column is `expected MSE × measured batch time`, normalized to source-only for that scene and frequency; lower is better. It is a throughput efficiency indicator, **not a measured speedup** at a separately rerun equal-time budget. Small differences are not meaningful given timing variation.

## Setup cost

| Scene | Nodes | Connections/readback ms | Baseline tables ms | Guided tables ms | GPU bytes per proposal pair |
|---|---:|---:|---:|---:|---:|
| box | 384 | 4.915 | 4.586 | 10.036 | 2,365,440 |
| staggered | 768 | 2.294 | 14.628 | 31.133 | 9,449,472 |

These are single setup measurements, not medians; the first connection dispatch can include shader initialization. Tables use O(nodes²) storage and rebuild work. This is a transparent research implementation, unsuitable as a live architecture. All geometry and frequency-independent setup is reused across seeds, sample budgets, methods, and frequencies in the measurement.

## Target discretization and reciprocity

| Scene | Hz | Change from 32 to 64 points per edge | Largest path reversal discrepancy |
|---|---:|---:|---:|
| box | 62.5 | 0.00000% | 4.33e-07 |
| box | 250 | 0.00002% | 1.67e-06 |
| staggered | 62.5 | 1.20944% | 4.06e-07 |
| staggered | 250 | 1.93024% | 1.55e-06 |

Reversal discrepancy is relative L2 over individual path contributions. It is **not** validation against a wave solver. The staggered target is not converged: sampling rankings concern the enumerated 32-point discretization. The equal-height coplanar variant fails reciprocity and is retained separately as a known production-visibility defect.

## Calculation

For complex path contribution `F(a,b)` and proposal probability `q(a,b)`, the enumerated response is `H = sum F`. Expected squared error after N independent samples is `(sum |F|²/q - |H|²)/N`. Observed error uses 96 estimates per row (32 seeds × 3 repeats). The table probabilities use the actual float CDF interval widths.
