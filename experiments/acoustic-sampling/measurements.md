# Matched-error acoustic sampling measurements

Apple M5; Unity 6000.4.1f1. Silent, low-priority Metal EditMode run.

**The prototypes are promising after setup, but none is ready to replace the live renderer.** These measurements cover one static source/listener pair and its two-edge diffraction response. They exclude full-scene audio synthesis, moving listeners, and sharing across pairs.

Prototype sample budgets come from enumerating the complete quadrature target's variance. The production budget near 20% was calibrated from a previous run and then evaluated with fresh seeds. The Cauchy-only control uses the same calibrated budget. Error is complex-response RMSE, not perceived audio quality or error against a wave solver.

Five repeats follow a discarded warmup. Each batch contains 32 estimates; observed error combines 160 estimates. A separate single-estimate dispatch is measured each repeat. Times include dispatch and blocking readback, with **median [25th–75th percentile]** shown. CPU final summation, proposal preparation, and SetData call time are outside these timings. This is wall time, not GPU timestamps; the wide single-estimate ranges should discourage precise speedup claims.

## 250 Hz, approximately 20% response error

| Scene | Method | Guided | Samples | Observed error | One estimate ms | Batch of 32 ms |
|---|---|---|---:|---:|---:|---:|
| box | Production trace | no | 12,544 | 21.76% | 1.264 [0.834–1.837] | 14.198 [13.933–14.309] |
| box | Cauchy proposal only | no | 12,544 | 22.14% | 0.765 [0.440–1.071] | 7.445 [5.561–8.668] |
| box | Source | no | 12,608 | 19.60% | 0.707 [0.590–0.934] | 2.548 [2.479–2.631] |
| box | Listener | no | 12,160 | 18.37% | 0.863 [0.781–0.992] | 5.525 [3.920–6.106] |
| box | Two-ended join | no | 1,600 | 21.12% | 0.578 [0.560–0.863] | 1.444 [1.315–1.735] |
| box | Source/listener/join mixture | no | 3,840 | 19.57% | 0.714 [0.484–0.876] | 1.392 [1.255–2.089] |
| box | Source | yes | 11,136 | 19.39% | 0.909 [0.730–0.982] | 2.562 [2.536–2.969] |
| box | Listener | yes | 11,072 | 19.80% | 0.610 [0.517–0.986] | 4.962 [3.457–7.047] |
| box | Two-ended join | yes | 1,600 | 19.80% | 0.391 [0.301–0.426] | 0.731 [0.573–1.293] |
| box | Source/listener/join mixture | yes | 3,648 | 19.13% | 0.598 [0.319–0.621] | 1.509 [1.293–1.535] |
| staggered | Production trace | no | 38,208 | 20.53% | 2.956 [2.194–3.242] | 58.307 [50.837–58.667] |
| staggered | Cauchy proposal only | no | 38,208 | 21.26% | 1.584 [1.479–1.684] | 31.596 [30.325–32.252] |
| staggered | Source | no | 6,144 | 20.13% | 0.472 [0.384–0.493] | 3.243 [3.180–4.663] |
| staggered | Listener | no | 5,184 | 19.62% | 0.743 [0.455–0.855] | 2.586 [2.509–4.398] |
| staggered | Two-ended join | no | 6,784 | 20.89% | 0.675 [0.448–0.927] | 2.905 [2.797–3.006] |
| staggered | Source/listener/join mixture | no | 4,800 | 19.90% | 0.454 [0.361–0.478] | 2.852 [2.805–3.956] |
| staggered | Source | yes | 6,272 | 18.97% | 0.601 [0.550–0.971] | 2.623 [2.339–2.689] |
| staggered | Listener | yes | 5,632 | 19.23% | 0.891 [0.624–0.927] | 1.941 [1.922–2.763] |
| staggered | Two-ended join | yes | 10,560 | 18.82% | 0.610 [0.584–1.004] | 4.410 [3.942–5.110] |
| staggered | Source/listener/join mixture | yes | 5,632 | 20.32% | 0.744 [0.566–0.991] | 3.064 [2.844–3.163] |

## 250 Hz, approximately 10% response error

| Scene | Method | Guided | Samples | Observed error | One estimate ms | Batch of 32 ms |
|---|---|---|---:|---:|---:|---:|
| box | Source | no | 50,304 | 9.63% | 1.375 [1.294–1.415] | 13.447 [13.303–13.457] |
| box | Listener | no | 48,576 | 9.55% | 1.204 [0.596–1.474] | 9.372 [8.909–12.395] |
| box | Two-ended join | no | 6,400 | 9.61% | 0.584 [0.491–0.615] | 3.298 [2.977–3.599] |
| box | Source/listener/join mixture | no | 15,168 | 9.63% | 0.630 [0.517–0.784] | 4.473 [4.190–6.711] |
| box | Source | yes | 44,416 | 9.88% | 0.858 [0.724–1.012] | 10.135 [9.537–11.450] |
| box | Listener | yes | 44,096 | 9.54% | 0.870 [0.692–1.070] | 13.206 [11.409–13.781] |
| box | Two-ended join | yes | 6,272 | 9.22% | 0.557 [0.481–0.635] | 2.427 [2.248–2.527] |
| box | Source/listener/join mixture | yes | 14,400 | 10.02% | 0.603 [0.596–0.628] | 3.823 [3.618–4.487] |
| staggered | Source | no | 24,448 | 9.58% | 0.767 [0.638–0.791] | 12.245 [11.793–12.383] |
| staggered | Listener | no | 20,736 | 10.55% | 1.038 [0.829–1.048] | 10.897 [10.354–11.796] |
| staggered | Two-ended join | no | 26,944 | 9.91% | 1.221 [0.837–1.815] | 13.621 [13.139–14.166] |
| staggered | Source/listener/join mixture | no | 19,136 | 9.36% | 0.860 [0.708–1.444] | 10.425 [9.513–12.271] |
| staggered | Source | yes | 25,088 | 9.42% | 0.894 [0.833–1.180] | 11.844 [10.735–12.838] |
| staggered | Listener | yes | 22,336 | 9.83% | 0.833 [0.805–1.222] | 12.408 [12.070–13.087] |
| staggered | Two-ended join | yes | 42,112 | 9.92% | 1.522 [1.338–1.637] | 18.831 [18.284–19.942] |
| staggered | Source/listener/join mixture | yes | 22,400 | 9.85% | 0.809 [0.800–0.812] | 11.278 [9.852–11.894] |

## 62.5 Hz, approximately 20% response error

| Scene | Method | Guided | Samples | Observed error | One estimate ms | Batch of 32 ms |
|---|---|---|---:|---:|---:|---:|
| box | Production trace | no | 1,792 | 21.07% | 0.668 [0.545–1.040] | 3.014 [2.945–3.622] |
| box | Cauchy proposal only | no | 1,792 | 20.71% | 0.763 [0.378–0.852] | 1.644 [1.008–1.711] |
| box | Source | no | 1,728 | 20.08% | 0.413 [0.403–0.450] | 1.389 [1.150–1.911] |
| box | Listener | no | 1,664 | 20.47% | 0.660 [0.422–0.670] | 0.861 [0.805–1.247] |
| box | Two-ended join | no | 256 | 17.60% | 0.630 [0.193–0.675] | 0.569 [0.435–0.828] |
| box | Source/listener/join mixture | no | 512 | 17.65% | 0.346 [0.230–0.346] | 0.388 [0.369–0.573] |
| box | Source | yes | 1,536 | 19.89% | 0.629 [0.340–0.806] | 1.036 [0.691–3.365] |
| box | Listener | yes | 1,472 | 19.97% | 0.633 [0.598–0.788] | 1.082 [1.022–1.292] |
| box | Two-ended join | yes | 192 | 20.58% | 0.260 [0.206–0.694] | 0.487 [0.394–0.672] |
| box | Source/listener/join mixture | yes | 512 | 17.24% | 0.750 [0.317–0.840] | 0.600 [0.548–0.947] |
| staggered | Production trace | no | 8,768 | 21.27% | 1.739 [1.005–2.039] | 13.395 [13.175–16.290] |
| staggered | Cauchy proposal only | no | 8,768 | 22.07% | 0.814 [0.586–0.834] | 6.079 [5.925–8.789] |
| staggered | Source | no | 1,408 | 20.56% | 0.897 [0.694–1.071] | 1.345 [1.206–1.376] |
| staggered | Listener | no | 1,216 | 20.02% | 0.455 [0.363–0.599] | 1.643 [0.830–1.796] |
| staggered | Two-ended join | no | 1,536 | 19.12% | 0.672 [0.405–0.679] | 1.217 [1.125–1.360] |
| staggered | Source/listener/join mixture | no | 1,088 | 20.67% | 0.431 [0.425–0.519] | 1.336 [1.280–1.478] |
| staggered | Source | yes | 1,408 | 20.51% | 0.719 [0.322–0.734] | 1.597 [1.565–4.264] |
| staggered | Listener | yes | 1,280 | 18.60% | 0.683 [0.330–1.020] | 1.079 [0.980–1.348] |
| staggered | Two-ended join | yes | 2,432 | 21.81% | 0.563 [0.274–0.612] | 1.650 [1.644–1.774] |
| staggered | Source/listener/join mixture | yes | 1,280 | 20.35% | 0.831 [0.422–1.080] | 1.528 [0.833–1.566] |

The 10% production budget was **not measured**. That table only compares prototypes against their discrete target. All 64/512/4,096/16,384-sample controls are retained in the raw CSV.

## Setup and transfer costs

| Scene | Warm connections/readback ms | Baseline tables ms | Guided tables ms | GPU bytes per proposal pair | First connection dispatch ms |
|---|---:|---:|---:|---:|---:|
| box | 1.443 | 4.537 | 7.509 | 2,365,440 | 2.824 |
| staggered | 4.551 | 15.045 | 26.066 | 9,449,472 | 2.887 |

Setup is measured once per fixture; it is not a median. The first connection dispatch is recorded separately because it can include shader initialization. Common scene upload/BVH creation and Unity startup are excluded. Dense O(nodes²) tables are frequency-independent and reused throughout the run; their construction must be paid or amortized. They are not suitable as the live architecture.

- box: median proposal-pair SetData call 0.175 ms; queued upload work may also affect the dispatch wait.
- staggered: median proposal-pair SetData call 0.714 ms; queued upload work may also affect the dispatch wait.

The production and Cauchy controls need no dense guide. Their `build_ms` field records common test-buffer/material setup, not guide creation; Cauchy-only inherits that allocation for comparison.

## What was isolated

1. Source, listener, join, and mixture use the same evaluator and quadrature target. Changing among them isolates proposal direction/combination; guidance changes only the proposal tables.
2. Cauchy-only calls the production `SampleEdge` twice with the same two 50% edge-event choices, then reduces contributions directly. No dense guide is used.
3. Production calls the unmodified `Trace3DBvh`, then reduces its second-event arrivals. Reflection and transmission are suppressed through material settings to isolate the two-edge contribution. The signed rigid-edge kernel and edge opacity remain unchanged. Surface calculations and full path writes still occur, so the gap to Cauchy-only measures more than sampling. Necessary surface transport cannot simply be removed from the complete renderer. Direct reduction here yields a static complex coefficient; changing delays, Doppler, and path continuity still need to be preserved in a live implementation.

## Validation and limits

The GPU test passed finite-output, positive-support, reciprocity, estimator-mean, zero-diffraction isolation, and exact single/batch packing checks. Across 20 continuous-control comparisons, mean differences stayed within 2.22 combined standard errors (six-standard-error regression threshold). This is statistical consistency, not proof of zero bias.

- box, 250 Hz: changing 32 to 64 quadrature points per edge changes the reference by 0.00002%.
- staggered, 250 Hz: changing 32 to 64 quadrature points per edge changes the reference by 1.93024%.

Prototypes use the enumerated 32-point target; continuous controls use the 64-point reference. The staggered continuum reference remains unconverged. Coplanar/grazing visibility still has a known reversal defect, reproduced by `ACOUSTIC_SAMPLING_COPLANAR=1`; these ordinary fixtures use unequal heights. No whole-church, moving-head, multi-source reuse, or training-reference claim follows.
