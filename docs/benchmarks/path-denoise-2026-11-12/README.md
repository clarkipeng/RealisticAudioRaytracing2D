# Path tracer: parallel connections, speed, the Church render and a denoiser — 12 November 2026

Follow-up to the [path tracer benchmark](../path-tracer-2026-11-12/README.md).
All numbers are measured on this Mac (Apple M5, 16 GB, Unity 6000.4.1f1 on Metal) under `tools/heavy_guard.py` (one heavy job machine-wide, 4 GiB guard, nice 19, one job worker, 50% pacing), often waiting for other projects' jobs.

## 1. Parallel connection trials

**Change.**
A queued specular-chain candidate used to retrace its exact image path and then draw trials one after another in one GPU thread until one followed the chain (the geometric count `N`).
`PathConnect` now gives each candidate 8 lanes of a 64-thread group: in round 0 lane 0 retraces the exact path and lanes 1–7 draw trials 1–7; in round r the lanes draw trials 8r … 8r + 7.
`N` is the smallest index of a successful trial, kept with an atomic minimum in group-shared memory, and each round decides whether to go on only from the previous rounds' results.
Because the trials are independent and identically distributed, the smallest successful index has exactly the distribution of the sequential count, so the estimator is unchanged; a candidate's latency falls from `N` retraces to about `N / 8`.

**Checks.**
All path checks pass unchanged in value (free field exact; rigid plane image 0.953; image lattice total 0.989, worst 20 ms window 6.1%; half-plane +0.30 dB; DD over a 2 m wall +0.49 … +1.58 dB; Eyring +4.8%; the clipped-region image 1.00 with the measured share).

Time per update (GPU busy per solve, one source and listener, all eight bands, 2 s horizon; 8 receivers per scene, one process, the measurement path with a fence per 1,024 rays):

| Scene | energy 2,048 | energy 16,384 | path 2,048 | path 16,384 | path 2,048 before | path 16,384 before |
|---|---:|---:|---:|---:|---:|---:|
| canopy (open) | 11 ms | 106 ms | 19 ms | 136 ms | 11 ms | 178 ms |
| barriers (open) | 9 ms | 59 ms | 9 ms | 81 ms | 13 ms | 110 ms |
| doorway | 56 ms | 575 ms | 104 ms | 834 ms | 140 ms | 1.30 s |
| wood shoebox | 20 ms | 329 ms | 70 ms | 596 ms | 244 ms | 1.92 s |
| stone shoebox | 50 ms | 614 ms | 109 ms | 844 ms | 283 ms | 2.33 s |
| corridor (two rooms) | 98 ms | 761 ms | 170 ms | 1.35 s | 1.10 s | – |
| furnished room | 168 ms | 1.39 s | 690 ms | 6.22 s | 3.31 s | – |

"Before" is the cost table of the first benchmark (sequential trials, all receivers).
At 2,048 rays the path tracer is now 1.0–2.2 times the energy method in the open scenes, doorway, stone shoebox and corridor, 3.5 times in the wood shoebox and 4.1 times in the furnished room; the furnished room fell from 3.3 s to 0.69 s per update, the corridor from 1.10 s to 0.17 s.
It is not realtime in rooms: 70–690 ms per update at 2,048 rays against a 23 ms budget (part 2).

**Trial cap.** With parallel trials the cap of 64 could be raised; at 1,024 the capped detections in the cluttered rooms fall from 112 to 3 (furnished room, 4,502 scored chains, 30 receivers) and from 44 to 0 (corridor, 9,785).
The level bias moved by +0.05 dB (furnished, 1.69 → 1.74 dB) and +0.02 dB (corridor), within the seed noise, while the cost rose about threefold (furnished room 2.0 s per update over its 30 receivers).
The cap stays at 64: its bias is below 0.1 dB in the worst scene of the suite, and the counter reports it.

## 2. Speed: profile, then the changes it pointed to

### Profile

`TakeChecks.MeasurePathProfile` (`Assets/Tests/Editor/PathProfileMeasurement.cs`) puts a GPU fence after every phase of a solve and repeats the solves with parts switched off (profiling switches only, `MethodTransport.ProfileFlags`): without the histogram writes, without the connection trials (exact retrace only), without the straight connections of scattering points, without their edge connections.
The differences give each part's share.
Times are per update (one source and listener, all eight bands), averaged over 4 poses, 2,048 rays; the fences add about 1 ms per phase.

Before (the step-1 commit; Church: the realtime listening settings, 5 s horizon):

| Scene | walk | candidate retrace | trials | histogram writes | edge distribution | readback |
|---|---:|---:|---:|---:|---:|---:|
| stone shoebox | 23 ms | 31 ms | 2 ms | not measurable (< 1 ms) | 2.7 ms | 0.2 ms |
| furnished room | 96 ms | 139 ms | 141 ms | not measurable | 0.7 ms | 0.2 ms |
| Church | **12,894 ms** | 20 ms | 0.5 ms | not measurable | 4.3 ms | 0.4 ms |

What it showed:
- **Histogram writes are not the cost**: switching them off changed nothing measurable in any scene, so group-shared accumulation was not done.
- **In the Church the walk is everything**, and within it the connections of scattering points: every ray lived the whole 5 s horizon, and each of its scattering points made a straight connection through the walls (the stone has a tiny transmission, so every connection ran the wall-crossing loop) and an edge connection whose mirror variant scanned every lower-index triangle of the mirror's object (`CompleteMirror`).
- **In the furnished room the connections dominate** (retrace and trials about equal), then the walk.
- **Hardware ray tracing is not available** to these kernels: Unity reports `SystemInfo.supportsRayTracing` and `supportsInlineRayTracing` false for Metal on this M5 with Unity 6000.4, so `RayTracingAccelerationStructure` and inline ray queries cannot be used; the BVH traversal was left as it is.

### Changes (all unbiased)

1. **Survival in time** (Russian roulette per time bin, weight-compensated): a path is alive at time t with probability `S(t) = min(1, T0 / t)`, applied at every surface event (specular, transmitted and scattering), with `T0 = pathSurvivalSeconds`.
   The histogram's bins widen in proportion to t, so a path population falling as 1/t keeps roughly the same number of contributions per bin, while the work over a horizon T falls from T to `T0 (1 + ln(T / T0))`.
   This is the variance-blind floor replaced by a per-time-bin schedule, not the full EARS: the survival is not learned from measured variance.
2. **Blocked straight connections by roulette**: a connection from a scattering point that hits a surface is traced through the walls only with probability `q` = that surface's largest energy transmission, and weighted by `1/q`; the transmitted energy is at most that factor, so the weight stays bounded (stone: q of 1e-6, so the crossing loop almost never runs).
   An unobstructed connection skips the crossing loop altogether.
   The source's direct sound stays exact.
3. **Mirror test in O(1)**: the edge connection's mirror leg no longer scans the object's triangles for a shared-edge contact; a hit on the triangle's boundary (a set of measure zero) is rejected instead.
4. **Transmission geometry only when chosen**: the partition exit and the medium behind a surface are traced only when the walk keeps a transmitted branch or transmits, not at every hit of a transmitting material.

Checks for the new parts (`PathChecks.cs`): leaked energy out of a scattering room through walls with t = 1e-3 is 4.24e-4 with the connection roulette, 4.41e-4 tracing every blocked connection and 4.37e-4 without survival in time (two seeds each), i.e. the same within noise.
The energy method gives 9.04e-4 for that room; the factor of two between the two methods' leakage is not explained (the energy method scores its walls' near field with a 0.5 m receiver sphere).
All other path checks pass unchanged; the image-lattice window check is run without survival in time, because it checks the connections at the energy method's sampling density.

### Accuracy against cost (wave suite)

`variants.py` scores each variant against the wave ground truth on the five walled scenes (two shoeboxes, doorway, corridor, furnished room; 2,048 rays, two seeds unless stated; time per update in the measurement harness, all receivers):

| Variant | level dB | level bias dB | EDT % | T30 % | ETC dB | time per update |
|---|---:|---:|---:|---:|---:|---:|
| energy method | 2.42 | +1.63 | 117 | 98 | 12.2 | 83 ms |
| path, before (every path to the horizon) | 2.66 | +1.47 | 104 | 122 | 16.1 | 989 ms |
| path, survival T0 = 16 ms | 3.44 | +0.91 | 113 | 103 | 24.7 | 101 ms |
| path, survival T0 = 16 ms, 16,384 rays | 2.65 | +1.49 | 66 | 77 | 19.6 | 849 ms |
| **path, survival T0 = 100 ms (new default)** | **2.66** | +1.43 | **88** | **70** | 20.4 | **156 ms** |

With `T0 = 100 ms` the level error is unchanged (2.66 dB), early decay and T30 improve, and an update is 6.3 times cheaper; the energy-time-curve distance is worse (16.1 → 20.4 dB): late 5 ms windows rest on fewer paths, and the ETC distance punishes their scatter in decibels.
`T0 = 16 ms` is 10 times cheaper but noisier (level 3.44 dB at 2,048 rays); at 16,384 rays it costs about what the old version cost at 2,048 and beats it on level, EDT and T30.
On the whole 11-scene suite at the default the level error is 4.03 dB (before: 4.05), closed rooms 1.99 dB (1.95), shadow 5.84 dB (5.88): unchanged within noise.

### Time per update now

Per update at the default (`T0` = 100 ms) and at `T0` = 16 ms, walk + connections (fences included):

| Scene | 256 rays | 512 rays | 2,048 rays | 2,048 rays, `T0` = 16 ms | before (step 1), 2,048 |
|---|---:|---:|---:|---:|---:|
| wood shoebox | 14 ms | 18 ms | 26 ms | 20 ms | 70 ms |
| stone shoebox | 18 ms | 15 ms | 28 ms | 18 ms | 109 ms |
| doorway | 13 ms | 14 ms | 28 ms | 27 ms | 104 ms |
| corridor | 22 ms | 23 ms | 48 ms | 33 ms | 170 ms |
| furnished room | 80 ms | 78 ms | 196 ms | 104 ms | 690 ms |
| Church (5 s horizon) | 38–72 ms | 38 ms | 217 ms | 215 ms | 12,900 ms |

At 256–512 rays the shoeboxes, doorway and corridor now fit the 23 ms budget; the furnished room (about 80 ms) and the Church (about 40 ms) do not.
Their time hardly falls from 512 to 256 rays: it is set by the longest paths and the longest retraces, one GPU thread each, not by the ray count; the Church times also vary by a factor of two between runs (other work on the machine).
**Not done and next**: splitting long paths and long retraces across threads (the wavefront idea applied to the walk: compact the surviving rays between bounces), and the EARS-style survival learned from measured per-bin variance; together they target exactly this latency floor.
