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

## 3. The level gap against the energy method

The first realtime Church render was about 4 dB quieter than the energy render inside and much quieter outside, and in the earlier leak check the path tracer leaked half as much as the energy method.
Before the Church goes on the scoreboard, three small measurements decided which method is right.

### An analytic leak reference

`MethodsLeakAsTheDiffuseFieldPredicts` (`PathChecks.cs`): in a closed room with a diffuse field the radiance at a wall is `E_in / 4π`, `E_in` the reverberant energy measured just inside it; a wall transmitting `τ = (1 − a) t` (undeviated) shows a listener outside the radiance `τ E_in / 4π` over the wall's solid angle `Ω`, so the reverberant energy outside is `τ E_in Ω / 4π` (direct sounds subtracted on both sides).
Room 6 × 5 × 4 m, a = 0.1, t = 1e-3, listener 0.8 m outside a wall (Ω = 4.32 sr), 2^16 rays, two seeds.

| Case | path, before | path, after the fix | energy method |
|---|---:|---:|---:|
| s = 1 (diffuse) | 0.994 | 0.994 | 1.012 |
| s = 0.5 | **0.501** | **1.043** | 1.119 |

**Cause.** A ray's small transmitted share is split off into one reservoir branch, traced after the ray ends.
The walk connects a specular chain to the listener after each of its events; the branch's chain ends with its transmission, but that last connection was never made, because the branch starts after the event.
So sound that reflects and then crosses a wall straight to the listener was lost: exactly the specular share of the leaked energy (half at s = 0.5).
The fix connects the branch's chain when it is resumed.
A second, smaller loss is fixed too: a transmission through a single sheet between air and air (no far face to exit) was treated like refraction and ended the chain's connections.
The connection roulette through walls was not the cause: its probability is already the largest transmission over the bands and every band is divided by it (traced always 0.88e-3, with the roulette 0.91e-3).
The energy method's +12% at s = 0.5 is most likely its 0.5 m receiver sphere next to the wall (it averages the field over a sphere whose near side sees more of the wall); not investigated further.

### The Church: fixed points, same window

`MeasureChurchLevels` (`ChurchLevelMeasurement.cs`) solves both methods at three Church points (doors closed) with 8,192 rays, a 2 s window and air absorption, and splits the energy into 0–50 ms, 50–200 ms and 0.2–2 s.
The energy method uses its listening settings (2 m receiver sphere, arrivals, 96 near planes).
Path minus energy (dB), at 125 / 500 / 2,000 Hz for each window:

| Point | path, 0.25 m region (as rendered) | path, 2 m region, three seeds |
|---|---|---|
| crossing, 10 m from the organ | −2 / −2 / −7 | 0 to +1 / 0 to +4 / −1 to −3 |
| far nave, 48 m | – / −8 / −2 to −7 | – / −5 to +7 / −3 to 0 |
| outside the closed doors | – / −30 to −45 / −20 | – / −2 to +2 / −18 to −24 |

**Inside: variance, not bias.**
With a 0.25 m region, 8,192 rays found 3–16 candidate specular chains in the whole 31,000 m³ church per solve (2–8 scored), against 1,368 receiver-sphere hits of the energy method with its 2 m sphere: the specular part of the field (stone scatters only 12–17% per bounce) was almost unsampled, so a single solve is low most of the time and rarely very high.
The region size only sets variance (the estimate divides by its measured probability), so with the energy method's 2 m the path tracer gets 1,060–1,190 candidates and agrees within about 1 dB at the crossing in every window and within the seed spread in the far nave; without survival in time, eight seeds give 0.088 ± 0.010 against the energy method's 0.082 for the crossing's late energy at 125 Hz.
The Church renders therefore use a 2 m region (as the energy method's listening render did); the default stays 0.25 m for the small suite rooms.
The region should scale with the room's size; that is not automatic yet.

**Outside, 50–200 ms: a sampling fix.**
That window is sound diffracted around the doorway's edges; the energy method enumerates every edge exactly, and the path tracer drew one edge uniformly out of 14,075 for each source connection, so two seeds of three missed it entirely.
The edge distribution of a solve (used by the source's edge rays and now by its edge connections too) now weights each edge by its angle and by the integral of 1/m² along it from the source *and* from the listener, with 0.1 for each end that cannot see it; every edge keeps a positive weight, so it stays unbiased.
Over eight seeds the path tracer now gives 2.77e-5 ± 0.44e-5 against the energy method's 3.22e-5 (125 Hz), within 1.1 standard errors.

**Outside, 0.2–2 s: not resolved.**
Both methods are dominated by rare large samples here (sound that leaks through the closed doors late): over eight seeds, at 125 Hz, the energy method gives 7.6e-5 ± 4.6e-5 (single seeds from 2.0e-6 to 3.7e-4) and the path tracer 0.9e-5 ± 0.7e-5 (3.0e-5 ± 2.8e-5 without survival in time).
The path tracer is lower in every band, by 1.5–2 standard errors; that is not significant with these spreads, and both are 20 dB or more below the inside level.
More seeds, or a better estimator for sound behind a partition, are needed to decide it.

All path checks pass with these changes (free field, rigid plane, image lattice, half-plane, walls, Eyring, the clipped-region image, the two leak checks, the synthesis check).
On the 11-scene wave suite (2,048 rays, two seeds) the level error falls from 4.03 to 3.52 dB with these fixes (in shadow 5.84 → 5.05 dB, from the edge distribution; closed rooms 1.97 dB, unchanged).

## 4. The Church walk with the path tracer

`church.py` (beside this file) renders the energy listening render's speech walk (`.context/energy-audio/church/speech-walk/take`) with the Path method: the same DSP as the energy render (spectral synthesis, FFT 1,024, 1,024-sample updates, binaural), a 6 s horizon, a 2 m detection region (part 3), and 512 rays (realtime budget, raw) or 8,192 rays (clean).
The page is `.context/energy-audio/listen/path.html` (both path renders beside the energy render and the static-IR baseline, one playback gain); the live view (`.context/live-view`) has "Path (realtime)" and "Path (clean)" buttons playing the same files.

### A synthesis loss, found and fixed

The first realtime render was 4 dB quieter than the energy render.
Part of it was the update crossfade: `PathReverb` crossfades linearly between the old and new response over each update, and its impulses keep their signs between updates, so a crossfade between two equal responses keeps the power; but two independent noisy estimates differ, and a linear crossfade between unequal amplitudes loses power.
From the realtime render's exported histograms the expected loss was a factor 0.75 (−1.2 dB) in every part of the walk.
The crossfade now scales each block of the blend so that its power is the linear interpolation of the two outputs' powers (measured over the block, with their cross term); equal responses keep a gain of one.
New Core check `PathReverbCrossfadeKeepsThePower`: updating every step between two histograms that split a late field over disjoint bins gives 0.70 of the fixed mean's output energy with the plain crossfade (−1.55 dB) and 1.001 with the correction.
The renders below use the corrected crossfade.

### Levels of the walk

Level of each render (dBFS, both ears, render gain), per part of the walk:

| Render | standing and turning (0–5 s) | walking to the organ and back (5–13 s) | doors open, outside (13–18 s) |
|---|---:|---:|---:|
| energy method | −43.8 | −43.4 | −51.9 |
| path, 512 rays | −45.1 | −46.6 | −60.9 |
| path, 8,192 rays | −45.4 | −45.3 | −60.5 |

The clean render is as loud as the realtime one (within 0.4 dB standing and outside), so the remaining difference from the energy render is not noise.
Inside the church the path renders are 1.6–1.9 dB (clean) quieter than the energy render; the static impulse responses at fixed points (below) agree in level within 0.4 dB at the crossing, so part of the walk's difference is probably the rendering of early arrivals (the energy method renders its 96 strongest arrivals as paths through the full HRTF; the path method renders everything from the histogram with diffuse-field ear weights), not the transport; this is not verified.

**Outside: a 10–12 dB difference in the late field, not resolved.**
At the walk's outside position at 16.5 s (listener at (12.0, 2.7, −46.3), `MeasureChurchLevels` with the new `ACOUSTIC_CHURCH_LEVELS_SAMPLE` and `POINTS=take`, 8,192 rays, four seeds each):

| 125 / 500 / 2,000 Hz | 50–200 ms | 0.2–2 s |
|---|---|---|
| energy, doors open | 1.31e-5 / 3.20e-6 / 6.86e-7 | 5.7e-4 / 2.9e-4 / 1.4e-4 |
| path, doors open | 0.94e-5 / 2.30e-6 / 4.93e-7 (one seed of four missed it) | 3.2e-5 / 2.1e-5 / 1.4e-5 |
| energy, doors closed (two seeds) | 1.31e-5 / 3.20e-6 / 6.86e-7 | 1.1e-4 / 3.8e-5 / 1.1e-5 |
| path, doors closed (two seeds) | 1.32e-5 / 3.22e-6 / 6.89e-7 | 1.0e-5 / 2.6e-6 / 0.7e-6 |

The diffracted sound (50–200 ms) agrees, and both methods respond to opening the doors; the late field outside is 10–13 dB lower with the path tracer in both door states.
Neither number is a sampling accident: the four energy seeds spread by a factor of two, the path seeds by 30%; the energy method gives the same within its noise with a 0.5 m or 1 m receiver sphere (3.4e-4 and 8.7e-4 at 125 Hz), and the path tracer the same with a 0.5 m region (2.5e-5), so its outside late field comes from next-event connections, not detected chains.
One of the two methods is biased for sound that leaves the building late; an analytic aperture case (a diffuse room with an open hole, the τ = 1 limit of the leak check) would decide which, and is the next check to write.
Until then the outside part of the path renders is not trustworthy, and neither is the energy method's.

### Against the scoreboard

Static impulse responses at the realism page's two points (doors closed, clean path settings: 8,192 rays, 2 m region; `church.py irs`, `church.py metrics`), now rows of [the scoreboard](../energy-audio-2026-11-09/scoreboard.md) (T30 s / EDT s / C80 dB):

| Response | level | 125 Hz | 500 Hz | 2,000 Hz | 8,000 Hz | echo density (150 ms) |
|---|---:|---|---|---|---|---:|
| church-near, energy | −19.9 dB | 6.8 / 5.1 / −0.1 | 6.2 / 5.2 / −2.2 | 5.0 / 4.2 / −0.1 | 2.3 / 1.1 / +4.8 | 0.78 |
| church-near, path | −19.5 dB | 6.3 / 3.0 / +1.7 | 6.3 / 4.4 / +1.5 | 4.8 / 4.6 / +1.4 | 2.9 / 0.7 / +5.6 | 0.30 |
| church-far, energy | −26.1 dB | 5.0 / 7.9 / −1.9 | 5.0 / 7.5 / −9.1 | 4.9 / 6.1 / −5.9 | 3.1 / 3.2 / −1.0 | 0.85 |
| church-far, path | −23.4 dB | 4.5 / 14.0 / −3.0 | 3.9 / 16.1 / −3.9 | 4.7 / 5.4 / −1.6 | 3.0 / 1.3 / +2.2 | 0.36 |
| St Paul's 9.5 m (measured) | | 9.0 / 6.9 / −4.7 | 10.7 / 6.9 / −1.8 | 7.0 / 3.5 / +6.1 | 2.0 / 0.1 / +13.0 | 0.79 |
| York Minster 24 m (measured) | | 7.8 / 7.4 / +0.7 | 8.2 / 8.1 / −3.6 | 5.8 / 5.0 / +0.1 | 1.4 / 1.3 / +7.2 | 0.91 |

T30 agrees with the energy method within 0.5 s at the crossing and 1.1 s in the far nave.
At the crossing the path response has more early energy (C80 1.5–3.6 dB higher, EDT shorter), and in the far nave its EDT is unreliable (10–16 s, longer than its T30): one 8,192-ray solve at 48 m is still noisy in the first 100 ms (part 3 measured −5 to +7 dB seed spread at 500 Hz in 50–200 ms there).
The echo density is 0.30–0.36 against 0.78–0.91 for the energy method and the measured halls: `PathReverb` places one impulse per 0.25 ms slot, which at 44.1 kHz is one sample in 11, and such a train measures about 0.3 by construction (normal noise measures 1).
That is a property of the synthesis, not of the transport; denser impulses per slot (several per slot sharing the slot's energy) would remove it, at the cost of more kernel spectra.
On the wave suite the scoreboard now lists the path method too (2,048 rays): level error 3.52 dB against the baseline's 8.85 and the energy method's 3.04.

### Cost

The realtime render's solves took 230 ms per update on average (238 s for 1,034 updates; 3.4 s worst), the clean render's 2.2 s, under 50% pacing and with the 6 s horizon and the 2 m region.
That is 3–6 times part 2's profile of the Church at 512 rays (40–70 ms, unpaced, 2 s horizon, 0.25 m region); the pacing accounts for a factor of two, the longer horizon and the larger region (about 100 times more chain candidates) for the rest.
At 512 rays the Church is not realtime in this configuration.

## 5. A denoiser

### Data

`tools/path_dataset.py --name path-v2` (plan, render, church, pack, verify) built 48 generated scenes (eight per family: shoebox, doorway, corridor, canopy, barriers, furnished; `tools/dlss_scenes.py`) plus the Church walk, 600 listener poses in all.
Each box scene has four static receivers and an eight-frame walk at 1.4 m/s (one frame per 23 ms update); the Church contributes 24 frames of the speech walk (standing, walking, outside), rendered with the 2 m region of the Church renders.
Inputs: the Path method at 256, 512, 1,024 and 2,048 rays, four seeds each (every walk frame its own seed, as live solves are).
Splits are whole families, by the frozen rule of `dlss_scenes.split_for`: corridor is validation, canopy (open) and the Church are test, and two doorway and two furnished scenes fall in test and validation; training has 28 scenes of the other families.
`verify` checks 1,715 payload hashes and that each split group sits in one split.
Rendering took about 75 minutes of guarded Unity time (one 4 GiB guard trip after 816 of 864 runs; the rest ran in batches of eight).

**Targets: more rays, not coarser bins.**
The target of a pose is the mean of two independent 8,192-ray solves (16,384 rays, 32 to 64 times the realtime inputs).
It is still noisy per fine bin (25% energy-weighted median, against 124% for one 512-ray input), but it is unbiased, and the loss is L2, so Noise2Noise applies: the minimiser is the conditional mean of the target, which is the clean answer because the target's noise has zero mean and is independent of the inputs.
Coarser target bins would have removed the noise only by blurring the early arrivals that the denoiser must keep in place; summing into perceptual windows happens in the loss instead (below), where it costs no resolution in the output.

### Model and training

`tools/path_denoise.py`: a 1D dilated convolutional network along the time bins (232,000 parameters, eight convolutions of width 96, receptive field 67 bins) on the last four frames of a pose (log energies of 6 directions × 8 bands, and log contribution counts, normalised by the four frames' total).
It predicts each cell as a gain on the four-frame mean plus a non-negative multiple of a smoothed level (so it can fill cells no frame reached), and then rescales every band's energy in each time segment (0–50 ms, 50–200 ms, later) to the four-frame mean's.
Those totals are therefore unbiased, the band levels are exactly the plain temporal average's, and the network only moves energy within a band and segment.
The loss is L2 on energies summed into perceptual windows (1 ms up to 16 ms, then about t/8; per direction and band), each divided by an input-only normaliser; one model is trained for all four budgets, with the eight rotations and reflections about the vertical axis as augmentation; the checkpoint with the best validation window error is kept (step 1,250 of 8,000; 3 minutes on the M5's GPU through MPS).
Inference takes 1 ms per pose on one CPU core.

**What did not work, in order.**
A per-fine-bin L2 loss (each bin divided by a smoothed input level) did not train: it was dominated by single specular arrivals that no input frame had detected, up to 10⁴ times the local level, and the validation loss stayed flat.
A gain on the smoothed level alone, with window loss, learned but overfit within 500 steps and lost 3–7 dB of level in the open canopy and the Church (held-out families); at 2,048 rays it was worse than the plain four-frame mean everywhere.
The residual form (identity is the starting point), training on all budgets, the symmetry augmentation and the per-segment energy rescaling fixed the level and most of the overfitting.

### Results on the held-out families

Errors against the target, mean over 264 validation and test poses (window dB / band level dB / decay curve dB; definitions in `metrics()`); "equal compute" spends the denoiser's time on rays instead (1 ms against 170–400 ms per guarded solve here, or against part 2's 13 ms per update at best: at most 8% more rays, under 0.2 dB on these curves):

| Rays | (a) one frame | (b) more rays, equal compute | (b′) four-frame mean | (bK) four frames at equal compute | denoised | (c) one 8,192-ray seed against the target |
|---|---|---|---|---|---|---|
| 256 | 9.05 / 2.34 / 5.79 | 9.11 / 2.31 / 5.84 | 6.77 / 1.67 / 4.25 | 6.48 / 1.66 / 3.92 | **5.48 / 1.67 / 3.51** | 1.61 / 0.43 / 0.90 |
| 512 | 7.67 / 1.86 / 4.58 | 7.77 / 1.96 / 4.74 | 5.62 / 1.34 / 3.26 | 5.33 / 1.39 / 3.15 | **4.77 / 1.34 / 3.00** | 1.61 / 0.43 / 0.90 |
| 2,048 | 5.35 / 1.36 / 3.12 | 5.34 / 1.39 / 3.16 | 4.00 / 0.96 / 2.33 | 3.63 / 0.89 / 2.06 | **4.18 / 0.96 / 2.68** | 1.61 / 0.43 / 0.90 |

At 256 and 512 rays the denoiser beats more rays at equal compute, both for one frame (b) and for the strict bar of four frames (bK): at 512 rays its window error (4.77 dB) is what plain averaging reaches with about 3,000 rays, 1.5 times the four frames' 2,048.
Its band levels equal the four-frame mean's by construction, about as good as (bK); its decay curves are better at 256 and 512 rays.
At 2,048 rays it is worse than plain averaging on every metric: the inputs are already good there, and the network's redistribution adds more error than it removes.
So it is useful at the realtime budgets only, and below 2,048 rays it should be switched off or retrained for that budget.

Per family at 512 rays (window dB / decay dB), four-frame mean → denoised: corridor 5.12 / 2.64 → 4.16 / 2.48; canopy 6.40 / 4.63 → 5.58 / 3.75; doorway 3.92 / 1.63 → 3.37 / 2.04; furnished 3.85 / 1.35 → 3.09 / 1.60; Church 7.93 / 3.78 → 7.12 / 4.44.
The window error falls in every family; the decay curve gets worse in the doorway, furnished and Church poses, where it smooths the late decay of single frames more than the targets support.

### Against the wave ground truth

`ground_truth.py` (beside this file) solved the 11 suite scenes at 512 rays with four seeds, ran the denoiser on each receiver's four solves (as a static listener's last four frames), and scored the variants with `tools/method_compare.py` against the wave ground truth (RMS over receivers and bands, as on the scoreboard; the suite's 2,048-ray solves are the same rays in one solve, i.e. bK):

| Variant | level dB | EDT % | T30 % | ETC dB |
|---|---:|---:|---:|---:|
| path, 512 rays, one solve | 4.96 | 293 | 77.1 | 20.5 |
| path, mean of four 512-ray solves | 3.45 | 466 | 68.2 | 18.4 |
| path, 2,048 rays, one solve (equal rays) | 3.52 | 435 | 72.0 | 18.3 |
| path, four 512-ray solves denoised | 3.45 | 416 | 64.6 | 14.8 |

Against the wave solution the denoised estimate is better than the same rays spent in one solve on decay time and on the energy-time curve (3.5 dB better), and equal in level (the four-solve mean's by construction).
EDT errors are large for every variant (and C50/C80 are not meaningful on this suite: their RMS runs to thousands of dB where a receiver's early energy is nearly zero; the scoreboard shows the same for every method).
One suite scene shares its geometry with a dataset scene (furnished-0005, with the presets' scattering in the dataset and zero scattering in the suite); that dataset scene is in the validation split, so it was not trained on but did help pick the checkpoint; the other suite scenes are distinct instances, and corridor and canopy are held-out families.

### The Church walk, denoised

The realtime render's exported histograms (512 rays, one per update) went through the denoiser with the three previous updates as context (`path_denoise.py church`), and `church.py render path-denoised` rendered the walk from them (`pathHistogramImport`); it is on the page and in the live view as "Path (denoised)".
The frame-to-frame change of the 500 Hz band energy, median over the walk, falls from 4.2 to 0.7 dB in 50–200 ms and from 3.8 to 0.7 dB after 200 ms; the plain four-frame running average gives the same 0.7 dB (consecutive averages share three of their four frames), so on this measure the improvement is the temporal averaging, not the network.
Its levels per part of the walk are −44.7, −46.8 and −61.3 dBFS (realtime −45.1, −46.6, −60.9; clean −45.4, −45.3, −60.5): the denoiser keeps the level, and so keeps the outside difference of part 4.
The Church is a held-out family here and its decay curves got worse than the plain four-frame mean's (above), so whether the denoised walk sounds closer to the clean one is for listening, not established by these numbers.

### Not done

The denoiser is trained once, on one seed of training, for one model size; there is no hyperparameter search, and validation picks the checkpoint from a single run.
The training set is small (28 scenes), and the best checkpoint comes early (step 1,250): more scenes are the most likely improvement.
