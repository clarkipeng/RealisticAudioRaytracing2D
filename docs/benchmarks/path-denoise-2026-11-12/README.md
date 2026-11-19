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

## 6. The late field outside the Church

Part 4 left the path tracer 10–13 dB below the energy method in the late field (0.2–2 s) at the walk's outside position (16.5 s, listener at (12.0, 2.7, −46.3), 1.7 m above the courtyard, doors open).
All numbers here are 125 Hz energies in that window from `MeasureChurchLevels` (8,192 rays per solve, 2 m region or sphere unless stated); 500 and 2,000 Hz behave the same.

### 1. An aperture check

New fixture (`ApertureRoom`, `PathChecks.cs`): the leak check's diffuse room (6 × 5 × 4 m, a = 0.1, s = 1) with a 1.2 m × 1.0 m hole in one wall and the listener 0.8 m outside it, the τ = 1 limit of `MethodsLeakAsTheDiffuseFieldPredicts`.
The prediction is the inside reverberant energy times the hole's solid angle over 4π.

| Case | energy method | path method |
|---|---:|---:|
| in view of the hole: outside / prediction (`MethodsRadiateThroughAHoleAsTheDiffuseFieldPredicts`) | 1.151 | 1.150 |
| in view, with a scattering ground outside (a = 0.3, s = 1) | 0.3227 | 0.3228 |
| behind an absorbing screen, only sound scattered by the ground (`MethodsAgreeOnSoundScatteredOutsideAHole`) | 3.25e-6 | 3.71e-6 |
| the same with transmitting walls (t = 0.01) | 3.69e-5 | 3.80e-5 |
| behind a screen raised 1 m over a specular ground (a = 0.01, s = 0), 12 seeds, radius 0.25 m / 1 m | 2.03e-3 / 1.61e-3 | 1.56e-3 / 1.87e-3 |

The two methods agree within 0.2% in view and within 15% behind the screen; both are 15% above the diffuse-field value in view (the field near a hole is not perfectly diffuse).
The first and third rows are now tests (Slow tier).
With one specular bounce the two agree within their spread (the energy method's sphere average moves by 25% with its radius there, so neither is a reference to better than that).

### 2. Refraction into media

Every Church material except glass has an acoustic index (0.064 for stone, 0.095 for wood), so its solids are media, not partitions, and a transmission into them refracts.
Stopping such rays (diagnostic bit 5, counted) found 75 per 8,192 rays: refraction from air into stone is total reflection beyond 3.6° from the normal, so almost every transmitted share comes back as a reflection.
The outside late field did not change, and setting every index to one (`ACOUSTIC_CHURCH_LEVELS_IOR1`) did not change it either: these chains carry no outside energy.

### 3. Where the outside late energy comes from

**Transmission: nowhere.**
Making materials opaque one group at a time (`ACOUSTIC_CHURCH_LEVELS_OPAQUE`: glass, door leaves, stone and vault and floor, copper roof, wood and pews, ground) left the median of either method unchanged.
The listener sees no surface inside the church (a ray cast from it: courtyard floor 5.2 sr, sky 4.5 sr, facade 1.9 sr, meadow 0.9 sr, door frame and leaves 0.09 sr); the door opens into a porch 7.6 m deep, and all the outside late sound leaves through that opening and then reaches the listener scattered or reflected by the courtyard, the facade and the porch.
Some of these runs first looked like large effects (all materials opaque: path ×5): they are not. Both estimates are heavy-tailed here, and changing any transmission changes the random number stream, so a different rare large sample lands in a different seed; drawing transmissions instead of keeping reservoir branches (bit 10) gives nearly the all-opaque values seed by seed (the same rare samples), and dropping the kept branches leaves the baseline unchanged (2.69e-5 against 2.73e-5).

**By estimator (no diffraction, so edges are out of it):**

| Part | energy method | path method |
|---|---|---|
| straight connections from scattering points (energy: 0.01 m sphere, 8 seeds; path: median) | 1.05e-5 | 1.0e-5 |
| everything (energy: track-length sphere for specular segments; path: chain detection) | 2.54e-4 ± 0.19e-4 (2 m, 64 seeds); 2.61e-4 ± 0.77e-4 (0.5 m, 64 seeds) | 8.8e-5 ± 1.2e-5 (2 m, 512 seeds); 8.0e-5 ± 2.4e-5 (4 m, 128 seeds) |
| median of one solve | 2.35e-4 (2 m) | 1.2e-5 |

The straight connections agree.
The difference is the specular part: the energy method's sphere gives 2.4e-4 with any radius from 0.5 to 2 m, the path tracer's chain detection 7.7e-5 with a 2 or a 4 m region, three times less (4.6 dB) and many standard errors apart, though the path estimate is so heavy-tailed (the top 5 of 512 solves carry 23% of its sum, and its mean rose from 3.8e-5 at 64 solves to 8.8e-5 at 512) that its error bar is optimistic.
A single path solve almost never scores an outside specular chain: of 41 candidates in eight 32,768-ray solves with a 0.25 m region, none scored (51% failed the exact path's last leg, 37% followed a different plane sequence, 10% missed the listener); with a 1 m region 12 of 1,194 scored.
So what the realtime render plays outside is the straight-connection part, 13 dB below the energy method.

### What this settles, and what not

The aperture checks show both methods right for sound leaving a room through an opening, both straight to the listener and scattered on the way; refraction into media and transmission through walls, doors and glass play no part outside the Church.
The remaining 4.6 dB is between the two specular estimators, and I could not decide it.
Both are unbiased in construction and each is independent of its own radius, which points at the path tracer, whose candidates almost all fail the exact retrace even with a region of 0.25 m; but the candidate test is directional only (a ray whose unfolded line passes near the listener but which hits something first is a candidate whose exact path is rightly blocked), so a high failure rate alone is not a bug.
The next check is a controlled exterior case with two or more specular bounces (a porch in front of the hole), where one of the two must disagree with the other, and then, if it is the path tracer, the energy method's sphere hits tagged with their plane sequence to see which exact paths the retrace rejects.
Nothing was fixed in either method.
I tried grouping coplanar triangles with a tolerance instead of rounded keys (a split plane would make a chain fail its sequence test); it changed nothing here and was reverted.

## 7. A controlled porch

### The case

The aperture room of part 6 (6 × 5 × 4 m, a = 0.1) with its 1.2 m × 1.0 m hole opening into a porch: a duct of the hole's own cross-section, 3 m deep, whose floor, roof and two side walls are specular (a = 0.05, s = 0).
The listener stands 4 m beyond the mouth and off its axis, at (10, 2.1, 2.2), out of view of the hole: every ray it sends back through the mouth meets the porch walls 1 to 5 times (2.97 on average, energy-weighted) before it reaches the hole.
The porch walls are parallel to its axis, so they return no sound into the room (the inside energy is the same with and without them, to the last digit).
Fixture and reference: `ApertureRoom(porch: true)`, `PorchReference`, `MeasurePorch` (opt-in) in `PathChecks.cs`.

**Reference.**
Unfolded, the porch walls form a rectangular lattice of images of the hole, and the mouth is the window through which the listener sees them.
The reference integrates over the mouth the radiance through the hole times (1 − a)^k, k the reflections of each unfolded ray, plus the source's own image paths through the hole and porch (there are none at this listener).
The radiance through the hole is κ E_in / 4π, E_in the inside reverberant energy, with κ measured in view of the bare hole in the porch's direction (38° off its axis; both methods agree there within 0.1%).
κ is 1.04 there and 1.15 on the axis, so the radiance is not isotropic and the reference, which uses one κ for all the porch's directions, is good to about 5%.

### Diffuse room (s = 1): both right

128 seeds per method, 65,536 rays per solve, 0.5 m region or sphere; ratio to the reference, with the running mean by doubling:

| Seeds | energy method | path method |
|---:|---:|---:|
| 16 | 1.069 ± 0.039 | 1.049 ± 0.035 |
| 32 | 1.063 ± 0.026 | 1.069 ± 0.029 |
| 64 | 1.045 ± 0.017 | 1.086 ± 0.023 |
| 128 | 1.050 ± 0.013 | 1.089 ± 0.016 |

Both converge 5–9% above the reference, within its own uncertainty, and within 3.7% (1.9 standard errors) of each other.
Neither is heavy-tailed here: the standard deviation of one solve is 14% (energy) and 17% (path) of the mean, and the largest of 128 solves is 1.4 and 1.6 times the mean.

### Specular room (s = 0.12, like the Church's stone): the energy method is biased

With a mostly specular room the radiance through the hole is no longer diffuse, so there is no reference; but each method's estimate must not depend on its own radius, and that decides it:

| Radius | energy method (sphere) | path method (region) |
|---:|---:|---:|
| 0.25 m (64 seeds) | 0.00951 ± 0.00031 | 0.00932 ± 0.00032 |
| 0.5 m (128 seeds) | 0.00826 ± 0.00009 | 0.00939 ± 0.00017 |
| 1 m (64 seeds) | 0.00705 ± 0.00006 | 0.00931 ± 0.00019 |

The path tracer gives the same value with every region (its mean is flat by doubling: 1.610, 1.569, 1.566, 1.598 in units of the diffuse reference at 16, 32, 64, 128 seeds).
The energy method's sphere average falls below the path tracer's by 12% at 0.5 m and 24% at 1 m and approaches the path tracer's value as the radius shrinks.
The field behind the porch is a set of beams (the images of the hole seen through the mouth), and a sphere averages a beam with its darker surroundings; the exact connection does not.
So in this case the energy method is biased low by its sphere and the path tracer is unbiased, with a moderate variance (21% per solve at 65,536 rays, largest of 128 solves 1.5 times the mean).
No path-tracer chains are rejected wrongly here, so there was nothing to tag.

### Back to the Church

The porch shows the mechanism that can bias the energy method near specular beams, but not its sign at the Church listener: a sphere centred in a beam reads low (the porch), one centred beside a beam reads high.
At the Church's outside listener (no diffraction, 125 Hz, 0.2–2 s) the energy method gives 2.54e-4 ± 0.19e-4 with a 2 m sphere (64 seeds), 2.61e-4 ± 0.77e-4 with 0.5 m (64) and 2.2e-4 ± 1.3e-4 with 0.25 m (128 seeds; the top 5 of 128 solves carry 93% of the sum); the path tracer gives 8.8e-5 ± 1.2e-5 (2 m region, 512 seeds) and 8.0e-5 ± 2.4e-5 (4 m, 128).
Below 0.5 m the energy method becomes as heavy-tailed as the path tracer, so its radius test cannot be carried to the radius where a beam would show; the two remain 4.6 dB apart and the Church is still not decided by measurement.
What is decided: the path tracer is unbiased in a controlled multi-bounce porch, where the energy method is not; and at the Church its estimate is heavy-tailed (one solve's median is 7 times below its mean, 13 dB below the energy method), which is what the realtime render plays.

New slow-tier checks: `MethodsSeeThroughAPorchAsTheImageReferencePredicts` (both methods within 20% of the reference; 1.05 and 1.04 with 8 seeds) and `PathPorchEstimateIsIndependentOfTheRegion` (specular room, 0.25 m against 1 m; 0.983).

### Sketch: a lower-variance estimator for specular arrivals (not implemented)

The path tracer is unbiased but impractical where the listener receives most of its late sound by specular chains through small openings: an update scores such a chain rarely, with a large weight.
The proposal combines three estimators of the same specular arrivals, weighted by multiple importance sampling as in vertex connection and merging (Georgiev et al. 2012, *Light transport simulation with vertex connection and merging*):

1. **Connection (as now).** A source-side chain whose unfolded ray passes within r of the listener image is retraced exactly and divided by its region probability times the followed share. Density p_c = (region solid angle × walk density at the chain's first vertex) × share; unbiased.
2. **Merging.** The same walk segments scored by a small receiver sphere (radius r_m, e.g. 0.25 m), as the energy method does, with density p_m = π r_m² × the segment's density per unit area at the listener. Biased by the sphere average (within 2% of the exact value at 0.25 m in the porch above, 12% low at 0.5 m), low variance.
3. **Listener subpaths by reciprocity.** Trace a few hundred rays from the listener (energy transport is reciprocal), follow them through specular events to their first scattering vertex y, and connect y to the source-side scattering vertices x of the same update by straight connections (as the existing next-event connections do, with transmission). A path source → … → x → y → specular chain → listener is then found from both ends: the listener subpath carries the specular chain exactly (a ray from the listener follows it with probability one), so the arrivals that the forward walk scores rarely become a connection of two diffuse vertices.

Each full path is reachable by all three techniques; with the balance heuristic, its weight for technique i is p_i / Σ p_j, with every p_j evaluated for that path (the forward chain's region density, the sphere's area density, and the listener subpath's density of reaching y times the connection's).
The merged estimator keeps the result consistent rather than unbiased; a fixed small r_m bounds the bias (the porch measures it), and r_m can shrink with the update count for static listeners.
In the porch case technique 3 alone would score every listener ray that leaves through the mouth; in the Church, every listener ray that reaches the porch or the nave through the doorway.

**Cost estimate**, against today's Church update at 512 rays (40–70 ms unpaced in part 2):
- listener subpaths: 256 rays to their first scattering vertex, mostly specular steps without connections, about 10–15% of a forward update;
- connections: one shadow ray per listener vertex to one sampled source-side vertex per subpath pair (a buffer of the update's ~5,000 source-side scattering vertices, sampled by their flux), with transmission when blocked: as many shadow rays as today's next-event connections of 256 rays, about 25–35%;
- merging: a hash grid of the update's walk segments that pass within r_m of the listener (tens per update), negligible;
- MIS weights: each technique's density must be evaluated for paths found by the others, which needs the forward region probability for listener-found chains (one cone test) and the listener-subpath density for forward-found ones (stored per vertex), a few percent.
In total about 1.4–1.6 times today's update, or the same cost at about 350 forward rays plus 160 listener rays.
The gain cannot be estimated without building it; the porch and Church fixtures are the tests it would have to pass (unchanged means in the porch, a stable median at the Church listener).

## 8. Wide beams instead of the exact specular connection

### The rule

Geometric acoustics cannot resolve the field more finely than the width of a sound beam, so the Path method now gives every ray a physical width instead of connecting its specular chains exactly.
After a ray's first specular reflection, every segment of it scores at the listener with a normalised two-dimensional Gaussian kernel of its distance ρ from the listener:

K(ρ) = 2 / (π w²) · exp(−2 ρ² / w²), with ∫ K dA = 1.

ρ is measured from the segment's line (extended) to the listener, which in the unfolded space of the chain is the distance from the straight ray to the listener's image; each image is scored once, by the segment that belongs to it, whether or not the closest approach falls inside that segment.
The deposit's time is the image's: the chain's origin time plus sqrt(L_c² + ρ²)/c, L_c the unfolded length to the closest approach.
Every band applies its own kernel to the same geometric ray.
The direct sound stays exact; scattering points and the source keep their next-event connections (straight, through walls, and through edges); UTD edge rays and edge connections are unchanged.
The kernel does not score the first segment after the source, a scattering point or an edge, which those connections already count, so each path is counted once.
Detection regions, the geometric trials and their cap, the image retrace and its parallel connection pass (`PathConnect`) are deleted.

**Width.**
A Gaussian beam of waist w₀ has radius w(z) = w₀ sqrt(1 + (z/z_R)²), z_R = π w₀²/λ.
The narrowest beam that spans a path of length L is the one whose largest radius over the path is smallest: its waist sits at mid-path, and minimising w(L/2)² = w₀² + (λL / 2πw₀)² over w₀ gives w₀² = λL / 2π and an end radius w² = λL / π.
With the waist at one end instead (a beam leaving a source), the end radius would be sqrt(2λL/π); for comparison, the first Fresnel zone at mid-path has radius sqrt(λL/4) = 0.89 sqrt(λL/π).
So the brief's derivation is right for a beam that connects two regions of equal size (the source's and the listener's), and this is the width used: w² = λ L / π with the band-centre wavelength and L the unfolded path length from the chain's origin (source, scattering point or edge) to the listener's image.
No constant was tuned: K's 1/e² radius is w.
At 1 kHz, w is 0.8 m after 6 m and 1.5 m after 20 m; at 125 Hz, 2.3 m and 4.2 m.

**Two rules the first version needed, both from geometry.**
- *Truncation at the listener's clearance.* In the unfolded space a room is a lattice of copies, one per image; a ray that passes an image closer than the listener's distance R to its nearest surface has necessarily crossed that image's copy, but a ray farther out may belong to a neighbouring image and is never scored for this one. A kernel wider than R therefore lost energy: the shoebox image lattice fell to 0.33 of the reference by 120 ms. The kernel is now truncated at R and renormalised (divided by 1 − exp(−2R²/w²)); R is measured once per solve by 256 rays from the listener (`PathClearance`). With it the lattice is within 1.5% in every 20 ms window.
- *Visibility from the segment.* A kernel reaches through walls; each scored segment needs the listener visible from its point nearest the listener. That point is kept 4 RayEpsilon off the walls at both ends of the segment, measured perpendicular to them: a point within the wall's tolerance saw through it, and the leak check read 5.7 times its prediction until this was fixed.

### 1. Wave-ground-truth suite (the deciding test)

11 scenes, two seeds, RMS errors against the wave solution (`beams.py`; same scorer as the scoreboard).
Seconds per run are the guarded measurement's busy time for all receivers of a scene.

| Variant | s/run | level dB | EDT % | T30 % | C80 dB | ETC dB |
|---|---:|---:|---:|---:|---:|---:|
| exact connection, 2,048 rays | 2.67 | 3.52 | 435 | 72.0 | 2019 | 18.3 |
| beams, 2,048 rays | 2.12 | **3.27** | **226** | **65.7** | **1763** | **13.9** |
| beams, 8,192 rays | 8.04 | 3.09 | 217 | 67.5 | 1654 | 12.8 |

| Group | exact, 2,048: level / EDT / T30 / ETC | beams, 2,048: level / EDT / T30 / ETC |
|---|---|---|
| in shadow | 5.05 / 223 / 74.4 / 19.3 | 4.68 / 72 / 73.3 / 16.0 |
| closed rooms | 1.97 / 37.3 / 38.3 / 20.5 | 2.02 / 19.0 / 53.5 / 6.2 |

The beams are cheaper than the exact connection at equal rays (2.12 against 2.67 s), so the equal-rays comparison is also an equal-or-better-time comparison for them; at both they are more accurate on level, EDT, T30 and ETC overall and in shadow.
By scene the largest gains are the corridor (level 4.00 → 1.94 dB, EDT 431 → 32%) and the doorway (EDT 264 → 48%, ETC 22.1 → 13.8 dB); the energy-time curve improves in every scene, most in the closed rooms (20.5 → 6.2 dB), where the exact connection's late field was a few large spikes.
Worse: T30 in the closed rooms (38 → 54%) and in three scenes (furnished room 60 → 88%, the two barrier scenes 63 → 79% and 61 → 66%), and the level in one shoebox (2.12 → 2.34 dB); at 8,192 rays the closed-room T30 is 42%, so most of it is noise in the late decay at 2,048 rays, not bias.
C80 stays meaningless on this suite (thousands of dB where a receiver's early energy is nearly zero), for every method.

### 2. Analytic checks

With the beams every Path check of the Methods category passes: free field, rigid plane (image 1.013, total 1.005), shoebox image lattice (within 1.5%), half-plane (+0.30 dB), far side of a solid, Eyring (T60 0.606 s against 0.585 s), the diffuse-field leak checks (0.994 and 1.058), the aperture (1.151, as before) and the porch (1.040), and `MethodsAgreeOnSoundScatteredOutsideAHole`.
`PathDetectionIsUnbiasedAtVisibilityBoundaries` tested the trials and is deleted with them; in its fixture (a panel clipping half of a specular image's neighbourhood) the beams give 0.63 of the geometric image, which is their smoothing across the panel's edge, as intended.
`PathPorchEstimateIsIndependentOfTheRegion` is deleted too: there is no region.

### 3. Porch

Diffuse room: 1.040 times the image reference (exact connection 1.089, energy method 1.040), inside the reference's 5%.
Specular room: 0.00685 against the exact connection's 0.0093 (−26%), close to the energy method's 1 m sphere (0.00705).
That is Fresnel smoothing and plausibly the physical answer: the "beams" behind the porch are images of a 1.2 × 1.0 m opening seen through a 1.2 × 1.0 m mouth 4–7 m away, a Fresnel number a²/λd of about 0.1 at 500 Hz, so at that frequency the geometric beams do not exist and the wave field is smooth across them.

### 4. The Church outside

The walk's outside position at 16.5 s, 125 Hz, 0.2–2 s, 64 seeds of 8,192 rays (`MeasureChurchLevels`):

| | without diffraction | with diffraction |
|---|---|---|
| energy method (2 m sphere) | 2.54e-4 ± 0.19e-4 | 9.1e-4 ± 3.8e-4 (8 seeds) |
| exact connection | 8.8e-5 ± 1.2e-5 (512 seeds), median 1.2e-5 | 2.0e-4 ± 1.3e-4 (16 seeds), median 2.8e-5 |
| beams | **2.55e-4 ± 0.43e-4**, median 1.20e-4 (0.47 of the mean) | 4.3e-4 ± 1.1e-4, median 3.9e-5 (0.09 of the mean) |

Without diffraction the beams agree with the energy method, and one solve's median is now half the mean instead of a seventh: the 4.6 dB gap of parts 6 and 7 is closed, and it was the exact connection's.
With diffraction the noise is back (median a tenth of the mean): the source's edge rays take half of its rays (`methodEdgeProbability` 0.5) and their outside contributions are heavy-tailed; the level is between the energy method's (whose own estimate is just as noisy at 8 seeds) and the exact connection's.
So the walk's outside sound is resolved for the specular part and not yet for the diffracted part.

### 5. Beams and UTD at a reflection boundary

A 2 m × 2 m rigid plate under the source and listeners on a line across its reflection boundary at x = 2 m (`MeasureReflectionBoundary`; energy without the direct sound, at 250 Hz / 1 kHz / 4 kHz):

| x | exact | exact + UTD | beams | beams + UTD |
|---:|---|---|---|---|
| 1.5 m (lit) | 0.171 / 0.171 / 0.171 | 0.213 / 0.179 / 0.163 | 0.125 / 0.147 / 0.163 | 0.183 / 0.168 / 0.165 |
| 1.9 m | 0.127 / 0.127 / 0.127 | 0.194 / 0.168 / 0.156 | 0.075 / 0.079 / 0.093 | 0.139 / 0.116 / 0.110 |
| 2.1 m (beyond) | 0 | 0.018 / 0.019 / 0.017 | 0.055 / 0.047 / 0.036 | 0.073 / 0.065 / 0.051 |
| 2.5 m | 0 | 0.009 / 0.007 / 0.003 | 0.028 / 0.011 / 0.001 | 0.037 / 0.018 / 0.004 |
| 3.0 m | 0 | 0.005 / 0.003 / 0.001 | 0.011 / 0.001 / 0.000 | 0.016 / 0.003 / 0.001 |

Summed over the line, beams + UTD carry 3% less energy than exact + UTD at 250 Hz: no net double counting.
Locally there is: just beyond the boundary the beams' smoothing and UTD's diffracted field add, 3–4 times exact + UTD at 250 Hz from 2.1 to 3 m (+5 to +6 dB, on a level 10–20 dB below the lit side), and 25–30% too little just inside it.
UTD already models the boundary's transition and the kernel smooths it again, over the beam's width; a principled fix would weight UTD's reflection-boundary term down where the kernel is wide, or narrow the kernel near edges; neither is done.

### 6. Cost per update

GPU time per update (walk and connections, one listener, fences included; `MeasurePathProfile`, the Church at the realtime render's settings):

| Scene | exact, 512 rays | beams, 512 | exact, 2,048 | beams, 2,048 |
|---|---:|---:|---:|---:|
| shoebox (0020) | 9.3 ms | 5.9 ms | 14.1 ms | 11.3 ms |
| shoebox (0010) | 11.9 ms | 10.5 ms | 25.6 ms | 26.2 ms |
| furnished room | 60 ms | 46 ms | 173 ms | 100 ms |
| Church (6 s horizon) | 248 ms | 270 ms | 609 ms | 658 ms |

The beams cost one visibility ray per scored segment and save the whole connection pass: cheaper in the furnished room (where the connections dominated) and the shoeboxes, 8–9% dearer in the Church, whose cost is the walk itself (rays that live for 6 s).

### Decision

The beams are at least as accurate as the exact connection on the wave ground truth, better on level, EDT, T30 and ETC overall and in shadow, and cheaper in rooms, so they are the Path method now; the exact connection's machinery is removed.
Open: the late decay's noise in closed rooms at 2,048 rays, the double counting next to reflection boundaries (section 5), and the heavy-tailed diffracted field outside the Church.

## 9. Echo density of the synthesis

`PathReverb` placed one impulse per 0.25 ms slot, which at 44.1 kHz is one sample in eleven: such a train measures about 0.3 on the normalised echo density (Abel and Huang), whatever the room, against 0.79–0.91 for the measured cathedrals.

**Change.**
Each slot now holds two fixed noise-like sequences (one per ear) instead of one impulse, scaled by the square root of the slot's histogram energy as before:
- Gaussian per sample, so a filled late field is Gaussian noise (normalised echo density near one), while an isolated arrival stays a 0.25 ms burst and an empty bin stays silent;
- the first sequence has area ±1 and the second area zero, both unit energy and orthogonal (Gram–Schmidt), so bands whose period is long against a slot see the slot's energy exactly, as from an impulse;
- per group of slots and per band, the two sequences are mixed by W = sqrt(target) G^(−1/2), where G is the Gram matrix of the band-filtered pair (computed exactly from the 2P-point spectra the convolution already uses) and the target is what an impulse train of the same energy would give through that band; so each band's energy is exact for every group, at any interaural coherence, and not only on average.

The plain version (Gaussian noise per sample, normalised per slot in broadband energy) failed the synthesis check: a 0.25 ms burst has a sinc-like spectrum and random detail, and a single arrival kept 0.73 of its 1 kHz band energy.
With the band-wise mixing `PathReverbReproducesTheHistogram` passes: through the comparison's band filters (not the synthesis's own), a single arrival keeps its band energy within 4% at 125–500 Hz and 4 kHz, +9% at 1 kHz and −13% at 2 kHz (tolerance 15%), and the late field within 10% at 125 Hz and 3% above; the crossfade check gives 0.998.

**Church static responses** (the realism page's two points; these are also the first with the beams of part 8, which change T30, EDT and C80, while the echo density is the synthesis's):

| Response | echo density (mean over 150 ms) | mixing time | T30 / EDT / C80 at 500 Hz |
|---|---:|---:|---|
| path, near, before (impulses, exact connection) | 0.30 | never | 6.3 / 4.4 / +1.5 |
| path, near, now | 0.76 | 66 ms | 6.5 / 5.1 / +0.6 |
| energy method, near | 0.78 | 41 ms | 6.2 / 5.2 / −2.2 |
| path, far, before | 0.36 | never | 3.9 / 16.1 / −3.9 |
| path, far, now | 0.77 | 72 ms | 5.0 / 5.4 / −5.0 |
| energy method, far | 0.85 | 50 ms | 5.0 / 7.5 / −9.1 |
| St Paul's 9.5 m / York 24 m / St Paul's 32 m (measured) | 0.79 / 0.91 / 0.89 | 41 / 35 / 35 ms | |

The echo density is now within 0.02–0.15 of the measured halls and the energy method; it takes 66–72 ms to reach 0.9 against 35–50 ms for them, because the early part of the histogram is sparse (single arrivals) for longer.

## 10. The Church walk again, with the beams and the new synthesis

**Renders** (`church.py`; same settings as part 4: 512 rays realtime with histogram export, 8,192 rays clean, 6 s horizon):
level in dBFS per part of the walk, with part 4's renders (exact connection, impulse synthesis) in brackets.

| Render | standing and turning | walking | doors open, outside |
|---|---:|---:|---:|
| energy method | −43.8 | −43.4 | −51.9 |
| path, 512 rays | −44.2 (−45.1) | −46.4 (−46.6) | −62.2 (−60.9) |
| path, 8,192 rays | −44.4 (−45.4) | −44.5 (−45.3) | −60.9 (−60.5) |
| path, 512 rays denoised | −43.8 | −46.9 | −63.2 |

Inside the church the clean render is now 0.6–1.1 dB below the energy method (1.6–1.9 dB before).
Outside it is still 9 dB below: these renders run with diffraction, and part 8 measured that with the source's edge rays the outside late field is heavy-tailed (one solve's median a tenth of its mean); a render plays something close to the median.
That is step 3(b), below.
The realtime solves took 249 ms per update on average (257 s for 1,034, under 50% pacing), the clean 2.6 s.

**Denoiser.**
`tools/path_dataset.py --name path-v3`: the v2 recipe with half the scenes (four per family, 24 generated scenes plus the Church, 312 poses), rendered with the beams; `verify` checks 875 payload hashes.
The same model and training as part 5 (`tools/path_denoise.py`, now reading the dataset named by `PATH_DENOISE_DATASET`, default `path-v3`), on the held-out families:

| Rays | four-frame mean | four frames at equal compute | denoised |
|---|---|---|---|
| 256 | 6.33 / 1.52 / 4.72 | 5.93 / 1.55 / 4.33 | **5.24 / 1.52 / 3.38** |
| 512 | 5.30 / 1.24 / 3.44 | 4.96 / 1.33 / 3.41 | **4.42 / 1.24 / 2.73** |
| 2,048 | 3.71 / 0.98 / 2.28 | 3.44 / 0.97 / 2.10 | 3.61 / 0.98 / 2.17 |

(window dB / band level dB / decay curve dB.)
It still beats more rays at equal compute at 256 and 512 rays and still loses at 2,048; the Church family again improves in window error (7.99 → 7.42 at 512 rays) and loses in decay curve (3.29 → 3.99).
So the walk was rendered denoised; as in part 5, the frame-to-frame change of the 500 Hz band (4.4 → 0.7 dB in 50–200 ms, 3.4 → 0.6 dB later) is the plain four-frame average's, not the network's.

**Published.**
The listening page `.context/energy-audio/listen/path.html` and the live view (`build.py` rerun) have the three new path renders; the stale denoised track of part 5 was removed.
The scoreboard (`scoreboard.py`) now shows the beams on the wave suite (`beams.py publish` replaced the suite's path runs at 2,048 rays: level 3.27 dB, EDT 226%, T30 65.7%, ETC 13.9 dB against 3.52 / 435 / 72.0 / 18.3) and the new Church responses with their echo densities (part 9's table).

## 11. Edges

### (a) UTD's reflection-boundary terms against the beam width

UTD's two reflection-boundary terms (those in φ + φ′) now carry the weight 1 − exp(−2 X s′/s) in the Path method's edge connections (`PathUtdEnergy`), X = kLa their transition argument and s, s′ the source- and listener-side distances.
The weight is one minus the beam kernel at the listener's offset x from the boundary: with w² = λ(s + s′)/π and, near the boundary, X ≈ π s x² / (λ s′(s + s′)), the kernel's exponent 2x²/w² equals 2X s′/s; no constant enters.
Where the beam reaches past the boundary UTD's transition is not added a second time; far from it UTD is unchanged; the incident-boundary terms are untouched, because the direct sound is exact and not a beam.
Double-edge (DD) connections keep plain UTD (there is no beam to compare with); the energy method is unchanged.

On the 2 m plate (`MeasureReflectionBoundary`, 250 Hz / 1 kHz / 4 kHz):

| x | beams | beams + UTD before | beams + UTD now | exact + UTD (part 8) |
|---:|---|---|---|---|
| 1.5 m | 0.125 / 0.147 / 0.163 | 0.183 / 0.168 / 0.165 | 0.151 / 0.158 / 0.165 | 0.213 / 0.179 / 0.163 |
| 1.9 m | 0.075 / 0.079 / 0.093 | 0.139 / 0.116 / 0.110 | 0.094 / 0.082 / 0.085 | 0.194 / 0.168 / 0.156 |
| 2.1 m | 0.055 / 0.047 / 0.036 | 0.073 / 0.065 / 0.051 | 0.060 / 0.047 / 0.034 | 0.018 / 0.019 / 0.017 |
| 2.5 m | 0.028 / 0.011 / 0.001 | 0.037 / 0.018 / 0.004 | 0.031 / 0.012 / 0.004 | 0.009 / 0.007 / 0.003 |
| 3.0 m | 0.011 / 0.001 / 0.000 | 0.016 / 0.003 / 0.001 | 0.013 / 0.003 / 0.001 | 0.005 / 0.003 / 0.001 |

Past the boundary UTD now adds 0.005 at 2.1 m and 250 Hz instead of 0.018; what remains above exact + UTD there is the beam's own smoothing.
On the lit side the result is now lower than exact + UTD, which is not a reference there: it adds |D|² on top of the full geometric reflection, while the coherent field at the boundary has a quarter of the geometric intensity; the energy integrated over the line is 0.83 against 1.07 (exact + UTD) and 0.69 for the beams alone.
A coherent reference would decide the lit side; the Pressure method's coherent sum (also printed by the measurement) interferes with the direct sound at this geometry and gives negative "reflected" energies, so it is not one.
On the wave suite (`beams.py`, `PATH_BEAMS_RAW=raw-path-beam-utd`, 2,048 rays, two seeds) the weighting improves the level error from 3.27 to 3.17 dB (shadow 4.68 → 4.57 dB) and EDT from 226 to 213%, with T30, C80 and ETC unchanged (`beams-utd.json`).

### (b) The heavy tail of the diffracted sound outside the Church

At the walk's outside position (125 / 500 / 2,000 Hz, 0.2–2 s, 64 seeds of 8,192 rays), one solve's median against the mean:

| Configuration | median / mean |
|---|---|
| no diffraction (part 8) | 0.47 / 0.66 / 0.52 |
| diffraction, no edge rays, no edge connections | 0.47 / 0.66 / 0.52 |
| diffraction, no edge rays | 0.29 / 0.15 / 0.10 |
| diffraction (edge rays half of the source's rays, the default) | 0.09 / 0.12 / 0.13 |
| ... with scattering points also drawing edges from the listener-aware distribution | 0.12 / 0.15 / 0.18 |
| ... and RD mirrors drawn by area | 0.11 / 0.09 / 0.09 |
| ... and edge-ray azimuths drawn by |D|² (half the time) | 0.06 / 0.09 / 0.11 (reverted) |

The tail comes from the edge connections of scattering points and from the source's edge rays, not from the beams.
Two unbiased changes are kept: scattering points draw their edge half the time from the solve's listener-aware edge distribution (the other half local or uniform, weighted by the mixture's probability), and the mirror of an RD connection is drawn in proportion to its area instead of uniformly among 17,345 triangles (`PathMirrorCdf`); drawing edge-ray azimuths in proportion to |D|² made the 125 Hz tail worse and was reverted.
With 64 seeds the medians themselves are uncertain to perhaps 30%, so the kept changes are justified by construction (each makes the probability of a sample closer to its contribution) more than by these numbers, and the tail is not fixed: the outside of the walk still plays about 9 dB below its mean in a single update.
What remains is the 1/(m l (m + l)) geometry of a D connection from a scattering point close to an edge (the door frame's own faces), which no edge distribution changes; sampling the edge point by that geometry, or splitting the edge's energy over the walk's next segments, is the next thing to try.

## 12. Speed in the Church

### Profile

`MeasurePathProfile` on the Church at the realtime render's settings (512 rays, 6 s horizon, four poses of the walk), GPU time of the walk per update with parts switched off and at three survival times T0 (survival in time S(t) = min(1, T0/t), part 2):

| T0 | as rendered | no histogram writes | no straight connections | no edge connections | neither connection |
|---:|---:|---:|---:|---:|---:|
| 100 ms (default) | 247 ms | 246 ms | 299 ms | 510 ms | 108 ms |
| 30 ms | 60 ms | 60 ms | 256 ms | 91 ms | 68 ms |
| 10 ms | 40 ms | 40 ms | 166 ms | 72 ms | 29 ms |

Switching parts off does not reliably make a solve faster: the time of a dispatch is the time of its slowest group, which is set by the few rays that live longest, and which rays those are changes with every switch and seed (the same configuration varied by 2× between runs in part 2).
The histogram writes cost nothing measurable.
The one dependable lever is how long rays live: from T0 = 100 ms to 30 ms the walk falls from 247 to 60 ms per update, and to 40 ms at 10 ms.

### Efficiency of the survival time

Survival in time is unbiased, so T0 trades variance for time.
Per-solve spread of the late field (0.2–2 s) at 512 rays, 32 seeds, standing inside the Church (the walk at 3 s), standard deviation over mean at 125 / 500 / 2,000 Hz, and the efficiency 1 / (relative variance × time) at 500 Hz:

| T0 | sd / mean | time | efficiency (relative) |
|---:|---|---:|---:|
| 100 ms | 1.39 / 0.77 / 0.70 | 247 ms | 1 |
| 30 ms | 1.14 / 1.07 / 1.27 | 60 ms | 2.1 |
| 10 ms | 1.65 / 1.59 / 1.56 | 40 ms | 1.4 |

At 125 Hz the gain of 30 ms is 6×, at 2 kHz 1.2×; outside the Church every setting is dominated by the edge tail of part 11 and the comparison says nothing.
On the wave suite, T0 = 30 ms (2,048 rays, two seeds) is cheaper (1.86 against 2.23 s per run) but not better: level 3.38 against 3.17 dB, T30 76 against 66% (shadow 92 against 74%), EDT 153 against 213%, ETC 14.7 against 13.9 dB (`beams-s03.json`); the suite's rooms decay in a fraction of a second, where 100 ms already cuts little.
So the default stays 100 ms; for the Church at a realtime budget 30 ms is the efficient choice (60 ms per update instead of 247), which is a property of the room's long decay, not a constant to change for every scene.
A rule that scales T0 with the room (for instance with the decay measured by the previous update) would make that choice automatically; it is not built.

### Splitting long paths across threads

Not done.
The walk is one loop per ray with about forty live variables (the ray, its chain, its branch reservoir, the pending deposit and connection state); splitting it into slices of a few events with compaction of the surviving rays between dispatches means writing and reading that state to a buffer at every slice, and restructuring the loop around it.
The profile says what it would buy: the latency is the longest-lived rays', so with T0 = 30 ms (60 ms per update, against a total of rays × mean events that a fully occupied GPU would do in a fraction of that) it is the remaining factor between the Church and the 23 ms target, and it is the next step.

### Before and after, per update at 512 rays

| | before | after (T0 = 30 ms) |
|---|---:|---:|
| Church | 247 ms | 60 ms |
| furnished room | 46 ms | (default unchanged) |
| stone shoebox | 6 ms | (default unchanged) |

## 13. Survival time set by efficiency: derivation, and why it waits for step 2

**The quantity to optimise.**
Survival in time keeps a path alive at t with probability S(t) and divides its weight by S, so every positive S is unbiased; what S changes is variance and cost.
Efficiency is 1 / (variance × cost) (EARS, Rath et al. 2022, for splitting and roulette factors).

**Variance.**
For the late field the measured relative variance of one solve fits var(T0) = a + b / T0 for S = min(1, T0/t): standing in the Church at 500 Hz, a = 0.37 (the path-to-path variance that roulette does not add) and b = 0.022 s (what roulette adds), from part 12's T0 = 100, 30 and 10 ms (the fit predicts 1.09 at 30 ms, measured 1.14).

**Cost.**
The cost that matters is the time of an update, and that is not the work: a dispatch lasts as long as its slowest group, whose time is the longest-lived of its G rays, about min(G·T0, path lifetime) in path time for S = T0/t.
Fitted to the Church's measured times, L(T0) ≈ c + d·T0 with c ≈ 30 ms (everything that does not depend on survival) and d ≈ 1,000 ms per second of T0 (G = 64 rays per group times 15.6 ms per second of path time).

**Optimum.**
Minimising (a + b/T0)(c + d·T0) gives the square-root rule of EARS:

T0* = sqrt( (b / a) · (c / d) ).

For the Church, sqrt(0.058 s × 0.030 s) = 42 ms, close to the 30 ms that measured best.
In the suite's rooms paths end by absorption well before G·T0 (their path lifetime is about the reverberation time, under a second), so d is near zero there and T0* goes to the path lifetime: no time roulette, which is what the suite preferred (100 ms better than 30 ms).
Both the room dependence asked for and its direction follow without a tuned constant, from two measured ratios: b/a, the variance that roulette adds relative to what is there anyway, and c/d, the fixed time relative to the time per unit of path life.

**Why it is not built yet.**
The ratio c/d is a property of the dispatch, not of the room: it is large because one long-lived ray holds a whole group.
Step 2 (wavefront passes with compaction) removes exactly that, and with it d changes from "longest ray in the group" to "average work per ray"; with the work model (cost proportional to the total path time, ∫S(t)L(t) dt) and the histogram's own per-bin relative error (one count per distinct path, bins in constant relative width), the same minimisation has no interior optimum at all: T0* is the path lifetime, i.e. no time roulette (computed for lifetimes from 0.5 to 6 s; the optimum is within 10% of the lifetime in every case).
Building the rule now would tune it to a dispatch shape that the next step is meant to remove, so the order is reversed: step 2 first, then, if its update time leaves a gap to 23 ms, the same derivation with the measured cost of the new dispatch.
The two measured ratios needed per solve are available to the tracer: b/a from the spread between consecutive solves of the same pair (or their counts and second moments), c/d from the solve's own phase times.

## 14. Wavefront passes: exact, and slower

**What was built** (`wavefront.patch` beside this file, not applied).
The walk ran in passes of K iterations of its loop; a ray still alive at the end of a pass saved its whole state (133 words: the ray, its connection vertex, its chain and branch, the band vectors, the pending deposit) and appended itself to a compacted list, and the next pass ran only the listed rays, from the first threads of the dispatch.
The result was identical to the uninterrupted walk (same random numbers, order-independent fixed-point sums): a new check gave a sum of differences of exactly 0 over all bands for K = 8 against one pass.
On the way: on Metal, buffer writes made inside a `[loop]` just before a `return` were silently dropped (the state was never saved and the rays vanished, 85% of the energy at K = 8); the save now happens after the loop.

**Time per update**, the walk's GPU time, one listener, fences included:

| Scene | rays | one pass | K = 32 | K = 8 |
|---|---:|---:|---:|---:|
| stone shoebox | 256 / 512 | 5.2 / 5.6 ms | 6.0 / 6.2 ms | 8.6 / 6.9 ms |
| furnished room | 256 / 512 | 46 / 48 ms | 61 / 63 ms | 61 / 66 ms |
| Church | 256 / 512 | 221 / 240 ms | 450 / 613 ms | 484 / 645 ms |

**Why.**
In the Church at 512 rays, K = 32 took 80 passes and K = 256 took 16; the time per pass was about K times 0.14–0.28 ms, i.e. a pass lasts as long as K iterations of its slowest ray, however few rays remain.
So the update time is not idle lanes waiting for the longest ray: it is the longest ray's own chain of iterations (about 2,500 here), each a BVH traversal or a connection, run one after another on one thread; compaction cannot shorten a single ray's serial chain, and its passes add a dispatch and a state round-trip each.
The shader with the state code also raised the test runner's peak footprint to 3.39 GiB.
The patch is therefore not applied.

**What would split a path across threads.**
Two things shorten the serial chain:
- doing each event's connections (the straight and edge connections, more than half of an iteration's cost: part 12 measured 247 → 108 ms with both switched off) in parallel by other threads, from a queue filled by the walk and drained between its passes, which is what the wavefront machinery above makes possible;
- making the longest paths shorter, which is the survival time of part 13, with the cost model now confirmed: the update time is the longest path's length, so c/d in T0* = sqrt((b/a)(c/d)) is a property of the room's path lifetime and does not go away.
Neither is done here; with the connections deferred the Church's walk would be about 108 ms at the default survival time and about 30 ms at T0 = 10 ms (part 12's "neither connection" column), so reaching 23 ms needs both.

## 15. The diffracted tail outside the Church: located, not fixed

The brief's hypothesis was scattering points right next to an edge, where the D connection's 1/m would blow up.
Measured at the walk's outside position (no edge rays, so only connections are in play; 64 seeds of 8,192 rays; median over mean at 125 / 500 / 2,000 Hz):

| Change (diagnostic, then reverted) | late level 125 Hz | median / mean |
|---|---:|---|
| as committed (part 11) | 2.48e-4 | 0.19 / 0.19 / 0.18 |
| connections from within 0.1 m of the edge dropped | 2.48e-4 | 0.19 / 0.19 / 0.18 (identical) |
| mirror-and-edge (RD) connections dropped | 2.48e-4 | 0.19 / 0.19 / 0.18 (identical) |
| double-edge (DD) connections dropped | 2.38e-4 | 0.20 / 0.19 / 0.17 |
| edges drawn from a listener-side distribution (angle × ∫1/m² from the listener × visibility) | 2.80e-4 | 0.18 / 0.18 / 0.18 |
| all edge connections dropped (part 11) | 2.55e-4 | 0.47 / 0.66 / 0.52 |

Near-edge points contribute nothing measurable: UTD's transition functions already regularise the 1/m (the transition argument kLa is proportional to the distance, and |D|² falls with it), so there is nothing to fix there.
Mirror-and-edge and double-edge connections are not it either, and a third edge distribution did not change the tail.
What remains is the single-edge connection itself, from points far from the edge: the listener sees no surface inside the church, so all interior sound reaches it diffracted by the porch's edges, and |D|² is largest for the few interior points whose line to the listener grazes an edge (its shadow boundary); the walk visits such points rarely and scores them with the full boundary energy.
Variance there is set by where the walk's vertices fall, which no choice of edge changes.
The principled remedy is to sample from the listener's side as well (edge rays from the listener through the porch's edges, joined to the walk's vertices by reciprocity, with the two techniques weighted by multiple importance sampling); it is a bidirectional technique and is not built.
The level outside agrees with the energy method within its noise (2.5e-4 against 2.54e-4 without diffraction, part 8), so the remaining problem is noise per update, not level.

## 16. The Church walk with part 11's edges

Parts 13–15 changed no code, so there is no new speed to render at; what changed since part 10's renders is part 11 (UTD weighted by the beam width, the mixed edge pick at scattering points, mirrors drawn by area).
The walk was re-rendered with that code, realtime and denoised, with part 10's settings (`church.py render path-realtime`, `path_denoise.py church`, `church.py render path-denoised`).
The denoiser was not retrained: path-v3 was rendered before part 11, and part 11 changes only the edge terms.
The clean render (8,192 rays) is still part 10's.

**Levels** (dBFS per part of the walk, part 10's renders in brackets):

| Render | standing and turning | walking | doors open, outside |
|---|---:|---:|---:|
| energy method | −43.8 | −43.4 | −51.9 |
| path, 512 rays | −44.8 (−44.2) | −46.0 (−46.4) | −64.8 (−62.2) |
| path, 8,192 rays (part 10) | −44.4 | −44.5 | −60.9 |
| path, 512 rays denoised | −45.1 (−43.8) | −44.6 (−46.9) | −64.4 (−63.2) |

Inside, the realtime render moved by 0.4–0.6 dB and the denoised one by 1.3–2.3 dB in opposite directions in the two parts, which is the size of one render's noise at 512 rays rather than an effect of part 11.
Outside, both are 1–3 dB lower than in part 10 and 13 dB below the energy method: a single update still plays close to the median of the heavy-tailed diffracted field (part 15), and part 11 did not change that.

**Speed.**
The realtime solves took 274 ms per update on average (284 s for 1,034 updates, 6.1 s worst), against 249 ms in part 10, under the same 50% pacing; peak memory 3.17 GiB for the realtime render, 2.68 GiB for the denoised one, 0.40 GiB for the denoiser.
The 10% rise is not separated from load on the machine; the target is still 23 ms (parts 12–14).

**Static responses and scoreboard.**
The realism page's two static responses (8,192 rays, doors closed; `church.py irs`, `realism.py page`) were re-rendered with part 11's code; the part-10 responses are kept in `.context/energy-audio/realism/renders/before-utd`.

| Response | T30 / EDT / C80 at 500 Hz | T30 at 125 Hz | echo density / mixing time |
|---|---|---:|---|
| path, near, part 10 | 6.5 / 5.1 / +0.6 | 7.2 s | 0.76 / 66 ms |
| path, near, now | 5.5 / 5.4 / −0.5 | 4.7 s | 0.79 / 39 ms |
| path, far, part 10 | 5.0 / 5.4 / −5.0 | 5.0 s | 0.77 / 72 ms |
| path, far, now | 5.9 / 4.8 / −3.4 | 7.3 s | 0.76 / 38 ms |
| energy method, near / far | 6.2 / 5.2 / −2.2 and 5.0 / 7.5 / −9.1 | 6.8 / 5.0 s | 0.78 / 41 and 0.85 / 50 ms |

These are single solves: the 125 Hz T30 moved by 2.3–2.5 s in opposite directions at the two points, which is solve-to-solve noise in the low band, not a trend.
The mixing time fell from 66–72 ms to 38–39 ms at both points, now within the measured halls' 35–41 ms; this coincides with part 11's denser edge and mirror sampling, but one solve per point does not establish the cause.
The scoreboard's wave suite now uses part 11's runs (`PATH_BEAMS_RAW=raw-path-beam-utd beams.py publish`): level 3.17 dB, EDT 213%, T30 65.8%, ETC 13.9 dB (part 10: 3.27 / 226 / 65.7 / 13.9).

**Published.**
The listening page `.context/energy-audio/listen/path.html` (`church.py page`) and the live view (`build.py`) play the new realtime and denoised renders; the scoreboard has the new suite rows and Church responses.

## 17. Persistent rays

**What changed** (`settings.pathPersistent`; `MethodPath.hlsl`, "Persistent rays"; `MethodTransport.cs`).
The rays are now the sound in flight.
Their state (160 bytes per ray: position and path time, direction and the cohort's age, the chain's flux and attenuation per band, the latest event's plane, survival, the chain's unfolded length and origin time, flags) stays in GPU buffers across updates.
Every update advances each live ray by one update period of path time (it stops in mid-segment and resumes there), launches the update's new rays from the source's current position, and writes the survivors compactly to a second pool.
What a ray meets during the update is scored against the listener's current position, with the full solve's estimators unchanged: next-event connections at scattering points, edge connections, and the beam kernel, which is scored in the update during which the ray passes the segment's point nearest the listener image.
The direct sound stays exact.
The full solve is the same loop with an infinite period, and stays the default.

Three details decide whether this is right:
- **When a contribution is heard.** A connection made now is heard after its travel time, so each contribution goes to the histogram of the update in which it is heard: a ring of future histograms per pair (one per update of the horizon, 44 MB for the Church), whose current slot becomes the update's histogram. The coordinator's brief put connections into the current update; through a door that has just opened that would be heard before the sound could get there.
- **Where an update sits in time.** New rays advance half a period in their first update, so update n covers, for every cohort, the path times that are heard within half a period of the time of update n's state. The response is centred on that time instead of lagging by half a period.
- **Transmitted branches.** The full solve keeps one reservoir of transmitted shares per ray and traces it when the ray ends, which can be seconds after the transmission. Here the reservoir is per update: a ray that stops while holding one queues it as a ray of its own (after the full solve's roulette), so a branch starts at most one update late. A ray that ends continues as its branch, as before.

Pools are sized from the rays that survival in time can keep (at most N Σ S(t) over the cohorts, plus a quarter, plus a cohort), or by `pathRaysInFlight`.
Overflow drops rays and is counted (`MethodTransport.RaysOverflowed`); every check asserts that there is none.
A scene cut in the take empties the pools and the ring (`TransportSolver.FlushRays`).
On Metal the state is written after the walk's loop (part 14's dropped writes).
A ray is never stopped within 4 RayEpsilon of its hit: the resumed trace starts RayEpsilon out and would miss the wall.
The per-ray medium lookup at the start of the walk is kept: reading the source's and listener's media from a buffer filled by the clearance kernel made the compiler fail to build the walk ("can't unroll loops marked with loop attribute").
`run.json` now describes the path method's synthesis as it is (noise per slot, part 9) instead of impulses, and names the persistent variant (`path-histogram-persistent-v1`).

**Checks** (`PathPersistentChecks.cs`; all pass, as does the full tier).

| Check | Result |
|---|---|
| Full solve cut into updates: the aperture room with ground, half-specular walls and diffraction, 2,048 rays, 16 ms updates, the same seed for every update's new rays | sum of differences 0.9–2.0e-6 of the total per band, against the full solve |
| Same room with transmitting walls (t = 0.05, branches), independent seeds, 40 seeds per mode | windows 0–20 / 20–60 / 60–500 ms within 1.4%, within one standard error, equal standard errors |
| Source at 100 m/s past a rigid plane, listener still, 4 ms updates | direct path 39.2908 ms (retarded 39.2908 ms); image heard at 40.88 ms against a retarded 41.05 ms and a frozen 30.00 ms; level 1.011 × 1/(cτ)² |
| A door filling the aperture room's hole opens after a horizon closed; listener outside, 8 ms updates | closed: 0; per update after the opening, relative to the open room's full solve: 0, 0, 0, 0, 0, 0.71, 1.03, 1.02, 1.04, 1.00, … ; last 10 updates 1.001 |
| Free field, rigid plane image, shoebox image lattice, half-plane (BTM), far side of a solid, Eyring decay, leak through walls (s = 1), in persistent mode | pass with their usual tolerances (the leak check with 0.75 s updates) |

The first check is the strongest: with the seed fixed, every update's cohort is the full solve's rays, and the persistent response after the warm-up is the full solve's to rounding, so the machinery (stopping and resuming, the beam's scoring point, the ring, the compaction) loses and duplicates nothing.
In the second check the 0–20 ms window first looked 23% high with six seeds; with 40 it agrees to 0.9%: the early field outside is heavy-tailed in both modes.
That 40-seed run reached 3.59 GiB, over the 3.5 GiB limit; the checks as committed (six seeds) peak at 3.13 GiB in the full tier.
In the door check nothing is heard until the first sound through the hole can arrive: the direct sound, at 42 ms, since the only surfaces the listener sees through the hole are on the far wall, 16 m from it.
After that the level overshoots the open room's by 2–4% for about a horizon: the room rang with the door closed, so the first sound out is that of the closed room, which is louder.
A connection is checked against the scene at the time it is made, not when the sound passes each point of it; through a door that moves during a connection's flight (at most a few tens of milliseconds) this is the remaining approximation.
A static measurement in persistent mode costs one full solve per update of its warm-up (one horizon of updates), because each update advances every cohort by one period; the checks use 16–750 ms updates for that reason.

## 18. Rays trapped in the Church's geometry, and the time per update

With persistent rays (part 17) an update in the Church took 36 ms at 256 rays and 46 ms at 512, and hardly depended on the number of rays or on the connections: the update waits for its slowest ray, which ran 1,166 iterations of the walk's loop in one update, against an average of 4.
New counters record the longest chain of iterations in a dispatch and their total (counters 7 and 8 of the path method).

**What the long chains were.**
A temporary probe recorded the rays that passed 400 iterations in an update.
They were not in clutter: they bounced between two parallel faces a few millimetres or centimetres apart, in air, every centimetre or so, for hundreds of events.
Three mechanisms put them there, each a meeting of the model's geometry with a fixed tolerance:

| Where | Gap | How rays got in | Fix |
|---|---|---|---|
| window panes (closed glass boxes, crossed by stone mullions) | 3.5 cm | a ray leaving a mullion's face where it meets the glass started its next segment within RayEpsilon (0.1 mm) of the pane's face, and the trace skips anything nearer than RayEpsilon | a segment that leaves a surface now starts RayEpsilon out along that surface's outgoing normal and is traced from zero distance |
| the floor slab under the front door frame | 8 mm | a transmitted branch crossing the door frame (a partition) left it at its far face, which lies inside the floor slab: the crossing skipped the other object's faces on the way | a partition crossing ends only once every other object entered on the way has been left too: overlapping solids are crossed as one |
| the door frame's threshold, whose top face lies 8 mm above the floor in an open mesh (48 boundary edges) | 8 mm | through the mesh's open ends | a ray that bounces between two facing parallel faces closer than half the shortest band's wavelength (1.07 cm) ends |

A gap tolerance in the ray–triangle test changed nothing (the same rays, at the same places), and so did switching edge rays off for the window and slit cases, so neither seams between triangles nor edge rays were the cause.
The slit rule is the only one that removes energy.
It is the physical limit of the ray model: a gap narrower than half the shortest wavelength has no transverse mode in any band (the first cuts on at c / 2g, above 16 kHz), so the field in it is a plane wave along the slit, which rays bouncing across it do not represent; here the slits are defects of the mesh, and the energy that enters one is that of the slit's mouth (8 mm high).
A fourth rule guards the others: a ray in air cannot meet a closed solid's face from inside, so if one does it got there through a leak and ends.
These ends are counted (counter 13 of the path method).

**What remains.**
After the fixes the slowest ray in an update runs about 250 iterations (236 at 256 rays, 260 at 512).
A probe of those found no more pockets: they are rays inside solid media (glass, stone, wood), where sound travels 6–16 times faster than in air, so a ray covers up to 125 m of path inside a 0.5 m slab in one 23 ms update, and rays among the pews' and organ's small parts.
Both are what the model describes, not defects, and their energy is small (a stone wall transmits 1e-4); the population control of part 20 (weight windows) is the principled way to stop spending the update on them, so there is no iteration cap.

**Edge distribution.**
Persistent rays keep each pair's edge weights (part 11) and refresh 2,048 of them per update, the whole set after a flush; every update rebuilds its distribution from the kept weights.
The walk stays unbiased with any positive weights, because a sample's probability is read from the same distribution it was drawn from; weights a few updates old only steer less well.
In the Church (14,075 edges, refreshed every 7 updates, 0.16 s) the edge phase fell from 3.6 to about 2 ms per update.

**Time per update now** (persistent, 23–24 ms updates, GPU time with fences, `MeasurePathProfile`; part 17's in brackets):

| Scene | rays per update | rays in flight | time per update | slowest ray (iterations) | ray and histogram state |
|---|---:|---:|---:|---:|---:|
| stone shoebox | 256 / 512 | 2,152 / 4,291 | 0.9 / 0.8 ms (3.1 / 1.6) | 8 / 8 | 14 / 16 MB |
| furnished room | 256 / 512 | 2,970 / 5,944 | 3.5 / 4.3 ms (8.9 / 6.5) | 35 / 36 | 14 / 16 MB |
| Church | 256 / 512 | 5,141 / 10,304 | 20.8 / 28.7 ms (35.6 / 45.7) | 236 / 260 (1,166 at 512) | 46 / 48 MB |

The Church is inside the 23 ms budget at 256 rays and not yet at 512.
Most of its state is the ring of future histograms (44 MB), which part 19 resizes.
The full tier passes; its quick part peaked once at 3.75 GiB, over the 3.5 GiB limit, and at 3.07 GiB when run again with the same tests.

## 19. The depth of the ring of future histograms

Part 17's ring held a histogram for every update of the horizon (262 in the Church, 44 MB), but a contribution is heard at most one connection's travel time after the update that makes it.
The ring is now sized when the first update's state is known (`MethodTransport.EnsureRing`): RingSlots = ceil(3 D n / (c Δt)) + 1, with D the diagonal of the scene's bounding box at that state (every triangle, source and listener), n the largest acoustic index of any material (1: no medium is slower than air), and never more than the horizon needs.
The factor 3 is the longest connection: a straight connection or a beam's offset is one line of sight, but an edge connection is two (vertex, edge, listener) and a mirror-and-edge or double-edge connection three, each no longer than D.
A contribution heard later than the ring reaches is clamped to its last slot and counted (`ConnectionsBeyondRing`); every persistent check asserts that there is none, and none occurred.
(Contributions later than the horizon are dropped as before and not counted.)
Timing is unchanged: the ring only loses slots that no contribution can reach.

| Scene | diagonal | ring slots | ray and histogram state |
|---|---:|---:|---:|
| stone shoebox (256 / 512 rays) | small | 2–3 | 13.8 / 15.6 MB → 2.5 / 4.3 MB |
| furnished room (256 / 512 rays) | small | 2–3 | 13.8 / 15.6 MB → 2.5 / 4.3 MB |
| Church | 1,247 m | 262 (unchanged) | 45.5 / 47.8 MB (unchanged) |

The Church does not shrink: its scene includes a meadow 1.2 km across (128 m without it), and a connection from the meadow's far end is heard up to 3.6 s later, inside the 6 s horizon.
A ring of 13 slots (one line of sight across the church itself, as the brief estimated) would need those contributions clamped, i.e. heard early, so the exact ring stays; in the Church its 44 MB is about 1% of the memory limit.

## 20. Beams only against shortcuts, and a fixed population of rays

### Population control (both modes)

**Rule.**
Persistent rays can be held at a fixed number N in flight (`settings.pathPopulation`).
Every cohort (the rays launched in one update) is brought, after each update's walk, to its share of N: cohort k, of age a = (k − ½)Δt, to n̄ S(a) rays with S(a) = min(1, T_f / a), T_f the end of the histogram's fine bins (64 × 0.25 ms = 16 ms).
A cohort of A rays is brought to its target by splitting each ray into ⌊ρ⌋ or ⌊ρ⌋ + 1 copies (ρ = n̄ S(a) / A, the extra copy with probability ρ − ⌊ρ⌋) or, for ρ < 1, by keeping it with probability ρ; every kept copy's weight is divided by ρ.
The walk itself then applies no survival in time.
n̄ = N / (1 + Σ_k S(a_k)) new rays are launched every update; in the Church (23 ms updates, 6 s horizon) that is 17% of N.

**Why this profile: survival in time and the share of new rays are one rule.**
A histogram bin at delay τ has width b(τ) (0.25 ms in the fine region, τ / 64 after it) and receives contributions from the rays of age τ at a rate proportional to their number n(τ); with a constant relative variance per contribution its relative variance is proportional to 1 / (n(τ) b(τ)).
The cost of an update is the number of rays in flight, N = ∫ n(τ) dτ / Δt.
Minimising the sum of the bins' relative variances, ∫ dτ / (n(τ) b(τ)²) (bins per unit delay 1 / b), at fixed N gives n(τ) ∝ 1 / b(τ) (Cauchy–Schwarz), i.e. S(τ) = min(1, T_f / τ): every bin then gets as many contributions as a fine bin.
The new rays are the age-0 end of the same profile, so their share follows; neither T0 nor the share is chosen by hand.
Part 13's T0* = sqrt((b/a)(c/d)) weighed variance against a dispatch whose time was its longest ray; with the cost now proportional to the rays in flight, the per-bin argument above replaces it, and T0 becomes T_f.
If the relative variance per contribution varied with delay as κ(τ), the same argument gives S ∝ sqrt(κ) / b; measuring κ per update is not done.

**Unbiasedness.**
Splitting and the weight lottery leave each ray's expected weight unchanged for any ρ fixed before the ray's own future is drawn, and ρ depends only on the cohort's count after the walk, i.e. on the past.
Conditioned on the counts every cohort is unbiased, so the response is unbiased; at most 16 copies per ray (a cost guard; any limit keeps the weights exact).
The launch count n̄ is fixed by the rule, so the cohort-size argument (a cohort's rays sharing the source power by their actual number) is trivial here: a cohort's rays each carry 1 / n̄ of it.
(A launch count of N minus the rays alive would depend on past randomness; conditioned on it a cohort is still unbiased, as long as it is at least one ray, but holding each cohort at its share already keeps N in expectation.)
Check: in the aperture room with transmitting walls, a population of 13,000 (4,060 new rays per update) matches the full solve in every window (ratios 0.96–1.17 with six seeds, within three standard errors; the persistent and full solves differ as much from each other there, part 17).

### Beams only

`settings.pathBeamsOnly` removes the next-event connections at scattering points and every edge connection (the ring keeps one line of sight); every arrival but the exact direct sound is a ray segment passing the listener, scored by the beam kernel, now also the first segment after a scattering point or an edge.
After a scattering point the beam's width uses the unfolded length from that point: a diffuse reflection keeps no phase relation with the path before it, so it is the new source of the Gaussian beam argument (w² = λL/π); after an edge ray, from the edge.
An edge ray that escapes takes the horizon as its UTD distance.

**What is lost without the connections** (analytic checks with the switch on): free field, rigid plane, shoebox lattice, Eyring decay, leak through walls, the hole and the porch pass; the half-plane fails at 250 Hz (+1.90 dB against BTM, tolerance 1.5 dB) and the far side of a solid (double diffraction) at 500 Hz (−5.7 dB; only the source's edge rays diffract, once).

**Cost.** Time per update against the rays in flight, population control, 23–24 ms updates:

| Scene | N in flight | shortcuts | beams only |
|---|---:|---:|---:|
| stone shoebox | 2k / 8k / 33k / 131k | 0.9 / 1.6 / 2.3 / 7.0 ms | 0.8 / 1.0 / 2.3 / 7.0 ms |
| furnished room | 2k / 8k / 33k / 131k | 2.6 / 4.4 / 11.7 / 41.9 ms | 2.7 / 4.3 / 11.5 / 42.2 ms |
| Church | 2k / 8k / 33k / 131k | 11.8 / 19.2 / 51.4 / 173 ms | 9.3 / 13.3 / 39.1 / 118 ms |

At 23 ms: the Church holds about 11,000 rays with shortcuts (1,900 new per update) and 17,000 beams only (2,900 new); the furnished room about 70,000 in both modes; the stone shoebox more than 300,000.
State and ring at N = 131k: 69 MB in the rooms, 93–110 MB in the Church.
Beams only is cheaper only in the Church (by 1.3–1.5×, the connections it skips are its edge connections through 14,075 edges); in the rooms the walk's own tracing dominates.

**Accuracy on the wave suite** (full solves, which persistent rays equal in expectation, part 17; 2,048 rays, two seeds, current code; RMS errors as on the scoreboard; level dB / EDT % / T30 % / C80 / ETC dB):

| Mode | time per run | all | in view | shadow | closed |
|---|---:|---|---|---|---|
| shortcuts, 2,048 rays | 2.36 s | 3.37 / 199 / 74 / – / 13.8 | 1.78 / 250 / 63 | 4.91 / 66 / 88 | 1.93 / 19 / 40 |
| beams only, 2,048 rays | 1.89 s | 5.47 / 346 / 63 / – / 15.0 | 1.99 / 430 / 58 | 8.75 / 60 / 72 | 1.95 / 18 / 23 |
| beams only, 8,192 rays | 11.7 s | 5.31 / 176 / 95 / – / 13.8 | 1.97 / 217 / 100 | 8.40 / 63 / 71 | 1.96 / 18 / 21 |

(C80's RMS error runs to thousands of dB on this suite for every method, part 5.)
In the shadow beams only is 3.5–3.8 dB worse in level and four times the rays do not recover it: the lost energy is diffraction, not noise.
In view it is 0.2 dB worse at 20% less time and its EDT error is 1.7 times larger; in closed rooms its T30 is better (23 against 40%).

**Church, per update, at equal time** (persistent, population control, N at 23 ms: 11,000 shortcuts, 17,000 beams only; 64 updates after a horizon of warm-up; late field 0.2–2 s; 125 / 500 / 2,000 Hz):

| Point | Mode | level (125 Hz) | sd / mean | median / mean |
|---|---|---:|---|---|
| outside (16.5 s) | shortcuts | 5.1e-4 | 5.2 / 5.5 / 5.2 | 0.03 / 0.05 / 0.12 |
| outside | beams only | 1.9e-4 | 7.0 / 7.8 / 7.8 | 0.00 / 0.00 / 0.00 |
| inside (3 s) | shortcuts | 6.1e-2 | 1.40 / 0.94 / 0.94 | 0.53 / 0.68 / 0.73 |
| inside | beams only | 5.1e-2 | 1.00 / 0.85 / 1.25 | 0.68 / 0.73 / 0.67 |

Outside, both are heavy-tailed per update (part 15's tail, now with about 1,900 new rays per update instead of 8,192), and beams only more so, at less than half the level (the diffracted sound it no longer connects).
Inside, the late field's noise per update is alike; between 50 and 200 ms beams only holds 2.6 times less energy at 125 Hz.
The denoiser's error at 256 and 512 rays was not measured for beams only: the denoiser was trained on the shortcut mode's histograms, and a fair number needs a dataset rendered beams only.

**Decision.**
Shortcuts stay: they are more accurate per unit of compute in level everywhere, by 3.5 dB in the shadow, and beams only is cheaper only in the Church and only by 1.3–1.5×.
Diffraction as an event of the walk (a ray whose beam reaches an edge diffracting a share of its energy by UTD) was not built: it could recover the shadow, but beams only already loses in view, where diffraction plays no part (EDT error 1.7 times larger at 20% less time), and with a cost advantage of at most 1.6× in rays there is no room for it to win overall; it is the next step if beams only is to be pursued.
The switch stays (a few conditions in the walk), so the comparison can be repeated.
Population control is kept for both: at 23 ms it gives the Church 1,900 new rays per update with shortcuts, against 256–512 with survival in time (part 18).

## 21. The Church walk with persistent rays

`church.py render path-persistent`: the walk with shortcuts and persistent rays, 11,000 in flight under population control (part 20's 23 ms point; about 1,900 new rays per update), 1,024-sample updates, the other settings as the realtime render.
It is on the listening page (`.context/energy-audio/listen/path.html`), in the live view as "Path (persistent)", and in a new section of the scoreboard that lists every walk render's level per part of the walk and its solve time per update.

| Render | standing and turning | walking | doors open, outside | solve per update |
|---|---:|---:|---:|---:|
| energy method | −43.8 | −43.4 | −51.9 | 4 ms |
| path, 512 rays per update | −44.8 | −46.0 | −64.8 | 274 ms |
| path, 8,192 rays per update (clean) | −44.4 | −44.5 | −60.9 | 2,574 ms |
| path, persistent, 11,000 in flight | −44.7 | −46.4 | −61.7 | 32 ms |

(dBFS; solve time per update in the render, 50% pacing, upload and readback included: the GPU's own time per update is part 20's 23 ms.)
The persistent render is 8.6 times faster than the per-update solve at 512 rays and, outside, 3 dB closer to the clean render: a single update's outside estimate is still heavy-tailed (part 20), but the new rays per update are almost four times as many.
Inside, its levels match the other path renders within their noise.
The walk starts with no rays in flight, as the sound does: the speech starts at the take's start, so nothing is missing.

The denoised persistent render was not made.
The denoiser was trained on independent solves; persistent updates are not independent (consecutive updates share the rays in flight, and population control splits rays), so whether it still beats averaging at equal compute needs a reference set of persistent histograms, which this round did not build.

**Open: the suite's level since part 11.**
The current code (shortcuts, full solves, 2,048 rays, two seeds) scores 3.37 dB in level on the wave suite against 3.17 dB at part 11 (in view 1.78 against 1.81, shadow 4.91 against 4.57).
The change is concentrated in the diffraction-dominated open scenes (barriers +0.6 dB in both material variants, canopy-bem +0.5 dB), while rooms improved slightly (furnished −0.14, shoebox −0.14, canopy-0009-rigid −0.33).
An ablation of part 18's three rules on the barrier and canopy scenes was inconclusive: with any of them switched off the outputs were identical to one another and differed from the run with all three on, yet scored the same to two decimals, so the switches' effect is not established.
These scenes' shadow fields are heavy-tailed per solve (part 15), so a 0.5 dB shift between random sequences is plausible, but it is not shown; the scoreboard keeps part 11's published suite runs until it is.

## 22. The suite's level since part 11: a measurement artefact and seed noise

Part 21 left open why the current code scores 3.37 dB in level on the wave suite against part 11's 3.17 dB, mostly in two scenes (barriers-0020-rigid +0.6 dB, canopy-bem +0.5 dB).
`beams.py` now takes `PATH_SCENES` (a subset of scenes) and `PATH_SEEDS` (other seeds) so that such probes are cheap.

**Bisection.**
The two scenes were re-rendered with the code of each commit since part 11 (657b36b, 09f1044, 798432c; full solves, 2,048 rays, seeds 0 and 1, as the scoreboard):

| Code | canopy-bem | barriers-0020-rigid |
|---|---:|---:|
| part 11's published runs | 3.40 | 4.64 |
| 657b36b (part 11's commit) | 3.93 | 5.24 |
| 09f1044, 798432c, HEAD | 3.93 | 5.24 |

Part 11's own commit already scores like HEAD, so no later commit moved these scenes.
The published runs came from part 11's step (a) alone (UTD weighted by the beam width): 657b36b with step (b) taken out (scattering points drawing edges from the listener-aware mixture, mirrors drawn by area) reproduces them bit for bit.
Part 11 measured the suite after (a), then added (b), which was justified as unbiased, without measuring the suite again.

**Seed noise or a real change.**
Both parts of (b) are unbiased by construction (the mixture's probability is the one `MethodPickEdge` draws with, and a mirror's is its share of the area), so the question is whether eight seeds instead of two still show a difference:

| Code, 8 seeds | canopy-bem | barriers-0020-rigid | shadow (both scenes) |
|---|---:|---:|---:|
| (a) only | 4.39 | 4.40 | 5.71 |
| (a) and (b) (HEAD) | 3.96 | 4.75 | 5.55 |

With eight seeds (b) is better in one scene and worse in the other, and better in the shadow over both; the two seeds of the scoreboard had happened to favour (a) in both.
Per receiver and band the eight-seed mean energies agree: the median |z| of their difference is 0.65 (0.67 for pure noise), with no more large deviations than noise gives (none and 0.8% beyond |z| = 3), so (b) changed the variance of these diffraction-dominated shadow fields, not their expectation.

**Scoreboard.**
The suite's path rows now come from the current code (`raw-path-shortcut`, 2,048 rays, seeds 0 and 1 like every method there): level 3.37 dB, EDT 199%, T30 74%, ETC 13.8 dB (in view 1.78 dB; shadow 4.91 dB; closed rooms 1.93 dB), against part 11's 3.17 / 213 / 66 / 13.9.
The extra seeds of the two scenes are kept in `raw-path-shortcut-extra-seeds` and not published, so every scene is scored with the same two seeds.
Two seeds are too few for the diffraction-dominated scenes: their level error moves by half a decibel between seed sets.

## 23. Listener-side sampling for the outside tail: built, measured, not kept

**What was built** (`listener-side.patch` beside this file, not applied).
Path space was partitioned by the events next to the listener, in one loop and without weights:
- the listener side counted the paths whose last event before the listener is one edge, preceded by a surface event or by the source itself: edge rays launched from the listener (the source's edge rays mirrored, by reciprocity), traced by the same walk with source and listener swapped, which then connects to the source as usual, and at launch makes only the single-edge connection to the source;
- the source side counted everything else, so its edge connections were only the double-edge ones (a listener edge ray cannot diffract twice in a row);
- the listener side's deposits arrived from its rays' first leg, and its rays were traced to their end within the update, in the update's scene (reciprocity per update: the listener side is not retarded, it uses the scene and both positions as they are at the update).

**The partition is not exact against the method's own path set.**
The walk draws edge rays only at its start, and connects through an edge only from a scattering point (with at most one mirror before the edge).
So the source side covers "edge, then any number of mirrors, then a scattering point" at the source's end but only "at most one mirror, then the edge" at the listener's end; swapping the sides for the class above swaps these limits.
The partitioned estimator therefore counts a different set of paths: more mirrors after the last edge, fewer after a first edge.
Making it exact needs either edge diffraction as an event anywhere in the walk (so both ends can diffract after any number of mirrors) or restricting the listener side to "at most one mirror after the edge", which needs a cap the walk does not have.

**Measured** (Church outside at 16.5 s, late field 0.2–2 s, 125 / 500 / 2,000 Hz; 1,024 listener rays per update):

| Measurement | current method | with the listener side |
|---|---|---|
| persistent, N = 11,000, 64 updates: median / mean | 0.03 / 0.05 / 0.12 | 0.16 / 0.28 / 0.32 |
| same: mean level at 125 Hz | 5.1e-4 | 1.3e-4 |
| full solves, 8,192 rays, 16 seeds: median / mean | 0.12 / 0.16 / 0.22 | 0.07 / 0.08 / 0.14 |
| same: mean at 125 Hz | 6.6e-4 ± 4.9e-4 | 4.5e-4 ± 1.9e-4 |
| Church, time per update at N = 11,000 | 23 ms (part 20) | 78 ms (walk 38 ms, listener side 38 ms) |

The outside field stays heavy-tailed: per update the median over the mean rose, over full solves it fell, and neither comes near the 0.5–0.7 seen inside.
The tail moved rather than went: the listener's edge rays carry the same UTD peak near their shadow and reflection boundaries as the source's (part 11 found drawing their azimuth by |D|² made the tail worse).
The cost tripled: the listener side's rays run to their end in every update (their longest chain alone is 30–37 ms), and the source side's edge connections, now all double-edge, are dearer.

On the wave suite (full solves, 2,048 source and 1,024 listener rays, two seeds) the open scenes improved (barriers 5.24 → 4.46 and 5.31 → 4.56 dB, canopy-bem 3.93 → 3.60, the canopies 2.60 → 1.77 and 2.54 → 1.89; shadow 4.91 → 4.60 dB) and the rooms got much worse (doorway 3.07 → 4.87, corridor 1.91 → 3.35, furnished 2.07 → 3.32; in view 1.78 → 2.08 dB; overall 3.37 → 3.31).
That is the different path set at work: after the listener's edge the walk follows any number of reflections, which the method never counted before, and in a closed room UTD energy added that way is counted on top of the reflected field.
The analytic checks pass with it (half-plane 0.29 against 0.30 dB, double diffraction 0.23 dB closer to BTM).

**Decision.**
Not kept: it is three times the cost, does not bring the outside tail near the target, and changes the method's path set in a way that costs the rooms more than it gains outdoors.
The code is reverted and the persistent Church walk is not re-rendered (the method is unchanged).
What the measurements point to instead is the one mechanism both problems share: edge diffraction as an event of the walk, where a ray whose beam reaches an edge diffracts a share of its energy, so that diffraction is sampled where the energy is, at either end of a path, and the single-edge connections (the tail's source, part 15) are no longer needed; that is part 20's open step, now with a second reason.

## 24. Diffraction as a walk event: a prototype that does not yet pass the analytic checks

**Test tiers first.**
The quick tier had reached 3.55–3.75 GiB in one Unity process and a slow batch the 4 GiB guard: each GPU case leaves its compiled kernels behind in the process.
`tools/unity_tests.py` now runs the quick tier in three processes (test names A–L, M–P, Q–Z) and the slow renderer and method batches in two each; every batch peaked at or under 2.9 GiB (commit 3d77c93).
The next full tier, run after the shaders had been changed and changed back, peaked higher in two batches (3.56 GiB for A–L, 3.47 GiB for the slow renderer cases M–Z) with the same tests; the first process after a shader change re-imports and compiles every shader, so the limit is met in steady state but not yet after a shader change, and that is not solved.

**What was built** (`walk-diffraction.patch` beside this file, not applied; `settings.pathWalkDiffraction`).
A ray segment deposits its beam on a nearby edge as it does on the listener: the edge receives the energy per unit length Φ K1(ρ) / sin γ, with K1 the one-dimensional marginal of the beam kernel (w² = λL/π), ρ the segment's closest offset from the edge line and γ their angle; summed over rays this is the incident intensity integrated along the edge.
One candidate edge per segment piece is chosen in proportion to that deposit, and the ray turns, with probability q, into a diffracted ray leaving the edge's closest point on Keller's cone with the UTD energy flux the source's edge rays use (weight 1/q), or goes on undiffracted (weight 1/(1 − q) after the edge).
The diffracted ray's own first segment is scored at the listener by the beam kernel; the source's edge rays and every edge connection are off.
UTD's transition terms get part 11's beam weight 1 − exp(−2X s′/s): the reflection-boundary pair always, the incident-boundary pair when the incident field is a reflected beam.
The candidate search tests every edge against every segment, which is enough for the analytic scenes and the suite (at most a few dozen edges) and not for the Church (14,075 edges, which would need a spatial index of the edges).

**What was found on the analytic checks** (configuration (a): shortcuts with walk diffraction; dB against BTM):

| Variant | half-plane, 250 / 500 Hz | far side of a solid (double diffraction), 125 / 250 / 500 / 1,000 Hz |
|---|---|---|
| current method (edge rays and connections) | +0.30 / +0.29 | +1.79 / +1.50 / +1.19 / +0.91 |
| q from the expected diffracted share (about 2.6%) | −11.9 / – | no energy |
| q = ½, UTD's distance to the first hit | +2.87 / – | +1.49 / +0.51 / −2.45 / – |
| q = ½, UTD's distance to the listener for the first segment, incident weight only for reflected beams | +1.50 / +1.84 | −0.10 / −0.98 / −4.01 / – |
| same, incidence taken from the chain's origin instead of the passing ray | +4.68 / – | +3.76 / +3.27 / 0.00 / −7.78 |

(A dash: the check stopped at its first failing band.)
- With q set by the expected diffracted share, about 1/k of the incident energy, only 656 of 131,072 rays diffracted and 17 twice, too few to reach the listener's kernel; any q is unbiased, and q = ½ fixed the starvation.
- Taking UTD's distance to the diffracted ray's first hit (or the horizon for a ray that escapes) instead of to the listener overweights |D|² near the shadow boundary: +2.9 dB.
- With both corrected the half-plane is within 2 dB at every band shown, but 1.5–1.8 dB high, and double diffraction falls with frequency to −4 dB at 500 Hz: a diffracted ray that grazes the top face toward the second edge is blocked before its closest approach, so the second edge sees only the half of its incident field that passes above it.
- Taking the incidence from the chain's origin, which is exact for UTD, made both worse (+4.7 dB, −7.8 dB). So the near-passing variant above was right by compensating errors, and the event's normalisation, the UTD edge-ray flux per unit length and azimuth scored at the listener by a beam kernel, has never been checked against the stationary-point formula of the edge connections. It needs a derivation of its own before anything else is measured.
- A Metal compiler finding on the way: atomic counter increments made just before a `continue` in the walk's loop were dropped once the loop grew (the escape and edge-event counters read zero while the events happened), the same pattern as part 14's dropped writes before a `return`. Deposits and ray state are unaffected; counters placed so are not to be trusted.

**Not measured:** the wave suite with four seeds, the Church's outside tail, and the time per update. The analytic checks fail, and the Church needs the edge index first.
**Decision:** the current method stays; the prototype is kept as a patch.
What it needs next, in order: a derivation of the walk event's normalisation, checked on the half-plane at every band (the beam-kernel scoring of a line source's rays against UTD's stationary-point formula); grazing incidence handled (the reflected half of the field at the second edge); then an edge index for the Church; then configurations (a) and (b) on the suite and the Church at equal time.

## 25. A denoiser for persistent rays

**The question.**
With persistent rays (parts 17–21) every update's histogram is the retarded-time response at the population that fits 23 ms of GPU time, noisy, and consecutive updates share history.
Can a small network running beside the tracer do better than the best equal-compute temporal filter on the same inputs, an exponential moving average with its decay chosen on validation, with band levels kept unbiased, and run live in Unity?

**Data** (`tools/path_persistent_dataset.py`, Unity side `MeasurePathPersistentSequences` in `Assets/Tests/Editor/PathPersistentDataset.cs`).
Three scenes per family of `tools/dlss_scenes.py`, 18 in all, chosen so that the families the frozen split rule holds out in part (wide doorways, crowded furnished rooms) contribute one held-out scene each, plus the Church.
Per generated scene three listener tracks: two static listeners and one walking back and forth at 1.4 m/s along a clear segment; per track a flush, a horizon of warm-up updates (89 at the 2 s horizon), then 64 recorded updates of 1,024 samples at 44.1 kHz.
The Church contributes its take's own walk in three windows (standing, walking, outside), each after 261 warm-up updates (its 6 s horizon).
Inputs: persistent rays under population control at each scene's 23 ms population N, calibrated in Unity by three probes (GPU time per update at a guess, then at the estimate), two seeds.
N came out at 79,000–138,000 in the shoebox and doorway rooms, 46,000–105,000 in the corridors, 22,000–40,000 in the furnished rooms and at the 262,144 cap in the open canopy and barrier scenes (14 ms there); the Church keeps part 20's 11,000.
Targets: the same tracks at 8 N (at most 2^19 rays, so 2–8 N) with two other seeds, the same retarded-time quantity at every update, so a walking listener's target is right for that update and no frozen full solve is used.
57 tracks, 1.0 GB packed; the render took 42 minutes in one Unity process, peak footprint 3.03 GiB (the measurement is a `UnityTest` that lets the editor tick between solvers: without the ticks Unity kept every released ray buffer until the process ended and the footprint grew past 4 GiB).
Splits as before: corridor and the crowded furnished room validation; canopy, the wide doorway and the Church test; the rest train (24 tracks).

**Model** (`tools/path_persistent_denoise.py`).
The part-10 network, a 1-D dilated convolution along the bins (271,608 parameters), with persistent inputs: the last K = 4 updates' histograms and counts plus two exponential moving averages of the histogram (decays 0.7 and 0.9, bias-corrected), all scaled by the K-update mean's total.
A persistent update's bin at delay t holds the cohort emitted t earlier, so the same delay in consecutive updates comes from different cohorts and the same rays reappear one update later at t + 23 ms; the convolution's 66-bin receptive field sees both.
It predicts non-negative weights on the K-update mean and the two averages plus a floor, as before.
Band levels: each band's energy per time segment (0–50 ms, 50–200 ms, later) is rescaled to a convex combination of the three inputs' totals there, with the mixing weights a softmax of the network's output averaged over the segment.
Each input's totals are unbiased for a static listener, so the combination is too; the network only chooses how much history to trust.
Conserving the K-update mean's totals instead, as in part 10, left the denoiser's band levels 0.1 dB worse than the moving average's (0.60 against 0.50 dB over the held-out families), because the moving average averages more updates.
Noise2Noise L2 loss in perceptual windows, AdamW, 8,000 steps of 16 updates on the GPU (3 minutes); the checkpoint with the lowest sum of window, level and decay-curve errors on validation is kept (step 2,000).

**Baseline.**
The moving average of the persistent input, bias-corrected, with its decay chosen on the validation tracks from 0 to 0.97: 0.8 (window errors 1.32 dB at 0.8, 1.38 at 0.9, 1.50 at 0.95).
Every estimate is scored on updates 32–63, so every average has 32 updates of history.

**Results** (errors in dB against the mean of the two high seeds: window / band level / decay curve; "target" is one high seed against that mean, the target's own noise; 2 low seeds × 32 updates per track).

| Held out | noisy input | moving average (0.8) | denoiser | target |
|---|---|---|---|---|
| all held out (2,112 updates) | 3.38 / 0.92 / 1.66 | 2.08 / 0.50 / 0.98 | **1.96** / 0.54 / **0.93** | 1.35 / 0.31 / 0.62 |
| test (canopy, wide doorway, Church) | 4.33 / 1.31 / 2.24 | 2.72 / 0.70 / 1.36 | **2.54** / **0.69** / **1.31** | 1.91 / 0.46 / 0.87 |
| canopy, static | 2.57 / 0.24 / 1.49 | 1.66 / 0.14 / 1.03 | **1.55** / **0.13** / **0.94** | 1.33 / 0.12 / 0.73 |
| canopy, walking | 2.46 / 0.16 / 1.65 | 2.31 / 0.28 / 1.23 | **2.04** / **0.22** / 1.24 | 1.29 / 0.06 / 0.72 |
| wide doorway, static | 1.42 / 0.20 / 0.51 | **0.58** / 0.09 / 0.25 | 0.65 / **0.08** / **0.23** | 0.46 / 0.05 / 0.16 |
| wide doorway, walking | 5.61 / 0.95 / 2.17 | 2.70 / 0.43 / **1.10** | **2.52** / **0.39** / 1.12 | 2.11 / 0.35 / 0.80 |
| Church, standing | 10.78 / 5.17 / 4.87 | 7.47 / 2.06 / 3.10 | **6.82** / **1.79** / **2.84** | 4.46 / 1.84 / 1.90 |
| Church, walking and outside | 13.71 / 6.87 / 7.57 | 8.38 / **3.82** / 4.18 | **7.93** / 4.07 / **4.11** | 6.01 / 2.33 / 2.47 |
| corridor, static (validation) | 1.50 / 0.39 / 0.72 | 0.80 / 0.17 / 0.38 | **0.72** / **0.15** / **0.34** | 0.47 / 0.09 / 0.22 |
| corridor, walking (validation) | 4.33 / 1.01 / 1.72 | **2.72** / **0.78** / 0.93 | 2.75 / 1.33 / **0.86** | 1.38 / 0.35 / 0.63 |
| furnished, crowded (validation) | 1.93 / 0.27 / 0.83 | 1.14 / 0.12 / 0.45 | **1.04** / **0.10** / **0.40** | 0.55 / 0.07 / 0.25 |

- Against the noisy input the denoiser removes 1.4 dB of window error over the held-out families and 1.8 dB on the test families; the moving average removes 1.3 and 1.6 dB.
- Against the moving average it gains 0.12 dB of window error over all held-out updates and 0.18 dB on the test families, with band levels about equal (0.54 against 0.50 dB, 0.69 against 0.70 dB on test) and decay curves slightly better.
- It loses in one place: a listener walking along the corridor, where its band levels are 1.33 dB off against the average's 0.78 dB; it trusts the long average there while the level changes with every step. It is told nothing about motion, and the corridor is a validation family, so nothing was tuned on it beyond the checkpoint choice.
- The Church's errors are large for every estimate because its target is itself noisy (88,000 rays in flight, 4.5–6 dB window error of one seed against the two); the ordering is the same.
- At equal compute: the kernel takes 1.6 ms per update at the Church's 447 bins (below), 7% of 23 ms, which would buy the moving average 7% more rays, about 3% less noise; that is worth about 0.06 dB of its window error over the held-out families. So the denoiser's margin over the best temporal filter is real but small, about 0.06 dB over all held-out updates and 0.1 dB on the test families.

**Runtime** (`Assets/Scripts/Simulation/PathDenoiser.cs`, `Assets/Resources/Acoustics/PathDenoise.compute`).
The network is small enough for a hand-written compute kernel: one kernel per step of the Python model (convert the tracer's fixed-point histogram, update the history and the two averages, features, normaliser, eight convolutions with exact GELU, prediction, band totals, rescaling), recorded into the audio's command buffer so it stays ordered between the tracer and the reverb.
The weights (1.1 MB) ship in `Resources/Acoustics/PathDenoise`, exported with a reference fixture by `path_persistent_denoise.py export`.
`PathDenoiserKernelMatchesTheModel` replays the fixture (eight updates of a validation walk): the kernel matches the PyTorch model to a worst relative error of 1.9 × 10⁻⁵.
`PathDenoiserFitsTheUpdateBudget`: 1.2 ms per update at 376 bins (2 s horizon) and 1.6 ms at 447 bins (6 s), including the upload of the test's histogram and the dispatch; its Sentis/ONNX alternative was not needed.

**Not measured:** the wave suite.
The suite scores single static poses of the generated families at fixed ray budgets against the wave solver; with persistent rays a static pose's answer is the long-run average of the updates, which the moving average and the denoiser both approach, and the table's static rows already measure that against an 8 N target.
**Decision:** the denoiser is kept and runs live in the playable Church (part 26), switchable at runtime; the moving average is the bar it has to keep clearing.
What would improve it: telling it the listener's and the sources' speed (the corridor walk), and a target with less noise for the Church.

## 26. The path method live: the reverb on the GPU, and a playable Church (in progress)

**Why the reverb moved to the GPU.**
`MeasurePathReverbCpuCost` (opt-in, `ACOUSTIC_PATH_REVERB_COST=1`) times the CPU reference, `PathReverb`, at the live Church's size: 48 kHz, two ears, eight bands, a 6 s horizon, updates of 1,024 samples.
It spent 432 ms building one update's kernels and 8.9 ms convolving one 1,024-sample block.
An update lasts 21.3 ms, so the CPU cannot keep up, by a factor of about 20.
Smaller partitions do not help the kernel build: it is a weighted sum of fixed slot spectra, about 24,000 slots × 2 sequences × 1,025 bins × 8 bands × 2 ears per update.

**What was built** (`Assets/Scripts/Simulation/GpuAudio.Path.cs`, kernels `PathEar`, `PathGroups`, `PathBand`, `PathPieces` and `PathPowers` in `GpuAudio.compute`).
`PathReverb` filters the source into eight bands and convolves each band with its own kernel.
Convolution is linear, so the same output comes from one kernel per ear: the sum over bands of each band's slot sequences convolved with its band filter.
That kernel is built on the GPU at every update, from the same slot sequences, interaural mix, band filters and ear weights (`PathReverb.BuildSlots` is now shared by both).
Per partition: one FFT per band of the slot sequences (both ears packed as real and imaginary parts), the product with each band filter, summed over bands, one inverse FFT, then an overlap-add of the filters' tails into the next partition and the partition's transform.
The convolution, the crossfade and the PCM output are the late field's (`LateInput`, `LateSum`, `LateInverse`, `LateWrite`).
Two things were added to them: the input is read half a partition ahead (the band filters' delay, inside the DSP latency), and the crossfade gets `PathReverb`'s power correction.
The persistent-ray denoiser (part 25) is recorded into the same command buffer between the tracer and the kernel build, one per source–listener pair, behind `settings.pathDenoise`.
Under GPU audio the method no longer reads its histogram back to the CPU.
The live renderer stops a batch at a partition boundary when the source has not yet got the samples the read-ahead needs (`GpuAudio.InputEnd`).

**Checks.**
- `GpuPathReverbMatchesTheCpuReference`: against `TakeRender.Mixer` with `PathReverb` on the same persistent-ray solves (14 updates, a moving listener, a cut, partial blocks). The relative error is 3.7 × 10⁻⁵ binaural and 6.3 × 10⁻⁶ mono.
- `LivePathRendersOnTheGpuWithTheDenoiser`: `TakeRender.Live` with the path method on GPU audio, persistent rays and the denoiser, as the playable Church runs it. Its observations end inside partitions, and the denoiser is switched off half-way.
- `GpuAudioParity` (the other methods) is unchanged.

**The playable Church** (`Assets/Scripts/Demo/PathLiveChurch.cs`, `Assets/Editor/PathLiveChurchBuild.cs`, `tools/path_live_church.py`; how to run it is in the main README).
It turns the Church scene's session into the live path method:
- persistent rays at a population of 8,000;
- the denoiser;
- GPU audio, binaural, with a 6 s horizon;
- one source, the organ, looping;
- the sliding doors on E;
- first-person controls as in the authored scene.

A small HUD shows the GPU time per update from profiler samples around the tracer, the reverb and the denoiser. It also shows the rays in flight, read back asynchronously, the denoiser's state (N toggles it) and the audio underruns.
A macOS player built from the test runner's copy of the project (167 MiB, 11 minutes the first time).

**Found on the way.**
- With GPU audio the method's CPU readback was never completed, so the second batch threw.
- The read-ahead failed when a batch ended inside a partition.

Both are fixed and covered by `LivePathRendersOnTheGpuWithTheDenoiser`.

**Not done yet.**
- The scripted smoke run of the build: the automated walk, output to wav, frame times and underruns. The first attempts ran into the two bugs above, and then into critical memory pressure from other jobs, under which the guard pauses the player. No numbers yet.
- The CPU cost of the remaining main-thread audio work.
- The population that fits beside the Church's rendering. 8,000 is a guess below part 20's 11,000 for the offline tracer.
- The HUD's GPU timings in a release player.
- The method toggle (path against energy): it needs the session to restart its take, so it was left out.

## 27. Optimisation round: the GPU budget, a profile, and the mesh trees

**The budget.**
In a game the audio cannot have the whole GPU.
The target is about 5 ms of GPU time per 23 ms update in the Church for the tracer, the denoiser and the audio together, at today's accuracy (the same noise at the same N, the wave suite within seed noise); the stretch goal is 2–3 ms.

**Profile before any change** (`MeasurePathProfile` with the Church take, persistent rays, population 11,000; a fence after every phase; the variants switch parts off with `ProfileFlags`; medians of the warm-up's updates, ±1 ms between runs on this machine).

| Part | ms per update |
|---|---:|
| walk (all ray segments of the update: traversal, beam scoring at the listener, shortcuts, edge connections, histogram and ring writes, population control) | 18.3–22.6 |
| of which histogram and ring writes (bit 0 off) | about 3 |
| of which straight shortcut and shadow rays (bit 2 off) | about 1.5 |
| of which edge connections to the 14,075 edges (bit 3 off) | about 2 |
| edge distribution and the edge-weight slice | 1.8–2.1 |
| readback of the histogram | 0.1–0.4 |
| denoiser (part 25, its own fence) | 1.2–1.6 |
| path reverb on the GPU (part 26) | not yet measured |

- The walk is the update.
- With every ablation off it still took about 20 ms, so the cost is the walk's own segments, the BVH traversal above all.
- The counters show why it is latency, not work. An update traces about 40,000 segments, which takes under a millisecond at any plausible GPU ray rate. But the longest thread runs about 280 segments in series (a ray caught in a narrow gap between surfaces), so the dispatch lasts as long as 280 dependent traversals.
- Each traversal walked the mesh tree with a median split, children in fixed order, and every node's box rotated into world space on the fly (three quaternion rotations per node visit).

**Change 1: the mesh trees** (`DirectBvh.BuildSah`, `Geometry.hlsl` `TraceMesh`).
- **Build.** Binned surface-area heuristic over three axes (12 bins), with leaves of up to 8 triangles where the heuristic prefers them. Below depth 18 the split falls back to the median, so the tree stays within the 32-entry traversal stack.
- **Traversal.** The ray is brought into the mesh's frame once, and boxes are tested with the inverse direction. Each box is padded by more than the transform's rounding, so no box the world-space triangle test would hit is ever culled.
- **Order.** The nearer child goes first; the farther one waits on the stack and is culled when popped if a closer hit has been found since.
- **Planar scenes** keep the rotated world bounds. Their geometry is extruded along world z, which a mesh-frame box cannot express under an out-of-plane rotation; `BvhChecks` caught this.
- **Exactness.** Triangles are still tested in world space, and ties are still broken by primitive ID, so the hits are the same.

Results:
- **Church, persistent, population 11,000:** the walk went from 18.3–20.3 ms to 6.2–6.9 ms. The edge phase went from 2.0 ms to 0.8 ms (it traces the edges' clearance rays). The update's counters (segments, events, rays in flight) are unchanged.
- **Wave suite, full solves at 2,048 rays:** every scene's histograms for seeds 0 and 1 match the published runs to a relative 6 × 10⁻⁸ (18 of 22 differ in the last bits of the summation, 4 are bit-identical), with the same total energy.
- **Four seeds** (new runs in `raw-path-bvh`): level 3.38 dB, in the shadow 4.91 dB, closed rooms 1.95 dB, against 3.37, 4.91 and 1.93 dB with the scoreboard's two seeds.
- **Time per solve:** 0.66 s, against 2.1–2.7 s.
- **Checks:** the full tier passes; peak footprint 1.98 GiB.
- **Memory:** the tree is trimmed to its node count; at most 2n − 1 nodes per mesh, against n/2 for leaves of four before. The Church's tree grows by about 0.5 MB.

The batch tools now start the editor with `--burst-disable-compilation`: twice the editor's Burst compiler crashed at start-up in the runner, and no code here uses Burst.

**The live budget after change 1** (`MeasurePathLiveBudget`, opt-in with `ACOUSTIC_PATH_BUDGET`).
The Church as the playable Church runs it: persistent rays at population 11,000, GPU audio, binaural, one update of 1,024 samples at a time, with a fence after each phase; medians over 40 updates after a horizon of them, with the listener on the take's walk.

| Phase | ms per update |
|---|---:|
| tracer (edge slice, walk, population control, histogram and ring) | 6.2 |
| audio without the denoiser (path kernels, input transform, partitioned convolution, PCM, readback of 1,024 samples) | 2.3 |
| denoiser (the difference with it on) | 1.0 |
| total | 9.5 |
| CPU time of the audio calls on the main thread | 0.15 |

So the round has gone from about 27 ms (24 ms of tracer, 1.6 ms of denoiser and the audio) to 9.5 ms.
The target of 5 ms is not reached yet.

**Tried and not kept: a roulette on long chains** (`settings.pathIterationCap`, reverted).
Instrumenting the walk showed that about 180 of the update's 11,000 threads run more than 16 loop iterations and about 32 run more than 32; the longest runs about 280.
A ray past 32 iterations within an update survived each further reflection with probability 0.75 and was reweighted, which is unbiased.
The walk went from 6.0–6.2 ms to 4.4–4.6 ms, less than the chain length suggests: the dispatch is bound by many moderately long chains, each iteration's latency being a traversal, the beam scoring and the histogram atomics, not by the single longest one.
Its variance cost was not measured, and the timings on this machine moved by more than 1 ms between runs while other jobs ran, so it is not kept on this evidence.

**Hardware ray tracing on the M5, a feasibility assessment (not built).**
Unity 6.4 reports neither ray tracing nor inline ray tracing on Metal here (`supportsRayTracing`, `supportsInlineRayTracing`, recorded by `MeasurePathProfile`).
A native plugin can reach the GPU underneath:
- `IUnityGraphicsMetalV2` gives the device and the current command buffer.
- `ComputeBuffer.GetNativeBufferPtr` gives our buffers as `MTLBuffer`.
- An `MTLAccelerationStructure` can be built from the triangle buffer: one per object, plus an instance structure refitted when the doors move.

But intersection queries exist only in Metal Shading Language kernels, and the walk is Unity HLSL compiled by Unity.
The walk would have to be compiled outside Unity: HLSL through DXC to SPIR-V, then SPIRV-Cross to MSL, with `TraceScene` replaced by an MSL `intersector` call.
The plugin would then dispatch it with our buffers bound by pointer, ordered with Unity's work through `CommandBuffer.IssuePluginEventAndData`.
That is a second toolchain and a second binding layout for the walk, a large kernel through SPIRV-Cross, and maintenance on every shader change.
The gain is bounded by the traversal's share of each iteration's latency, which is not yet measured.
The measurement to make first is the walk with the traversal replaced by a trivial one (an analytic box), at the same chains.

**Next, in order of expected gain:**
- **The walk's per-iteration latency.** Measure the traversal's share as above. Then split the walk into stages (trace, score, connect), or give the histogram writes a local accumulation in place of global atomics.
- **The audio.** Precomputed twiddles in the shared FFT, which now evaluates a sine and a cosine per butterfly (about 2,500 FFTs of 2,048 points per update). For late partitions with one or two impulse groups, assemble the kernel from fixed group spectra instead of eight FFTs.
- **The denoiser.** Its first convolution reads 292 channels per output; tile it in shared memory and try half precision.
- **Edges.** The edge slice is 0.8 ms. Cull edges too short for UTD in a band and edges that cannot see the listener's region, then use a spatial index for the walk's candidates.
- **Overlap.** The tracer of update n+1 with the audio of update n.

None of these have been measured yet.

## 28. Toward 5 ms: trustworthy timing, the traversal's share, the long-chain lottery, the FFT's twiddles

**Timing.**
`MeasurePathLiveBudget` now measures every variant in one process, interleaved in rounds of 8 updates (25 rounds, 200 updates each, after a horizon of warm-up each).
Drift from other work on the machine therefore falls on every variant alike.
It reports the median and the 10th–90th percentile range per phase, and the load average before and after.
The heavy-job lock is held throughout, but other users' light jobs kept the machine's load at 10–30 during every run here; that cannot be switched off from this side.
So the absolute times below carry wide ranges, and the comparisons rely on the interleaving: within one run the medians of identical work agree to about 0.1 ms.

**The traversal's share** (a profiling bit, since removed, that traced each walk segment twice; the second query is identical, so the chains do not change and the extra time is one traversal per step).
Interleaved, at load 11–20:

| Variant | tracer, ms (median [10th–90th]) |
|---|---|
| as is | 7.90 [6.90–9.24] |
| each segment traced twice | 10.51 [9.41–12.42] |
| no histogram or ring writes | 7.88 [6.92–9.78] |
| no shortcut and no edge connections | 7.31 [6.30–8.69] |

- The traversal is about 2.6 ms of the tracer's 7.9 ms, a third.
- Hardware ray tracing could at best remove most of that, about 2 ms, and leaves the other two thirds (beam scoring, the walk's bookkeeping, population control, the edge slice).
- That is real, but not by itself the route to 5 ms. The native plugin's cost (a second shader toolchain for the walk) is not justified yet; it is the first lever after the cheaper ones below.
- The histogram and ring writes cost nothing measurable (7.88 against 7.90 ms), so accumulating them in threadgroup memory instead of global atomics would gain nothing; it was not built.
- The connections are about 0.6 ms.

**The long-chain lottery** (`settings.pathIterationCap`, `pathIterationSurvival` = 0.75).
After `pathIterationCap` loop iterations of a ray within one update, each further reflection survives with probability 0.75 and the survivor is reweighted, which is unbiased.
Time, interleaved at load 27–31 (wide ranges, stable medians):

| Variant | tracer, ms |
|---|---|
| population 11,000, no cap | 7.68 [6.43–12.03] |
| cap 32 | 6.63 [5.53–11.73] |
| cap 32, population 14,000 | 7.75 [6.67–12.08] |
| cap 32, population 17,000 | 8.70 [7.38–12.76] |
| cap 16 | 6.06 [5.13–10.81] |

Noise, Church standing and walking windows (2 seeds × 64 updates each), in dB.
"Seed against seed" compares two independent runs, so the target's own noise is absent; "against 8 N" compares with part 25's 8 N targets.

| Variant | seed against seed: window / level / decay | against 8 N: window / level / decay |
|---|---|---|
| population 11,000, no cap (part 25's runs) | 8.31 / 5.57 / 5.17 | 10.23 / 5.15 / 4.81 |
| population 11,000, cap 32 | 8.34 / 5.76 / 5.06 | 10.12 / 5.17 / 4.76 |
| population 14,000, no cap | 8.26 / 5.47 / 4.82 | 9.73 / 4.65 / 4.59 |
| population 14,000, cap 32 | 8.40 / 5.41 / 5.05 | 9.77 / 4.56 / 4.67 |
| population 14,000, cap 16 | 8.21 / 5.36 / 5.08 | 9.69 / 4.66 / 4.64 |

- At the same population the lottery's extra noise is below what these runs resolve (cap 32 at 11,000 against no cap: 10.12 against 10.23 dB).
- At equal time it wins: cap 32 at 14,000 costs what no cap costs at 11,000 (7.75 against 7.68 ms) and is about 0.5 dB closer to the target in window and level, the gain of 27% more rays.
- Cap 16 at 14,000 matches no cap at 14,000 in noise. In the dataset render it took 6.0 ms against 7.8 ms per update.
- It is kept. The playable Church uses cap 16. The default stays off, so the checks that compare persistent and full solves ray for ray keep their meaning.

**The FFT's twiddles.**
The shared FFT evaluated a sine and a cosine per butterfly. It now reads them from a table computed in double precision, 1,024 entries for every size up to 2,048.
The audio phase went from 3.53 to 3.39 ms, interleaved.
`GpuPathReverbMatchesTheCpuReference` and the GPU audio parity checks pass unchanged.

**Where the audio's time goes** (profiling switch `GpuAudio.SkipPathKernels`: build the path kernels once and then leave them).
3.39 ms with the kernels built every update, 1.93 ms without, so the kernel build is about 1.5 ms. The rest is the input transform, the partitioned convolution, the direct path, and the readback with its fence.
Rebuilding only kernels whose bins changed beyond noise was not built. Undenoised, the Church's late bins change by several dB between updates (part 25's seed-against-seed errors), so most kernels would be rebuilt anyway.
The exact alternative is cheaper: about 220 of the 283 partitions hold at most two impulse groups. Their kernels can be summed from fixed group spectra (16 multiply-adds per bin and ear) instead of eight band FFTs, which would cut the kernel build to about a third. It is the next change.

**Where this leaves the budget** (medians from these runs at load 10–30): tracer about 6.1 ms with cap 16, audio about 3.4 ms, denoiser about 1.2 ms, against 7.7, 3.5 and 1.3 ms at the start of this part.
On a quiet machine part 27 measured 6.2, 2.3 and 1.0 ms for the earlier code.

Still open, in order:
- the group-spectra kernel build (about 1 ms);
- edge culling per band and by visibility (the edge slice and the edge connections, about 1.4 ms together);
- the denoiser's dispatches (about 25 small ones);
- overlap of the tracer with the audio. Unity issues compute work serially on Metal, so overlap hides latency but does not lower GPU time; it needs a measurement of whether the tracer leaves the GPU idle.

## 29. Toward 5 ms, continued: group spectra, the denoiser's steps, edges, and whether 5 ms is in reach

All times are from `MeasurePathLiveBudget` (interleaved variants, medians with the 10th–90th percentile range). They were taken on a quiet machine this time (load average 4–7).
At that load the ranges are tight and earlier parts' loaded numbers run 1–2 ms high.

**Audio: kernels from the groups' spectra** (`PathBandGroups`).
Most of the late kernel's partitions hold one or two impulse groups.
For those, the filtered partition is summed in the frequency domain from the groups' fixed spectra (PathReverb's own zero-padded FFTs of their two slot sequences), weighted by the band filters and the groups' coefficients. That replaces eight band FFTs per partition.
The rest keep the band FFTs.
It is exact: `GpuPathReverbMatchesTheCpuReference` gives the same agreement as before (3.75 × 10⁻⁵ binaural, 6.3 × 10⁻⁶ mono).

| Variant (cap 16, population 11,000, no denoiser), 480 updates each | audio, ms |
|---|---|
| every partition through the band FFTs | 2.45 [2.30–2.77] |
| group spectra where a partition has at most two groups | 2.17 [2.00–2.49] |
| kernels built once, then left (a bound) | 1.34 [1.14–1.62] |

- The kernel build went from 1.11 to 0.83 ms. That is less than the FFT count suggested: the early partitions, the transforms of the pieces and the coefficient passes remain.

**Denoiser: where its time goes** (`MeasurePathDenoiserSteps`: a fence after each step at 447 bins, minus the fence's own round trip of about 0.2 ms).
- The first convolution takes about 0.26 ms, the other seven about 0.07 ms each, the scale reduction about 0.07 ms.
- Every other step is about 0.01–0.03 ms.
- The convolutions are about 0.75 of the denoiser's 1.0 ms. Fusing the small steps' dispatches would save at most about 0.1 ms, so it was not done.
- **Tiling, not kept:** a convolution staging its input in threadgroup memory (64 bins × 4 outputs per group, 32 channels at a time) matched the model (worst relative error 1.9 × 10⁻⁵) but was slower, 0.72 against 0.26 ms for the first layer and 0.26 against 0.08 ms for the others. Its barriers cost more than the reads it saved, so it was removed.
- **Half precision: not tried.** Rounding the activations to 11 bits gives relative errors of order 10⁻³ per layer, against the check's tolerance of 2 × 10⁻³; it would need the check relaxed, which is not what was asked.
- A smaller network, 64 channels wide instead of 96, would take the convolutions to about half; that needs retraining.

**Edges.**
- The Church has 14,075 diffracting edges, with lengths at the 10th, 50th and 90th percentiles of 0.13, 0.56 and 15.3 m.
- **Per band, UTD validity.** UTD's wedge solution assumes the edge is long against the wavelength, so in a band where it is not, the method's UTD energy is not the edge's. Dropping such an edge from that band biases that band low by the edge's contribution, and BTM, which handles finite edges, is the reference for that bias. The Church's share of edges shorter than one wavelength is 2.4% at 16 kHz, 5.3% at 4 kHz, 29% at 1 kHz, 71% at 250 Hz and 77% at 125 Hz. The edge work is the same for all eight bands at once (one sampled point, one visibility ray, one UTD evaluation per band), so it is saved only for an edge culled in every band, which is the 2.4% shorter than 2 cm. That culling would save nothing measurable, so it was not built. Its bias against BTM was therefore not measured; the half-plane and solid checks use edges of metres and would not see it.
- **By visibility of the listener's region.** The edge slice (about 0.8 ms) and the edge connections (part 28: about 0.6 ms with the straight ones) are the whole edge budget, so this culling could save at most about 1 ms. To stay unbiased it may drop only edges proven invisible from the whole region, which takes conservative occlusion queries per region. That was not built.

**The whole update, quiet machine** (cap 16, group spectra, 320 updates each):

| Population | tracer, ms | audio with the denoiser, ms | total |
|---|---|---|---|
| 11,000 | 5.15 [4.75–5.88] | 3.22 [2.90–3.93] | 8.4 |
| 8,000 | 4.47 [4.02–5.28] | 3.22 [2.91–3.72] | 7.7 |
| 6,000 | 3.59 [3.28–4.29] | 3.20 [2.90–3.81] | 6.8 |
| 4,000 | 3.40 [3.02–4.12] | 3.18 [2.88–3.91] | 6.6 |
| 11,000, without the denoiser | 5.17 [4.73–5.77] | 2.15 [1.91–2.59] | 7.3 |

- Each phase's figure includes one fence's round trip (about 0.2 ms), so the GPU's own work is about 0.4 ms less than the total.
- **Is 5 ms in reach?** Not with the present structure.
- The audio side, the kernels, convolution and readback (about 2.2 ms) plus the denoiser (about 1.0 ms), does not depend on the number of rays.
- The tracer has a floor of about 3.3 ms below 6,000 rays: the edge slice, the population control and the dispatches, and the chains of the rays that remain.
- At 4,000 rays the update still costs 6.6 ms.

**The two remaining routes, with numbers** (neither built; the user decides).

(a) **Hardware ray tracing through a native Metal plugin.**
- Part 28 measured traversal at about a third of the tracer. At 11,000 rays on the quiet machine that is about 1.7 ms of 5.15 ms; at lower populations it is less.
- Hardware traversal would recover most of it but not all, since the queries remain.
- Expected: the update from 8.4 to about 6.9–7.2 ms at today's accuracy.
- Cost: a second shader toolchain for the walk (HLSL through DXC and SPIRV-Cross to Metal, with the intersector call patched in), a native plugin that builds and refits the acceleration structures, and both kept in step with every shader change.

(b) **Fewer rays, the denoiser compensating** (`route_b.py` beside this file, after `lottery.py plan` and `MeasurePathPersistentSequences`; Church standing and walking, updates 32–63, two seeds, against part 25's 8 N targets; window / band level / decay curve in dB; the target's own noise is about 4.5–6 dB in window, so all rows are inflated alike):

| Population (cap 16) | update, ms | noisy input | after the denoiser |
|---|---|---|---|
| 11,000 (today) | 8.4 | 10.37 / 5.09 / 4.68 | 6.43 / 2.03 / 2.74 |
| 6,000 | 6.8 | 11.81 / 5.81 / 5.87 | 7.98 / 3.35 / 4.45 |
| 4,000 | 6.6 | 13.01 / 6.79 / 6.56 | 8.43 / 2.36 / 4.59 |

- No population fits 5 ms: the update still costs 6.6 ms at 4,000 rays.
- Halving the rays costs about 1.5 dB of window error and 1.7 dB of decay curve after denoising, and the denoiser was trained at the 23 ms populations, not at these.
- So fewer rays alone cannot reach 5 ms.
- With (a), and the denoiser retrained 64 channels wide (about −0.4 ms), the update would come to about 5.6–6 ms at 6,000 rays, still at (b)'s accuracy cost.

**5 ms needs the fixed costs to fall.** About 3.2 ms of audio and denoiser, and the tracer's 3.3 ms floor below 6,000 rays, do not scale with the number of rays.
Candidates, unmeasured:
- the convolution's partitions run at a coarser block for the late tail (non-uniform partitions; the late field changes slowly);
- the late kernel rebuilt every second update with the crossfade kept (a stated approximation of 23 ms in the late tail's timing);
- the denoiser narrowed and retrained;
- the edge slice refreshed less often (its weights are an importance distribution, so any refresh rate is unbiased).

## 30. The next round: what the convolution costs, and the edge distribution refreshed less often

Interleaved `MeasurePathLiveBudget`, cap 16, population 11,000, no denoiser, 320 updates per variant, load average 12–19:

| Variant | tracer, ms | audio, ms |
|---|---|---|
| as committed | 5.00 [4.55–5.56] | 2.06 [1.84–2.21] |
| the late convolution left out (`GpuAudio.SkipLateConvolution`, a profiling bound) | 4.97 [4.56–5.55] | 1.90 [1.66–2.02] |
| edge distribution refreshed every 4th update | 4.70 [4.25–5.29] | 2.03 [1.84–2.22] |
| edge distribution refreshed every 16th update | 4.62 [4.05–5.13] | 2.05 [1.84–2.21] |

**Non-uniform partitioned convolution: not built.**
- The whole late convolution (input transform, the products over 283 partitions, the inverse and the output) costs about 0.16 ms per update.
- That bounds what bigger late blocks can save, so the expected 1 ms is not there.
- The products are memory-bound and small: about 12 MB of kernels and inputs per update.
- The audio phase's remaining time is the path kernels (part 29: about 0.8 ms), the direct path, the denoiser and the readback with its fence.

**The edge distribution, refreshed every few updates** (`settings.pathEdgeSliceUpdates`; one source–listener pair).
Each refresh recomputes a slice of 2,048 edge weights and rebuilds the edge-ray distribution with a single-group scan over all 14,075 edges.
In between, the last distribution stays.
The walk samples edges from that distribution and divides by its probability, so any refresh rate is unbiased; only the variance can change.
Noise, Church standing and walking against the 8 N targets (window / level / decay, dB; `route_b.py`):

| Edge distribution refreshed | tracer, ms | noisy input | after the denoiser |
|---|---|---|---|
| every update (cap 16) | 5.00 | 10.37 / 5.09 / 4.68 | 6.43 / 2.03 / 2.74 |
| every 4th | 4.70 | 10.35 / 5.21 / 4.57 | 6.49 / 2.09 / 2.76 |
| every 16th | 4.62 | 10.38 / 5.13 / 4.84 | 6.53 / 2.11 / 2.89 |

- Every 4th update is within these runs' noise (about ±0.05 dB between configurations of equal noise) and saves 0.3 ms. The playable Church uses it.
- Every 16th update saves another 0.08 ms but is 0.15 dB worse in the decay curve after denoising. It is not used.

**Not done in this round** (in the order asked; each has a measurement to make first):
- **Sorting the pool by Morton cell and direction octant, fused with population control.** Part 28 put traversal at about a third of the tracer. Coherence would speed the traversal part and the histogram writes, so the walk can gain at most about 1.7 ms.
- **Energy-aware selection and systematic resampling.** The first measurement is CV² of the weights per cohort, in the Church, the shoebox and the furnished room.
- **A 64-channel denoiser retrained across 4k–23k populations.** It needs a dataset rendered at those populations, about 40 minutes.
- **The auto-budget:** N adjusted each update from the measured GPU time.
