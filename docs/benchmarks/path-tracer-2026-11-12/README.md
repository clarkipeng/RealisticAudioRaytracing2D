# One energy path tracer (`path` method) — 12 November 2026

**Verdict: one loop, one output, one estimator.**
The `path` method replaces the energy method's stages (exact enumeration of low-order arrivals, a receiver sphere for the rest, the 96 nearest planes, tail extrapolation, one-second seed averaging) with one unbiased Monte Carlo path tracer whose only output is an energy histogram per source/listener pair, and audio is synthesised from that histogram alone.
It passes every analytic check the energy method passes, and one it could not: a specular image whose detection region is half blocked is now estimated without bias (1.00 against 0.70 for the plain estimator).
On the wave suite it beats the static-IR baseline in level everywhere (4.05 against 8.85 dB RMS at 2,048 rays, 3.26 at 16,384) and, with 8 times the rays, matches the energy method at 2,048 (3.26 against 3.13 dB) without its biases; at equal rays it is noisier, which is what the denoiser is for.
Its cost is the open problem: 1–18 times the energy method per ray, most in closed specular rooms (see Cost).
The energy and hybrid methods are unchanged and remain selectable until their removal is approved.

All numbers are measured on this Mac (Apple M5, 16 GB, Unity 6000.4.1f1 on Metal) in the isolated runner under `tools/heavy_guard.py` (one heavy job machine-wide, 4 GiB guard, nice 19, one job worker, 50% pacing).

## The loop

```text
for each ray (all eight octave bands carried together):
    vertex v = source; weight = 1
    loop:
        connect v to the listener:                                   next-event estimation
            straight, through any walls (transmission per partition)
            through one edge, one mirror and one edge, or two parallel edges (UTD energy kernel)
            (ray 0 does the source's straight connection for all rays: it is deterministic)
        if v is the source, with probability p_edge: an edge ray       edge diffraction as an event
            (edge and point drawn with density proportional to 1/m^2, azimuth uniform on Keller's cone)
        else: a direction (uniform at the source, cosine at a scattering point)
        follow the ray; at each surface:
            air absorption over the segment (ISO 9613-1, per metre)
            choose absorb / transmit / scatter / specular by the material's energy fractions
                (a small transmission is split off into one reservoir branch per ray, traced later)
            scatter: this point becomes v, break to "connect"
            transmit through a partition or reflect specularly: the chain goes on, and
                connect the chain to the listener (below)
            Russian roulette once the weight is below 1e-6 of a fresh ray (unbiased)
        until the ray is absorbed, escapes, leaves the 2 s horizon or loses the roulette

connect a specular chain to the listener (the same loop's connection step):
    image = the listener mirrored through the chain's reflections
    if the ray left its vertex inside the region about the image direction
        (a cone of half-angle asin(r / distance); for an edge ray, a box in edge point and azimuth):
        retrace the exact image path; if valid, score
            its exact energy x N / (probability of the region)
        where N = number of fresh rays drawn in the region until one follows the same plane sequence
```

**Why N.**
A detection region is only unbiased if every ray in it follows the chain; near a visibility boundary part of the region does not, and the old estimator (energy method, 6 November) was low by that share (95% failed retraces in the furnished room).
The share `p` that does follow is unknown, but the number of independent draws until the first success has mean exactly `1/p`, so multiplying by it removes the bias without computing `p`.
The expected extra cost per image is the region's own probability (one region's worth of trials), whatever `p` is; the trials are capped at 64 (`pathMaxTrials`), and a capped detection is counted.
The region size `r` (`specularDetectionRadius`, 0.25 m) therefore changes variance and cost but not the mean: it is not a receiver volume the estimate averages over.

**The only constants.**
`p_edge = 0.5`, the roulette floor `1e-6`, the transmission-branch floor `0.05`, `r` and the trial cap change variance and cost, never the expectation (the cap only where it is reached, and it is counted).
`MinimumDistance` (0.1 m in audio) clamps 1/r² next to the listener, as every method does.
Materials are the scene's measured absorption, scattering and transmission per octave band.

**Assumptions** (shared with the energy method unless stated):
energy (incoherent) transport, no phase; Lambert scattering with the ISO 17497-1 coefficient; specular reflection with the random-incidence absorption (or `|R(θ)|²` of an impedance material); transmission applied once per partition (an object with two faces within 3 m) and undeviated through it; UTD for rigid wedges as an energy kernel, double diffraction only over parallel edges and only as a connection; geometry frozen during a solve.

**What it leaves out** (all small or rare in the scenes here, each a bias): chains that refract into a closed medium are not connected (no exact image through refraction); paths with two or more reflections before a diffraction (only one mirror before an edge is connected); sound that crosses a wall before it diffracts (edge rays need a clear line from the source); double diffraction over skew edge pairs.

**Implementation.**
`Assets/Shaders/MethodPath.hlsl` (kernel `Assets/Resources/Acoustics/PathMethod.compute`, its own shader), driven by `MethodTransport.cs`.
The chain connections are queued by the walk after a cheap region test and run by a second kernel, `PathConnect`, one candidate per thread: done inline, one long retrace stalled its whole SIMD group (4x slower in the stone shoebox).
Every heavy helper (BVH traversals) has one call site; the BVH variant compiles in about 5 minutes and the shader compiler stays below 330 MB.

## What was removed, compared with the energy method

| Energy method (11 November) | Path method |
|---|---|
| exact enumeration of direct, first- and second-order images and D/DD/RD/DR routes (a separate kernel) | the same paths come out of the one loop; only the direct sound is still rendered as an exact delayed path, and it is the loop's first connection |
| second order only for the 96 planes nearest the source and listener (`methodNearPlanes`) | no plane selection: every order, every plane |
| a receiver sphere (0.25 m; 2 m in the Church) scoring every other specular segment (biased: an average over the sphere) | exact image retrace for every specular chain, unbiased (measured share) |
| tail continuation past the horizon from a fitted decay (`lateTailSeconds`, `LateTail`) | none: the horizon is simulated (5 s in the Church renders) |
| moving average over 1 s of solves with new seeds (`lateAccumulationSeconds`) | none: one solve per update with a new seed; the noise is left for the denoiser |
| room-energy reference for the tail fit | none |
| late histogram in 23 ms bins, audio as noise under a smooth envelope | 0.25 ms bins, audio as one impulse per 0.25 ms |

## Output

One histogram per solve and source/listener pair: energy `[direction 6][band 8][bin]` in the direct-sound convention (free field at 1 m = 1), plus the number of contributions per bin.
**Time bins:** 0.25 ms up to 16 ms (64 bins), then each bin is 1/64 of its start time (constant relative resolution; 0.78 ms at 50 ms, 3.1 ms at 200 ms, 15.6 ms at 1 s), 376 bins to 2 s and 455 to 5 s (`pathBinSeconds`, `pathFineBins`).
Single reflections, which the ear hears as echoes, arrive early and get sub-millisecond bins; later the reflection density makes the field diffuse and a fixed fraction of the arrival time is finer than its envelope changes.
**Directions:** the six axes of the listener's head frame (world axes for measurement), each arrival split by the squares of its direction's positive components (they sum to one), so an arrival moving round the head changes its weights smoothly instead of jumping between bins.
**Counts:** contributions per time bin (all bands share one geometric path), a direct measure of how many samples a bin rests on, for the denoiser.

## Audio from the histogram

`Assets/Scripts/Simulation/PathReverb.cs` (CPU reference; the GPU audio backend falls back to it for this method):
- Time is cut into 0.25 ms slots; each holds one impulse at a fixed random position inside the slot with a fixed random sign. Its energy is the histogram energy in the slot (a bin spreads its energy evenly over its span), so a fine bin is one impulse with amplitude √E and a late, wide bin is a train of impulses sharing its energy.
- Band filtering: each band of the source signal is convolved with its own impulse train (same positions and signs, that band's amplitudes) and the bands are summed; the band weights are cos² in log frequency between neighbouring centres, which sum to one, so equal band energies give one flat click. The filters are linear-phase FIRs with half a partition of lookahead inside the DSP latency.
- Ears: the six directions combine per ear with the existing diffuse-field equalised HRTF band energies, and two sign sequences are mixed per band to the HRTF set's diffuse-field interaural coherence. This is interaural level, not interaural delay: early echoes are lateralised by level only.
- Positions and signs are fixed per source/listener pair, so an update changes amplitudes only; updates crossfade sample by sample over one response step (two kernel banks); uniform partitioned convolution, with each partition's kernel a weighted sum of fixed spectra (no FFT per update).
- **The direct sound** is the exact arrival of the loop's first connection (retarded delay, band energies, direction, Doppler), rendered by the existing delayed-path synthesis with the full HRTF.
- **Doppler of reflections is not rendered.** A per-bin delay drift could only be derived by matching bins between successive noisy histograms, which is not principled; the direct path is exact, and Doppler of reflected sound needs the denoiser's temporal model (or exact arrivals).

Check `PathReverbReproducesTheHistogram` (CPU, quick tier): a single fine-bin arrival comes out as one click at its delay with the band energies of its interpolated spectrum to within 0.5% from 250 Hz to 4 kHz (125 Hz: 0.95, the 125 Hz band's filter is longer than the window the check reads), and a stationary late field over 0.1–1.4 s keeps its band energies within 3–5% (one realisation of random signs; about 10% spread expected at 125 Hz).
Because one impulse's spectrum interpolates between band centres, a sloping spectrum's band energies come out slightly low: about 0.15 dB for −1 dB per octave, 0.4 dB for the steeper slope of a decaying room.

## Validation

Analytic checks (`Assets/Tests/Editor/MethodChecks.cs` with `TakeRender.Method.Path`, `Assets/Tests/Editor/PathChecks.cs`; categories `Methods` and `Slow`):

| Check | Path method | Energy method (for reference) |
|---|---|---|
| Free field, every band | 1.000000 × 1/r², all in the arrival bin | same |
| One rigid plane, 2^18 rays | image 0.953 × 1/rᵢ², total 0.981 | 0.95, 0.98 |
| Rigid shoebox image lattice (α = 0.1), 2^16 rays, 200 ms | total 0.991, worst 20 ms window 8.1% (5.4% before the deferred pass, another draw) | 0.9995, 8% |
| Diffuse shoebox (α = 0.2, s = 1) vs Eyring | T60 0.613 s vs 0.585 s (+4.8%) | same |
| Rigid half-plane (shadow) vs BTM | +0.29 … +0.31 dB, 250 Hz–2 kHz | +0.29 … +0.31 dB |
| DD over a 2 m wall vs BTM | +0.49 (2 kHz) … +1.58 dB (125 Hz) | same |
| DD over a 0.25 m wall (reported, not asserted) | +4.6 … +8.1 dB (UTD double diffraction, known) | same |
| **Image whose detection region is clipped by a panel** (new) | **0.996** with the measured share; 0.699 with one trial (the plain estimator) | (biased like the plain estimator) |
| Audio synthesis reproduces the histogram (new, CPU) | see Audio | – |

The full regression tier is in Regression below.

## Scoreboard

Full tables: [scoreboard.md](scoreboard.md) (`scoreboard.py`, one command).
RMS error against the wave ground truth over the 11-scene suite, all bands with ground truth pooled, both seeds.
All methods at 2,048 rays; the path tracer also at 16,384 rays on the nine scenes where that render finished (corridor and furnished room were cut for time, see Cost).

| Scenes | Method | level dB | EDT % | T30 % | C50 dB | C80 dB | ETC dB |
|---|---|---:|---:|---:|---:|---:|---:|
| all 11 | baseline (static IR) | 8.85 | 420 | 38.0 | 5.9 | 5.9 | 10.1 |
| all 11 | energy, 2,048 | 3.04 | 269 | 78.7 | 1410 | 1925 | 15.1 |
| all 11 | **path, 2,048** | 4.05 | 1012 | 88.2 | 1530 | 2026 | 16.4 |
| closed rooms | baseline | 1.73 | 15.4 | 6.1 | 1.95 | 1.60 | 4.9 |
| closed rooms | energy, 2,048 | 1.99 | 18.4 | 23.9 | 1.86 | 1.70 | 5.5 |
| closed rooms | **path, 2,048** | 1.95 | 25.2 | 24.8 | 2.00 | 1.90 | 11.6 |
| 9 scenes | energy, 2,048 | 3.13 | 285 | 79.8 | 1496 | 2042 | 15.4 |
| 9 scenes | **path, 16,384** | 3.26 | 325 | 84.1 | 1416 | 1916 | 14.7 |
| 9 scenes, shadow | energy, 2,048 | 4.41 | 209 | 88.8 | 1857 | 2341 | 17.9 |
| 9 scenes, shadow | **path, 16,384** | 4.66 | 90 | 81.6 | 1815 | 2282 | 17.2 |
| 9 scenes, closed (two shoeboxes) | energy, 2,048 | 1.93 | 19.0 | 10.3 | 1.93 | 1.80 | 4.7 |
| 9 scenes, closed (two shoeboxes) | **path, 16,384** | 1.82 | 23.8 | 11.0 | 1.99 | 1.88 | 5.8 |

Reading:
- **Against the baseline** the path tracer wins on level everywhere (4.05 against 8.85 dB at 2,048 rays; 3.26 at 16,384), and in shadow also on EDT; it loses on T30, clarity and the energy-time curve, as the energy method does: the baseline carries the true reverberation, which in closed rooms hardly depends on position, and in the open scenes every geometric method writes almost nothing after about 100 ms where the ground truth keeps about 1%, so clarity errors are unbounded there.
- **Against the energy method** at equal rays the path tracer is noisier (level 4.05 against 3.04 dB, energy-time curves 16.4 against 15.1 dB, closed-room ETC 11.6 against 5.5 dB): the energy method's receiver sphere averages every late segment over a 0.25 m sphere (a bias, but a smooth one), while the path tracer scores exact image paths, which are rarer.
  With 8 times the rays it is level with the energy method at 2,048 (3.26 against 3.13 dB) and better in shadow (EDT 90% against 209%, T30 82% against 89%) and on clarity and energy-time curves.
- **Closed-room levels are unbiased**: 1.95 / 1.82 dB RMS against 1.99 / 1.93 for the energy method; the remaining closed-room error is the low-frequency modal behaviour no geometric method has (energy-audio benchmark, Part A).
- The per-scene detail (`.context/method-compare/score/report.json`) shows the furnished room as the hardest case: at 2,048 rays its energy-time curves are 17–21 dB off, from noise, not bias (its levels are within 1.3–2.8 dB): its perfectly specular boxes give long, narrow image chains, which the region detects rarely.
  The suite has no scattering at all, the worst case for an exact specular estimator; with real scattering coefficients most late energy reaches the listener by next-event connections from scattering points, which have no such sparsity.

## Cost

GPU time per solve (one source, one listener, all eight bands; nice 19; busy time of the measurement solves):

| Scene | path, 2,048 rays | path, 16,384 rays | energy, 2,048 rays |
|---|---:|---:|---:|
| canopy (open) | 11 ms | 178 ms | 10 ms |
| barriers (open) | 13 ms | 110 ms | 5 ms |
| doorway | 140 ms | 1.30 s | 45 ms |
| wood shoebox | 244 ms | 1.92 s | 17 ms |
| stone shoebox | 283 ms | 2.33 s | 42 ms |
| corridor (two rooms) | 1.10 s | – | 113 ms |
| furnished room | 3.31 s | – | 188 ms |

**The path tracer is not realtime yet, and is 1–18 times slower than the energy method at equal rays**, most in closed specular rooms.
The walk is the same; the cost is the chain connections.
Each queued candidate retraces its chain (up to hundreds of reflections in a 2 s closed room) and then draws trials until one follows the chain, one after the other in one GPU thread; a chunk of 1,024 rays leaves only tens of candidates, so a few long trial sequences set the time of the whole pass (furnished room: 6 trials per scored chain, 112 capped at 64).
Moving them out of the walk (`PathConnect`) already cut the stone shoebox from 1.05 s to 0.28 s per solve.
**Next step (not done):** run a candidate's trials in parallel, one GPU thread per trial index: the index of the first successful trial among independent trials is the same geometric count `N`, so the estimate is unchanged, and the latency of a candidate falls from `N` sequential retraces to one.
A second, smaller cost is register pressure: the walk carries the chain state and a copy for the transmission branch.

Shader compile: the BVH variant of `PathMethod.compute` takes about 5–7 minutes in the runner (the compiler stays below 330 MB); the non-BVH variant about 2 minutes.
Memory: the method checks and suite renders peaked at 2.3–3.3 GiB process-tree footprint.

## Denoiser data (pilot)

`tools/path_dataset.py` (plan, render, pack, verify); bulk data under `.context/path-dataset/` (ignored by git); dataset `.context/path-dataset/datasets/path-pilot-v1/`.

**One example** is one listener pose in one scene: a static receiver, or one frame of a moving-listener sequence (8 consecutive frames one response step apart, 1,024 samples at 44.1 kHz, walking at 1.4 m/s, so 3.3 cm per frame).
- **Inputs:** the path tracer's histogram `[direction 6][band 8][bin]` and its contribution counts `[bin]` at the realtime budget (1,024 rays), two seeds; every frame of a sequence has its own seed, as live solves do.
- **Target:** the same pose at 8,192 rays, two independent seeds (never the input seeds); `target-mean.f32`, `target-variance.f32` (unbiased, across the two seeds) and the summed target counts.
- Bins: the 376 path bins to 2 s (edges in `bin-edges.f32`); air absorption on; materials are the presets' absorption **and scattering** (unlike the wave suite, which needs zero scattering).
- Each scene: 4 static poses and one 8-frame walk, chosen with a named SHA-256 stream (`dlss_scenes.stream`), clear of boxes by 0.3 m and of the source by 0.5 m.
- **Layout and identity** as `docs/audio-dlss/dataset.md`: `dataset.json` (recipe, budgets, cost, noise, rejections), `bundles.jsonl` (one per scene: split, split group, every payload with dtype, shape, axes, byte size and SHA-256), `examples.jsonl` (scene, split, pose, static/moving, sequence and frame); `verify` re-hashes every payload and checks that each split group lies in one split and that the held-out families are where the frozen rule puts them.
- **Splits** (frozen `dlss_scenes.split_for`): corridor = validation and canopy = test are whole held-out families; doorways wider than 1.5 m and dense furnished rooms also go to test/validation; the Church is test.

**Pilot as generated: 10 scenes, 120 examples** (train: 2 shoebox, 2 doorway, 2 barriers; validation: 2 corridor; test: 2 canopy).
The two furnished rooms have inputs but no targets (they were cut for time: about 13 s per target solve) and are listed as rejected; the Church frames (3 windows of 8 frames: standing, walking, outside) are planned in `path_dataset.py church` but were not rendered.

| Family | Generation cost per scene (inputs + targets, GPU busy) | Target noise (energy-weighted relative standard error per bin) | Input noise (relative error per bin vs target) | Target level standard error |
|---|---:|---:|---:|---:|
| barriers | 0.1 + 0.4–0.9 s | 5–7% | 16–36% | 0.04–0.07 dB |
| canopy | 0.1–2.2 + 1–17 s | 9–10% | 32–57% | 0.4 dB |
| shoebox | 8–16 + 61–68 s | 16–28% | 71–120% | 0.07–0.10 dB |
| doorway | 7–14 + 55–63 s | 33–34% | 105–123% | 0.4–0.5 dB |
| corridor | 12 + 64–103 s | 20–23% | 83–102% | 0.11 dB |
| furnished (inputs only) | about 13 s inputs; targets estimated 300 s | – | – | – |

**Targets at 8,192 rays are not yet clean per bin in rooms** (16–34% per fine bin) although their band levels are within 0.1–0.5 dB: per-bin targets need about 16 times more rays (or, cheaper, coarser target bins).
Scaling: at the current speed one closed-room scene costs about 1–2 minutes of GPU for this recipe, so 1,000 scenes would be about a day; the parallel-trial change above is the lever.

## Listening render

Planned: `listen.py` renders the first 6 s of the Church speech walk (standing, turning the head, first steps) plus 3 s of tail, binaural, 5 s horizon, at realtime rays (2,048, a new seed every 23 ms update) and clean rays (16,384), and builds a page at `.context/path-tracer/listen/index.html`.
**Not delivered.** The realtime render was stopped after 12 minutes without its first batch of four updates finished (part of that was the shader compile in the render process, about 5–7 minutes); the clean render was not started.
In the Church (17,345 triangles, 14,075 diffracting edges, 31,000 m³ with 5–8 s of reverberation) every ray lives the whole 5 s horizon and its specular chains reach hundreds of reflections, which is exactly where the chain connections are slowest (see Cost).
Estimate: at least 1–3 minutes per update at 2,048 rays with this implementation, i.e. several hours for the 388 updates of the excerpt; the parallel-trial change is needed first.
The audio path itself is exercised by `PathReverbReproducesTheHistogram` (synthesis from a histogram) and by the method's audio wiring (`MethodTransport.DispatchAudio`, `PathExpand`), which this render was the first end-to-end use of.

## Regression

`tools/unity_tests.py full` (three fresh processes) after the last shader change:

| Batch | Result | Wall | Peak footprint |
|---|---|---:|---:|
| Quick (all non-Slow `TakeChecks`, including `PathReverbReproducesTheHistogram`) | 322 passed, 12 skipped (opt-in measurements), 0 failed | 154 s | 3.09 GiB |
| Slow renderer checks (this run also picked up the two Slow path checks; `unity_tests.py` now leaves them to the method batch) | 24 passed, 0 failed | 316 s | 3.28 GiB |
| Method checks (category `Methods`, energy, pressure and path cases) | 33 passed, 1 skipped (pressure DD, known gap), 0 failed | 71 s | 2.85 GiB |

The default method is unchanged (`Hybrid`); the energy and pressure kernels are unchanged; `RayTransport.Layout` treats the path method like the energy method with one arrival slot (the direct sound).
Unity time for this whole task was about 3 h 10 min of wall time (over the 3 h budget by the Church render attempt), most of it in shader compiles (5–7 minutes per change of the BVH kernel) and the suite renders.

## What is not verified

- **The Church listening renders** (realtime and clean) were not produced (see Listening render); the audio pipeline has not been run end to end on a take, only its synthesis on a synthetic histogram.
- The GPU audio backend does not synthesise the path histogram; the method uses the CPU reference (`PathReverb`) for every render, live included, and is far from realtime.
- Interaural time differences of reflected sound: the ears get per-band levels and diffuse-field coherence, not delays; only the direct sound has the full HRTF.
- Doppler of reflections is not rendered (direct sound only).
- Suite at 16,384 rays covers 9 of 11 scenes (corridor and furnished room cut for time); two seeds at 2,048, one at 16,384.
- Dataset: furnished targets and the Church frames are missing; targets are noisy per fine bin (16–34% in rooms).
- The trial cap (64) is reached in long chains of cluttered rooms (furnished room: 112 of 4,502 scored chains at 2,048 rays per 30 receivers), where the estimate is biased low by an unmeasured amount.
- Edge-chain regions (box in edge point and azimuth) are checked only through the half-plane and wall checks, which mostly exercise the next-event edge connections; a dedicated check of an edge ray followed by reflections is missing.
- The left-out path classes listed under the loop (refracting media, more than one reflection before a diffraction, transmission before a diffraction, skew edges) are not quantified.
- Register pressure and occupancy of the walk kernel were not profiled.

## Reproduce

```sh
PY=.context/fdtd-teacher/venv/bin/python
B=docs/benchmarks/path-tracer-2026-11-12
$PY tools/unity_tests.py run 'TakeChecks\.(Path.*|Methods.*\(Path\))'          # analytic and audio checks
$PY tools/method_runs.py jobs suite-path --scenes <suite ids> --methods path --budgets 2048 16384 --seeds 0 1
# render (one Unity process for all scenes): see the launch line in this README's history, or tools/method_runs.py render suite-path
$PY tools/method_runs.py pack && $PY $B/scoreboard.py
$PY tools/path_dataset.py plan && nice -n 19 $PY tools/path_dataset.py render && nice -n 19 $PY tools/path_dataset.py church
$PY tools/path_dataset.py pack && $PY tools/path_dataset.py verify
nice -n 19 $PY $B/listen.py render realtime && nice -n 19 $PY $B/listen.py render clean && $PY $B/listen.py page
```
