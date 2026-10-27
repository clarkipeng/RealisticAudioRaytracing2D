# Numerical wave reference versus current renderer

This review responds to the request to see the wave ground truth beside what the
renderer can reproduce. It includes a fresh measurement of the live 3D sampler
against the independently refined **250 Hz finite-3D rigid-block benchmark**, a
new **125 Hz rigid-canopy wave reference**, and the preserved improved **2.5D
complete-route prototype** results. These implementations and physical problems
are explicitly separated. None of these is a broadband rain recording.

The improved prototype was never integrated into the finite-3D rain renderer.
The October 21 results verified point-source/infinite-extrusion fields; October
22 runtime changes accelerated the older finite-3D sampler while preserving its
pressure, including large errors. See the [experiment history](../../acoustic-experiment-history.md)
and [original measurement ledger](../gpu-route-reuse-2026-10-21.json). This review
does not establish a regression from a previously accurate finite-3D renderer.

The generated `index.html` offers common-scale pressure and level maps, a complex
residual map, and one-cycle pressure traces at selectable receivers. It compares
2 and 128 rays per frequency bin, seeds 0 and 1, and diffraction disabled/enabled.
Animation is opt-in, capped at 30 fps, and stops when the tab is hidden. There is
no audio playback. `comparison.png` is a static scientific export of the 2-ray,
seed-0, diffraction-off case; `report.json` records all measured cases.

The two adjacent seeds share much of their ray sequence at higher budgets. They
are a sensitivity check, not independent statistical replicates or error bars.

## Current finite-3D benchmark result

| Rays/bin | Diffraction | Complex error, seed 0 | Complex error, seed 1 |
|---:|---|---:|---:|
| 2 | Off | 42.17% | 280.06% |
| 128 | Off | 27.80% | 27.40% |
| 128 | Experimental on | 1450.67% | 1450.70% |

With diffraction off, all 712 shadow positions are silent in this fixture at
both budgets. Enabling it at 128 rays fills almost all positions but introduces
very large contributions (maximum pressure amplitude 69.65, reference maximum
1.86). Both low-budget first-reflection sampling and experimental diffraction
need further work. A passing measurement test means the data are complete and
finite, not that the renderer passes a physical-fidelity threshold.

## Matched physical problem

- Point source at `(-3, -0.4, 0.2)` m, pressure amplitude 1 at 1 m in free space.
- Two rigid finite boxes, centered at `(-0.8, -0.9, 0)` and `(0.8, 0.9, 0)` m,
  with half-extents `(0.3, 1.1, 1.4)` and `(0.3, 1.1, 1.7)` m.
- 1,178 sampled air positions on the `z = 0.2` m slice, sound speed 343 m/s.
- Rigid, lossless boundaries; no scattering, transmission, or air attenuation.
- Complex convention `exp(+iωt)` and outgoing free-space pressure `exp(-ikr)/r`.
  No per-field gain fitting or normalization is applied.

The reference is the Helmholtz boundary-element solution implemented in
[`compare_wave_field.py`](../../../tools/compare_wave_field.py), independent of
the ray/edge kernels. Its finest 250 Hz mesh has 8,000 triangles and 4,004 DOFs.
The last mesh refinement changes the complex field by **0.260% overall** and
**0.983% in the geometric shadow**. An independent analytic rigid-sphere control
has **0.348% relative error** at 250 Hz. These are numerical checks, not rigorous
error bounds or real-world measurements. The 500 Hz reference has weaker shadow
convergence and is deliberately outside this review.

The renderer retains **32 bounces and a 1.25 s path horizon**. Its 1,000 Hz sample
rate / 8-point FFT configuration computes five bins; only the exact 250 Hz bin is
evaluated here. This is a complex transport-field measurement, without source
mixing, audio synthesis, temporal interpolation, HRTF, or playback. The wave
reference is steady-state; differences combine finite native support, sampling
variance, and propagation-model error.

## Rain-canopy reference

The seven boxes and source position come unchanged from the frozen native rain
fixture. Patch 39 is at `(3.25, 0.001, -1)` m. This diagnostic drives that one patch
with a unit 125 Hz tone and samples 1,166 positions on the listener-height plane
`y = 1.6` m. It does not mix the 98 recorded source textures. Both solvers use
rigid surfaces: the rain fixture's absorption/scattering values do not uniquely
define the complex boundary conditions a wave solver requires. No reflecting
floor is added, and there is no roof transmission or structural vibration.

The BEM uses the **exact exterior of the box union**, removing internal touching
faces. A tensor grid splits at every box boundary before surface subdivision;
all surface edges have exactly two oppositely oriented incident faces. Signed
surface volume matches independently integrated occupied-cell volume: 28.0625
m³, with exterior area 233.25 m² at every mesh size. The native renderer retains
the original seven boxes, exposing the same physical exterior through its usual
scene representation. Reference meshes have 2,448 / 4,960 / 7,904 triangles at
0.5 / 0.35 / 0.25 m maximum axis spacing. Refinement changes are reported in the
canopy ledger and viewer. They are not rigorous total-error bounds.

The final refinement changes complex pressure by **0.430% overall** and **1.363%
in the shadow**. The current sampler's seed-0 error is **55.92% at 2 rays**, or
**52.06% at 128 rays**, with diffraction disabled. At 128 rays, experimental
diffraction raises it to **294.64%**. These are single-frequency propagation
errors; they do not score the rain source texture or the full recording.

One receiver lies exactly on a geometric grazing line. Rounding scene inputs to
Unity's float32 representation changes its visibility classification. The
independent visibility check uses those rounded inputs; all field metrics retain
the point, and the ledger records its index and position. It is not silently
removed or treated as a general propagation failure.

## Reproduce

Use a closed, disposable Unity runner with the current project runtime and test
dependencies. The script verifies runtime file hashes against the working tree
and refreshes only the opt-in measurement fixture. It rejects the main project
and an existing output directory. The reference cache is
`.context/wave-field/run`; its original scene and NPZ hashes are checked during
packaging.

```sh
python3 docs/benchmarks/wave-review-2026-10-27/render.py \
  .context/gpu-audio/runner .context/wave-review-new

OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 OMP_NUM_THREADS=1 \
nice -n 19 .context/wave-planar/venv/bin/python \
  docs/benchmarks/wave-review-2026-10-27/package.py \
  .context/wave-field/run .context/wave-review-new .context/wave-review-view
```

For the canopy, prepare the exact fixture once, solve the three meshes
sequentially, then use the same renderer and packager:

```sh
OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 OMP_NUM_THREADS=1 \
nice -n 19 .context/tone-artifacts/venv/bin/python \
  docs/benchmarks/wave-review-2026-10-27/canopy.py prepare .context/canopy-wave-new

# Repeat reference for --mesh .5, .35, then .25 (one worker at a time).
OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 OMP_NUM_THREADS=1 \
nice -n 19 .context/tone-artifacts/venv/bin/python \
  docs/benchmarks/wave-review-2026-10-27/canopy.py reference .context/canopy-wave-new --mesh .5

python3 docs/benchmarks/wave-review-2026-10-27/render.py \
  .context/gpu-audio/runner .context/canopy-native-new \
  --reference .context/canopy-wave-new --hz 125

OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 OMP_NUM_THREADS=1 \
nice -n 19 .context/wave-planar/venv/bin/python \
  docs/benchmarks/wave-review-2026-10-27/package.py \
  .context/canopy-wave-new .context/canopy-native-new .context/wave-review-view/canopy \
  --hz 125 --fine-mesh .25 --coarse-mesh .35
```

Native work uses background task policy, nice 19, one Unity job worker, a 20%
completed-work duty budget, and a 3 GiB sampled parent-process RSS guard. Pacing
occurs after approximately 50 ms of accumulated work, avoiding per-receiver
background timer coalescing. A partial first run with per-receiver pacing was
stopped; only the completed batch-paced measurement is used for this review. Timing
columns exclude the deliberate pacing. RSS sampling is not a whole-process-tree
memory limit. Packaging requires a passed native measurement, complete unique
receiver indices, finite values, direct-visibility agreement with an independent
segment/box check, identical physical inputs, and unchanged runtime/test hashes.

The test's `ACOUSTIC_WAVE_WORK_FRACTION` setting is opt-in and affects offline
measurement pacing only. No production propagation behavior changes in this
review.

## Saved evidence and verification

- [Block measurements](blocks-report.json) and [canopy measurements](canopy-report.json)
  retain all settings, source hashes, errors and the reference metadata.
- [Reproduction archive](evidence.zip) includes the measured CSVs, all three new
  canopy wave fields, reused block reference fields, source scripts, native test
  reports, resource samples, and a SHA-256 manifest. The interrupted per-point
  pacing run is retained separately. Its 4,053 shared pressure values match the
  completed run exactly.
- Both native measurement tests passed, covering 18,752 finite complex-pressure
  samples. Input/fixture hashes, complete unique receiver indices, and independent
  float32-geometry visibility checks passed during packaging. The three canopy
  meshes pass closed-manifold/winding/volume checks and solve to GMRES residuals
  below 1e-8. These are data and reference checks, not renderer-fidelity passes.
- Native wall times were 178.8 s for blocks and 206.9 s for canopy with deliberate
  pacing. Peak sampled native process RSS was 1,092 MiB; the finest wave worker
  peaked at 976 MiB. No speakers, microphone or foreground Unity session ran.
- Browser review exercised all 16 budget/seed/diffraction combinations, both
  display modes, receiver controls and animation start/stop. Metrics agree with
  the ledgers; images load; no console errors or horizontal overflow were found.
  Scientific figures and browser screenshots were visually inspected.
