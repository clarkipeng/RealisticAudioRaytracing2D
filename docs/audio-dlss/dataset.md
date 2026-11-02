# Audio-DLSS denoiser dataset

Written 29 October 2026.
This is the data pipeline for the low-budget → converged-tracer denoiser described in the [audio-DLSS plan](README.md) ("Revised direction, 29 October 2026").
Measured results of the first pilot are in the [denoiser pilot benchmark](../benchmarks/denoise-pilot-2026-10-29/README.md).

## What one example is

One example is one receiver in one generated scene, rendered by our own tracer at one realtime budget with one seed.
Its input is the complex pressure at 32 frequencies (125 Hz to 4 kHz in 125 Hz steps), split into eight path families with path counts and the largest single-path weight per family.
Its label is the same receiver at the converged budget, averaged over K = 8 independent seeds, with the unbiased sample variance stored beside it.
Inputs and labels are separate solver runs with separate seeds; nothing is decimated from the label.

| Family | Meaning | Estimator |
|---|---|---|
| `direct` | line-of-sight path, including transmission | deterministic |
| `R` | first-order image reflections | deterministic once candidates cover every triangle, sampled below that |
| `D`, `DD`, `RD`, `DR`, `RR` | complete early routes (one or two interactions) | Monte Carlo over edge points and, at low budgets, over routes |
| `walk` | residual random walk (later interactions) | Monte Carlo, heavy-tailed |

**The primary label is the early field: every family except `walk`.**
The residual walk does not converge at any affordable budget: its seed-to-seed spread does not shrink, and grows, as rays increase from 128 to 8,192 per bin, and single walk paths can outweigh the direct sound (pilot benchmark, "Residual walk").
It stays in every input as its own buffer, and an auxiliary `total_target_mean` (all families) is stored, but no current label can include it.
Fixing the walk's weights is a tracer change and comes before any late-field label.

## Budgets

The label budget is 512 walk rays per frequency bin and 65,536 complete-route samples, with 32 interactions and 1.25 s maximum delay, at 8 kHz with a 64-point FFT.
Inputs scale both knobs by 1/256, 1/64, 1/16 and 1/4, with four seeds each; 1/256 is the live preset (2 rays per bin).
Seeds come from named SHA-256 streams (`dlss_scenes.stable_seed`), never Python's `hash`.

## Scene generator

[`tools/dlss_scenes.py`](../../tools/dlss_scenes.py) builds seeded axis-aligned box scenes.
One `scene.json` drives both solvers: the Unity tracer merges the boxes with the automatic solid bake, and [`tools/fdtd_wave.py`](../../tools/fdtd_wave.py) voxelises the same `walls` list.
Every face lies on a 1/16 m lattice, which is exact in float32 (a 5 cm lattice produced sliver triangles in the bake) and is represented exactly by FDTD grids of 6.25, 3.125 or 1.5625 cm.

| Family | Varies | Split |
|---|---|---|
| `shoebox` | closed room 3–8 × 2.5–4 × 3–7 m | train |
| `doorway` | room with one door or window (30%) in a wall, 0.8–2.0 m wide; source inside or outside | train if width ≤ 1.5 m, **test** if wider (every third scene) |
| `corridor` | straight or L corridor open at both ends, or two rooms coupled by a door with an exterior door | **validation** (held-out family) |
| `canopy` | roof on posts, overhang on a back wall, or porch with side walls; no floor | **test** (held-out family) |
| `barriers` | 2–6 walls and pillars in a 6–9 m field; no ground | train |
| `furnished` | closed room with 2–10 requested tables, cabinets and partitions | train if fewer than 8 requested, **validation** otherwise (every third scene) |

Walls, floors and roofs are 0.25 m thick.
Open-air scenes follow the canopy benchmark convention of no ground plane.
Receivers lie on a horizontal grid at 1.2–1.8 m (random per scene), at least 0.3 m from any box and 0.5 m from the source; the grid indices are stored so neighbouring receivers can be averaged.
The source is at least 0.5 m from any box.
An analytic line-of-sight flag per receiver is computed independently of the shader.

Surfaces use the existing presets in `Assets/Materials/Church <name>.asset` (stone, wood, glass), one preset per role (floor, walls, roof) per scene and one per clutter box; no coefficients are invented.
The presets carry different interior media, so the bake forms one solid group per preset.
FDTD has no wall impedance yet, so the FDTD subset is additionally rendered with every surface rigid (`tracer-rigid`) and only that variant is compared with the wave solver.
Closed rooms (`shoebox`, `furnished`) are excluded from the FDTD subset: a lossless closed room has no finite-time single-frequency response.
Each wave bundle records how much its response still changes between 0.75 T and T (`truncationChange075`); responses changing by more than 5% have not settled and should not be used as labels (in the pilot: every corridor and most doorways).

The split rule lives in `dlss_scenes.split_for` and is frozen with the generator version.
Every seed, budget and receiver of a scene stays with its scene, and the split ancestry group (`splitGroup`) maps to exactly one split.
`tools/fixtures/dlss-scenes-v1.json` freezes the scene hashes; any generator change must bump `GENERATOR_VERSION`.

## Layout

Bulk data lives under `.context/audio-dlss-data/` (ignored by git), following [simulation contracts §9](../simulation-contracts.md):

```text
.context/audio-dlss-data/
  generation.json                    generator version, code hash, scene hashes
  scenes/<scene-id>/scene.json       the one description both solvers read
  scenes/<scene-id>/tracer/          Unity run: run.json, responses-<budget>.f32, counts-<budget>.i64,
                                     peaks-<budget>.f32, seconds-<budget>.f32, direct-visible.u8,
                                     boundary.json, provenance.json
  scenes/<scene-id>/tracer-rigid/    same, every surface rigid (FDTD subset only)
  scenes/<scene-id>/fdtd/            run.json, wave.f32 [P,2,2], truncation.f32, pressure.f32, source.f32
  logs/render-*/                     jobs, provenance (hashes of every synced source), resources, Unity log
  datasets/<name>/
    dataset.json                     identity = SHA-256 of this file
    generation.json                  resolved recipe, acceptance rule, rejections by family
    bundles.jsonl                    one bundle per scene: hashes, split, quality, descriptors
    examples.jsonl                   one row per (scene, budget, seed, receiver)
    derived/<scene-id>/              target-mean.f32 [P,Q,2], target-variance.f32 [P,Q],
                                     target-validity.u8, total-target-*.f32, context.f32 [P,6],
                                     rigid-target-*.f32 (FDTD subset)
```

Tracer payloads are raw little-endian arrays: `responses` is float32 `[seed, receiver, frequency, family, real/imag]`, `counts` int64 and `peaks` float32 `[seed, receiver, frequency, family]`, `seconds` float32 `[seed, receiver]`.
Bundles reference them in place, so run payloads are never copied.
Every descriptor declares file, dtype, shape, axes, units, C order, byte offset and length, and the whole-file SHA-256.
`run.json` records the SHA-256 of the exact `scene.json` bytes the renderer read, and each FDTD run does the same; packing fails if either differs.

Deviations from §9, all for a static experiment: `history_frames` is 1, availability is `static-offline-v1`, and the quantity is complex pressure per family rather than energy.
The context schema `static-geometry-v1` is source-minus-listener xyz, distance, analytic line of sight and the renderer's direct-path visibility.

## Acceptance

A label entry is valid when its standard error `sqrt(var/K)` is at most 10% of `|mean|` plus 5% of the scene's receiver-RMS at that frequency.
A scene is accepted when every value is finite and at least 90% of entries are valid; rejected scenes stay in `bundles.jsonl` with reasons and are counted by family in `generation.json`, but contribute no examples.

`tools/dlss_dataset.py verify` checks the index hashes, every payload hash and size, scene/run identities, the frozen split rule, group-to-split isolation, a whole held-out family, and every example's bundle, variant and receiver.
`tools/dlss_dataset.py identity` checks that the Unity bake, the FDTD voxels at 6.25 and 3.125 cm and the exact box union have the same solid volume.

## Reproduce

```sh
PY=.context/fdtd-teacher/venv/bin/python          # tools/fdtd-requirements.txt
uv venv --python 3.12 .context/solid-python        # bake backend for the Unity runner
VIRTUAL_ENV=.context/solid-python uv pip install -r tools/solid-boundary-requirements.txt
nice -n 19 $PY -m unittest discover -s tools -p test_dlss.py
$PY tools/dlss_jobs.py test TakeChecks.CompleteSampleFamiliesMatchDebugCapture
sh docs/benchmarks/denoise-pilot-2026-10-29/run_queue.sh
$PY tools/dlss_dataset.py pack denoise-pilot-v1
$PY tools/dlss_dataset.py verify .context/audio-dlss-data/datasets/denoise-pilot-v1
$PY tools/dlss_dataset.py identity
$PY docs/benchmarks/denoise-pilot-2026-10-29/analyze.py .context/audio-dlss-data/datasets/denoise-pilot-v1
```

The Unity stages use the isolated runner at `.context/gpu-audio/runner` (never the working editor), Unity 6000.4.1f1 on Metal, `nice 19`, one job worker, 20% work pacing, a 4 GiB process-tree memory guard, and suspend while the Mac is on battery.
A scene is accepted by `dlss_jobs.verify_render` (payload sizes, finite values, scene and budget identity), not by the test runner's exit code; invalid outputs are set aside and re-rendered.
The measurement ignores audio-device and service logs, which a long suspend can trigger, and refuses to publish a scene after any other error log.
The FDTD stage uses the same pacing, a 4 GiB GPU/RSS guard and the FDTD study's battery pause.

## Renderer export hook

The only renderer change is a diagnostic: `RayTransport.CompleteSampleFamilies()` (exposed as `TransportSolver.CompleteSampleFamilies()`) reads back the route tables of the ordinary dispatch and repeats the shader's sample-to-route mapping, so each complete-route sample can be labelled D/DD/RD/DR/RR without the slower debug capture.
It is synchronous, single pair and single site, and changes no physics or shader.
`TakeChecks.CompleteSampleFamiliesMatchDebugCapture` checks it sample by sample against the existing debug capture, and the dataset measurement asserts that the family split reconstructs the ordinary summed response at every frequency.
