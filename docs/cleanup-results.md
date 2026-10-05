# Cleanup validation and measured bottlenecks

## 2026-10-05 compact capture cleanup

- Removed unreferenced `TrainVisualLoader` and `AudioConvolve`, plus the duplicate FFT scene/component. Automated FFT/direct GPU checks remain.
- Reduced `ComputeHelper` from 553 to 101 lines. Legacy manager/audio code now lives in explicit partial files, retaining serialized component references; current Inspectors omit legacy controls.
- Station capture writes input states, sample/tick timing, source events, identity metadata and small diagnostics by default. Full poses/geometry/PCM/output audio are optional diagnostic reference data.
- `StationReplay` reconstructs in Edit mode and is shared by Inspector rendering and batch jobs. It uses public scene/controller/audio preparation APIs and the same initialized simulation step. Explicit ground queries replace reliance on Play-only collision callbacks.
- 397 core checks and Metal FFT/direct/batched/live-versus-offline checks passed. The three-lap, 4,801-tick recording reconstructed byte-identical states/events and stereo audio across 4,225,025 frames; its five replay files occupied 39,176 bytes. Artifact: `Recordings/station-20261005-192306-273/`.
- A clean Git checkout at `2ad3c7e` and a fresh batch Editor (`-nographics`, no live audio) rebuilt a normal six-file recording. State/event hashes and all 896,001 non-silent stereo frames matched the original Editor's offline reconstruction exactly. The generated comparison opened successfully in the native comparison window. Inputs: `Recordings/station-20261005-194749-220/`; job, output and equality report: `.context/cleanup/checkout-job-final.json`, `checkout-render-final/`, `checkout-comparison-final.json`.
- That recovery test exposed unstable compiler module IDs across checkout directories. New Editor recordings validate relative project paths and file contents instead; older recordings and Player builds retain their original identity checks. Matching Git history supplies the files, but automatic checkout is still not implemented.
- Batch parsing now preserves absent/null comparison and reference jobs. Unity's inline JSON serializer previously instantiated those optional objects, triggering unrequested work. The parser uses the already-installed Json.NET 3.2.2 package, now declared directly. Regression checks cover optional jobs; the final batch run produced only `low/` and `high/`, with no reference render.
- A separate replay cadence test measured 4.90 and 9.99 FPS with identical final scene positions. The test deliberately throttles the Editor; it checks simulation accuracy, not uninterrupted output at those cadences. Artifact: `Recordings/station-20261005-193400-102/`.
- The default station now uses CPU tracing and 8,192 lead frames (185.8 ms at 44.1 kHz), down from 24,000 (544.2 ms). A 32.6-second standalone Mac run recorded only six compact files, with zero underrun/late/overflow samples. Maximum observation gap: 1,025 samples; maximum direct solve: 0.996 ms. Built app and report: `.context/cleanup/Station.app`, `.context/cleanup/player-result.json`.
- The background Editor remained near 10 FPS even with higher requested frame-rate caps. The three-lap capture had one 1,024-sample deadline miss and a 9,215-sample observation gap. Replay can also miss deadlines at this cadence or during script compilation. The smaller lead is not a guarantee against Editor stalls. Standalone measurements are reported separately.

## Earlier measurements

Historical measurements from 2026-09-29: Unity 6000.4.1f1, Apple M5 CPU/GPU, Metal, using an isolated validation project. At that time the main project was still on Unity 2022.3.62f3; it has since migrated to Unity 6000.4.1f1. The numbers below are unchanged, were not rerun for the documentation consolidation, and do not establish Windows/DirectX results. See the [README](../README.md) for current operation and [design contract](simulation-contracts.md) for planned work.

## Correctness

- 263 CPU checks passed, including input mutation rejection, immutable observed knots, bounded long-session history, analytic fractional-delay FIR accuracy, live/offline equality, recorder wrap/overflow, concurrent capture finalization, and reference acceptance/rejection.
- GPU FFT, transformed 2D/3D direct fixtures, batched asynchronous completion/disposal and live/offline equality with both delay filters passed on Metal.
- CPU and GPU Play-mode walkthroughs completed with editable native objects, moving listener/occluder, source events, saved input/output files and subsequent offline comparisons. Disabling the manager released its session and output queue.
- The actual Editor CLI completed a three-budget comparison. An exact-agreement reference gate rejected the moving take while leaving its completed measurement outputs available.
- No runtime files, packages or assembly boundaries were added for this cleanup. The existing implementation files were extended.

## Scene validation

The warmed mesh/material fixture performed 1,000 validations with zero managed allocations. Its total duration was about 3.2 ms. Snapshots compare shared mesh data and authored shape/material inputs; they do not regenerate scene geometry.

The earlier walkthrough spent 778 ms validating 26,632 polls, including duplicate audio timestamps. The final CPU walkthrough spent **2.76 ms** across **130** validations; 28,451 duplicate polls serviced pending rendering without revalidating or rewriting accepted poses. Maximum validation cost was 43.8 microseconds. Preparation, including snapshot warmup, is reported separately (35.4 ms in this run). These are observations from comparable three-second runs, not a general speedup guarantee.

## Live output

Both walkthroughs used one source/listener, 60 triangles, 44,100 Hz audio, a 256-sample response step and a 4,096-sample monitoring lead (92.9 ms). They recorded 129 distinct pose observations each. Both reported zero underrun, late, overflow and over-budget callback samples/counts at producer stop; output capture finalized without overflow.

The CPU run used linear delay filtering; the GPU run exercised a 16-sample-radius causal windowed-sinc filter. Their DSP times therefore represent different configurations. GPU maximum submit-to-consumption completion time was 14.74 ms. Output WAVs preserve component PCM before downstream Unity mixer/device processing, and callback records include the DSP clock relative to the take origin.

## Direct traversal throughput

Controlled all-miss fixture: 16 source/listener pairs, static triangles outside every path, warmed backend, eight measured calls per case. CPU and synchronous GPU calls each solve one response frame. GPU batches contain eight frames per call; their column divides elapsed batch time by eight. All GPU figures include readback/wait overhead. This measures throughput; batch latency and live scheduling must be assessed separately.

| Triangles | CPU ms/response | GPU sync ms/response | GPU batch ms/response |
|---:|---:|---:|---:|
| 64 | 0.209 | 0.388 | 0.051 |
| 4096 | 13.010 | 3.633 | 0.204 |
| 65536 | 207.502 | 25.780 | 2.939 |

Traversal still scans every primitive per pair. The near-linear scaling makes object/geometry acceleration the next measured scaling target. Batching reduces dispatch/readback overhead but does not change that traversal algorithm. Configurable FFT/STFT, validated indirect responses and converged ray-budget training references remain future work.

## Reproduction

Run `TakeChecks.Run` or `TakeChecks.RunGpu` with Unity's `-executeMethod` flag; omit `-nographics` for GPU checks. The walkthrough and controlled traversal automation, logs, CSV, takes, monitored PCM, offline renders and CLI outputs are retained locally under `.context/validation/cleanup/`. The temporary automation is outside the runtime codebase.

Unity Editor SearchDatabase logged its previously observed startup indexing exception during Play-mode entry. The capture, finalization and comparison checks above completed. This issue is recorded in the retained logs; no project upgrade was made during these cleanup measurements. The subsequent main-project migration is separate from this report.
