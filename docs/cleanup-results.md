# Cleanup validation and measured bottlenecks

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
