# Small baked-reflection experiment

The user asked to try precomputation/interpolation without designing a larger architecture. This experiment changes no production renderer or source model.

**Listening outcome: rejected.** The user said the audition sounds bad. Whether that applies to both halves or specifically to the cached half is pending clarification. The actual fine-cache rain WAV differs from its fresh reference by 1.95% relative RMS (difference −34.20 dB relative to reference), with correlation 0.99981 and level difference +0.00043 dB. This makes the shared source/room model the first numerical suspect, but does not establish perceptual equivalence or rule out audible interpolation error. The single-emitter closed-box control is not spatially distributed rain. Keep the interpolation measurements; do not promote this as a successful rain audition. [Listening audit](listening-audit.json).

An 8 × 6 × 3 m closed shoebox provides a cheap independent geometric reference using the [image-source method](https://jontalle.web.engr.illinois.edu/Public/AllenBerkley79.pdf). All walls use the same illustrative frequency-independent pressure reflection coefficient, 0.65. Enumerating up to 32 surface reflections gives 45,824 reflected image paths. Direct sound is evaluated exactly for every query. This is not a general measured-material, diffraction, moving-boundary or Church model.

The bake samples both source and receiver positions. Two alternatives use the same bilinear-source × bilinear-receiver weights: blend complete reflected impulse responses, or interpolate each matched image path's delay/gain and then assemble the response. All methods use the same 17-tap windowed-sinc delay deposition. Path identities are known in this box; matching and visibility changes in arbitrary geometry remain untested.

Four held-out point pairs were tested with 1 m and 0.5 m grids. One pair also received a local 10 cm refinement. The fine bake covers only one 10 cm square at each endpoint, not the whole earlier domain or whole room.

| Sampling | Raw-response total energy error | Delay/gain interpolation total energy error | Reflected-delay weighted RMS error |
|---|---:|---:|---:|
| 1 m, four queries | −3.09 to −1.95 dB | −0.083 to +0.111 dB | 0.095–0.150 ms |
| 0.5 m, four queries | −4.18 to −2.65 dB | −0.038 to +0.016 dB | 0.031–0.043 ms |
| 0.1 m local, one query | −2.59 dB | +0.024 dB | 0.00196 ms |

Energy agreement alone hid substantial waveform errors: total-IR relative L2 error for delay/gain interpolation was 1.01–1.22 on the coarser grids, falling to 0.119 at the one refined query. The fine case's four band-energy errors are between −0.036 and +0.045 dB. These are approximation measurements, not phase-exact reconstruction or a listening acceptance. Smaller raw-IR blend cells did not uniformly improve energy: arrival misalignment causes cancellation.

Controls passed for six independently constructed first-order mirror distances/gains, source/receiver reciprocity and interpolation's exact grid-node selection/partition of unity. The nine case results and all audio hashes are in [report.json](report.json). Cached delay/gain arrays use float32; raw IRs are also retained for comparison. The 1 m and local 10 cm path caches each use 5,865,472 bytes; the 0.5 m cache uses 29,693,952 bytes. Their coverage differs, so equal storage does not imply that the whole scene can be refined for free. Cache files stay under `.context/`.

The reference is already a cheap vectorized analytic solver. Timings are recorded, but lookup excludes rebuilding the interpolated IR, and none measures Unity/GPU performance. This experiment establishes interpolation behavior, not a runtime speedup. It uses nice 19 and one numerical worker, and all computation finished.

The listening files use the same existing recording-based rain test signal in every version: five seconds of source plus one second of response support. This is a test emitter in the room, not a claim that weather lands on its sheltered floor. One common gain is applied, with no per-method normalization or limiter. The main comparison plays fresh → interpolated 10 cm bake, six seconds each with a half-second gap. The second plays fresh → raw 1 m blend. All nine WAVs are finite and below full scale. Delivery is the Conductor workspace's `.context/rain-source-review/cache-test/index.html`.

```sh
OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 OMP_NUM_THREADS=1 \
  python docs/benchmarks/rain-cache-2026-10-26/run.py \
  <mono-float-44100Hz-test-signal.wav> <new-output-directory>
```

The script refuses an existing output directory and includes the controls. It exports caches, reports, WAVs and the listening page. No live audio device or Unity process is started.
