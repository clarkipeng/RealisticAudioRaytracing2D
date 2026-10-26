# Weather shelter comparison — 25 October 2026

Church rain now uses seeded, spatially distributed one-shot impacts instead of eight stationary copies of an ambience recording. Wind uses overlapping tree-rustle grains over a canopy mask. Both remain single logical sources and share the existing raytracer, material propagation and DSP. No indoor gain switch, structural vibration model, extra weather kernel or per-drop GameObject was added.

The comparison holds the weather setting steady while the first-person listener walks outside → doorway → closed interior → outside. The final 0.2 s fades the weather; a 2 s tail follows. The picture camera uses the exact recorded head pose. The minimap shades emitting areas and draws seeded birth locations; it shows emissions before acoustic travel delay, not received loudness.

## Profile and limits

- 44.1 kHz mono, FFT 128 / hop 32, 2048-sample response steps, one response per batch.
- 32 bounces retained; 1.25 s maximum path delay. This is a bounded preview, with coarser frequency resolution and shorter late tails than the previous FFT-1024 / 10 s profile.
- Ground: 58 × 86 m mask, one sampled impact per 344 samples (~128/s), capacity 192, gain 0.006. The original dry kernel is 1058 samples; regenerate it with `tools/rain_impact.py`.
- Canopy: 80 × 70 m density mask from foliage bounds at 6 m height; one 0.5 s grain per 4096 samples (~10.8/s), capacity 24, gain 0.06 before the shared gust envelope.
- Every burst source receives at least its own retained capacity in indirect rays per bin. Other sources keep their configured budget. Overflow remains an error.

Impacts are energy-weighted acoustic samples, not a calibrated count of all physical droplets. The dry kernel is a procedural noise transient, not a measured material response. Canopy height is a planar approximation. Roof impacts, roof vibration, water/bubble mechanics and material resonance remain absent. The film therefore demonstrates exterior impacts reaching the listener through the existing propagation model, including the doorway and walls. Its numerical renderer remains an unvalidated geometric preview.

## Validation

[44 selected regression cases across two runs](regression.json) passed with no failures or skips. They cover one-shot timing/energy/gating, unequal source ray budgets and mirror gains, CPU/GPU/live agreement, multiple response batches, legacy planes, packed source/listener offsets, and existing mirror/batch regressions. A route-specific audit found that Unity’s approximate vector comparison discarded tiny nonzero turns. The recorder now uses exact component equality; its new regression covers a sub-tolerance input. [Compact replay](replay.json) reproduces all 1,802 states and events byte for byte, with zero position/rotation/velocity drift. The corrected capture has identical state/event hashes, camera-view bytes and first-frame image bytes to the completed image sequence, allowing [verified frame reuse](frame-reuse.json).

Existing Metal potential-initialization and integer-division warnings remain.

The higher-gain preflight exposed 47 samples above full scale (peak 1.7214). The final scene uses a lower rain gain, 0.006 versus 0.015. That six-second preflight reserved 1,650,944,356 numeric bytes (~1.54 GiB), with a sampled process RSS peak of 739.6 MiB. It is not the delivered audio. A process RSS guard alone does not measure all unified GPU storage; the numeric allocation report is recorded separately.

All exports run offscreen with audio disabled, nice 19, one Unity worker and a 20% work fraction. The launcher stops its own process above 3 GiB sampled RSS. Pacing occurs after completed image work or GPU audio readback; startup/import work and OS scheduling mean this is not a hard whole-system utilization ceiling.

Final render, replay and level measurements are recorded alongside the delivered artifact. Observed window levels use dBFS, not calibrated SPL or a controlled insertion-loss measurement; different windows contain different seeded impacts and wind gust phases.

## Final mixed render

The 38 s, 1280 × 720 / 30 fps film uses the recorded head camera and mono 44.1 kHz audio. Only the 144-sample DSP latency is trimmed. Master gain is 1.0 throughout; there is no segment normalization. The raw WAV peak is **0.69614**, with zero samples above full scale. Decoded AAC peaks at **0.68987**, also with no samples above full scale. The audio and picture scene/state/event hashes match, and the geometry hash is unchanged from the earlier Church film.

| Window | RMS | Level |
|---|---:|---:|
| Outside, 1–5 s | 0.019246 | −34.31 dBFS |
| Closed interior, 17–21 s | 0.002452 | −52.21 dBFS |
| Outside again, 32–35.5 s | 0.019626 | −34.14 dBFS |

The closed-interior window averages **17.90 dB quieter**; the two exterior holds differ by **0.17 dB**. Interior transients remain visible in the quarter-second measurements; these averages do not claim a perfectly uniform field. The interior’s high-frequency band loses more power than its low-frequency band. [Level/spectrum plot](levels.png) · [Full measurements](diagnostics.json) · [Native renderer report](render.json) · [Resource measurements](resources.json).

The full ten-source scene reserves **1.69 GiB** of numeric working storage. This is a different preview profile from the old 7.62 GiB job, with the frequency/delay tradeoffs described above. Rendering took **1546.68 s** under the 20% work budget; this is an offline demonstration, not a real-time performance result.

Reproduce the level report with `python analyze.py <capture-directory> <audio-directory> diagnostics.json` (NumPy/SciPy). It checks the take/render hashes before measuring.

## Wind-only probe

The separate 38 s `wind-shelter.wav` keeps the same listener route, geometry and wind emissions, with unity playback gain and the same 144-sample latency trim. Source-index compaction changes the transport random stream, so this is an independently traced probe rather than an exact extracted stem. Its peak is 0.08127, with no samples above full scale.

The outside / closed-interior / outside-return windows measure −42.62 / −115.15 / −47.33 dBFS respectively. Wind becomes nearly silent behind the closed door; the approximately 72.53 dB difference is an observed result of this preview, not a validated physical attenuation target. Exterior quarter-second levels also vary substantially with the sparse canopy grains and their propagation. This remaining behavior needs listening review; the mixed render's more consistent exterior averages do not establish wind-field convergence. [Wind measurements](wind-diagnostics.json) · [Wind renderer report](wind-render.json).

The wind probe reserves 204,589,426 numeric bytes and took 189.37 s under the same work budget. Sampled process RSS peaked at 1,432.77 MiB including startup/import work. All owned export processes exited. [Artifact hashes and delivery properties](manifest.json) identify the exact movie and trimmed wind probe; [preview](preview.png) is frame 18 s from the delivered movie.
