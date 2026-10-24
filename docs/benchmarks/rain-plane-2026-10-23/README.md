# Masked rain plane evidence

Parent checkpoint: `0659ec5` (two point rain sources). Current implementation replaces them with one masked plane; it removes the unused garden clip. See [scene-presentation.md](../../scene-presentation.md#masked-rain-plane--23-october-2026) for contracts, Inspector settings, costs and limits.

- `regression.json`: 204/204 selected TakeChecks cases pass, including ten new plane cases. Original full NUnit XML is compressed in `authoring.zip`. Selection: `-runTests -testPlatform EditMode -testFilter TakeChecks -testCategory 'Core;GPU;Integration'`.
- `verification.json`: actual Church extraction, compact replay rig equivalence, old v4 take opening, capture guard checks, sample positions and the reduced-profile GPU render report.
- `scene-diff.json`: unchanged serialized collider blocks and the exact changed scene object IDs.
- `rain-plane-map.png`: mask footprint with the actual stable source sites. This is not a pressure map.
- `rain-smoke.wav`, `waveform-spectrogram.png`, `audio-check.json`: direct offline output and numerical inspection. No audio-device playback or listening evaluation occurred.
- `authoring.zip`: one-off Unity asset authoring/check script, plotting script and logs. The authoring script consumes the **old scene from `0659ec5` with the current plane code**, not the already-converted scene; its local output/legacy-take paths need adapting on another machine. It is intentionally not installed as production/editor tooling.
- `manifest.json`: source/artifact hashes and deleted paths. Source history is in Git, not duplicated in the archive.

Unity 6000.4.1f1 / Metal. Every Unity job ran separately in an isolated project with `nice -n 15`, `-batchmode -force-metal -disable-audio -job-worker-count 1`. The renderer wrote PCM directly. No live microphone, device callback, multiplayer peer or full-profile deadline validation was run.

Two early authoring-check failures are retained: the first counted 58 serialized colliders but overlooked the instantiated player controller (59 loaded); the second attempted to test texture mutation using a no-op `Apply`. The corrected checks compare the loaded collider count before/after and actually change a pixel. No runtime workaround was added for either check mistake.
