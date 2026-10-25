# Direct sound, 24 October 2026

Church and its builder now set `maxBounces = 0`. The shared Raytraced/Spectral pipeline keeps direct point/plane arrivals, travel delay, occlusion, material transmission and listener filtering. It allocates no indirect path/filter rows and skips the trace dispatch, including reflections and diffraction. Seeded planes still evaluate every retained burst; their `Samples` capacity remains enforced, while an indirect-ray coverage floor is unnecessary at zero bounces.

## Verification

**45 renderer regression cases passed, with zero failures/skips**, plus the Church integration case below. The selected run totals 46 cases in 12.03 seconds. After noticing the organ's silent opening, the scene probe was refined to use seven-second clip offsets for every clip and passed again in 8.14 seconds. Those offsets are probe input, not scene edits.

- An analytic impulse arrives at exactly the expected delay/gain, with no reflected tail, on CPU and GPU. Opaque walls block it; 25% energy transmission produces the expected 50% pressure coefficient. The reflection-enabled control produces a later tail. These checks cover 2D/3D and brute-force/BVH traversal, with a positive diffraction factor in the zero-bounce case.
- Seeded short bursts agree with an independent delayed-PCM oracle, CPU/GPU synthesis, live/offline schedules, one/three-response batches, 64/128-sample response steps and linear/sinc resampling. They exercise movement, frozen birth positions, gain, stop and restart with only one indirect ray configured at zero bounces.
- Streamed and clip PCM agree in CPU/GPU Spectral and IR synthesis at zero and positive bounces. This checks the stream renderer, not microphones or network delivery.
- Existing plane, material and mirror checks also pass. Twelve zero-bounce cases were added to the repository tests.

The isolated scene probe compiles the saved Church: **17,321 triangles, 59 loaded colliders, eight registered sources**, including four silent captured voice slots, and four playing clips. Five of the rain plane's eight fixed sites emit through its authored mask. It renders two seconds of the clips plus the saved three-second tail, at an outdoor receiver and the saved indoor player position. Every clip excerpt has nonzero RMS. No saved settings are overridden during rendering: 44.1 kHz, mono, 1024 FFT, 256 hop, 1024 response step, up to 64 responses per batch, GPU Spectral, BVH, zero bounces.

| Receiver | Peak | RMS | Total render time |
|---|---:|---:|---:|
| Outdoor `(-20, 2.7, -25)` | 0.047882 | 0.004409 | 1.912 s |
| Saved indoor `(0, 2.65, -30)` | 0.016425 | 0.002269 | 3.071 s |

Both float WAVs are finite and unclipped; their final second is exactly silent. Reported numeric working storage is **248.08 MiB** per render, not measured process/GPU residency. Actual path capacity is **7,695 slots**, versus the calculated **278,559** for the same source layout with the previous 32 bounces. No old-profile runtime comparison was performed. These are single offline runs on the reported Metal device, not a real-time deadline or perceptual-quality benchmark.

The tests ran in an isolated Unity 6000.4.1f1 batch with Metal and audio disabled. No Play mode, foreground scene, speaker playback, microphone or network session was used. Existing shader warning families and environment/license diagnostics remain in the archived logs. Source recordings retain their own environmental acoustics; this change does not dereverberate them or integrate the deferred physical drop/object-radiation model. Direct travel delay and finite source/material/HRTF filtering remain valid after source stop.

## Evidence and reproduction

- [Scene settings and render reports](church-probe.json), [regression cases](regression.json), [PCM checks](audio-checks.json), [source/artifact hashes](manifest.json).
- Silent-generated [outdoor WAV](outdoor.wav) and [indoor WAV](indoor.wav); `python verify.py` rechecks their finite samples, levels, length, final silence and recorded zero-bounce capacity (NumPy/SciPy).
- [Scene probe](DirectSoundChurchProbe.cs); [Unity XML results and logs](unity-evidence.zip). The archive retains the first 46-case run and the final scene-only rerun.

Use a separate Unity project copy with this revision's `Assets`, `Packages` and `ProjectSettings`. Copy `DirectSoundChurchProbe.cs` into its `Assets/Tests/Editor`, and supply a fresh output directory through `ACOUSTIC_DIRECT_PROBE`. Run Unity with `-batchmode -force-metal -disable-audio -job-worker-count 1 -projectPath <copy> -runTests -testPlatform EditMode`, selecting:

```text
TakeChecks.ZeroBounces;TakeChecks.Emission;TakeChecks.Plane;TakeChecks.MirrorChecks;TakeChecks.TransportMaterials;TakeChecks.StreamedAndClip;DirectSoundChurchProbe
```

Use `-testResults <results.xml> -logFile <unity.log>`. Do not use `-nographics`: the GPU cases require Metal. The scene probe opens but does not save Church, requires batch/edit mode and records numeric takes without starting the runtime session.
