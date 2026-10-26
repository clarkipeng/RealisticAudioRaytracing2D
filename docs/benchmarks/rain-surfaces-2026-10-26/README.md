# Rain on the first acoustic surface — 26 October 2026

Both Church scenes now sample rain above the scene and place each impact on the first acoustic triangle below it. Roof hits replace the ground hits beneath them. The mask describes incoming rain; there is no hand-painted roof exclusion. Emissions remain one logical source, use the existing dry impact PCM and share spectral transport/DSP.

`AudioSourcePlane.projectImpacts` selects generator 3 when one-shot impacts are enabled. The event clock means impact time, not release time. Query recorded geometry at that time with the existing two-level BVH; cache both hits and misses. A miss stays silent without resampling or renormalizing the remaining impacts. Move a hit 1 mm along its incoming-facing surface normal, then freeze its world position through the acoustic tail. This keeps launch points on the air side of the transport epsilon. The implementation adds no weather transport kernel, per-drop GameObject, material parameter or extra logical source.

## Checks

**47 Unity cases and 3 Python cases passed**, with no failures or skips in the Unity runs. [Regression list](regression.json).

- First-hit geometry covers roofs above ground, reversed winding, rotation, moving/disabled surfaces and misses.
- Event positions/energy stay stable across batches and ray budgets; cached hits and misses are not recast. Takes authenticate generator 3, reject planar projection and reject nonzero one-shot clip offsets.
- Projected impacts agree across CPU/GPU synthesis, single/multiple response batches and live/offline scheduling, including an independent direct-sound delay/PCM oracle.
- For a receiver below a roof, both flat and BVH GPU transport apply the assigned energy transmission: 0.25 gives a 0.5 pressure multiplier; an opaque roof gives zero direct pressure. No listener-zone gain is involved.
- Existing impact, masked-plane, mirror, packed-source and GPU batch cases still pass.
- HUD queries handle roof coverage, geometry flags, moving/rotated geometry and payload authentication.

## Church audit

A fresh 36 s numeric take follows the existing shelter-comparison route. The audio engine's emission preparation generated these first-surface hits:

| Acoustic object | Impacts |
|---|---:|
| CopperRoof | 1,201 |
| StoneRoof | 131 |
| Stone floor and courtyard | 3,057 |
| StoneBrickWalls | 197 |
| StoneBeams | 24 |
| FrontDoorFrame | 5 |
| **Total** | **4,615** |

All 4,615 candidates hit exactly one surface; none missed this scene footprint. Each emitted point was checked against the nearest surface at its own birth, with no duplicate ground impact. Peak retained rain occupancy was 189 of 192 slots. Scene geometry remains the same 17,321 triangles and geometry hash as the previous film. Source count remains 10. The old generator-2 take still opens with its original authenticated scene hash.

Emission preparation over the whole route took **1.326 s wall time** at background priority; this includes the existing source preparation and excludes GPU transport/audio synthesis. Its numeric staging/BVH storage reports **782,808 bytes**, excluding managed dictionary overhead and the separately owned geometry. This is not a live-audio performance claim. [Audit](audit.json) · [Impact positions](impacts.csv) · [Artifact identities](manifest.json) · [Capture/audit process resources](resources.json).

The independent Python minimap reconstruction finds the same 4,615 birth samples. Its maximum position difference from the C# points is **0.109 mm** (4.35 µm RMS); float precision differs between implementations. This calculation only drives markers, never audio. [Comparison](hud-comparison.json).

## Limits

The assigned acoustic material governs propagation, while every hit still shares the source's uncalibrated 24 ms dry transient. All hit surfaces in this Church snapshot currently use the **Church stone** acoustic material, including the mesh named CopperRoof; the object name does not supply a copper response. Material-specific impact timbres, physical drop forces, roof vibration/drumming, splashes, canopy throughfall and slanted rain remain deferred. Geometry must be in the acoustic snapshot to intercept rain, and the sampling plane must be above the intended surfaces.

The visual rain streaks still use the presentation's static shelter-height texture; they are not one-to-one particles for the acoustic events. Acoustic reflections remain configured at 32 bounces. The previous film and its indoor/outdoor level measurements predate roof impacts. This revision validates numerical placement and targeted audio behavior; it does not include a newly rendered full Church soundtrack or a listening-quality claim.

## Reproduction

Use an isolated project with the changed assets. Run `ChurchSequenceAuthoring.UpdateRainSurfaces` in silent batch mode to author both scenes. Run the Unity EditMode filter `TakeChecks.ProjectedImpact;TakeChecks.Impact;TakeChecks.Emission;TakeChecks.Plane;TakeChecks.GpuBatches;TakeChecks.MirrorChecks`, plus `python -m unittest discover -s tools -p test_sequence_movie.py -v` in the NumPy/Pillow environment.

For the numeric scene audit, copy [Audit.cs](Audit.cs) into the isolated runner's `Assets/Tests/Editor/RoofImpactAudit.cs`. Invoke `RoofImpactAudit.CaptureAndRun` in silent batch mode with:

```text
-sequenceScene Assets/Scenes/ChurchWeatherComparison.unity
-sequenceOutput <fresh-capture-directory>
-sequenceFirstPerson 1 -sequenceEndFrame 1 -sequenceHeight 360
-sequenceRoute weather -sequenceTail 2 -sequenceWorkFraction .2
-impactTake <fresh-capture-directory>/take
-impactOutput <fresh-audit-directory>
-impactLegacyTake <previous-generator-2-take>
```

The audited process used `taskpolicy -b`, nice 19, one Unity worker, disabled audio and a 3 GiB RSS guard. Only one picture was rendered; the complete numeric route was recorded. `RoofImpactAudit.Run` can audit an already captured take with the last three arguments. Remove the copied audit helper after use. All owned workers exited.
