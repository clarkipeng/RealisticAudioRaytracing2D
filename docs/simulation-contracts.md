# Acoustic simulation and reconstruction: design contract

Updated 2026-10-05 against the current Unity 6 working tree. This is the single design specification for scenes, simulation, capture, comparison, multiplayer, datasets and learned reconstruction. It replaces the separate architecture, comparison, remaining-work, scene-assets and training plans. Examples describe proposed contracts unless explicitly marked implemented.

[README](../README.md) remains the operating guide for actual APIs and workflows. [Cleanup results](cleanup-results.md) contains the current station offline benchmarks and explicitly dated historical measurements.

**Workflow:** edit a native 2D/3D scene → walk or run a scenario → record once → render the same take at several budgets → compare sound and cost → generate validated examples → train and deploy optional reconstruction.

**Invariant:** the take determines scene and source inputs; the render configuration determines computation. Changing quality never changes the recorded experiment.

Contents:

1. [Status and decisions](#1-status-and-decisions)
2. [Code and dependencies](#2-code-and-dependencies)
3. [Unity scenes and assets](#3-unity-scenes-and-assets)
4. [Capture and saved inputs](#4-capture-and-saved-inputs)
5. [Physics, representations and configuration](#5-physics-representations-and-configuration)
6. [Shader contracts and ownership](#6-shader-contracts-and-ownership)
7. [Live, offline and audio rendering](#7-live-offline-and-audio-rendering)
8. [Reconstruction and temporal upscaling](#8-reconstruction-and-temporal-upscaling)
9. [Example generation and dataset format](#9-example-generation-and-dataset-format)
10. [Models, training and deployment](#10-models-training-and-deployment)
11. [Audio comparison and benchmarking](#11-audio-comparison-and-benchmarking)
12. [Multiplayer](#12-multiplayer)
13. [Implementation sequence and acceptance](#13-implementation-sequence-and-acceptance)

The [system audit below](#system-audit-2026-10-05) sets the immediate repair order. Audio correctness and trustworthy recording completion precede the planned recordings browser, new scene features and training. The browser section remains a future specification.

## 1. Status and decisions

| Capability | Current implementation | Planned work |
|---|---|---|
| Unity authoring and capture | Native 2D/3D geometry export, rigid motion, activation, stable IDs, source events, immutable take files; editable station, train service with stops and walking aboard, and door with two sources and one binaural listener. | AcousticLab and scripted dataset scenarios. |
| Replay and computation | Pose replay/scrubbing, CPU/HLSL direct solvers with optional static BVH plus direct tests for moving objects, shared live/offline timeline and mixer, WAV/response export, batch jobs and measured station offline benchmarks. | Take browser, scene-independent playback, shared visual snapshots and continuous bounded playback; then validated indirect transport and richer saved response payloads. |
| Directional spectral preview | Shared 2D/3D GPU transport with multi-bounce reflection/scattering, transmission/refraction, air/material losses and edge diffraction heuristics. Configurable FFT/hop WOLA, interpolated per-arrival KEMAR orientation, response crossfades and delay-derived Doppler; same live/offline implementation. Closed 2D/3D objects use per-medium speed/damping, including nested solids and interior sources. | Physical normalization/convergence, exact convolution, directional dataset export and validation of complex moving boundaries. The preview is not an approved training reference. |
| Multiple endpoints | Every source/listener pair; mono or binaural mix per head; one monitored listener. | Larger workload benchmarks and explicit local receiver selection for multiplayer. |
| Comparison | Coarse/fine schedules, metrics/reference gate, native audio A/B/Difference, waveform/spectrogram plots and synchronized listener/top-down/free-camera views. | Per-pair target checks, learned outputs and validated reflection comparisons. |
| Runtime | Bounded PCM/capture queues, live/offline GPU response batches, fractional-delay FIR, provenance and diagnostics. | Lower measured latency, convolution, reconstruction scheduling. |
| Learning and networking | Neither implemented. | Dataset loader, models, inference and actual host/join gameplay. |

The current project uses Unity **6000.4.1f1**, C# and HLSL. Earlier CPU/Metal checks and the repaired legacy FFT are documented in the README. Windows/DirectX still needs its own checks. The legacy reflection shader has different physics and is not a validated reference.

The first physically validated indirect dataset model remains geometric **energy** transport. The implemented spectral preview instead synthesizes pressure using idealized square-root energy coefficients and path phase; it is explicitly excluded from that validation gate. Signed pressure/complex transport is a separate phase-bearing capability, followed by models that reconstruct those quantities. Full rendered-audio spectrogram enhancement is another experiment, not a prerequisite for improving acoustic responses.

Temporal upscaling is now an explicit requirement: low-cost solves may run less frequently than reconstructed response updates. Ray budget, solve cadence, response output cadence, response resolution and audio sample rate are separate controls. Test each reduction independently before combining them.

Critical constraints:

- Dense replay cannot recover motion never captured. High-rate labels initially describe the recorded interpolated trajectory, not unobserved physical ground truth.
- More rays reduce estimator noise; they do not supply absent diffraction, transmission, room modes or material phase.
- Sparse history may contain no evidence of a newly opened path. Limit prediction age, retain fresh direct visibility and count fallbacks; do not treat a network's uncalibrated confidence as proof.
- History duration, IR support, STFT window and lookahead are different time scales. Report each.
- Valid silence/zero energy is different from missing data. Clip inactivity does not erase the scene's unit-input response.
- A speedup must include inference, transfers, synthesis, DSP, graphics contention and deadlines. Extra output frames or FFT bins alone do not demonstrate recovered information.

Still unresolved before their respective milestones: the indirect estimator's receiver/PDF normalization; phase-bearing reference physics; measured latency/memory thresholds on target workloads; and model sizes/quality thresholds selected from baseline evidence. These are explicit deliverables, not silently chosen defaults.

### System audit, 2026-10-05

Audited production revision `1586cbb` in the actual Unity project, not the Conductor checkout. This audit changes documentation only. Numerical probes ran in Unity 6000.4.1f1 on Apple M5/Metal without entering Play or dirtying TrainStation. [Recorded measurements](benchmarks/system-audit-2026-10-05.json) distinguish new probes from earlier station performance measurements. Local repro scripts are `.context/system-audit/Probe.cs`, `.context/system-audit/ExistingChecks.cs` and `.context/renderer-audit/Probe.cs`.

**Verdict:** keep the scene, recording and shared execution boundaries. The spectral preview has confirmed synthesis errors and an unvalidated reflection estimator. It is unsuitable as a high-fidelity training teacher. A GPU implementation of the same formulas would retain those errors. No new packages, compiler migration, service layer or second scene representation is justified by this audit.

The actual current flow is:

```text
Unity scene / reconstructed inputs
  → numeric geometry + sample-timed states and source events
  → shared GPU transport → individual path readback
  → CPU source FFT history, per-path ear filters, IFFT and WOLA
  → live: completed PCM ring → Unity audio callback
    offline: the same PCM → WAV
```

The older flow used GPU spectral accumulation/IFFT, then CPU overlap-add and an additive CPU PCM ring. Current code already has circular source-spectrum history, overlap storage and a PCM handoff queue. Circular storage and GPU/CPU placement are separate decisions. The old implementation also had a fixed 1024-point IFFT, blocking `GetData`, mono playback and a lock around its audio ring; reverting it wholesale would restore those constraints.

| Boundary | Audit decision |
|---|---|
| Native Unity objects → numeric scene | Keep. Local triangles/segments, baked scale, stable object IDs and separate rigid poses are appropriate. Native collider proxies remain editable. Scale/topology/material/roster changes require a new segment; arbitrary deformation is not implemented. |
| Static BVH + moving primitives | Keep. Objects leave the static tree after motion and use direct tests thereafter. GPU batches classify all their poses before dispatch. Add a dynamic hierarchy only if moving geometry becomes a measured bottleneck. |
| Source commands and simulation/audio clocks | Keep integer sample events and the tick-to-observation map. Inputs alone do not preserve the original relationship between gameplay ticks and audio samples. |
| Compact station replay | Keep its narrow scope: scene initialization plus station/walking inputs, verified against matching project/engine/platform. It is not a general deterministic Unity physics recorder. |
| Shared live/offline solver and DSP | Keep one implementation and the same sample timeline. Live must meet playback deadlines; offline can advance without an audio device. Batching may differ without changing output. |
| One receiver per head | Keep world arrival direction and paired HRTFs. Two ear channels do not require two separately traced listeners. Multiple heads multiply the source/listener work; live currently computes every registered head even though it monitors one. |
| PCM callback boundary | Keep the existing single-producer/single-consumer queue. The callback copies completed PCM and records bounded diagnostics. It must not run transport, wait for GPU work, allocate or write files. |

Confirmed correctness findings, in repair order:

1. **Spectral filtering is not correct linear delay/convolution.** `SpectralAudio.Synthesize` and `SetResponse` apply a residual delay as frequency phase inside a windowed transform, then apply the original synthesis window. Zero-delay identity works; arbitrary delays do not. The new test uses a unit impulse, no HRTF, no geometry, 128-point FFT and 32-sample hop. At input sample 31 with a 31-sample propagation delay, the declared output is sample 190 with gain 1. Actual gain is 0.7324865, with an early contribution of 0.008240453 at sample 62. Of 28 integer-delay/impulse-position combinations, 16 fail absolute tolerance `1e-5`; delays divisible by the hop pass. This is a DSP error independent of acoustic realism. Fix block support, padding and overlap against an independent time-domain convolution reference. Storage modulo arithmetic must not become circular convolution. Declared latency must describe the actual support.

2. **Pending overlap can be cut off by listener state.** `SpectralAudio.Render` gates already accumulated output using the latest `enabled/oldEnabled` pair. Once the second disabled response arrives, both are false although earlier windows still contain valid output. The existing repro was rerun: disabling at 512 with transition 64 gives sample 639 = 0.13143453, then sample 640 = 0, where the pending overlap is 0.125. Choose explicit listener stop semantics and implement them once; the current mixture of fading and abruptly discarding pending samples is inconsistent. Source Stop already stops future emission through `Signal`; it must still allow earlier sound to arrive.

3. **Path-array indices are incorrectly treated as physical path identity for Doppler.** `SpectralAudio.SetResponse` differentiates delay between equal slots/bins. An indirect ray can hit a different face or edge at the next response. A new valid path with a 30 ms delay increase over a 128/8192-second response interval is discarded because the computed Doppler ratio is negative; the new probe observes its contribution count change from 1 to 0. Retain Doppler only where correspondence is established. A direct path can still need a reset when its transmission topology changes; other paths need real correspondence or an explicitly unsupported Doppler mode. Never interpret an unrelated arrival as velocity. The current approaching/receding pure-tone check does not establish general moving-path correctness.

4. **Indirect frequency allocation leaves holes or unequal weights.** `TracePaths` uses `bin = ray * FrequencyBins / RaysPerPair` and weight `FrequencyBins / RaysPerPair`. At FFT 1024/rays 512, the 513th bin receives no indirect ray; at FFT 2048/rays 512, 513 of 1025 bins receive none. Even FFT 1024/rays 1024 gives aggregate per-bin weights from 0.50098 to 1.00195. Direct spectral paths still cover every bin, so direct audio can hide this error. Use the agreed configurable integer rays per frequency bin: `totalRays = bins * raysPerBin`, with correct per-bin sampling normalization. This removes the allocation error but does not fix the physical estimator. Keep acoustic frequency resolution separate from internal FFT padding for convolution.

5. **The reflection control does not describe the first reflected arrival.** `Transport.hlsl` connects each sampled surface hit directly to the listener using `cosine * 0.5`, regardless of its scattering coefficient. Scattering only affects the continuation ray. In a one-bounce planar wall fixture, scattering 0 and 1 produce byte-identical arrival records: 109 nonzero reflected paths, spread over 8.31–60.88 ms. The geometric specular image path is 8.246 ms. With diffraction disabled, this is not a specular wall response. Branch probabilities, surface sampling and pressure/energy weights also lack a complete estimator derivation. Treat these as geometric preview heuristics. Establish specular and diffuse reference fixtures and normalization before calling more rays a more physically accurate teacher. Material reflection phase and wave diffraction are still absent; increasing budget cannot recover them.

6. **Compact-recording failure can be hidden behind a successful acoustic save.** `StationDemo.RecordSafely` catches input-recorder construction/tick/observation errors, logs them and sets `recording = null`. `RayTraceManager.EndCapture` can still finish its separate writer and show Saved. With `retainFullReference=false`, there is no full pose payload to recover the failed input recording. This is a code-path finding, not a newly simulated disk failure. Make completion mean that both required recording parts succeeded; on input failure propagate an explicit failed session/save state. Do not silently claim a usable compact take.

7. **Offline reconstruction synthesizes some flags instead of reproducing their semantics.** `StationReplay.Observe` assigns Geometry from the binding and Source from membership. Live capture additionally checks primitive count, `AcousticSurface.enabled` and `RaytracedAudioSource.enabled`. The station adapter does not log arbitrary changes to those components. At minimum preserve supported initial enabled states and reject changes outside the adapter's input contract. A same-project hash does not make omitted runtime state reproducible. This is a code-path finding; the current all-enabled station fixture does not exercise it.

Other constraints that should be explicit rather than hidden behind quality labels:

| Area | What the code currently means | Required adjustment |
|---|---|---|
| Tail and history | The station permits arrivals up to 10 seconds but renders only a 1-second tail after source cutoff. `RenderDirect` and `Live.Finish` use the requested tail without proving it contains the response. | Drain the retained response support plus DSP support, or record intentional truncation in the output. History horizon, DSP overlap and playback queue are separate quantities. No recording-duration cap is needed. |
| Moving geometry | Responses use one frozen scene pose per solve; source history supplies older PCM. Acceleration and angular velocity are recorded but are not transport inputs. | Keep this declared approximation until retarded source/receiver/boundary timing is implemented and tested. More frequent solves do not by themselves turn it into a moving-wave solution. |
| Head motion | HRTFs interpolate continuously in angle, but refresh with the whole acoustic response and are baked into queued PCM. At 44.1 kHz the station's 16,384-frame lead is 371.5 ms, plus 23.2 ms declared FFT latency and device/physical delay. | Refresh head filtering independently of expensive transport where possible, preserve direction until ear synthesis, and measure head-to-audio latency. Smoother interpolation alone cannot fix stale head orientation. Do not bake far-future ear audio against today's head pose. |
| Materials | The shared shader evaluates the actual FFT bins. However `AudioMaterial.EnergyKnot` only exposes arbitrary knots for absorption/scattering; transmission/IOR/damping still come through legacy eight-band data. | Extend the existing coefficient contract when those curves are needed, migrating authored assets explicitly. The old eight-band authoring limitation is not a necessary solver limitation. Physical-property sliders are heuristic, not calibrated material measurements. |
| Low/High comparison | `TakeComparison.Run` permits only a different response step; FFT, ray count, bounces and other settings must match. | Label this temporal comparison. Add ray-budget comparison only with covered/normalized bins and independent convergence checks. Neither current preset means validated high fidelity. |
| Training export | `responses.bin` and captured response applications contain direct opaque probes even when PCM is spectral. All sources are mixed in final WAVs. | Add only the selected per-pair target when its teacher is valid; current files cannot reconstruct the indirect directional response. IR, complex-transfer/directional-target export and training remain unimplemented. Fixed shader RNG is reproducible, but there is no configurable independent seed for convergence testing. |
| Replay portability | Compact replay hashes all Assets/Packages/ProjectSettings, not only motion dependencies. A shader-only change can prevent reconstruction of an old input-only run. | For now reconstruct with its matching revision and preserve the generated numeric take as an offline artifact before changing renderer code. Git commit alone does not preserve uncommitted files. Do not weaken identity checks without proving the smaller dependency boundary. |
| BVH coverage | Nearest-hit queries use the static BVH; nearby-edge searches still scan every edge and containment checks scan candidate volumes/primitives. | Profile those kernels before adding another hierarchy. The direct-only traversal benchmark does not establish indirect traversal cost. |

Performance and instrumentation conclusions:

- Current station settings allocate 6,146 path records per response: 196,672 bytes read back, including empty ray/bounce slots. At 44.1 kHz and response step 1024 that is approximately **8.47 MB/s of path data**, versus **0.353 MB/s of stereo output PCM**. Each 64-response path array is about 12 MiB, on CPU and GPU separately. These are layout calculations, not measured transfer bandwidth. GPU synthesis can avoid routine path readback; diagnostics/dataset exports should request it deliberately.
- The [existing same-revision station smoke runs](benchmarks/material-head-fixes-2026-10-05.json) put CPU audio work at approximately 5.07–10.94 seconds for a 21.18-second take. It is the dominant reported component in those runs. Main-thread scheduling varied substantially, so this audit makes no new speedup claim. Halving solve cadence does not halve the source FFT/per-hop path accumulation/IFFT work.
- `DirectBenchmark.Run` explicitly forces Direct transport. `backend = GPU-Metal` reports the transport backend, not an all-GPU renderer. `pairPrimitiveTestUpperBound` counts only one pair/primitive diagnostic pass and is not an upper bound on spectral tracing, bounces, edges or medium queries. `dispatchReadbackSeconds` also has different synchronous/asynchronous timing scopes. Rename or scope these counters before treating them as bottleneck attribution. Record whole-job wall time and stage times; use GPU timing when available rather than presenting dispatch CPU time as shader execution time.
- Live rendering waits for two observations to interpolate an interval and synthesizes on the Unity control thread. This favors live/offline agreement, but observation gaps, CPU DSP, GPU completion and the playback lead all matter. A finite offline render faster than duration does not establish deadline-safe realtime playback. The selected-listener output queue is sound; main-thread stalls must be measured and addressed upstream.
- `SceneToTake.Guard` deliberately rereads authored meshes to reject in-place edits. Keep the correctness boundary; profile its cost on larger authoring scenes before inventing a dirty-state system. Batch sizes, bounded event queues and maximum acoustic delay are resource/response settings, not recording-duration caps.

The lean target remains the existing numeric scene and one shared block renderer:

```text
source blocks + scene response
  → GPU directional spectral work and ear filtering
  → correct GPU IFFT / overlap accumulation
  → completed PCM blocks
      live: asynchronous readback → existing PCM ring → callback
      offline: same blocks → file
```

Use absolute sample/frame keys with modulo storage, one clear owner of each output range and explicit clearing before reuse. An additive accumulation ring owns unfinished contributions; the existing PCM queue owns finalized samples and must not be overwritten by late arrivals. The first GPU port can retain source-spectrum history and a short output overlap ring; it need not introduce a second long future-audio ring or duplicate the entire delay horizon. If forward accumulation is chosen instead, retain direction until the relevant head pose is available. Both approaches must apply travel time once, preserve complete filter support, and respect the declared frozen-geometry time convention. The existing per-path representation is sufficient for a correctness-first port; no arbitrary angular grid is mandated.

**Repair sequence and gates:**

1. Make the numerical oracles authoritative: arbitrary-offset integer/fractional delays, independently convolved FIR/HRIR output, complete source/listener stops, cuts, long arrivals and block/hop invariance. Fix synthesis, pending-output gating and invalid Doppler correspondence. Make compact save failure visible and replay activation semantics explicit.
2. Correct per-bin allocation and define the reflected estimator. Validate a single mirror, a diffuse surface, absorption/transmission and per-medium delay independently. Keep diffraction labeled heuristic until it has its own reference. Exclude the preview from training eligibility throughout.
3. Port the corrected synthesis to GPU behind the existing live/offline boundary, compare against the small reference fixtures, then measure the actual station workload. Retain the CPU direct solver as an independent baseline. Scope diagnostics properly; measure deadlines/head latency and memory as well as throughput before reducing lead or increasing ray count.
4. Add the selected directional/per-pair export and separate ray/temporal experiments, then dataset generation/training. The recordings browser, multiplayer and extra scene features can follow; they do not repair the renderer.

**Test coverage finding:** all **9,686 existing transport assertions passed again**, while **16/28 independent impulse cases failed**. Existing identity tests use zero delay, and transition tests reproduce the current window formula without exercising a second disabled-listener response. They prove implementation agreement in those fixtures, not general convolution correctness. The fresh wall test and reused-slot probe expose additional gaps. Earlier capture/BVH/replay checks remain useful; they were not all rerun by this audit. No Windows/DirectX validation or new gameplay/performance run was performed.

Files to change in those repairs are `SpectralAudio.cs` and `RayTransport.cs`/`Transport.hlsl`, the existing `TakeRender`/`Live` tail and block boundaries, `StationDemo.cs`/`StationReplay.cs`, and the corresponding independent tests. `AudioManager.PcmQueue`, scene ABI and replay format do not need replacement to fix the audio errors. Keep names aligned when touching ownership: `DirectSolver` currently orchestrates both direct and spectral transport, and the generic render entry point is still named `RenderDirect`.

Legacy cleanup is a later deletion, not another renderer fork. `Big Room.unity` and `SmollRoom.unity` still reference `Raytrace2D.compute`; migrate or deliberately retire those scenes before deleting its manager/audio partials and packed material helpers. The station's `RaytraceControlPanel` is still active UI, so it is not wholly dead legacy code. Preserve old implementations in Git instead of adding new copies or expanding legacy features. README claims about the old CPU default, next-step browser priority and missing new files in the map were corrected with this audit.

## 2. Code and dependencies

Keep one codebase and reuse existing components:

```text
Assets/
  Scripts/
    Scene/         AcousticSurface, AudioMaterial, SceneHelper
    Simulation/    RayTraceManager, Take, TakeRender, DirectSolver, DirectBvh, TakeComparison
    Audio/         AudioManager, RaytracedAudioSource, Hrtf
    Demo/          existing movement, scene controls and legacy fixtures
  Shaders/         Direct.compute and existing legacy shaders/includes
  Editor/          TakeEditor, TakeDemo, TakeComparisonWindow, TakeComparisonAudio/View,
                   StationReplay, DirectBenchmark
  Tests/Editor/    TakeChecks and numerical fixtures
  Scenes/         editable Unity scenes
  Prefabs/        shared environment and actors
  Materials/      visual and acoustic assets
  Audio/          clips and mixer
  Resources/      KEMAR HRIR data and attribution
  External/       imported assets and provenance
Packages/         Unity package manifest
ProjectSettings/
tools/            import_kemar.py; later take_dataset.py, train_reconstruction.py
experiments/      resolved experiment configurations and small fixtures
docs/
  simulation-contracts.md
  cleanup-results.md
```

No new package hierarchy, assemblies, generic plugin framework, scene database or orchestration service. Preserve Unity `.meta` GUIDs. Retire legacy code/assets only after checking references and replacement coverage.

Proposed additions have concrete owners: `Simulation/Bvh.cs`, `Simulation/IndirectSolver.cs`, `Shaders/Indirect.compute`, shared geometry helpers when reused, `Audio/AcousticDsp.cs`, and one reconstruction adapter at the existing response boundary. Do not create empty files in advance.

Dependencies flow Unity authoring → numeric scene/state → solver → response → audio rendering. Capture and normal simulation work without ML. Offline uses the same C#/HLSL solver and DSP. A small Python job loop is sufficient if existing Editor/batch jobs become tedious. NumPy/PyTorch serve training; SciPy is optional for independent DSP checks. Test ONNX → Unity `com.unity.ai.inference` before substantial training. Networking adds Netcode for GameObjects/Unity Transport only at its milestone. Pin compatible versions then.

Current binaural rendering adds no runtime package: measured KEMAR coefficients are bundled as a Unity resource. The standard-library Python importer verifies the source archive hash and reproduces the asset. Retain [Gardner/Martin attribution and conversion details](../Assets/Resources/Acoustics/ATTRIBUTION.md) when distributing data or derived outputs.

Start hardware measurements on the available Metal machine; verify the same layouts and fixtures on DirectX before claiming support. HLSL sharing is useful but does not prove equal performance. A different compiler/backend needs a measured reason, not a parallel architecture. [Unity compute documentation](https://docs.unity3d.com/6000.0/Documentation/Manual/class-ComputeShader.html).

## 3. Unity scenes and assets

### Components and geometry

Use ordinary Unity GameObjects, native primitives, transforms, colliders, meshes and animation. Extend `AcousticSurface`, `RaytracedAudioSource` and `RayTraceManager`; do not introduce parallel replacements. A receiver remains an explicitly registered transform until a separate component earns its cost.

| Role | Contract |
|---|---|
| Geometry | `AcousticSurface` selects mesh/collider/proxy and acoustic material, optionally per submesh. |
| Source | `RaytracedAudioSource` owns clip settings and issues sample-timed events to its manager. |
| Receiver | Registered head transform; configurable omnidirectional mono or binaural left/right output. One head is one receiver, distinct from Unity's native AudioListener. |
| Session | Manager owns identity tables, capture/replay state and numeric resources; stays active outside the captured hierarchy. |
| Motion | Exactly one controller, animator or physics owner per moving object; acoustics observes its result. |

Auto geometry prefers MeshFilter then a supported collider. Explicitly select Collider on proxies; never export coincident mesh and collider surfaces. Readable selected meshes produce triangles. Supported 3D colliders include boxes, MeshCollider meshes and uniformly scaled spheres/capsules with declared curve tolerance. A cooked convex collision hull is not necessarily the mesh asset. Native cylinders use their mesh if their capsule collider is an unsuitable approximation.

Planar2D uses supported Box/Polygon/Edge/Circle/Capsule Collider2D shapes in XY, including offsets. Open edges remain open; out-of-plane movement fails validation. No projection of arbitrary 3D rooms into approximate line segments. Unsupported compound/scaled shapes need explicit proxy children or meshes.

Fixed authoring scale is baked into local geometry. During a take, translation/rotation and activation may change; topology, material, scale/shear, roster, hierarchy and clip bindings remain fixed. Reject changes visibly instead of silently changing labels. Capture actual world poses and estimated velocity/acceleration, including angular quantities and validity; replay disables motion writers and applies the observations rather than rerunning physics.

IDs are unique per placed instance, persisted before capture, sorted into stable numeric tables, and never derived from display names or Unity instance IDs. Disabled objects retain their indices. Source/listener tables map to object indices; pair order is `sourceIndex * listenerCount + listenerIndex`.

### Scene composition

Build two useful environments first:

- **TrainStation (implemented):** one platform with boarding steps, shelter/columns and an enclosed waiting room with a sliding door. Walking between the open platform and room gives repeatable visibility changes. An editable perimeter oval drives the locomotive and walkable carriage along the same route used to bake the rail mesh. Train service and the original train recording/room chime run automatically. The walking listener records a binaural head; additional sources/listeners remain supported but are not instantiated without a scene purpose. The Game view shows only the recording control while walking; mouse sensitivity appears when the cursor is released. Authoring/diagnostics remain in the Inspectors. Platform 2, obsolete markings/signs, the static reference receiver and unused synthesized engine were removed from both scene and builder. The operating controls and measured preview limitations are in the README.
- **AcousticLab:** two connected spaces, adjustable opening, source, shutter and marked receiver positions. Save the existing walkthrough generator's result as an editable scene. Adapt `SmollRoom.unity` as the XY/orthographic counterpart with appropriate movement.

Preserve FFT/legacy regression scenes. Make the train visible in Edit mode; runtime spawning is reserved for actual network actors. Show acoustic proxies through optional gizmos/lines. Place horn/speaker origins outside opaque housings. The initial player capsule supplies collision, not acoustic body occlusion around its listener.

```text
TrainStation
  Simulation          RayTraceManager + AudioManager, always active
  World               captured root
    Station           shared static prefab, track and actor anchors
    Train             visual, rigid proxy children, horn
    Door              fixed frame and moving leaf
    Announcement      emitter
    Player            controller, camera/receiver, optional emitter
  Scenario            routes, cue settings and reset state
  Presentation        lighting and existing Unity UI/text
```

`Prefabs/Station/Platform.prefab` owns static architecture, not players or managers; `Train.prefab` owns the locomotive/carriage visuals, proxies and attached horn/engine emitters. A scene-level `StationLoop` supplies distance-based poses; `LoopTrack.asset` is its baked Kenney-derived visual mesh. Rebuild the mesh after authoring route dimensions, before capture. The carriage root is explicitly registered because its world pose differs from a rigid train child through bends. Reuse these in solo and multiplayer. Keep the unique door, room emitter and player as native scene objects until another scene needs shared prefabs. Network player/train variants add ownership/replication later. Each proxy child has one supported collider.

Use a clean unit-scale train motion root, nested visual calibration, proxy children and horn. The existing locomotive prefab has scene-specific offsets/scale; rebuild the shared assembly from its FBX while preserving old references until migrated. Calibrate train/track proportions together in metres, with no accumulated scale corrections or shear. The emitter receives its manager through scene/session wiring.

Controls remain minimal: automatic recording/train/audio on Play, ordinary movement and nearby door interaction, then Save run. Replay starts from scene initialization through the Inspector; detailed numerical settings stay there too. Test manual movement and authored routes: distance changes, wall crossings, passing obstructions, independent source events and multiple listeners.

### Selected assets, lighting and materials

Keep the Built-in Render Pipeline and existing Kenney Train Kit 1.1. Select existing FBX assets `train-diesel-c`, `railroad-straight`, optional `railroad-curve`, `train-carriage-box` and `train-connector`. Use native geometry for platforms, shelters, rooms and doors. Detailed rails/bolts are decorative unless an experiment requires them. Imported models may stay unreadable when explicit collider proxies supply acoustics. No new model importer is required. The kit is CC0; retain its local license. [Kenney source](https://kenney.nl/assets/train-kit).

Optional visual imports, not downloaded by this specification: Poly Haven [Kloofendal 48d Partly Cloudy Pure Sky](https://polyhaven.com/a/kloofendal_48d_partly_cloudy_puresky), initially 2K EXR, and [Concrete Floor 02](https://polyhaven.com/a/concrete_floor_02), selected maps up to 2K. Preserve CC0/source/author/resolution/hash records. Keep unused vendor duplicates untouched until a dependency audit.

Use shared Standard materials for concrete/painted metal/safety paint, Skybox/Panoramic for the sky and existing unlit/text support for markers. Normal maps use normal import; roughness/masks use linear data and the Standard workflow's proper channel packing, including `smoothness = 1 - roughness`. Visual textures do not define acoustic coefficients.

`StationLighting.prefab` contains a directional sun, light probes, reflection probe and a few local lights. Share a LightingSettings asset and sky material between station scenes; RenderSettings and baked data remain scene-specific. Bake fixed architecture after layout settles; moving trains/doors/players use probes and realtime direct shadows. Check open/closed doors because a static bake cannot represent every state. Keep the lab neutrally lit. No graphics pipeline migration or post-processing package is required.

Visual materials and `AudioMaterial` assets stay separate. Preserve existing materials; add numerical fixtures `IdealRigid` (absorption/scattering 0/0) and `IdealAbsorber` (1/0) for the later energy solver. Both remain opaque in today's direct solver. Measured real materials need sourced curves; guessed values are not measurements.

Generate documented impulse, sweep and seeded-noise fixtures. Preserve the horn for preview while resolving its provenance before distributed datasets; existing wind/meme clips lack adjacent source records and stay out of controlled datasets. Identify already-reverberant recordings. Prepare dry PCM before capture; avoid simultaneous native AudioSource playback duplicating the custom renderer. No purchased audio library, speech service or asset catalog is required.

## 4. Capture and saved inputs

### Numeric scene and time

Unity objects resolve into existing `AcousticTake.Scene` tables: objects, local primitives, material curves, source bindings, receiver bindings and prepared clips. Per-observation `State[]` follows the object table. No GameObject, Transform, AudioClip or GPU padding crosses the file boundary.

Units are metres, seconds, Hz and radians; quaternions are xyzw. One fixed sample rate and clock origin belong to each take. Live capture uses integer sample timestamps mapped from `AudioSettings.dspTime`; silent capture keeps its configured rate. Device-rate changes or clock discontinuities end a segment. All future adapters must preserve sample-time identities.

Existing capture observes in LateUpdate. It preserves the first observation at each sample, counts duplicate polls and never rewrites a knot already consumed live. At recorded samples return stored float32 poses/motion exactly; between them use linear position and shortest-path quaternion interpolation. Hold the prior pose across a marked cut. Record observation gaps and interpolation version. More frequent reads of unchanged frame state do not constitute finer observations.

Velocity comes from consecutive observed poses; acceleration from velocity estimates; angular derivatives use relative quaternions. Initial/cut derivatives are invalid. Between observations, the declared interpolation determines velocity, not independent acceleration integration. Timestamp-mismatched Rigidbody velocity and visual pose cannot be combined; endpoint motion includes rotation about an offset where applicable.

One voice per source is implemented: Play restarts it, Stop ends it, Gain changes it. Multiple simultaneous voices on one source are deferred. Same-sample events retain append order. Events apply inside half-open audio blocks at exact samples, with clip offset/loop policy recorded; reject commands for already consumed audio.

Source inactivity stops new emission while preserving already-emitted tails. Geometry/listener activation follows the response schedule. An initial or mid-session segment must explicitly capture source phase/state and any required pre-roll; the current UI starts voices after capture begins. Playback, timing and source content are independent of reconstruction quality.

### Implemented files and versions

New take inputs use version 2, adding authenticated transmission/IOR/damping material curves. Version 1 remains readable and reconstructable using its original scene checksum; missing coefficients use zero transmission/damping and IOR one. New direct render and monitored-output manifests use version 2 for directional response records and channel metadata. Existing version-1 response files contain delay/gain only; readers must dispatch on the manifest version rather than infer row widths. Old take inputs remain usable for mono or binaural rendering.

```text
take/
  take.json           schema/completion, scene tables, clock, provenance, hashes
  geometry.bin        local primitives
  states.bin          timed observations
  events.bin          source events
  clip-0.f32          prepared mono PCM per unique clip
  monitor/            optional output.wav, timing/application logs and manifests
run/
  run.json
  responses.bin       existing direct responses only
  listener-0.wav      one summed mono/stereo float32 WAV per listener
```

Binary types are little-endian, with explicit serialization and no padding:

| Payload | Record |
|---|---|
| Geometry | Three float3 vertices plus int32 object/material indices: 44 bytes. |
| State frame | int64 sample followed by 80 bytes/object: position3, quaternion4, velocity3, angularVelocity3, acceleration3, angularAcceleration3 as float32; uint32 flags. |
| State flags | Active=1, Geometry=2, Source=4, Listener=8, Cut=16, VelocityValid=32, AccelerationValid=64. |
| Event | int64 sample/clipOffset, int32 sourceIndex/action, float32 gain, byte loop: 29 bytes. |
| Direct frame v2 | int64 sample then float32 delaySeconds/pressureGain/directionX/Y/Z per pair: `8 + 20*P` bytes. Direction points from the listener toward the arrival's origin in world coordinates. |
| Direct frame v1 | Legacy delaySeconds/pressureGain only: `8 + 8*P` bytes. |
| Clip | Mono float32 at take rate, content-hashed. |

The current downmix is channel averaging and resampling is linear. Replace or independently validate that preparation before high-fidelity audio claims; paired runs must always share exact prepared PCM and processing identity.

Run/output manifests bind `channelsPerListener`, `hrtfDataset`, `hrtfHash`, `hrtfTaps` and the DSP contract. WAV frames order left then right for binaural output; internal blocks order [sample,listener,ear]. Sample counts, queue capacities and lead always measure frames. Output-capture v2 application rows contain int64 scheduledSample/consumedSample/applicationTicks, int32 pairCount, then the same five float32 values per pair. Callback rows remain unchanged. Capture preserves both ear channels; overflow, channel mismatch or interrupted shutdown leaves it incomplete.

Capture validates geometry/materials, bindings, scale and hierarchy without rebuilding geometry every observation. Mesh scanning still costs O(mesh data) and must be profiled. Complete manifests publish only after payloads close and hash successfully. Reject incomplete files, unsupported formats, mismatches and non-finite data.

Keep two replay meanings separate: a new render evaluates the saved take with new settings; original-run review plays captured component PCM and uses actual response-application/queue logs. Component PCM includes underruns but excludes downstream device processing. Exact original playback does not imply bit-identical simulation across GPUs.

Offline numeric replay needs no original Unity scene. The current in-scene pose replay needs matching persisted take object IDs/assets and suspends all motion writers, including configured extra scripts; stopping restores them. The planned browser instead constructs isolated objects from the take or a saved visual snapshot, as specified in section 11. Seeking cached output PCM only reads the requested audio range; seeking a new simulation must reconstruct sufficient dry/response history and tails, or be explicitly marked a cold start.

### Compact station recording

`StationRecording` is the default station recorder. It stores consumed input changes, sample-stamped source events and the audio-sample/completed-tick mapping from scene initialization. It references the authored scene and runtime revision instead of retaining runtime snapshots. Train, door, look and walking share `StationDemo.InitializeSimulation` / `Step` in live playback and reconstruction.

| File | Contract |
|---|---|
| `input.json` | `station-input-from-init-v2`: scene identity, project content hash (Editor), runtime module, engine/platform, Git revision/dirty status, timestep, motion settings, counts/completion and checksums. |
| `input.bin` | Little-endian 25-byte changes: int64 tick, float32 movement X/Y and look X/Y, byte jump/interact flags. Held axes persist; look/buttons apply once. |
| `input-observations.bin` | Little-endian int64 audio sample plus int64 completed tick. Begins at (0,0); sample times increase and tick counts never decrease. |
| `take.json`, `events.bin` | Scene/source identity tables, source PCM and geometry hashes, audio rate/duration, and source-event records. Payload hashes do not require storing the payloads. |
| `live.json` | Small timing and deadline report. Not needed to replay. |

Full poses, geometry, prepared PCM and monitored audio are omitted by default. **Retain Full Reference** explicitly adds those diagnostic payloads. Existing full takes remain readable. Hashing, source events and timing metadata use the same writer in both modes; there is no second acoustic file format.

Play begins capture at initialization. Save completes it; another run starts by re-entering Play. Stopping Play also finalizes input data. Scene-reload playback runs sequentially from zero; there are no seeks, checkpoints, cached playback or artificial duration limits. Reopening requires matching scene/code/engine/platform and simulation settings. Live output checks device sample rate; offline reconstruction uses the saved rate independently.

`StationReplay` is an Editor operation available in Edit mode, from the Inspector and in batch jobs. It builds an isolated [Unity preview scene](https://docs.unity.com/en-us/engine/6000.3/script-reference/unityeditor/scenemanagement/editorscenemanager/newpreviewscene), uses the same initialization/step methods and regenerates an ordinary acoustic take. It checks authored geometry/source hashes and optionally compares original full observations. Runtime playback does not require this generated take. Walking support is queried explicitly so reconstruction does not depend on Play-only collision callbacks.

The current adapter covers station train/door/walking inputs. Arbitrary scene edits, rigid-body behavior and cross-platform determinism are not promised. Editor project identity hashes all files under `Assets`, `Packages` and `ProjectSettings` with relative paths; it excludes generated `Library` contents and checkout location. Older manifests without that hash retain their original module/scene checks; Player recordings require the matching runtime build. Preserve the matching Git revision/assets; identity hashes cannot recover missing source files. The README describes current usage and validation.

## 5. Physics, representations and configuration

### Solver families and independent controls

The supported direct baseline computes opaque visibility, delay `distance / soundSpeed` and pressure gain `referenceDistance / max(distance, minimumDistance)`. Planar2D uses the same point-source spreading convention with planar paths; it is not a cylindrical 2D wave equation. Surfaces are two-sided; epsilon/normal policies belong to the model version.

The first indirect estimator adds opaque geometric reflection/scattering. Before accumulation is implemented, derive emission/directional PDFs, path weights, reflection/scattering sampling, receiver acceptance measure and dimension-specific normalization. Validate free field, full absorption, a reflecting plane and an ideal specular room. A small independent image-source oracle is appropriate for those restricted fixtures, not another scene engine. [Allen–Berkley image-method paper](https://users.umiacs.umd.edu/~ramanid/cmsc828d_audio/AllenBerkley79.pdf).

Energy absorption/scattering coefficients do not determine complex pressure reflection coefficients. Phase-bearing transport needs measured/derived complex data or a declared idealized model. Transmission, air loss, source directivity, diffraction, room modes and moving-boundary wave propagation are separately validated extensions. Initial solves freeze geometry at a scene sample.

| Setting | Meaning |
|---|---|
| `dimension`, `physicsVersion` | Geometry and physical approximation. |
| `raysPerPairPerBand`, `samplerVersion`, `seed` | Sampling cost/noise. |
| `maxReflectionOrder`, `supportSamples` | Truncation/bias; changing these is not just reducing noise. |
| `solveStepSamples` or explicit solve samples | How often actual transport is solved. Today's direct API calls this `responseStepSamples`; preserve that API. |
| `outputStepSamples` or explicit output samples | How often the renderer requests a held/reconstructed response. Proposed, independent of solves. |
| Frequency grid and delay-bin edges | Response resolution; explicit arrays and units, not fixed bands. |
| `sampleRate` | Dry/output audio sample clock; fixed by the take. |
| DSP/STFT settings | Transform, block/window/hop, synthesis and buffering. |
| History, prediction limit and deadlines | Available model context and maximum accepted age/horizon. |
| Memory/batch limits | Execution capacity; never silently redefine physics or normalization. |

Author material curves at arbitrary ordered frequency knots, with interpolation/out-of-range conventions. The current energy curves interpolate linearly in Hz and clamp outside their knots. Transport bands have ordered nonoverlapping edges within [0, Fs/2] and a declared coefficient evaluation rule. Validate coefficients and grid bounds; interpolate once into the requested material-by-band table. More sampled bands do not create new measured material detail. Keep legacy eight-band packing in its adapter.

Sampling keys use named seed stream, snapshot identity, pair, band, global ray index and bounce. Do not derive them from frame count, worker ordering or dispatch size. Record actual counts. Start with independent sampling per pair/band; shared paths require correct PDFs. Low/reference seeds are independent. Tiling normalizes against the complete estimator budget, not each tile.

### Response quantities

For one scene sample, all response quantities describe unit excitation; source gain is applied once in DSP. The indirect target excludes deterministic direct sound. Every payload declares quantity, dimensions, units, grids, phase/time origin, normalization, origin and `includesDirect`.

| Quantity | float32 layout | Semantics |
|---|---|---|
| DirectPath | `[P,5]` | Implemented v2: delaySeconds, pressureGain, world directionX/Y/Z. Unit listener-to-arrival-origin direction; zero for coincident endpoints. |
| BandEnergy | `[P,F,D]` | Nonnegative incoherent sum/integral of squared unit-input path gains per delay bin, evaluated at the declared band model. |
| PressureIR | `[P,K]` | Signed response to a discrete unit impulse; `K` is support in audio samples. |
| ComplexTransfer | `[P,Q,2]` | Real/imaginary pressure transfer bins for one response; `Q=N/2+1` for even real FFT size N. |
| AudioSTFT | `[channel,frame,Q,2]` | Real/imaginary coefficients of actual source/listener audio with declared frame starts. This includes the sound content. |

BandEnergy is not a sound spectrogram, per-second density or calibrated SPL. Summing frequency bands does not establish broadband physical energy without a source/filter weighting convention. A blocked direct path can coexist with nonzero indirect energy. Missing output is never valid zero.

PressureIR and ComplexTransfer are equivalent representations only with matching full transform/support conventions. Use forward sign `exp(-i2πkn/N)`, inverse factor `1/N`, real DC/Nyquist and conjugate symmetry. Energy → pressure needs explicit synthesis; `sqrt(energy)` or FFT(energy) does not recover physical phase. Tag origin as estimated energy, synthesized pressure or validated traced pressure.

An IR's sample axis is propagation delay. The sequence of IRs/transfer functions is scene time. Audio STFT frames are analysis time. No code may conflate these axes because their tensors happen to have the same dimensions.

### Configurable DSP and STFT

A STFT configuration declares sample rate, window length W, FFT size N, hop J, analysis/synthesis windows, overlap normalization, causal/centered alignment, exact frame origin and boundary padding. Initially accept only validated supported sizes with `0 < J <= W <= N`; capabilities need not be fixed at 1024 or one thread-group size.

Bin spacing is `Fs/N`; effective resolution depends on window duration/shape. Increasing N with unchanged W zero-pads the same observation. Reducing J adds overlapping frames, not automatically new physical detail. Centered analysis needs future audio and its lookahead counts toward latency. SciPy provides an independent transform/window reference, not a Unity runtime requirement. [ShortTimeFFT](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.ShortTimeFFT.html).

Partitioned FFT convolution has independent block/partition/FFT lengths and must implement linear convolution without circular wrap. Validate a short CPU convolution oracle, impulse/tone/noise responses, absolute level, block invariance, transform round trips and first/last frames. Merely multiplying matching STFT bins is not a general exact implementation of long, time-varying convolution.

## 6. Shader contracts and ownership

### Existing direct ABI and proposed indirect ABI

The current CPU/HLSL direct ABI is:

| Direct struct/buffer | Offsets/stride |
|---|---|
| Primitive | float4 a/b/c at 0/16/32; uint object/material/pad/pad at 48/52/56/60; 64 bytes. Local vertices; 2D uses a/b. |
| State | float4 position/rotation at 0/16; uint flags/padding at 32; 48 bytes. |
| Source/listener bindings | int32 object indices, 4 bytes each. |
| Response | float4 direction at 0 (xyz world listener-to-arrival-origin, w=0); float32 delay/gain at 16/20; reserved zero at 24/28; 32 bytes. Disk serialization is separately packed to five floats. |
| Optional BVH node | float4 boundsMin/boundsMax at 0/16; uint left/right/first/count at 32/36/40/44; 48 bytes. World-space bounds; a separate int32 array maps leaves to original primitive IDs. |

The future indirect world-space ABI uses full 16-byte record lanes. It is separate from direct's local-geometry/state layout:

```hlsl
struct GpuTriangle { float4 a, b, c; uint4 ids; };    // 64 bytes
struct GpuSegment  { float4 a, b; uint4 ids; };       // 48 bytes
struct GpuMaterialBand { float4 coefficients; };    // 16 bytes
struct GpuBvhNode { float4 boundsMin, boundsMax; uint4 links; }; // 48
struct GpuEndpoint { float4 positionRadius, rotation; uint4 ids; }; // 48
struct GpuEnergyArrival { float4 directionDelay; uint4 ids; };     // 32
```

C# mirrors use sequential Pack=4, Vector4 and an explicit four-uint struct; assert every size/offset and verify distinctive values with a GPU echo fixture on each claimed backend. No bools, float-encoded IDs or implicit layout. Scalar arrays remain 4-byte elements. Absolute time stays int64 on CPU/files; kernels use relative times and deterministic sample keys.

Primitive ids are object/material/stablePrimitive/flags (active bit 0); unused vertex lanes zero. 2D coordinates are XY with unused lanes zero. Normals derive from geometry with the declared hit-side rule. Material table is `[material*F + band]`, coefficients absorption/scattering/0/0; band records contain lower/center/upper Hz/0.

BVH links are left/right/firstPrimitive/count. Leaves have nonzero count and UINT_MAX children; internal nodes have valid children and count zero. Empty geometry has zero nodes. A separate primitive-index array preserves primitive identity and per-object ranges through BVH ordering.

Endpoint fields are world position + receiver radius (source radius zero), quaternion xyzw, and object/flags/0/0. Radius is an estimator parameter whose normalization must converge, not an unexplained gain control.

Energy arrivals hold world listener-to-arrival-origin direction + delaySeconds and source/listener/reflectionOrder/flags, valid bit 0. This direction is opposite the final ray's propagation direction; a hit position alone cannot identify it. Indirect order starts at 1. A separate buffer holds `[arrival, activeBand]` energy; global band equals batch base plus local band. Initialize every valid element. Coherent pressure arrivals require their own descriptor, not reinterpretation.

### Geometry, memory and scheduling

**Implemented direct BVH:** compile local geometry once, cache its first world pose and build one balanced median-split hierarchy with at most four primitives per leaf. Before each solve, any geometry object whose position or quaternion differs from its first pose permanently leaves the static set for that solver session. Rebuild only when membership changes; moving objects are tested directly at the current pose, even after stopping or returning to their initial pose. Activation/geometry flags remain per-query checks. This classification uses observed poses, independently of Unity's authoring Static flags, so old recordings need no schema migration or hand-authored mobility tags.

The CPU traverses cached static world primitives; HLSL traverses uploaded world bounds and retains the original local-primitive transform/intersection math. Conservative bounds account for CPU/GPU rounding. The primitive-index array stores the static tree's sorted prefix and the moving list's suffix, preserving original primitive IDs. GPU submission classifies every frame before dispatch: if any frame moves an object, every frame in that batch tests it directly. All frames share one static tree, uploaded only after a membership change. No per-frame node snapshots or moving-bound refits remain. `bvhUpdates` counts tree builds including initialization, `bvhUpdateSeconds` includes classification and later rebuilds, and `bvhStaticPrimitives` records the final static count. Brute force remains the default and correctness oracle in 2D/3D; [station benchmarks](cleanup-results.md) measure the tradeoff. Per-instance dynamic BVHs remain deferred until measured workloads justify them.

Memory estimates precede submission:

```text
energy response      4 * P * F * D bytes
pressure response    4 * P * K bytes
complex response     8 * P * Q bytes
arrival scratch      capacity * (32 + 4 * activeBandCount) bytes
```

Include masks, reduction, history, DSP and model tensors in the full budget. Tile pair/band/ray work and offline output; resident live responses must fit. Initialize/bind valid storage even for zero logical counts, pass explicit counts, and guard every kernel index before access. Use a race-free reduction and no CPU hit-count readback merely to launch the next GPU pass.

An overflow invalidates a solve. Offline retries fitting partitions with identical sampling identity/normalization; realtime uses a declared fallback. No partial response can masquerade as complete.

One session owns buffers; submitted work keeps immutable inputs/output slots until completion and readback retirement. Keep the current one-in-flight direct batching until more slots have measured value. Cancellation prevents publication but does not free in-use buffers. Copy request-owned readback before release.

Proposed boundary, implemented as concrete methods rather than a plugin framework:

```text
SolveRequest:
  generation, scene/state/config hashes, requested scene sample
  ordered pairs, physics, quantity/grid, budget, seed
ResponseFrame:
  same identity + pair validity/activity
  deterministic direct result + indirect payload
  solve counts, overflow/truncation, timing and publication information
```

A request hash includes evaluated state bytes, not only timestamp. Reconfiguration/seek changes generation. Actual transport completions publish monotonically by scene sample; stale completions are counted and cannot replace newer source data. Predicted outputs have a separate output-sample cursor: a late actual solve may refresh predictor history for future outputs without rewriting audio already consumed.

## 7. Live, offline and audio rendering

| Mode | Inputs | Execution |
|---|---|---|
| Live | Captured Unity observations and effective source events. | Bounded work against output deadlines; select one monitored receiver. |
| Offline replay | Evaluate immutable take at requested sample positions. | Same solver/DSP; wait and stream results. Wall time never advances scene time. |
| Scripted generation | Explicit simulation ticks and virtual sample clock. | Record one take first, then reuse it for all render budgets. |

Scripted routes evaluate motion from declared time; physics scenarios record actual stepped results and tick-to-sample mapping. Do not accumulate wall-clock deltaTime or promise cross-machine physics determinism. Fix parameters before capture and preserve independent scene/motion/signal seeds.

Current live direct rendering waits for a closed observation interval before interpolation. Record observation, submission, completion, publication and actual application timestamps. Prediction is a new explicit policy; it must not silently change existing live/direct replay semantics.

```text
capture or evaluate scene state
submit due actual solves while a slot is free
retire completions; validate identity and publish to response history
for each due output sample:
    choose compatible direct and indirect response/reconstruction
    apply source events, synthesize/convolve, produce bounded PCM
audio callback:
    copy prepared PCM; fill missing range with silence; count underruns
```

No scene access, allocation, locks, inference, GPU waits or file I/O in the audio callback. Current control-thread mixing may move to one owned worker if measured necessary. GPU features/inference/DSP compete with graphics; CPU execution may win for small workloads. Capturing detailed rays is an opt-in diagnostic cost.

For time-varying rendering, the conceptual contract is `y_l[n] = sum_s sum_k h_sl[n,k] * x_s[n-k]`, where n is the output sample and k propagation delay. Our frozen-scene responses approximate h under a declared application/transition policy. They do not solve retarded-time propagation through moving boundaries. Preserve source history across response updates and render already-emitted tails.

Today's direct path holds responses with optional tap crossfades and linear/windowed-sinc fractional delay. Continuous delay/Doppler handling is future work: interpolate delay at audio rate under a declared motion approximation and do not add a duplicate pitch shift. Validate movement, time-origin and absolute arrival errors before labeling it improved physical fidelity.

Energy synthesis uses a documented filter bank, normalized band overlap and reproducible phase/noise whose response energy matches the target. Maintain temporal continuity; independent random synthesis each frame causes audible changes unrelated to the geometry. Pressure/complex responses use compatible convolution. Bounded old/new filter transitions preserve history; cuts/seeks obey their declared reset/pre-roll policy.

**Implemented direct binaural rendering:** each receiver is one head pose with two ear channels. Direct paths retain world listener-to-source direction, delay and pressure gain. Inverse head rotation converts direction into +X right, +Y up, +Z forward. The shared live/offline mixer interpolates the bundled MIT KEMAR compact HRIRs on their measured azimuth/elevation grid, composes the result with the fractional-delay FIR once per response and crossfades old/new filters under the existing transition policy. Source gain is applied once; pressure/occlusion gain is applied once. No additional native Unity spatialization is applied to the output driver. Mono remains the default for old settings; the station explicitly selects Binaural.

The measured dataset contains 710 mirrored directions, 128 taps/ear at 44.1 kHz, with paired ear gain/timing preserved. Filters resample once at preparation for other rates. Both outputs sum all source paths; there are no independently traced ear receivers. Source history includes the HRIR support, and offline tail settings must cover the desired propagated/filter tail. Geometric delay is added once. The dataset's bulk measurement delay was cropped by its authors; its remaining timing is not a calibrated absolute eardrum arrival. Do not add an ear-offset ITD on top of measured ITD. `dspLatencySamples` reports fractional-delay FIR latency separately from the HRIR's time response. Coincident endpoints use the frontal filter convention; unmeasured elevations clamp to the measured elevation range. Interpolation wraps azimuth continuously across the rear seam. The spectral mixer uses the same angular interpolation and crossfades old/new responses over `transitionSamples` at synthesis hops, with overlap-add; a cut resets that transition and Doppler history.

This is a generic far-field mannequin model for headphones, not personalized HRTFs, near-field head scattering, headphone equalization or complete wave acoustics. Directional reflection/reverberation rendering remains future work: preserve the final segment's incoming direction before accumulation, plus any hit position needed for diagnostics. Scalar band energy cannot recover directional reverberation without extra arrival data or an explicit diffuse assumption. Keep mono reference checks. Binaural tests verify ear timing/level, head rotation, stereo routing/capture and live/offline sample agreement on CPU and Metal.

Use the active Editor for authoring-time jobs, or batch mode when that project is closed. A second Editor must not share the open project. Existing CPU batch jobs can use -nographics; HLSL jobs need a graphics device. A built worker is an option when simultaneous authoring/jobs justify it, not required infrastructure. Interrupted runs restart initially; resumability later must restore DSP history and verify all identities.

## 8. Reconstruction and temporal upscaling

### Experiments and model boundary

“Audio DLSS” here means our learned acoustic reconstruction, without dependency on NVIDIA's graphics DLSS SDK. Its useful analogy is reusing noisy current/history information with motion and validity, not copying an undocumented frame count. NVIDIA describes temporal reconstruction and stability/responsiveness tradeoffs, not a universal context size we can adopt. [NVIDIA technical overview](https://research.nvidia.com/labs/adlr/DLSS4/).

| Experiment | Inputs → target | Prerequisite |
|---|---|---|
| Direct temporal pilot | Coarse delay/gain → dense direct responses. | Existing renderer plus per-pair checks; deterministic baselines first. |
| Indirect denoising | Low rays → checked high-ray energy at identical samples/grids. | Validated energy estimator. |
| Temporal response upscaling | Sparse solves/history/current observed motion → responses on a denser scene-time grid. | Same-model dense references and availability-aware query context. |
| Combined reconstruction | Sparse low-ray solves → dense high-ray targets. | Isolated denoising and temporal reports. |
| Pressure/complex reconstruction | Low-cost phase-bearing response → matched IR/complex response. | Pressure/material-phase teacher and consistency checks. |
| Audio spectrogram reconstruction | Cheap rendered audio STFT, known dry signal and available context → high-quality audio STFT. | Validated audio targets, synthesis and streaming transform schedule. |
| Cross-grid reconstruction | Explicit coarse response/STFT grid → explicit finer grid. | Separate input/output descriptors and trained mapping; no silent resize. |

Ray budget, response cadence, delay-bin resolution and frequency grid can all be reduced, but changing bounce/support/physics is a different bias problem. Do not claim denoising evidence proves reconstruction of omitted physical effects.

PressureIR and ComplexTransfer preserve the reusable scene response. AudioSTFT contains the played content and requires held-out source recordings/content families as well as scenes. A complex spectrogram predicts real/imaginary values, not just an image of magnitude. Spectral-domain audio super-resolution is demonstrated by AERO; its result is evidence of feasibility, not a chosen realtime architecture or verified acoustic solver. [AERO](https://arxiv.org/abs/2211.12232).

A temporal request explicitly names:
```text
ReconstructionRequest:
  generation, take/scene identity, pair IDs
  targetSceneSample, inputDecisionSample, target quantity/grid
  actual low response history + scene/publication samples + masks
  available query observations + their timestamps
  model/preprocessing identity, reset epoch and prediction-age limit
ReconstructionOutput:
  targetSceneSample, quantity/grid, pair validity, prediction
  completion/application timing, model identity, fallback reason
```

Actual solve history and generated responses are separate. Initially feed actual low solves only, not earlier predictions. Finite external history avoids uncontrolled recursive drift. If a recurrent/prediction-feedback model is later used, train/evaluate full unrolled sequences with its own predicted history and explicit reset rules.

For illustration, solve every 20 ms and request output every 5 ms. These are example cadences, not presets. At each target, evaluate one requested response using available history and its age; several target times may share the latest actual solve. Current observed motion can help, but a closed door/new reflection invisible in the inputs remains unidentifiable.

### Causal and buffered modes

Realtime forecasting uses low responses with scene sample s <= target t and publication time a <= decision d. Offline/buffered interpolation may use a later endpoint only as a separately labeled lookahead mode with its waiting latency. It cannot qualify a no-lookahead model.

Define all timestamps in the take clock:
```text
d = t + playoutLeadSamples - inferenceAndDspReserveSamples
eligible low: s <= t AND a <= d
prediction horizon = t - latest eligible low scene sample
response age at decision = d - latest eligible low scene sample
```

Budgets are nonnegative, reserve <= lead. Freeze model inputs at d; include inference, synthesis and publication overhead in the measured reserve. Physical propagation delay remains in the response. Late model completion cannot rewrite consumed samples or shift output to make metrics look better.

Two availability modes are supported in the dataset design:

- **replay-delayed-v1:** a low result at s becomes available at the first captured observation strictly after s, plus saved solve delay. This reproduces the need for a closed interpolation interval under an idealized schedule, not measured GPU speed.
- **measured-publication-v1:** record actual accepted publication samples and the last captured-observation index used by each solve. Reconstruct features only from that bounded prefix. Existing offline run timings are insufficient; this log is new work.

Both modes must validate timestamps against their capture/pose availability policy. Restrict training targets to before the final captured observation; held terminal tails are DSP evaluation. A target without enough closed input history is unavailable, not padded observed silence.

Keep latest H eligible valid actual responses in increasing scene time within the reset epoch; use actual spacing. Left-pad with index -1 and zero values/validity. Reset on seeks, incompatible configurations, pair changes, teleports and relevant discontinuities/activation changes. Sound Play/Stop alone changes excitation and does not reset the unit-response model. Previous-target loss windows use their own deadlines.

Temporal query context uses **latest-observation-hold-v1**: choose the latest observation q <= t received by d and use that actual pose/motion, with age (t-q)/Fs. Do not silently interpolate it from a future observation or extrapolate it to t. More sophisticated prediction is a different recorded input policy. Measured mode needs observation receipt times; idealized mode assumes receipt at its sample and labels that assumption.

Fresh direct visibility can gate stale indirect prediction or request an early actual solve. Its cost belongs to the budget, and the same gate/schedule must be replayed for comparisons. First temporal models do not receive a new raw geometry tensor or assume arbitrary moving obstacles are inferable from endpoint motion. Evaluate those failures explicitly.

Runtime has configured maximum prediction horizon, observation age and response age, with validation-selected limits. Reject unsupported grids, non-finite outputs, stale generation or missed deadlines. Validate quantity constraints, support and fixture-derived gain/energy limits; a finite tensor alone is not validated acoustics. Prefer a compatible valid low response within its age limit; otherwise retain current deterministic direct output with the declared bounded indirect transition/fade. Never hold stale indirect output indefinitely. With cross-grid models, the fallback needs a validated explicit mapping or that configuration is unsupported.

### Three time scales and phase consistency

History spans scene updates; IR support spans propagation delay; audio/STFT context spans signal samples. Past history does not itself add steady-state lookahead, although startup warmup and actual solve/processing delay remain. A long IR can be reconstructed at one current scene time without future scene observations.

For phase-bearing targets, retain absolute delay and phase origins. Moving arrivals must shift consistently; averaging unaligned pressure IRs can cancel them and erase energy. Known bulk-delay alignment may be an explicit reversible model preprocessing step, but restore it for absolute timing metrics and never align evaluation outputs opportunistically. Preserve real-signal spectral symmetry and check the resulting IR's support/causality.

For AudioSTFT, validate inverse reconstruction and phase continuity through overlapping frames. A denser STFT hop is not a higher acoustic solve rate. A larger FFT with unchanged analysis window is zero-padding. Sample-rate/bandwidth extension is a separate learned inference task, not guaranteed recovery of missing frequencies.

For moving scenes, snapshot transfer functions cannot simply be multiplied by source STFT bins and claimed to implement arbitrary time-varying propagation. Use the declared convolution/transition renderer and score audible clicks, smearing, phase/arrival errors and latency alongside tensor losses.

## 9. Example generation and dataset format

### Collection and teacher acceptance

A resolved recipe records scene/layout family, dimension, geometry/material variation, endpoint placements, motion curves, source-event schedule, capture/tick schedule, length and independent scene/motion/signal seeds. Resolve ranges before capture using authored traversable bounds. Manual and scripted captures use the same take format.

The recordings browser and batch entry point share the job contract in section 11. Selecting a take for replay never changes its inputs or its dataset membership. Generating pairs is explicit, and successfully rendered comparisons become training examples only after the acceptance and availability checks below.

Start with open paths, wall/door crossings, moving obstructions and multiple pairs; add reflection rooms after solver validation. Include stationary/slow/fast motion, valid zeros, activity transitions, long gaps, cuts and cold/warm starts. Keep difficult valid examples even when the low result is poor.

```text
assign related layout/trajectory/session families to splits
for resolved recipe:
    take = capture once
    low runs = render sparse and/or low-ray variants
    teacher = render dense and/or high-ray reference
    validate identity, physics, quality and publication assumptions
    retain failures with reasons
    for accepted pair and target timestamp:
        build available history and query context at its own deadline
        append an index row referencing original payloads
validate descriptors, hashes and split isolation; publish complete dataset
```

Generate low-ray runs separately; merely decimating the teacher does not simulate low-ray noise. Pure cadence experiments can reuse rows only when budget, estimator, state and sample identities truly match. Use named RNG streams derived through a versioned stable hash, never Python's built-in hash.

Direct teachers require per-pair delay/gain tolerances in addition to the existing finer-run mixed-WAV gate; cancellation can hide pair errors. Dense references resolve only the captured interpolation. Physics/rapid-motion validation needs finer capture or analytic trajectories as well.

For stochastic targets, predeclare K >= 2 independent reference seeds at identical sample/grid/physics settings. Compute mean and unbiased sample variance; do not cherry-pick seeds. Failed jobs rerun under the same identity or leave the target unaccepted. Check increasing budgets, receiver/support convergence and analytic fixtures; small variance alone cannot certify low bias. Use explicit absolute/relative tolerances and a near-zero floor. Report rejection counts by scene/category, not just accepted-set scores.

For pressure teachers, average only samples estimating the same phase-aligned physical quantity. For synthesized audio, share synthesis version/seed and mark synthesized phase. Live underrun PCM is diagnostic evidence, never the offline target.

### Files and identities

Preserve existing take/run payloads; no file per window, copied Unity scene, database or dataset pickle:

```text
<data-root>/
  takes/...                        existing take files
  runs/...                         runs may also remain inside take folders
  datasets/<name>/
    dataset.json
    generation.json                resolved recipe and quality configuration
    bundles.jsonl
    examples.jsonl
    derived/<bundle-id>/           availability and teacher statistics only
  training/<run-id>/
    config.json, preprocessing.json, metrics.json, checkpoints/, export/
```

Metadata paths are relative to the dataset directory; artifact paths are relative to artifact_root, itself relative to that directory. Hash exact bytes with SHA-256, binding manifests as well as payloads. Do not recreate Unity JsonUtility scene hashes with Python JSON serialization. No cyclic self-hash: dataset identity is the finalized dataset.json file hash; that manifest binds index hashes and bundle payload identities.

| dataset.json field | Required contract |
|---|---|
| schema | acoustic-training-dataset-v1; explicit failure for unsupported formats. This remains a draft until a writer exists. |
| experiment, quantity, dimension | One compatible response experiment per dataset. |
| artifact_root, sample_rate, history_frames | Portable root, fixed positive rate and H. |
| grid | Direct channel order; or complete frequency/delay/FFT/STFT grids, units, origin and direct-inclusion policy. |
| context_schema, query_schema | Exact feature order/units; query_schema is none for initial denoising. |
| availability | Timing mode, lead/reserve, delays or publication-log identities, context receipt policy and maximum ages. |
| split_groups | Explicit group → train/validation/test mapping. |
| bundles, examples | File, row count and SHA-256. |
| generation, complete | Resolved generation.json filename/SHA-256 and completion status. |

Each bundle binds take/scene/state/event hashes, stable pair mapping, run/config/code hashes, low variants/seeds, reference ensemble, descriptors, quality report and split ancestry. The generation configuration is saved with its descriptor/hash, not an unresolvable hash alone. Link layout-family, trajectory-family and multiplayer-session ancestry; all connected related groups remain in one split.

Every numeric descriptor declares file, dtype, shape, axes, units, logical C order, byte_offset, byte_length and whole-file SHA-256; explicit byte strides describe interleaving. byte_length is the addressed span beginning at byte_offset. Validate sizes/offset products before mapping. Use <f4, <i8 and |u1 for float32, sample/index int64 and byte masks. Timestamps remain integers; the direct propagation-delay field remains float seconds.

Current version-2 direct scalar address:
```text
frame r, pair p, channel c:
offset = r*(8 + 20*P) + 8 + 20*p + 4*c
c=0 delaySeconds; c=1 pressureGain; c=2,3,4 world directionX,Y,Z

[R,P,5] strides = [8+20*P, 20, 4]
initial offset = 8
nonempty addressed span = (R-1)*(8+20*P) + 20*P
full file length = R*(8+20*P)
```

Use an unaligned structured dtype/explicit strides, not float32 reshape of timestamps. Version-1 direct rows have two channels and an 8-byte pair stride. A mono temporal experiment may select only delay/gain from version 2, but its descriptor must state that selection. Preserve directions and recorded head rotations for binaural experiments. Directions are unit vectors, with the coincident zero convention; they require directional normalization/validity rather than generic scalar interpolation.

Future response runs use response-samples.i64 [R], responses.f32 [R,P,...quantity axes...] and pair-valid.u8 [R,P], described in run.json. Energy uses [R,P,F,D], pressure [R,P,K], complex [R,P,Q,2]. Separate active endpoints from numeric validity. Declared grid sizes can differ only in a named cross-grid experiment with both descriptors.

Availability data uses low-available.i64 [R]. Measured mode adds low-observation-cutoffs.i64 [R] and observation-received.i64 [capturedObservationCount], all bound to the producer's exact run. Record decisions/skips/resets needed to reproduce adaptive scheduling. Publication refers to a whole response unless the producer actually publishes pairs independently.

Energy teacher aggregates are target-mean.f32 and target-variance.f32 [R,P,F,D], with K recorded. Accumulate offline in float64, store finite float32; variance has squared-energy units, standard error sqrt(variance/K). Original runs stay referenced. Other quantities require their own uncertainty semantics.

Example rows identify bundle/low variant/pair, target frame/sample, decision sample, H low indices, reset epoch, optional prior target and query observation index. Invalid history uses -1; index boundaries and hashes are checked. The previous target window can be built on demand without a duplicate row.

This JSON is a **timing fixture**, not production tuning; placeholders and complete:false make it non-usable:
```json
{
  "schema": "acoustic-training-dataset-v1",
  "experiment": "direct-temporal-resolution-v1",
  "quantity": "DirectPath",
  "dimension": "Spatial3D",
  "artifact_root": "../..",
  "sample_rate": 44100,
  "history_frames": 4,
  "grid": {"channels": ["delaySeconds", "pressureGain", "directionX", "directionY", "directionZ"]},
  "context_schema": "relative-motion-v1",
  "query_schema": "none",
  "availability": {
    "mode": "replay-delayed-v1",
    "solve_delay_samples": 128,
    "playout_lead_samples": 768,
    "inference_and_dsp_reserve_samples": 256,
    "max_prediction_samples": 512,
    "max_response_age_samples": 1024,
    "max_observation_age_samples": 512,
    "observation_receipt": "sample-time-assumed"
  },
  "split_groups": {"fixture-family": "train"},
  "bundles": {"file": "bundles.jsonl", "rows": 1, "sha256": "<sha256>"},
  "examples": {"file": "examples.jsonl", "rows": 1, "sha256": "<sha256>"},
  "generation": {"file": "generation.json", "sha256": "<sha256>"},
  "complete": false
}
```

With observations every 256 samples, low step 512 and high step 128: target frame 12 is t=1536, d=2048. Low frames 0–3 have samples [0,512,1024,1536] and availability [384,896,1408,1920]. Low frame 4 is excluded:
```json
{"example_id":"fixture-p0-t1536","bundle_id":"fixture","low_variant":"low-0","pair_index":0,"target_frame":12,"target_sample":1536,"decision_sample":2048,"low_frames":[0,1,2,3],"reset_epoch":0,"previous_target_frame":11,"query_observation":-1}
```

Training-only scales and checkpoints stay outside Assets. Memory-map payloads read-only with a bounded worker-local handle cache, copying selected minibatches before normalization. Do not load all overlapping windows into RAM.

The proposed **acoustic-audio-training-dataset-v1** reuses bundle/hash/split/availability rules with quantity AudioSTFT and a listener/channel-frame example unit. Required input_grid and output_grid contain the full STFT configurations, explicit frame starts, sample rates and analysis/synthesis delays. Store spectra as float32 [frame,listener,channel,frequencyBin,realImag] and validity as bytes [frame,listener,channel]; explicitly select one mono channel or paired left/right channels. Binaural datasets bind the HRTF identity, head orientations and renderer settings; preserve interaural timing/level and split related ear examples together. Example rows replace pair_index with listener_index/channel_index and reference low frame indices plus the target frame. A target timestamp identifies the declared analysis-frame origin, not automatically its availability: the whole required audio window must exist by the decision deadline.

Bind original low/high rendered PCM, exact dry PCM/events, pre-roll, renderer/phase origin and any optional source-conditioning ranges. If dry conditioning is selected, its samples/availability and source order are part of the feature schema; do not expose future streaming audio. Use the same audio sample rate initially while varying acoustic budget/cadence or explicit STFT grids. Bandwidth/sample-rate extension adds an explicit sample-clock mapping as its own experiment. Missing phase is not silently filled with teacher phase. The response-v1 reader must reject this audio schema unless it implements these distinct axes and timing rules.

## 10. Models, training and deployment

### Exact initial response features

For energy denoising, minibatch axis P represents independently processed pair examples:
```text
lowEnergy        float32 [P,H,F,D]
historyValid     mask    [P,H]
context          float32 [P,H,12]
targetEnergy     float32 [P,F,D]
targetValidity   mask    [P,F,D]
targetUncertainty        [P,F,D]   supervision/quality metadata only
prediction       float32 [P,F,D]
```

History schema **relative-motion-v1**, oldest to newest:

| Columns | Values |
|---|---|
| 0–2 | Source minus listener world position at the low response's scene sample. |
| 3–5 | Relative linear velocity in m/s; zero if either endpoint's velocity is invalid. |
| 6 | Both velocities valid, 0/1. |
| 7–8 | Source/listener active with their role enabled, independent of clip playback. |
| 9 | (target sample - low scene sample)/Fs. |
| 10 | (decision sample - low publication sample)/Fs. |
| 11 | log1p(raysPerPairPerBand); zero for deterministic direct data. |

Planar features have zero third spatial/velocity component. First energy experiments use a uniform budget per pair/band; nonuniform band budgets require their own feature schema. Poses used to build low-response context are bounded by that solve's observation cutoff. No future teacher, scene ID, camera image or dry spectrum appears in this input.

Temporal models add **temporal-query-v1 [P,12]**, built from the latest available held observation q:
```text
0–2   current observed source-minus-listener position, metres
3–5   observed source velocity, m/s
6–8   observed listener velocity, m/s
9     both velocities valid; otherwise zero columns 3–8
10    (target_sample - q)/Fs
11    (target_sample - latest_low_scene_sample)/Fs
```

Individual velocities distinguish common motion through the world from a stationary relative pose. Current endpoint activity and cuts gate/reset outside the graph. Query receipt and freshness checks are mandatory. Visibility gates can use the deterministic direct path outside this feature schema, with their cost counted; new geometry features require a named schema instead of hidden inputs.

No valid low history means unavailable/fallback. Valid zero response remains observed data. Initial padding is zeroed after preprocessing with validity appended. Do not normalize Boolean masks.

### Candidate graphs and losses

Begin with a small energy network, using two exported inputs:
```text
low_log_energy [P,H,F,D] → padded convolution H→W
context        [P,H,13]  → flatten → linear projection to W → broadcast-add
                         → L residual blocks (two 3×3 convs + ReLU each)
                         → 1×1 head + Softplus
                         → normalized linear energy [P,1,F,D]
```

The 13th context value is history validity. The temporal variant adds the 12 query values to the conditioning projection through a declared third input. H, W, L and grids are config values chosen against measured cost; no hardcoded band count or unbounded tensor. Padding is across frequency/delay, never future scene history.

Multiply output by saved per-band scale s[f] and apply activity masks outside the graph. Record this adapter in the model manifest. Float32 energy history costs 4*P*H*F*D bytes before context, activations, gradients and optimizer state.

Fit s[f] as RMS over unique accepted training pair-target frames, with an explicit positive floor. Multiple low variants do not multiply a target's weight. Fit continuous context normalization only on training examples; retain invalid-velocity flags, zero invalid groups after normalization, and record all-zero bands/features. Validation/test use frozen statistics.

Energy loss is a configured weighted sum of masked absolute error in E/s, log1p(E/s) error, and reference-matched temporal derivative error. Use actual sample spacing; exclude resets/discontinuities. Score absolute energy, decay and arrival timing separately. Never hide failed targets as zeros or allow one aggregate loss to mask level errors.

Direct pilot input is [P,H,2], with distinct delay/gain units and deterministic hold/motion-aware baselines. A network is optional if a cheap baseline suffices.

Pressure reconstruction uses signed outputs [P,K]; complex reconstruction uses real/imaginary outputs [P,Q,2]. Use compatible finite-history convolution candidates, preserving transforms and phase origins. Supervise absolute waveform/complex error plus delay/energy and multi-resolution spectral checks; phase losses in near-zero bins need masks/floors. Do not reuse the energy Softplus/log transform for signed/complex data. Final architecture/weights require the validated pressure teacher and a measured pilot.

Full-audio STFT reconstruction needs its own streaming model experiment with real/imaginary outputs, known source conditioning where used, explicit analysis/synthesis latency and waveform reconstruction losses. Hold out dry recording families; random crops of one clip cannot cross splits. Model families like spectral encoder-decoders are candidates, not accepted realtime implementations. Do not add a GAN/diffusion pipeline before a small baseline shows a need.

Each checkpoint initially supports one quantity, physics/dimension and declared input/output grids/history policy. Simulator configurability does not imply arbitrary model configurability. Pair counts can be chunked into verified batch sizes with explicit padding masks; padding pairs cannot affect other outputs. Start separate 2D/3D checkpoints unless held-out evidence supports combining them.

### Training and evaluation loop

Python, NumPy and PyTorch are the proposed tools. Use MPS where verified on this Mac, CPU for tiny checks, or CUDA on another measured machine; backend identity belongs in results. Cross-backend bitwise reproducibility is not promised. [PyTorch MPS](https://docs.pytorch.org/docs/main/notes/mps.html).

Split before windowing: all related layouts/trajectories/takes, alternate budgets/seeds/listeners and multiplayer segments/client views remain grouped. Sampling chooses groups/takes then windows so long captures do not dominate. Persist category weights and RNG state. Evaluation uses the full accepted distribution and includes rejection/fallback rates.

```text
verify completed dataset, payload hashes, split isolation and availability
fit training-only normalization; save preprocessing identity
initialize small model + configured AdamW
for each epoch:
    for group-balanced minibatch:
        load causal histories, query context and targets
        also build previous target's window at its own deadline if needed
        predict, restore units, compute masked losses
        zero gradients, backpropagate, optional configured clipping, optimizer step
    evaluate fixed validation sequences without gradients
    save last checkpoint and best under the frozen selection rule
    stop under the declared training/early-stopping budget
reload best; evaluate held-out test with frozen baselines
export; verify Python/Unity numerical parity and total live cost
```

Save model/optimizer, epoch/update, sampler/RNG state, dependency/backend versions and dataset/config/preprocessing hashes. Resume only against matching identities. Record failed/over-budget trials. Test once after model selection; changes prompted by test results require fresh held-out evidence.

Loader starts with one process and bounded read-only mmap handles; tune workers only from RAM/throughput evidence. Overfit one tiny fixture to prove units/targets/gradients connect, then evaluate generalization. Test realistic variable publication delay, gaps, warmup, cuts and long sequences; an ideal timing fixture does not qualify a live system.

Compare raw/held low output, causal deterministic reconstruction, the model and spending the model's cost on more rays/more solves, all at matched information and allowed latency. For temporal experiments, report error versus prediction horizon and transition regime, not only average error. Combined budgets require the isolated reports first.

### Deployment

Before expensive collection/training, export an untrained representative graph and verify actual Unity import, supported operators/shapes, CPU/GPU numerical parity and realistic tensor cost. Pin an explicit compatible ONNX opset: the reviewed Unity 2.4 documentation advertises 7–15, so newer exporter defaults cannot be assumed compatible. [PyTorch exporter](https://docs.pytorch.org/docs/main/onnx_export.html), [Unity model support](https://docs.unity3d.com/Packages/com.unity.ai.inference@2.4/manual/index.html).

Export weights/graph plus tensor names/shapes, quantity/grids/phase origin, preprocessing, history/query/reset/fallback policy, supported dimensions/physics, ages/lookahead, code/data hashes and parity fixtures. Put only a selected deployment model and its manifest in Assets/Models; bulk data/checkpoints stay outside Assets/source commits.

The adapter runs low solve → available history/query → bounded inference → output validation → response renderer. Measure feature construction, upload/readback, inference, postprocessing and DSP under scene graphics load. A model is accepted only with held-out improvement over simple/equal-cost compute baselines and declared latency/memory/deadline limits. No guaranteed speedup follows from the architecture.

## 11. Audio comparison and benchmarking

**Live station controls:** the train service and original `train_horn.mp3`/room chime start automatically on Play. Service cruises at 12 m/s with 3 m/s² acceleration/braking, stopping at the same platform for 10 seconds each lap, including startup. Capture starts automatically before movement and sound, and the Game view exposes one Save run action (R). The world keeps moving during audio drain, and saving resumes live monitoring with restarted clips. Re-enter Play for a new recording. Walking starts on Play. Esc releases the cursor and reveals mouse sensitivity; left- or right-click inside the scene resumes walking, while UI clicks leave the cursor available. E opens/closes the nearby waiting-room door. Entering and leaving the carriage uses ordinary walking across the level platform edge; there are no boarding teleports, prompts, pose cuts or special boarding state. The normal station dwell is unchanged, and player parenting stays stable. The walking controller tracks its supporting collider in local coordinates, applies the floor’s translation/yaw before sweeping walking and gravity, and inherits floor velocity on jumping or stepping off. Benches leave a clear aisle; normal collision remains active aboard. Detailed source controls, replay, rendering and diagnostics remain in the Inspectors/comparison window. The full take format is unchanged; the station adds the initialization input sidecar described in section 4.

The current roster is 53 objects, two sources and one binaural listener; older rosters need their matching scene for pose replay, while offline audio remains self-contained. Capture records actual late-frame poses, not a separately simulated trajectory. Movement and gravity use one CharacterController sweep per update, including sub-millimetre steps. Wheels align to the rails; emitters lie outside the opaque train proxy. The passenger carriage uses separate floor/side/roof geometry. Mouse sensitivity is independent of render-frame duration. Autonomous passengers remain future work.

**Implemented:** Tools → Acoustics → Compare audio opens comparison.json without a running scene. Select a recorded listener by name/index; new reports also retain its stable ID and relative take directory. Low/High waveforms and spectrograms use equal scales and a shared cursor. Low/High/Difference (low-high) audition preserves sample position on switches, with one shared gain initially unity. Preview restarts at the preserved sample and may have a gap; a 5 ms edge fade affects audition only. The original WAVs and numerical metrics are unchanged. Later add model output as another aligned run.

The window displays RMS/peak error, reference RMS/SNR state, input/settings/HRTF identities, reference eligibility, budgets and stage timings. Recorded live counters are explicitly attributed to the input take's capture, separately from offline budgets. Validate selected audio hashes, WAV headers/shapes, completion, quantities and low/high settings before inspection. Only a selected listener/range is loaded, bounded to 30 seconds and 2,000,000 frames. AudioUtil reflection is isolated and signature-checked, with clip cleanup; plots/files remain usable if preview is unavailable.

Display spectrograms use configurable power-of-two FFTs, periodic Hann, one-sided peak-amplitude dBFS, shared color limits and linear frequency. Time display decimation is explicit in the effective hop; pooled frequency rows retain the largest bin, and the final window zero-pads. These visualization arrays are not an ML dataset or streaming synthesis implementation. Complex stereo training remains a distinct stage requiring full phase, transform and timing contracts.

Optional listener and top-down views use an isolated preview scene cloned from the matching open scene's transforms, static mesh/text renderers, lights and solid 3D colliders. Evaluate recorded world poses parent-first at the same audio cursor, holding the last pose during tails. No gameplay, rigidbody simulation or audio runs in this copy. Free inspection uses a small spherical CharacterController against the copied colliders, so the camera stops/slides at ground and walls. Free-camera movement only changes inspection; Follow listener disables that controller and restores the exact recorded head view. Existing takes are never corrected or resimulated to match later scene changes. Top-down source/listener markers share that pose sample. Camera textures update at most 30 Hz and are released with the preview scene on close/reload/Play transitions. Matching IDs do not establish visual asset equivalence: current materials/meshes/lighting may differ from capture. This is interactive recorded-pose rendering, not encoded video or measured sound/display synchronization.

Per-pair responses remain essential: summed audio may conceal one source's error through cancellation or masking. Offline produces all listeners; live chooses a receiver. Two-source/two-listener parity fixtures exist, but arbitrary scaling has not been established.

Earlier train observations were about 10 FPS, maximum pose gap ~116 ms and GPU completion wall time up to ~201 ms. A 24,000-sample lead at 44.1 kHz (~544 ms) avoided underruns in that short run. These are historical smoke-test observations, not target latency or isolated GPU execution measurements. Diagnose observation cadence and scheduling in Editor and standalone before adding load.

| Area | Required measurements |
|---|---|
| Capture/scene | Validation, observations/gaps, extraction, BVH build/refit, changed/uploaded bytes. |
| Solver | Actual rays, pair/band/bounce/arrival counts, overflows/truncation, execution timing if available. |
| Scheduling | Submission/completion/publication/application, queue occupancy, scene/response ages, prediction horizon, stale/discarded updates. |
| DSP/model | Synthesis/convolution, features, transfers, inference, postprocess, memory, lookahead, fallback rates/reasons. |
| Audio/capture | Callback duration/deadline, underrun/late/overflow samples, allocations, writer throughput/incomplete capture. |
| Whole workload | Warm p50/p95/p99 and maxima, cold start separately, repetitions, real-time factor, actual end-to-end latency and peak memory. |

Do not sum overlapping durations to claim latency; correlate timestamps. GPU submission/completion wall time is not GPU execution. Mark unavailable counters unavailable. Run profiling without training competing for the same device; measure minimal graphics and actual station presentation separately. Detailed overlays/path capture must be identified in results.

Current worst-case direct work scales O(R*S*L*G); mixing about O(N*S*L*C*Ktap), where C is one or two output channels and Ktap includes the HRIR in binaural mode. Version-2 response bytes are R*(8+20*S*L), output about 4*N*L*C excluding headers. Binaural filter preparation also searches the measured directions and composes the FIR per response; profile that cost when increasing pair count or update rate. These guide sweeps, not arbitrary capacity guarantees. Tiling/pair batching preserves identity and normalization.

Record hardware, Unity/backend/build, device sample rate, audio buffer, visual workload, seeds and deadlines before a realtime acceptance claim. Use same take/physical model/output range across comparisons. Include static/open/blocked paths, moving doors/trains, fast motion, empty geometry, large counts and cuts. Score absolute level, arrival time, decay, phase where meaningful, clicks, staleness and deadline misses. No automatic time shifting or separate amplitude normalization.

### Planned recordings browser and portable playback

**Scope:** load any complete, supported take without opening its original Unity scene; play or scrub its movement and available audio; render quality variants and generate examples from those same inputs. Preserve existing v1 take payloads and v1/v2 render readers. This whole subsection is proposed work. The Inspector workflow and 30-second comparison ranges described above are the current implementation.

#### Navigation and controls

Use one entry point, **Tools → Acoustics → Recordings**, by extending `TakeComparisonWindow`. Its home view has **Record scene** and **Recordings**. Keep the existing Compare audio menu as a link to this same window during migration. A take opens into playback, with comparison and generation controls revealed when needed:

```text
Home:       [Record scene]  [Recordings]
Recordings: [Open folder…]  name / duration / sources / listeners
            selected take → [Play / Pause]  ───●──── whole-take timeline
                            [Recorded / Rendered / Low / High / Difference]
                            listener view | top-down view
                            [Generate pairs…]
```

List the configured recording root and explicitly opened folders; do not scan the entire project or import recordings into Assets. Discovery reads manifests only. A small rebuildable preferences cache remembers recent folders and selection; files remain authoritative. Incomplete/unsupported takes stay visible with an inline reason instead of disappearing. Show only available audio choices, and show listener selection only for multiple receivers. Put render settings, provenance and diagnostics in folded details. Playback can start from a take folder or `take.json`; an existing `comparison.json` resolves its take and selects the corresponding renders.

**Record scene** prepares/reuses the scene's visual snapshot in Edit mode, then enters the selected existing authoring scene using the normal Unity workflow; capture begins at scene initialization before movement or sounds. Selecting a different authoring scene follows Unity's normal unsaved-scene handling. While walking, keep the single Save run control. Esc releases input and exposes Resume/Recordings alongside sensitivity; it does not silently pause the captured world or stop a take. No additional gameplay commands or interaction prompts are needed.

**Stop & save** finishes output capture and selects the new take in the browser without stealing walking focus. It retains today's automatic return to live monitoring. **Recordings** explicitly switches activity: finish/save a take or stop live audio, let Draining finish, suppress the automatic live restart for this transition, then leave Play mode and open the browser. Do not freeze updates before draining. Preserve the pending take path across the Editor mode transition. A failed finalization keeps the browser open with the failure; it must not silently play a partially saved capture or discard it. Users do not operate the manager's internal states themselves.

This browser proposal is deferred behind working initialization recording and sequential input playback. Its first delivery would be the native Editor window, which can preview saved takes in Edit mode. A standalone Record/Recordings panel is a later thin host over the same take/audio session; it is not required to launch the browser. That host has a neutral startup view and unloads the live scene after draining before replay; the replay scene never starts train/gameplay/audio scripts. Add its UI to the existing control-panel code when building the standalone workflow, rather than a separate menu framework.

#### Inputs, identities and audio choice

Loading is independent of `RayTraceManager`, the active scene, object count and current device sample rate. `take.json` and its existing payload hashes identify the immutable acoustic inputs. The take manifest's SHA-256 identifies an exact take file; existing scene/state/event hashes continue to identify solver inputs. Validate the selected payloads and schemas before playback, with cancellable progress on first verification. Browsing a library does not hash every recording or load PCM/meshes. Each session owns its reader, buffers and preview resources; changing selection cancels and disposes the previous session before publishing the replacement.

Audio selection is explicit:

| Choice | Source and behavior |
|---|---|
| Recorded | Complete `monitor/output.json` and its verified WAV; only for the listener actually monitored. Preserve captured underruns and recorded latency. This is component output, not a microphone recording or downstream device capture. |
| Rendered | A compatible completed offline run, with listener/channel/HRTF/settings identities retained. |
| Low / High / Difference | An explicitly selected completed comparison; both sides must share its declared timeline and experiment. Difference remains Low minus High. |
| No cached audio | Show **Render audio & play** using the selected saved preset; mark the result Rendered. Until then, allow silent pose scrubbing. Never label regenerated audio Recorded. |

Prefer Recorded on initial open when available for the selected listener, otherwise the last explicitly selected compatible render. Select a recorded take's monitored listener initially. Switching to another receiver uses that receiver's offline audio or requests a render; recorded mono/stereo output cannot supply a different receiver. Generated response files retain all source/listener pairs. Listen to a receiver's summed output; source-isolated listening would require another declared render, not separating sources from a mixed WAV.

Incomplete monitored audio does not invalidate an otherwise complete input take. Keep it in diagnostics; normal playback can use a completed offline render. Unsupported/corrupt acoustic inputs fail validation. Old recordings are read in place; opening them never rewrites manifests, IDs or poses.

New monitored-output reports bind the final take-manifest hash when finalization completes. Existing reports lack this explicit association: retain them as legacy recordings with their original adjacent take and validated timing, and expose the missing binding in details. Moving a monitored WAV alone cannot establish which take/listener it belongs to.

The shared session flow is intentionally small; these are proposed operations, not new public APIs yet:

```text
open(path):
    close prior playback session
    take = validate and open saved acoustic inputs
    view = matching saved snapshot, otherwise saved acoustic geometry
    tracks = validate available recorded output and completed renders
    select listener/track; seek to its first valid sample; remain paused
play(): prefill bounded audio; use its consumed position to drive take.Evaluate
seek(sample): cancel old reads; apply poses; refill audio; restore play/pause
generate(recipe): enqueue the immutable take identity; leave input files unchanged
```

#### Visual replay without the original scene

**Basic view is always available for a valid take.** Create one unit-scale replay transform per recorded object; reconstruct local triangles in 3D or segments in XY from `geometry.bin`. Those coordinates already contain authoring scale: apply only recorded position and rotation, never multiply by `ObjectDefinition.scale` again. Use simple acoustic-material colors and source/listener markers. Apply activity/cut flags and the existing interpolation. This restores the saved acoustic representation, not omitted decorative assets. Source/listener-only objects still receive transforms. Replay needs no gameplay physics; free-camera collision is a separate inspection concern.

**Saved visual snapshots are optional, shared assets.** Export a whitelisted visual hierarchy before capture, bound to the persisted object IDs, fixed relative transforms/scales, static meshes/text/materials/textures, solid inspection colliders, lights and listener camera settings. Reuse the existing visual-copy logic. Build the copy from permitted components; never instantiate gameplay scripts and then remove them after lifecycle callbacks could run. Registered objects receive recorded world poses parent-first; unregistered children inherit fixed local transforms. Every independently moving or activating visual root must be registered. The snapshot cannot depend on mutable objects in the open scene.

Use Unity's built-in AssetBundle export for the whitelisted prefab and its dependencies, with one immutable bundle per visual revision and target platform. Keep the serialized object-ID → prefab-transform-path mapping and fixed camera/lighting metadata in `snapshot.json`; paths use child indices, not nonunique names. Capture and restore any global preview settings explicitly. Snapshot identity hashes the finalized manifest, which binds bundle/dependency bytes, exporter version, Unity version, render pipeline and build target. An authoring dependency key can locate an existing export, but is not proof that its bytes remain valid. Do not rebuild unchanged snapshots for every take. [Unity bundle build API](https://docs.unity3d.com/6000.0/Documentation/ScriptReference/BuildPipeline.BuildAssetBundles.html).

Build snapshots in Edit mode before entering the live scene, and reuse the prepared snapshot for subsequent takes. A standalone scene carries a previously prepared compatible snapshot; runtime bundle building is not required. Direct Unity Play without a matching prepared snapshot still permits acoustic capture with the basic view. Changes to supported static visuals or bindings create a new revision on the next authoring export. Validate the supported rigid/static contract and snapshot binding at capture boundaries; do not promise exact playback of animated materials, particles, skinned meshes, changing lights or other uncaptured state. A feature needs a recorded channel before being added to the supported contract. A snapshot failure leaves acoustic capture/replay usable and states that the take has only the basic view. Lighting/shader output can vary across devices, so saved assets and poses do not promise pixel-identical video.

AssetBundles are platform-specific. A Mac export is not a Windows replay asset; include a compatible export or use the basic view. Record and check the tested Unity/pipeline compatibility before loading a snapshot. Audio and acoustic geometry remain independent of the visual bundle. This avoids a custom material/shader serializer or an Addressables dependency. [Unity platform limitations](https://docs.unity3d.com/6000.0/Documentation/Manual/assetbundles-platforms.html).

Proposed additions to the existing recording root:

```text
Recordings/
  station-<id>/
    take.json, geometry.bin, states.bin, events.bin, clip-*.f32
    monitor/                         existing optional captured output
    visuals.json                     optional binding to a saved snapshot
    <render-or-comparison>/           existing outputs remain readable here
  visuals/<snapshot-id>/
    snapshot.json, scene.bundle       shared across takes of this revision
  datasets/<name>/                    section 9 manifests and indexes
```

`visuals.json` declares `schema: acoustic-take-visuals-v1`, `takeManifestHash`, a relative snapshot-manifest path and its SHA-256. The completed sidecar is published after the take manifest is final; it introduces no circular hash or change to v1 take files. `snapshot.json` declares `schema: acoustic-visual-snapshot-v1`, component/export versions, compatibility, bundle path/hash, mappings and fixed presentation data. Old takes without the sidecar use the basic view. Missing/incompatible visuals produce a short explanation beside that view; don't bind by name or silently substitute today's scene. Preserve the existing current-scene viewer only as an explicitly labeled diagnostic option.

#### One clock, continuous playback and seeking

The internal cursor is signed int64 samples at the take's sample rate. Seconds are presentation only. Rendering, seeking and pose evaluation share this cursor. During audio playback, advance from the audio player's consumed playback position, not frame count, decoder read-ahead or accumulated `deltaTime`. Paused scrubbing sets the cursor directly. Visual updates sample that clock at display rate; a slow visual frame must not stretch audio.

Each audio selection has an explicit mapping between file frame and take sample:

- **Offline runs:** file frame `n` corresponds to output scene sample `n`. Keep propagation delay and declared FIR/HRTF latency in the output; do not shift by waveform correlation. Low/high metrics continue to compare original samples unchanged.
- **Recorded output:** reconstruct the output clock from `callbacks.bin` (`firstSample`, block length and `dspTimeSeconds`) and `inputDspOriginSeconds`. For frame `c` in a callback starting at `b`, `sceneSample = round((callbackDspTime - inputDspOriginSeconds) * Fs) + (c - b)`. Validate callback continuity and clock mapping. Preserve the captured queue lead in the audio so the default recorded view shows what the component emitted while those poses occurred. `leadSamples` also maps scheduled solver output to its capture frame for diagnostics; subtracting it is a separately labeled alignment operation, not the default replay clock.

If an old capture lacks valid clock metadata, offer explicitly labeled unaligned audio and pose inspection independently, or use a completed offline render for synchronized replay. Do not assert measured synchronization from a guessed offset. Audio/display/device latency remains subject to measurement even with correct software timestamps. Before/after the pose interval, hold the first/last pose through valid audio lead/tails. Declare track coverage; unavailable intervals are not measured silence, and Difference is only available on a valid common timeline.

A seek or track/listener switch stops the current output, increments a session generation, cancels obsolete reads, fills the new position and resumes if previously playing. Keep absolute sample position; reject stale buffer completions. Small audition fades may suppress start/stop clicks without altering files or metrics. Apply fades at user discontinuities, not every streaming block. Cached PCM already contains propagation/DSP history, so seeking it does not rerun source events or physics.

Keep one continuous take timeline while loading bounded audio blocks and plot tiles automatically. The current `Preview` creates an entire in-memory clip for a manually chosen range through internal `AudioUtil`; it cannot simply be relabeled as streaming. Verify a streamed `AudioClip` and its seek/cursor behavior in Unity 6000.4.1f1 Edit mode first. Unity's public clip API supports small on-demand PCM blocks, but that does not establish compatibility with the internal Editor audition API. If that route fails, use an isolated preview audio host; gate delivery on continuous playback rather than concatenating clips with audible gaps. Keep this adapter behind the existing preview boundary. [Unity streamed clip API](https://docs.unity3d.com/6000.0/Documentation/ScriptReference/AudioClip.Create.html).

File reads, validation and decoding occur outside the audio callback. The callback consumes prepared bounded data without locks, file I/O, logging or allocation. Preview starvation is reported separately from silence already present in a captured WAV; request pause/rebuffer without blocking the callback, and preserve the first unconsumed content sample instead of advancing past unheard data. Any device silence inserted by the preview is a playback fault, not recorded content. Clip/device sample-rate conversion must retain the take clock and original files; don't require an old take's sample rate to equal today's device rate. No throughput or synchronization claim precedes the playback spike and native measurements.

`AcousticTake` already seeks fixed-stride frames and keeps a timestamp index. Reuse that reader: cache adjacent observations during sequential playback and binary-search on seek. Keep the existing O(observation-count) int64 timestamp index initially; pose/PCM payloads must not grow in memory with duration. Bound waveform/spectrogram caches by the visible range and prefetch window, with documented byte caps. Compute display spectra lazily; they remain distinct from ML tensors. Library scanning, hashing and expensive plotting must not run inside the playback callback or every repaint.

#### Storage, batch generation and automation

Retain the current binary float32/int64 files and seekable working directories. The first storage savings come from sharing snapshots and referencing existing takes/renders/windows. Use an optional lossless ZIP export to move/archive a selection, including its referenced snapshots and required audio once. Import extracts to a validated working directory before opening, with relative paths resolved inside the selected archive root. Validate paths, declared sizes and hashes; compression must preserve source bytes and cannot round poses, decimate observations or transcode training PCM. Keep archive/import outside the playback path. A custom delta codec, per-window files, database and content store are unnecessary for the first delivery.

**Generate pairs** accepts one or more selected takes and one resolved generation recipe. Start with the current direct-model coarse/fine cadence experiment and explicit reference tolerances. It calls the same render/comparison methods as `TakeBatch`, writes run reports, applies section 9 per-pair and causal-availability checks, and writes index rows only for accepted examples. Successful WAV rendering alone does not establish training eligibility. Reflection/ray-budget pairs remain unavailable until their solver and reference gates exist.

The recipe binds experiment/schema version, low/high/reference settings, seeds when applicable, renderer/shader/DSP/HRTF identities, output policy and quality/availability rules. A job key hashes the exact input identities and resolved settings. Reuse an existing result only when identity, payload hashes and completion all match; changed code/settings produce a new job. Preserve separate low variants while sharing a verified high result when its complete identity matches. Cancel between bounded batches, retain incomplete outputs with a reason, and resume at completed-job boundaries. Publish dataset manifests last. Split related takes/layouts/trajectories/ears/variants together before making windows, as already required in section 9.

Keep the queue as a saved JSON list plus completion reports, with one job active by default. The UI and batch entry point call the same C# job method; batch execution never constructs cameras, loads visual bundles or auditions audio. Extend the existing `TakeBatch.Job` compatibly to accept the same resolved recipe for a list of takes; keep today's single-job JSON valid. Run the queue in the idle Editor first. Do not launch a second Unity Editor against the project already open by the user; unattended workers need an isolated project copy or a later built runner. GPU jobs require a graphics-capable host; CPU-only jobs can retain the current `-nographics` path.

Manual and automatic collection share `BeginTakeRecording`, observed poses and sample-timed source events. A small scenario driver supplies a resolved movement/event schedule to existing scene controls; it records the actual outcome once. Save recipe, seeds, layout/trajectory family and capture schedule beside the take. Render all budgets from that take rather than rerunning the controller for each budget. Scripted collection is a separate queue step from rendering existing takes; no procedural-world framework is needed. Training loaders read windows on demand using the section 9 descriptors; they do not use preview plots or captured underruns as high-quality targets.

#### Code ownership and delivery gates

| Existing file / proposed addition | Responsibility |
|---|---|
| `Assets/Editor/TakeComparisonWindow.cs` | Take discovery/home view, open take/comparison, continuous timeline, existing A/B/plots and generation controls. Keep one window. |
| `Assets/Editor/TakeComparisonView.cs` | Replace required open-scene clone with basic-geometry or snapshot construction; retain cameras, pose application and inspection controls. |
| `Assets/Editor/TakeComparisonAudio.cs` | Accept recorded-output and render descriptors, validate clock/coverage, replace range-clip audition with bounded playback; keep preview-only analysis here. |
| `Assets/Scripts/Simulation/Take.cs` | Reuse versioned reader/interpolation and fixed-stride data; add adjacent-frame caching only with measured benefit. No payload migration for the browser. |
| `Assets/Scripts/Simulation/TakePlayback.cs` (new when implementing playback) | Small session owning selected take, track descriptors, sample cursor, seek generation and cleanup; no Editor UI or scene-manager dependency. Move reusable audio reading here or into an audio file only when a second host needs it. |
| `Assets/Editor/TakeVisualSnapshot.cs` (new at the snapshot milestone) | Whitelist export, dependency/compatibility identity, bundle build and optional take sidecar. Temporary build assets are cleaned after export; no copied scene per take. |
| `Assets/Editor/TakeEditor.cs` | Link Inspector to the shared browser; factor a callable batch job method from the existing CLI entry point. Preserve single-job compatibility. |
| `Assets/Scripts/Demo/StationDemo.cs`, `Assets/Scripts/Demo/RaytraceControlPanel.cs`, `RayTraceManager.Take.cs` | Recordings transition, completion notification and live-restart policy; preserve one capture/drain owner. Do not duplicate these states in UI code. |
| `TakeRender.cs`, `TakeComparison.cs`, later `tools/take_dataset.py` | Existing offline rendering, reusable verified runs and per-pair acceptance; section 9 dataset indexing/loading. The browser does not implement another solver or trainer. |

Delivery order and tests:

1. **Playback spike and raw-take browser:** prove native Edit-mode streaming/seek/cursor behavior, then implement discovery, isolated basic views and recorded/cached audio on a continuous timeline. Open old/current 2D/3D takes with no original scene, differing rosters and several sources/listeners. Corrupt/incomplete inputs produce specific failures; missing monitored audio still permits input replay/rendering. Demonstrate playback across several former 30-second boundaries, repeated seeks, channel changes and a different device sample rate, with bounded buffers and no steady-state callback allocations or block-edge gaps.
2. **Saved visuals:** capture a snapshot, record two takes sharing it, edit/delete the original scene assets, then reopen the recordings. Verify object mapping, scaled geometry, parent/child motion and activation. Move/export/import a take with its snapshot; missing bundles or incompatible platforms still allow basic views and audio. Unsupported animated presentation is identified. Verify no gameplay callbacks, duplicate audio listeners or live-scene mutations, and cleanup on close/reload/cancel.
3. **Generation from the browser and batch:** same inputs/recipe reach the same job identities and tolerance-checked outputs through both entry points. Test cancellation/retry, reuse of a common high run, rejected teachers, source/listener ordering, causal windows and split isolation. Add scripted capture through the same recorder after the saved-take path passes.
4. **Standalone host:** move genuinely shared loader/audio/view code out of Editor and add the thin home panel. Test record → save/drain → replay → return to scene, scene teardown and no accidental live restart. The Editor's internal `AudioUtil` cannot be a standalone dependency.

For timing acceptance, use a fixture with independently known pose cuts and audio impulses plus a nonzero queue lead. Check exact integer mapping at stored samples and seek endpoints, block-continuous PCM, recorded-versus-rendered timing semantics and bounded pose interpolation error. Measure cold open/hash time, warm Play/seek time, memory versus duration and audio/display delay on the actual Editor/backend; set shipping thresholds from this spike instead of inventing performance guarantees. Until these gates pass, describe this as planned browser/playback work rather than a working main menu or training pipeline.

## 12. Multiplayer

The first network scene uses the same Station prefab, network player/train variants and existing solver. Players host/join, move independently, see one another and hear shared clips from their own receiver positions. Test two separate processes then two machines; a configurable player limit is measured, not assumed.

Use Netcode for GameObjects and Unity Transport after verifying/pinning Unity-compatible releases. Local/direct-IP comes first; internet join codes add Multiplayer Services/Relay configuration and a real cross-network test. [Unity Relay integration](https://docs.unity.com/en-us/mps-sdk/tutorials/relay-and-ngo).

The host owns spawning, shared train/door movement and accepted sound ordering. Each owner drives its player through package replication; remote copies disable local input/camera/native AudioListener. Exactly one native listener per playing client, none on a non-playing server. Competitive prediction/reconciliation, host migration and matchmaking services are deferred.

Each client renders its own acoustic receiver from the replicated world. Replicate object state and sound commands, not response tensors, FFT buffers or final acoustic audio. Quality settings remain local. Add explicit live receiver selection at the manager boundary; retain all captured poses for offline work.

Network contracts:

- Map session-stable acoustic IDs to network IDs and verify scene/clip asset identity.
- Commands carry event ID, source/clip, action, gain, loop, clip offset and server emission time. Apply accepted commands once, including on the host.
- Replicate current voice state/start time for late join; event broadcast alone cannot restore phase.
- Map server time to each client's DSP clock/sample rate with declared delay/rounding and recorded clock corrections. Device sample indices are not network timestamps.
- Align remote transform presentation and sound to the same scene time. Late commands advance phase or skip ended clips; never rewind consumed audio or restart from sample zero.

Host takes record server-accepted state/events; client takes record the actual interpolated state/events its local acoustics saw. Low/high for a client use that same client take. Do not label host-versus-delayed-client differences as reconstruction error. Related session segments and views share a dataset split.

Roster changes require closing the current fixed-roster segment before rebinding/spawn destruction, then starting a segment with current voice phase. A small session index records offsets, gaps and warmup. Disconnects finalize what is valid; host departure visibly ends the initial session. Do not claim seamless continuous capture across roster changes yet.

Microphone voice needs a separate streamed-PCM/jitter/capture contract; fixed-clip gameplay is the initial scope. Proposed integration homes are MultiplayerSession.cs, NetworkPlayer.cs and NetworkAcousticSource.cs, with ordinary package transform replication for props rather than a networking class per object.

Acceptance covers independent ownership, differing audible occlusion per listener, no duplicate host playback, correct late-join loops, leave/disconnect cleanup, latency/jitter/loss, differing device sample rates and replayable segments. Measure network delay separately from acoustic reconstruction latency.

## 13. Implementation sequence and acceptance

Current status is listed in section 1. The [system audit repair sequence](#system-audit-2026-10-05) takes precedence: fix the renderer and recording failures before the broader milestones below. These milestones describe remaining scope, not permission to defer correctness while adding features. Proceed in small stages without empty scaffolding:

| Stage | Concrete work | Completion gate |
|---|---|---|
| 1. Recordings and authoring | Section 11 raw-take browser, bounded audio playback, basic/saved visuals and capture handoff; retain station/lab/XY authoring fixtures. | Old/current takes open without the original scene; continuous shared timeline, explicit audio clock, snapshot fallback, bounded cleanup and no gameplay execution. |
| 2. Timing/direct data | Profile active workload, per-pair reference gates, dataset loader/index and availability/query logging. | Causal windows, corrupt/mismatched rejection, exact pair mapping, measured ages/deadlines, no split leakage. |
| 3. Numerical foundation | BVH/oracles, energy estimator derivation, convolution and configurable transforms; tiny model import test before ML investment. | Absolute gain/normalization, brute-force agreement, transform/phase/tail checks and claimed-backend ABI parity. |
| 4. Indirect simulation | Bounded 2D/3D energy tracing, synthesis and low/high reference runs. | Independent seed/budget/receiver convergence, overflow-safe batching, audible transitions and uncertainty. |
| 5. Reconstruction | Separate denoising and temporal experiments, then combined; small training/export adapter. | Held-out quality/cost benefit, limited horizon, real publication schedules, valid fallback and total runtime budget. |
| 6. Phase/spectral extension | Validated pressure teacher, IR/complex models; audio-STFT/cross-grid experiments where justified. | Phase/causality/support/synthesis validity, streaming latency and held-out source/scene results. |
| Multiplayer extension | Shared scene, package integration and clock/roster adapter after capture timing is stable. | Real clients/machines and segment/clock/late-event tests; not blocked on training. |

Preserve the independent correctness oracles. Checks that matter:

- Existing/new disk offsets, GPU strides/echo, non-divisible dispatches, empty scenes and arbitrary supported band counts.
- Native shape offsets/scales, moving/rotating parents, velocity/acceleration validity, fixed membership and source/listener identities.
- Exact recorded observations/events, immutable same-sample knots, cuts, pre-roll/tails, source activation and two-source/two-listener replay.
- BVH/refit versus brute force, independent analytic transport, batch-size invariance, absolute pressure/energy and reference convergence.
- IR/complex/STFT round trips and bounds, block-size invariance, long moving-response playback, phase/arrival continuity.
- Deadline-window fixture, late/missing inputs, resets, true zeros versus padding, low-variant target deduplication, grouped split isolation.
- Queue exhaustion, arrival overflow, cancellation, teardown, out-of-order completion and stale predicted versus actual response publication.
- Python/Unity feature packing, preprocessing and graph parity at supported batch sizes; finite predictions and all fallback paths.

The next implementation locations are the existing Take/TakeRender/TakeComparison/manager boundaries, extensions to the existing Editor comparison window, the concrete BVH/indirect/DSP additions, and two Python entry points. Add render_jobs.py only if actual sweep volume needs it. Shared source/timeline/geometry code is retained.

Before collecting expensive labels, finish the estimator/phase derivation appropriate to that target and set tolerances from fixtures. Before model acceptance, fix hardware/workload, latency/memory, prediction-age and quality thresholds from baseline measurements. Bigger networks, longer history, more bands, compiler changes or additional services are not substitutes for those checks.
