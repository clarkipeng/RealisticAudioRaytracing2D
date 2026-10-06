# Acoustic simulation and reconstruction: design contract

Updated 2026-10-06 against the current Unity 6 working tree. This is the single design specification for runtime cleanup, scenes, microphone/clip sources, simulation, capture, comparison, multiplayer, datasets and learned reconstruction. It replaces the separate architecture, comparison, remaining-work, scene-assets and training plans. Examples describe proposed contracts unless explicitly marked implemented.

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

**Implemented next-scene foundation:** shared session, generic compact replay and Church blockout. **Remaining:** legacy/material migration → Church acoustic validation → local microphone → LakeForest → multiplayer on the same scenes. [Section 13](#13-implementation-sequence-and-acceptance) defines delivery gates. The remaining additions are specified, not implemented. The existing simulation/training contracts remain applicable; scene demos do not establish teacher accuracy.

Review the [runtime changes](#runtime-cleanup-and-shared-scene-session), [scene designs](#scene-composition), [recording formats](#scene-independent-input-implemented-and-stream-recording-proposed), [microphone pipeline](#microphone-and-streamed-source-audio-proposed), and [multiplayer contracts](#12-multiplayer).

## 1. Status and decisions

| Capability | Current implementation | Planned work |
|---|---|---|
| Unity authoring and capture | Native 2D/3D geometry export, rigid motion, activation, stable IDs, source events, immutable take files; editable station, train service with stops and walking aboard, and door with two sources and one binaural listener; Church blockout with the same session/player/door and compact recorder. | Church acoustic validation and LakeForest; migrate the old rooms as small 2D/3D fixtures. |
| Replay and computation | Shared compact scene-input reconstruction, pose replay/scrubbing, CPU/HLSL direct solvers with optional static BVH plus direct tests for moving objects, shared live/offline timeline and mixer, WAV/response export, batch jobs and measured station offline benchmarks. | Streamed source files; small recordings picker. Portable decorative snapshots and continuous audition remain deferred. |
| Directional spectral preview | Shared 2D/3D GPU transport with multi-bounce reflection/scattering, transmission/refraction, air/material losses and edge diffraction heuristics. Configurable frequency grid, selectable scene-IR or continuous path-clock spectral synthesis, interpolated per-arrival KEMAR orientation, samplewise response crossfades and first-order image-source reflections; same live/offline implementation. Closed 2D/3D objects use per-medium speed/damping, including nested solids and interior sources. | Physical normalization/convergence, higher-order mirror connections, directional dataset export and validation of complex moving boundaries. The preview is not an approved training reference. |
| Multiple endpoints | Every source/listener pair; mono or binaural mix per head; one monitored listener. | Larger workload benchmarks and explicit local receiver selection for multiplayer. |
| Comparison | Coarse/fine schedules, metrics/reference gate, native audio A/B/Difference, waveform/spectrogram plots and synchronized listener/top-down/free-camera views. | Per-pair target checks, learned outputs and validated reflection comparisons. |
| Runtime | Bounded PCM/capture queues, GPU transport and DSP with asynchronous PCM batches, selectable CPU reference, fractional-delay FIR, provenance and diagnostics. | Lower measured latency/memory and reconstruction scheduling. |
| Microphone and networking | Sources currently require prepared clips; no mic capture, voice codec or network gameplay. | Timestamped stream input, saved source audio, actual host/join and local acoustic rendering of remote voices. |
| Learning | Not implemented. | Validated teachers, dataset loader, models and inference. |

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


### GPU audio execution, 2026-10-06

The CPU DSP remains a correctness reference in `SpectralAudio.cs` and `ImpulseResponseAudio.cs`. `GpuAudio.cs` owns GPU buffers/readback and `Resources/Acoustics/GpuAudio.compute` owns GPU DSP. Shared code is limited to useful contracts and scheduling: scene/state layout, settings, source events, integer sample clocks, live/offline orchestration and the existing PCM queue. No new package or public scene representation is required.

```text
CPU: evaluate scene / source events; upload poses + new dry source samples
GPU: trace → group compatible arrivals → material + head filters
     → source-time resampling → parallel path FFTs → per-ear sum/IFFT
     → valid overlap-save samples → response fade → finished PCM
CPU: async PCM completion → live queue, or wait → offline WAV
```

Transport paths remain in their existing 32-byte buffer. Direct-only probes remain available for diagnostics; the CPU reference explicitly requests full path readback. GPU groups match delay, arrival direction and delay rate, plus topology identity in Spectral mode. Material bins become a short FIR, composed with the complete interpolated KEMAR ear response. Spectral uses continuous emission-time clocks; IR fixes the rate to zero and evaluates the equivalent sum of delayed path filters. The CPU IR reference instead uses a partitioned scene FIR. Both GPU modes use overlap-save, apply travel delay once and retain old/new response banks for samplewise fades.

Dry source samples are uploaded incrementally to a bounded GPU circular history. Response poses occupy separate batch slots until consumed; GPU metadata resets are dispatched in order, not overwritten by CPU writes while earlier responses are queued. Live submits all known intervals in a batch before reading PCM, and can submit DSP immediately after tracing without waiting for diagnostic probes. No audio-thread GPU work or waits are introduced. Offline uses the same DSP and explicitly waits for each output delivery. Larger FFTs use staged global-memory transforms when they exceed the 2,048-element shared-memory fast path; this is an implementation threshold, not a user FFT cap. GPU resource limits still apply.

Reports include the selected DSP contract, GPU-audio shader hash, streaming upload/readback bytes and numeric working storage. CPU dispatch/preparation time is not GPU execution time. Initialization uploads and transport transfers are excluded from the audio byte counters. The current filter allocation reserves worst-case path capacity; grouping uses one work item per source/listener pair, and per-bin reduction loops over groups. These are remaining optimization targets, not a claim of maximal acceleration. Windows/DirectX needs its own validation. Training export/inference, general moving-boundary physics and a low-latency guarantee remain outside this port. [Actual measurements and tests](cleanup-results.md#gpu-audio-port-2026-10-06).

### Continuous Doppler and synthesis methods, 2026-10-06

The previous renderer is preserved at `6cc47f1`. The new `Synthesis` setting separates scene-IR convolution from path-wise spectral rendering. No new dependency, scene representation or recorder is needed. Existing serialized settings select `ImpulseResponse` (0); `Spectral` (1) is explicitly opt-in; both modes now support GPU DSP and a CPU reference. Direct transport uses its own continuous-delay mixer and ignores this setting.

For a stationary homogeneous medium, let `n` point from the source at emission to the listener at reception. With velocities relative to that medium:

```text
q = receivedFrequency / emittedFrequency
  = (c - dot(listenerVelocity, n)) / (c - dot(sourceVelocity, n))
emissionTime = receiverTime - travelTime(receiverTime)
delayRate = 1 - q
```

The direct solver solves `c*delay = |listenerNow - sourceNow + sourceVelocity*delay|`, then evaluates the direction and ratio. Thus source and listener motion have distinct factors. At a moving surface, the outgoing/incoming frequency ratio is `(1-dot(surfaceVelocity,incoming)/incomingSpeed) / (1-dot(surfaceVelocity,outgoing)/outgoingSpeed)`. Surface point velocity is linear velocity plus `angularVelocity × (hit-center)`. Segment speed is `airSpeed/IOR`. These local factors assume subsonic, nondispersive propagation; moving-fluid advection and Mach shocks are not modeled. See [time-varying delay Doppler](https://www.dsprelated.com/freebooks/pasp/Doppler_Simulation.html) and [physical delay-line modeling](https://www.dafx.de/paper-archive/2002/DAFX02_Smith_Serafin_Abel_Berners_doppler_leslie.pdf).

The GPU receives 80-byte states: position, quaternion, linear velocity, angular velocity and flags, each in a 16-byte lane. `VelocityValid` is required; cuts invalidate velocity. CPU/GPU direct solves match. The 32-byte path keeps direction XYZ, delay-rate W, delay seconds, pressure gain, frequency-bin index and a topology/sample hash. It does not estimate velocity by differentiating unrelated slots.

`SpectralAudio` continuously reads source history at the path's emission time. Each solve supplies a fresh retarded-time anchor; its local delay slope drives source reads until the next update. This avoids accumulating an approximate slope into nonphysical drift. The 32-bit topology/sample hash separates path groups; it is not a collision-free persistent geometric identity. Changing trajectories/topology can still introduce phase discontinuities between solves; crossfades smooth those response changes. This is not arbitrary-trajectory retarded reconstruction. No additional pitch shifter is applied. Linear interpolation is inexpensive but aliases pitch-up; windowed sinc widens its source support and low-passes when `q>1`.

The native `N/2+1` material bins define an `N`-tap centered pressure basis. Each path applies that short response and its complete interpolated HRIR to the time-warped signal with zero-padded FFTs and overlap-save. The transform length covers `N + HRIR length + block - 2`; the valid last block contains no circular wrap. Propagation stays in the source clock, outside the short filter. This still performs short-filter convolution mathematically, but constructs no long scene IR or propagation-delay partitions. Declared DSP latency remains `N` plus sinc radius. `ImpulseResponseAudio` retains the previous partitioned scene-FIR algorithm; it deliberately ignores path delay rate and crossfades frozen responses. It is not a continuous moving-Doppler implementation.

Direct transmission brackets the emission-time root using per-medium segment travel times (fixed-point proposals alternate with bisection to resolve slow convergence) while other boundaries remain at their solve pose. First planar mirrors iterate the source emission and rigid mirror pose at interaction time. The validated moving-wall case is normal translation. Frequency-dependent material coefficients remain evaluated on the configured bin grid; frequency changes between successive interactions and the distinction between phase/group velocity in dispersive materials are not tracked. Oblique/rotating moving surfaces, acceleration and changing occlusion require further validation. Sampled diffuse/diffraction paths use local velocity factors but frozen geometry/delays; their pressure estimator and full moving-path timing remain preview approximations. The tail freezes final pose and velocity, instead of extrapolating motion beyond the recording. Reported support is not a proof of arbitrary moving-wave tail completeness.

Output/capture report version 3 adds delay rate to direct diagnostic rows (six float32 values per pair, after the frame sample). Comparison opens old v1/v2 reports. Take binaries and compact station input logs are unchanged. These diagnostic rows still cannot reconstruct indirect WAV content.

Validation includes analytic approaching/receding source and listener cases in 2D/3D, equal source/listener velocity, and approaching/receding planar mirrors. Rendered tone frequency/coherence checks disable crossfades; a separate sinc fixture rejects a tone shifted beyond Nyquist. Both synthesis methods pass the independent static convolution/long-HRIR oracles and streaming/batch tests. This establishes those numerical cases, not a validated room teacher or realtime performance. [Measurements](cleanup-results.md) record the station cost.

### Renderer repairs, 2026-10-05

The scene ABI, compact replay format, shared live/offline renderer and PCM handoff queue remain intact. No new package or runtime subsystem was added. The pre-repair implementation remains in Git at `1586cbb`; audit checkpoint `332f8f3` preserves its independently measured failures.

This historical repair at `6cc47f1` established the algorithm now retained by `ImpulseResponseAudio`. The optional spectral route is described above. The repaired IR DSP implements uniform partitioned **linear convolution with overlap-add**:

1. An acoustic grid of `N/2+1` real-signal bins defines a centered finite `N`-sample pressure FIR. Equal delay/direction arrivals combine on this grid; individual bins have the same finite-basis meaning.
2. Each directional FIR is linearly convolved with the complete interpolated HRIR. Propagation delay is deposited once using the selected linear or normalized windowed-sinc fractional-delay FIR. It is not a phase rotation inside a synthesis window.
3. The resulting FIR is partitioned into `B`-sample blocks; each uses a zero-padded `2B`-point FFT. A sample-keyed circular history retains source blocks. IFFT output halves overlap-add without circular aliasing.
4. Response transitions crossfade old/new **output samples**, preserving the same input history. Listener disable fades output to silence once; Source Stop gates future emission while allowing earlier sound to finish. Cuts switch responses immediately. Unchanged responses reuse their filter bank.

`N` is acoustic frequency resolution, `B` is convolution work granularity, and response step is scene-solve cadence. They are separate controls. Total declared DSP latency is `N` samples, plus sinc radius when enabled; propagation and measured HRIR timing remain separate. HRIR length is not constrained to fit `N`. Maximum acoustic delay bounds storage. `responseSupportSamples` is the conservative maximum observed FIR/block support; `tailCoversResponseSupport=false` explicitly identifies a requested tail that may truncate it. These fields do not automatically extend the requested render.

The mathematical guarantee is convolution of this declared finite response, not a wave-accurate material/room model. A finite band basis can spread a transient around its delayed center. No slot-derived velocity or pitch shift remains. Continuously crossfaded moving responses pass approaching/receding tone fixtures; coarse held responses and arbitrary changing path topology still need a better physical motion model.

Transport now uses an integer `raysPerBin`, with total rays `(N/2+1)*raysPerBin`, weight `1/raysPerBin`, and an explicit seed. Every bin is covered. Frequency bins share random samples to retain broadband coherence, while each still evaluates its own material interactions. Old serialized total-ray settings round up to complete per-bin budgets. Each ray also samples a first-order image-source mirror candidate, with primitive-sampling normalization and shared-edge deduplication. Specular pressure uses the square root of the unscattered reflected energy; the existing diffuse connection now uses scattering. A perfect mirror no longer creates the old arbitrary diffuse first arrival.

**Transport limitations remain:** first-order specular fixtures establish geometric gain/delay for a single planar mirror, not convergence of a complex scene. Low primitive budgets can miss mirrors or yield high weighted variance. Higher-order perfect-mirror connections are incomplete; diffuse pressure weights/branch PDFs and edge diffraction remain preview heuristics. The renderer is still excluded from the training-reference gate. Increasing rays does not change this status.

Compact saves now complete only after the required station input recorder finalizes. Construction, tick, observation or finalization failure invalidates `take.json` and reports failure. Offline station reconstruction preserves authored initial source/surface enabled flags and actual primitive membership. Unsupported runtime activation changes fail recording explicitly; adding a general activation event system is outside the current input contract.

Independent checks compare integer/fractional impulse delays and full ear-filter output against time-domain convolution over multiple grids, blocks and callback chunks; they also cover receiver fades, changed indirect slots, mirror gain/time, legacy settings migration, invalid saves and initial activation. Existing live/offline, medium and batch-invariance checks remain in place. [Measurements and validation](benchmarks/renderer-repair-2026-10-05.json) record the actual station workload. Numerical convolution correctness does not certify realtime scheduling. At that milestone, GPU transport read paths back for CPU synthesis. The GPU port above removes routine path readback; the station still has a 16,384-frame playback lead, and head filtering remains tied to response updates.

The padding/overlap distinction follows [linear FFT convolution](https://www.dsprelated.com/freebooks/sasp/Convolving_Long_Signals.html) and [zero padding](https://www.dsprelated.com/freebooks/sasp/Zero_Padding_Time_Domain.html). The separate mirror connection reflects the [delta-distribution nature of ideal specular reflection](https://www.pbr-book.org/4ed/Reflection_Models/Specular_Reflection_and_Transmission); it does not validate the diffuse estimator.

### System audit, 2026-10-05

The pre-repair audit of `1586cbb` is historical. Its impulse/convolution failures and CPU synthesis bottleneck were addressed by the repairs and GPU port above. Keep the raw [audit](benchmarks/system-audit-2026-10-05.json), [repair results](benchmarks/renderer-repair-2026-10-05.json) and [measured history](cleanup-results.md); do not treat old failing counts as current failures.

Current limits relevant to new scenes:

- Higher-order specular connections, diffuse-pressure normalization and diffraction are not validated room references. More rays alone do not approve training labels.
- The station uses five bounces, a one-second output tail and 16,384 samples of playback lead. These are scene settings, not universal fidelity/latency requirements. Tail coverage is reported, not extended automatically.
- Head filtering is tied to response updates and already queued PCM. Measure head-to-audio latency as well as throughput.
- Only absorption/scattering have arbitrary authored knots; other material coefficients still use eight-band conversion. Complete that migration rather than expanding legacy packing.
- Static BVH traversal works; moving objects leave it for direct tests. Nearby-edge and containment searches still have linear scans. Profile larger scenes before choosing BLAS/TLAS or another hierarchy.
- Compact replay is station-specific and depends on matching project/build identity. Hashes cannot recover missing assets or uncommitted code.
- Low/High currently changes response cadence only. Per-pair directional export, independent ray-budget comparisons and learned reconstruction remain planned.

The selectable test suite now covers 107 default cases plus an explicit device-deadline case; measured results and limits are in [cleanup results](cleanup-results.md#test-suite-cleanup--2026-10-06). These tests are numerical/software evidence, not physical calibration.

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
                   SceneReplay, DirectBenchmark
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

Use the existing runtime/editor/test assemblies. No additional package hierarchy, generic plugin framework, scene database or orchestration service. Preserve Unity `.meta` GUIDs. Retire legacy code/assets only after checking references and replacement coverage.

Keep `DirectBvh`, `DirectSolver`, `RayTransport`, the two CPU synthesis implementations and `GpuAudio`. Extend those boundaries when required; do not add duplicate Bvh/IndirectSolver/AcousticDsp layers or empty files. Rename misleading entry points such as `RenderDirect` when their callers are migrated together.

Dependencies flow Unity authoring → numeric scene/state → solver → response → audio rendering. Capture and normal simulation work without ML. Offline uses the same C#/HLSL solver and DSP. A small Python job loop is sufficient if existing Editor/batch jobs become tedious. NumPy/PyTorch serve training; SciPy is optional for independent DSP checks. Test ONNX → Unity `com.unity.ai.inference` before substantial training. Networking adds Netcode for GameObjects/Unity Transport only at its milestone. Pin compatible versions then.

Current binaural rendering adds no runtime package: measured KEMAR coefficients are bundled as a Unity resource. The standard-library Python importer verifies the source archive hash and reproduces the asset. Retain [Gardner/Martin attribution and conversion details](../Assets/Resources/Acoustics/ATTRIBUTION.md) when distributing data or derived outputs.

Start hardware measurements on the available Metal machine; verify the same layouts and fixtures on DirectX before claiming support. HLSL sharing is useful but does not prove equal performance. A different compiler/backend needs a measured reason, not a parallel architecture. [Unity compute documentation](https://docs.unity3d.com/6000.0/Documentation/Manual/class-ComputeShader.html).

### Runtime cleanup and shared scene session

This is a focused extraction while building the second scene, not a renderer rewrite.

| Existing code | Change | Deletion/reuse condition |
|---|---|---|
| `RayTraceManager.Legacy.cs`, `AudioManager.Legacy.cs` | Migrate `Big Room` and `SmollRoom` to the current 3D/2D renderer; retain their useful fixtures. | Then remove both legacy partials (~1,100 lines), old shader-only helpers, unused debug buffers/UI and legacy branches. Verify all scenes/prefabs first; Git retains old code. |
| `SceneHelper.cs`, `AudioMaterial.cs` | Keep numeric scene export; remove old 2D projection/packed material adapters after scene/material migration. | Preserve GUIDs and compare exported geometry/coefficients; never delete shared code by filename alone. |
| `StationDemo.cs` | Input, session startup/save/replay and cursor handling now live in `SceneSession`; train service/motion remains in the existing station component. | No `ChurchDemo` or `ForestDemo` copies of the recorder. `SlidingDoor` is shared by station and church. |
| `SceneRecording.cs`, `SceneReplay.cs` (renamed) | Evolve to scene input recording and reconstruction, retaining the station-v2 reader. | Same initialization/step code live and offline; remove train-specific settings from the shared contract. |
| `RaytracedAudioSource.cs`, `TakeRender.Signal`, `TakeRender.Live` | Add a stream source alongside clips, with bounded source history and explicit availability. | CPU/GPU and mic/network use this same sample-read boundary. |
| `RaytraceControlPanel`, `TakeEditor`, comparison window | Replace station-specific session calls with shared ones; preserve existing views/diagnostics. | One small control surface; no per-environment menus. |

`SceneSession.cs` and `SlidingDoor.cs` implement the shared scene clock and door; add `MicrophoneInput.cs` only at the microphone milestone. Shared source-buffer types can live with `Signal`; recording/replay are moves of existing files with their `.meta` files. Network adapters are added only at the multiplayer milestone. Keep the independent CPU oracle and genuine IR/Spectral algorithms.

The shared session owns the player/head, manager, fixed-step index, input mode and explicit ordered list of supported scene actors. Actor initialization and stepping form one small contract in the session file; no reflection registry or generic service framework. Use persisted actor IDs for order and interaction targets. Authoring changes require a new capture; arbitrary scripts/rigid bodies are not silently replayable.

```text
initialize authored scene:
    validate fixed bindings and actor settings
    initialize actors, player and head once; synchronize colliders
    open recording at tick/sample zero before accepting movement or sounds
step(tick, acceptedInput):
    record consumed input/accepted interaction at this tick
    advance train/doors/other supported actors in stable order
    synchronize changed collider transforms
    advance head and walking controller; complete the physics step once
    publish completed tick for the existing audio observation mapping
```

Unity performs the live physics step; isolated reconstruction explicitly simulates its physics scene once. Neither path double-steps controllers. Native physics repeatability remains a tested build/platform property. Reconstruction copies only supported components and remaps their references; it must not start a second live session or microphone. Unsupported actors fail before recording, with full numeric capture available only as an explicit alternative.

Extend `AudioMaterial`'s existing knot structure to include absorption, scattering, transmission, acoustic index and interior damping at arbitrary ordered Hz knots. Continue the declared linear-Hz/clamped interpolation; `c_material = c_air / index`, with damping in nepers/metre. Require finite values, positive index, nonnegative damping, coefficients in [0,1], and absorption + transmission <= 1 under the selected energy contract; scattering partitions reflected energy. Asset migration preserves the old evaluated values and records its version. Do not silently clamp invalid measurements or claim the physical-property sliders are measured material data.

Package cleanup follows actual dependency use: remove unused demo/import packages only after checking assets, serialization and an Editor/Player compile. Keep tooling packages in use. Source line count alone is not a reason to merge independent algorithms or remove validation.

## 3. Unity scenes and assets

### Components and geometry

Use ordinary Unity GameObjects, native primitives, transforms, colliders, meshes and animation. Extend `AcousticSurface`, `RaytracedAudioSource` and `RayTraceManager`; do not introduce parallel replacements. A receiver remains an explicitly registered transform until a separate component earns its cost.

| Role | Contract |
|---|---|
| Geometry | `AcousticSurface` selects mesh/collider/proxy and acoustic material, optionally per submesh. |
| Source | `RaytracedAudioSource` binds a prepared clip or proposed stream input and issues sample-timed source events; source identity is independent of storage. |
| Receiver | Registered head transform; configurable omnidirectional mono or binaural left/right output. One head is one receiver, distinct from Unity's native AudioListener. |
| Session | Manager owns identity tables, capture/replay state and numeric resources; stays active outside the captured hierarchy. |
| Motion | Exactly one controller, animator or physics owner per moving object; acoustics observes its result. |

Auto geometry prefers MeshFilter then a supported collider. Explicitly select Collider on proxies; never export coincident mesh and collider surfaces. Readable selected meshes produce triangles. Supported 3D colliders include boxes, MeshCollider meshes and uniformly scaled spheres/capsules with declared curve tolerance. A cooked convex collision hull is not necessarily the mesh asset. Native cylinders use their mesh if their capsule collider is an unsuitable approximation.

Planar2D uses supported Box/Polygon/Edge/Circle/Capsule Collider2D shapes in XY, including offsets. Open edges remain open; out-of-plane movement fails validation. No projection of arbitrary 3D rooms into approximate line segments. Unsupported compound/scaled shapes need explicit proxy children or meshes.

Fixed authoring scale is baked into local geometry. During a take, translation/rotation and activation may change; topology, material, scale/shear, roster, hierarchy and clip bindings remain fixed. Reject changes visibly instead of silently changing labels. Capture actual world poses and estimated velocity/acceleration, including angular quantities and validity; replay disables motion writers and applies the observations rather than rerunning physics.

IDs are unique per placed instance, persisted before capture, sorted into stable numeric tables, and never derived from display names or Unity instance IDs. Disabled objects retain their indices. Source/listener tables map to object indices; pair order is `sourceIndex * listenerCount + listenerIndex`.

### Scene composition

Use three environments with multiplayer as an optional mode on each. All remain editable Unity scenes, with metre-scale geometry and one shared player/session setup. Initial sizes below are authoring starting points, not solver limits.

| Scene | Editable layout and interactions | Acoustic purpose and acceptance |
|---|---|---|
| `TrainStation` (implemented) | Preserve the single platform, automatic perimeter train/stops, ordinary walk-on carriage and waiting-room door. | Moving source/receiver, Doppler, moving-platform replay and indoor/outdoor transitions. Keep the existing train sound. |
| `Church` (editable blockout) | Start with an approximately 30 × 12 × 15 m nave, side aisles/columns, pew rows and one adjoining room with an opening door. Place a test speaker at the front and walking paths down the nave/behind columns/through the doorway. | Separated early arrivals, decaying reverberation, material variation, occlusion and head rotation. Play an impulse/sweep/dry speech; later speak through the mic. Validate simple subfixtures before interpreting the full hall. |
| `LakeForest` | Start with an approximately 120 m outdoor mesh clearing, shoreline, water plane, rocks, a tree cluster and a cliff across the water. Walk a loop from open shore into trees and behind rock cover; use two independently positioned test sources. | Long arrival times and distinct cliff echoes, ground/water differences, occlusion, scattering and larger static workloads. Measure each effect by changing one scene element. |

The church must not fake success with a Unity reverb zone or reverberant source recording. The acoustic solver creates the response. Vary bounce budget, rays, delay horizon and render tail independently; retain complete measured decay or label truncation. Report early-arrival timing, band decay curves and EDT/T20/T30 only where their fitting ranges and noise floor support them. Do not call a one-second tail a measured multi-second RT60. Higher-order mirror/diffuse limitations may require transport work before this becomes a teacher; low-frequency room modes are outside geometric validation.

The outdoor scene uses mesh terrain/acoustic proxies because the current exporter does not support Unity Terrain. Water starts as a planar reflective approximation with declared coefficients; no underwater solver, wave motion, wind refraction or vegetation-volume model is implied. Tree trunks/large branches/rocks participate through simple meshes/colliders; decorative leaves initially do not. Forest reverberation is frequency-dependent and must be checked separately from hall reverberation ([measured bamboo-forest example](https://doi.org/10.1121/1.1354988)). Add canopy/ground detail only when its acoustic role and cost are measured.

```text
Church / LakeForest / TrainStation
  Simulation       RayTraceManager + AudioManager + shared SceneSession
  World            fixed geometry, movable actors, persistent acoustic IDs
  Player           reused controller + camera/head + optional mouth source
  Sources          test speaker(s), clip or streamed PCM
  Presentation     scene lighting and existing minimal UI
```

Create a shared player prefab and test-speaker prefab when the church uses them. Keep unique architecture in its scene; prefab only repeated pews/columns/trees and genuinely shared actors. No scene copies for Low/High or multiplayer. The current `Platform.prefab`, `Train.prefab` and `StationLoop` remain station assets. Visual calibration stays below unit-scale motion roots; acoustic proxies have explicit shapes/materials and no coincident duplicate surfaces. Keep the player's collision capsule out of acoustic geometry initially so it does not enclose its own head/mouth.

All scenes expose walking/look/jump, the existing Save run, and listener/top-down comparison views. Recreate these views from the replay timeline; storing two videos with every take is unnecessary. Video export can be an explicit later output. Use ordinary nearby door interaction. Mic controls add one explicit enable/mute control with a recording indicator; device/gain/self-monitor settings appear when the cursor is released. No boarding button, horn/restart buttons, repetitive interaction notifications or separate game menu is required. Detailed renderer/material/benchmark settings stay in the Inspector/comparison window. A scripted route feeds the same accepted-input path as manual movement.

Retain small 2D/3D room fixtures by migrating the existing rooms; a separate large AcousticLab environment is not required. They supply analytic single-wall/door/empty-space tests without loading scenery. Static scenes use the existing BVH first. Profile primitive, edge and volume queries before adding per-mesh BLAS plus instance TLAS; visual tree instancing alone does not accelerate acoustic tracing.

### Selected assets, lighting and materials

Keep the Built-in Render Pipeline and existing Kenney Train Kit 1.1. Select existing FBX assets `train-diesel-c`, `railroad-straight`, optional `railroad-curve`, `train-carriage-box` and `train-connector`. Use native geometry for platforms, shelters, rooms and doors. Detailed rails/bolts are decorative unless an experiment requires them. Imported models may stay unreadable when explicit collider proxies supply acoustics. No new model importer is required. The kit is CC0; retain its local license. [Kenney source](https://kenney.nl/assets/train-kit).

Optional visual imports, not downloaded by this specification: Poly Haven [Kloofendal 48d Partly Cloudy Pure Sky](https://polyhaven.com/a/kloofendal_48d_partly_cloudy_puresky), initially 2K EXR, and [Concrete Floor 02](https://polyhaven.com/a/concrete_floor_02), selected maps up to 2K. Preserve CC0/source/author/resolution/hash records. Keep unused vendor duplicates untouched until a dependency audit.

Use shared Standard materials for concrete/painted metal/safety paint, Skybox/Panoramic for the sky and existing unlit/text support for markers. Normal maps use normal import; roughness/masks use linear data and the Standard workflow's proper channel packing, including `smoothness = 1 - roughness`. Visual textures do not define acoustic coefficients.

A reusable lighting prefab, if needed, contains a directional sun, light probes, reflection probe and a few local lights. Share a LightingSettings asset and sky material between station scenes; RenderSettings and baked data remain scene-specific. Bake fixed architecture after layout settles; moving trains/doors/players use probes and realtime direct shadows. Check open/closed doors because a static bake cannot represent every state. Keep numerical room fixtures neutrally lit. Church uses window/door daylight and a small number of local lights; LakeForest reuses sun/sky/probes. Check performance with these visuals enabled. No graphics pipeline migration or post-processing package is required.

Visual materials and `AudioMaterial` assets stay separate. Preserve existing materials; add numerical fixtures `IdealRigid` (absorption/scattering 0/0) and `IdealAbsorber` (1/0) for the later energy solver. Both remain opaque in today's direct solver. Measured real materials need sourced curves; guessed values are not measurements.

Generate documented impulse, sweep and seeded-noise fixtures. Preserve the horn for preview while resolving its provenance before distributed datasets; existing wind/meme clips lack adjacent source records and stay out of controlled datasets. Identify already-reverberant recordings. Prepare dry PCM before capture; avoid simultaneous native AudioSource playback duplicating the custom renderer. No purchased audio library, speech service or asset catalog is required.

Asset scope for the new scenes: build church walls/arches/pews from native primitives and simple authored meshes first. Use shared stone/plaster/wood Standard materials; select only needed maps from the [Poly Haven texture library](https://polyhaven.com/textures) after layout acceptance. For outdoor visuals, the optional [Kenney Nature Kit](https://kenney.nl/assets/nature-kit) supplies CC0 trees/rocks; import only used models with its license. Use the existing Built-in material shaders and a simple Standard water surface initially. A custom animated water shader is decorative later work, not an acoustic dependency. No new asset download is implied by this document.

Store scenes in `Assets/Scenes`, shared player/speaker/door prefabs in `Assets/Prefabs`, environment-only prefabs under their environment folder, visual and acoustic material assets separately under `Assets/Materials`, and external models/licenses in `Assets/External`. Reuse existing locations rather than adding a package tree. Asset hashes/provenance bind training recipes. Placeholder acoustics must be labeled as such until sourced coefficients replace them.

## 4. Capture and saved inputs

### Numeric scene and time

Unity objects resolve into existing `AcousticTake.Scene` tables: objects, local primitives, material curves, source bindings, receiver bindings and prepared clips. Per-observation `State[]` follows the object table. The proposed stream table extends source storage without changing object/primitive GPU layouts. No GameObject, Transform, AudioClip or GPU padding crosses the file boundary.

Units are metres, seconds, Hz and radians; quaternions are xyzw. One fixed sample rate and clock origin belong to each take. Live capture uses integer sample timestamps mapped from `AudioSettings.dspTime`; silent capture keeps its configured rate. Device-rate changes or clock discontinuities end a segment. All future adapters must preserve sample-time identities.

Existing capture observes in LateUpdate. It preserves the first observation at each sample, counts duplicate polls and never rewrites a knot already consumed live. At recorded samples return stored float32 poses/motion exactly; between them use linear position and shortest-path quaternion interpolation. Hold the prior pose across a marked cut. Record observation gaps and interpolation version. More frequent reads of unchanged frame state do not constitute finer observations.

Velocity comes from consecutive observed poses; acceleration from velocity estimates; angular derivatives use relative quaternions. Initial/cut derivatives are invalid. Between observations, the declared interpolation determines velocity, not independent acceleration integration. Timestamp-mismatched Rigidbody velocity and visual pose cannot be combined; endpoint motion includes rotation about an offset where applicable.

One voice per source is implemented: Play restarts it, Stop ends it, Gain changes it. Multiple simultaneous voices on one source are deferred. Same-sample events retain append order. Events apply inside half-open audio blocks at exact samples, with clip offset/loop policy recorded; reject commands for already consumed audio.

Source inactivity stops new emission while preserving already-emitted tails. Geometry/listener activation follows the response schedule. An initial or mid-session segment must explicitly capture source phase/state and any required pre-roll; the current UI starts voices after capture begins. Playback, timing and source content are independent of reconstruction quality.

### Implemented files and versions

New take inputs use version 2, adding authenticated transmission/IOR/damping material curves. Version 1 remains readable and reconstructable using its original scene checksum; missing coefficients use zero transmission/damping and IOR one. Current direct render and monitored-output manifests use version 3, adding propagation delay rate to the version-2 directional records and channel metadata. Existing version-1 response files contain delay/gain only; readers must dispatch on the manifest version rather than infer row widths. Old take inputs remain usable for mono or binaural rendering.

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
| Direct frame v3 | int64 sample then six float32 values per pair: delay, pressure gain, direction XYZ and delay rate; `8 + 24*P` bytes. |
| Direct frame v2 | int64 sample then float32 delaySeconds/pressureGain/directionX/Y/Z per pair: `8 + 20*P` bytes. Direction points from the listener toward the arrival's origin in world coordinates. |
| Direct frame v1 | Legacy delaySeconds/pressureGain only: `8 + 8*P` bytes. |
| Clip | Mono float32 at take rate, content-hashed. |

The current downmix is channel averaging and resampling is linear. Replace or independently validate that preparation before high-fidelity audio claims; paired runs must always share exact prepared PCM and processing identity.

Run/output manifests bind `channelsPerListener`, `hrtfDataset`, `hrtfHash`, `hrtfTaps` and the DSP contract. WAV frames order left then right for binaural output; internal blocks order [sample,listener,ear]. Sample counts, queue capacities and lead always measure frames. Output-capture application rows contain int64 scheduledSample/consumedSample/applicationTicks, int32 pairCount, then six float32 values per pair in v3 (five in v2). Callback rows remain unchanged. Capture preserves both ear channels; overflow, channel mismatch or interrupted shutdown leaves it incomplete.

Capture validates geometry/materials, bindings, scale and hierarchy without rebuilding geometry every observation. Mesh scanning still costs O(mesh data) and must be profiled. Complete manifests publish only after payloads close and hash successfully. Reject incomplete files, unsupported formats, mismatches and non-finite data.

Keep two replay meanings separate: a new render evaluates the saved take with new settings; original-run review plays captured component PCM and uses actual response-application/queue logs. Component PCM includes underruns but excludes downstream device processing. Exact original playback does not imply bit-identical simulation across GPUs.

Offline numeric replay needs no original Unity scene. The current in-scene pose replay needs matching persisted take object IDs/assets and suspends all motion writers, including configured extra scripts; stopping restores them. For compact inputs the authored scene/build remains required; numeric full takes retain their acoustic geometry, while current decorative comparison views still require matching scene assets. A portable visual-snapshot system is deferred. Seeking cached output PCM only reads the requested audio range; seeking a new simulation must reconstruct sufficient dry/response history and tails, or be explicitly marked a cold start.

### Compact scene recording

`SceneRecording` is the default station/church recorder. It stores consumed input changes, sample-stamped source events and the audio-sample/completed-tick mapping from scene initialization. It references the authored scene and runtime revision instead of retaining runtime snapshots. Train, door, look and walking share `SceneSession.InitializeSimulation` / `Step` in live playback and reconstruction.

| File | Contract |
|---|---|
| `input.json` | `scene-input-from-init-v1` (also reads `station-input-from-init-v2`): scene identity, project content hash (Editor), runtime module, engine/platform, Git revision/dirty status, timestep, player and ordered actor IDs/types/version/settings, counts/completion and checksums. |
| `input.bin` | Little-endian 25-byte changes: int64 tick, float32 movement X/Y and look X/Y, byte jump/interact flags. Held axes persist; look/buttons apply once. |
| `input-observations.bin` | Little-endian int64 audio sample plus int64 completed tick. Begins at (0,0); sample times increase and tick counts never decrease. |
| `take.json`, `events.bin` | Scene/source identity tables, source PCM and geometry hashes, audio rate/duration, and source-event records. Payload hashes do not require storing the payloads. |
| `live.json` | Small timing and deadline report. Not needed to replay. |

Full poses, geometry, prepared PCM and monitored audio are omitted by default. **Retain Full Reference** explicitly adds those diagnostic payloads. Existing full takes remain readable. Hashing, source events and timing metadata use the same writer in both modes; there is no second acoustic file format.

Play begins capture at initialization. Save completes it; another run starts by re-entering Play. Stopping Play also finalizes input data. Scene-reload playback runs sequentially from zero; there are no seeks, checkpoints, cached playback or artificial duration limits. Reopening requires matching scene/code/engine/platform and simulation settings. Live output checks device sample rate; offline reconstruction uses the saved rate independently.

`SceneReplay` is an Editor operation available in Edit mode, from the Inspector and in batch jobs. It builds an isolated [Unity preview scene](https://docs.unity.com/en-us/engine/6000.3/script-reference/unityeditor/scenemanagement/editorscenemanager/newpreviewscene), uses the same initialization/step methods and regenerates an ordinary acoustic take. It checks authored geometry/source hashes and optionally compares original full observations. Runtime playback does not require this generated take. Walking support is queried explicitly so reconstruction does not depend on Play-only collision callbacks.

The current adapter covers shared walking/sliding-door inputs in Station and Church, plus the station train. Arbitrary scene edits, rigid-body behavior and cross-platform determinism are not promised. Editor project identity hashes all files under `Assets`, `Packages` and `ProjectSettings` with relative paths; it excludes generated `Library` contents and checkout location. Older manifests without that hash retain their original module/scene checks; Player recordings require the matching runtime build. Preserve the matching Git revision/assets; identity hashes cannot recover missing source files. The README describes current usage and validation.

### Scene-independent input (implemented) and stream recording (proposed)

Keep compact recording from initialization as the default for the new supported scenes. Record the scene/build identity, fixed actor settings, accepted input changes, interactions, source events and sample-to-completed-tick map. Train/door motion is reconstructed by the same actor code; static geometry stays in the authored scene. The new `scene-input-from-init-v1` header identifies its actor/input contract and preserves the old station-v2 reader. Keep the existing 25-byte local walking/look/button records and 16-byte observation mapping. The header binds the player and ordered actor IDs, component contract versions and immutable initialization settings; new action payloads require a versioned contract rather than untyped event blobs. Do not weaken project identity to make old recordings open against changed code.

**Microphone audio is new input data and must be saved.** Neither Git, input buttons nor an RNG seed can reconstruct speech. Compact mic takes therefore add the source PCM and timing metadata, while continuing to omit regenerable poses/geometry/asset clips and optional monitored output. Live mic preview without saved source data is allowed, but that run cannot be offered as replayable/renderable mic data.

Propose acoustic take **v3** for source kinds; current v1/v2 files retain their existing interpretation. `SourceDefinition` gains `kind: Clip | Stream`; Clip uses the existing clip index, Stream uses a stream index. Resolve bindings before capture; a disabled mic can occupy its source slot from initialization. Stream tables are separate from immutable clip assets and finalized independently of the geometry/scene identity, avoiding an end-of-recording change to the scene hash. The completed take manifest authenticates all stream descriptors and the shared source-block log path/hash; no unauthenticated sidecar determines rendering.

| Stream record | Meaning |
|---|---|
| Stream descriptor in take manifest | Stable source binding, relative payload path/hash, int64 first scene sample and sample count, canonical mono float32 at the take's fixed rate; original device/codec and processing identity. |
| `source-<index>.f32` | Little-endian canonical PCM entering source gain/transport, written sequentially; file length is `4 * sampleCount`. No per-block files and no load-entire-recording requirement. |
| `source-blocks.bin` | Shared append-only timeline: int32 stream index, int64 first scene sample, int32 frame count, int64 available-at scene sample, uint32 status = 28 bytes/record. Spans are ordered/nonoverlapping per stream and cover its payload. |
| Status values | Captured/decoded data; intentional silence; missing capture; concealed network loss. A bit identifies discontinuity. Unknown values or unaccounted ranges are invalid. |
| Clock metadata | Device/remote origin, source rate, mapping/resampler version, epoch changes and measured uncertainty. Availability denotes completed canonical input, not inferred acoustic capture time. |

Gaps occupy their declared timeline and contain the exact silence/concealment used live, with status preserved. Intentional mute is valid silence; missing input is not. Start/Stop/Gain remain separate source events and apply once; Stream Play/Stop gates emission without restarting or looping the sample cursor (`clipOffset=0`, `loop=false`). Muted device input is not retained. Restarting capture records a discontinuity and new clock mapping without changing past samples. Source data outside declared active/recorded intervals is known silence only where that policy is explicit.

Use checked 64-bit absolute sample/file positions; chunk lengths remain bounded integers. Memory depends on response/DSP history and pending I/O, not take duration. At 48 kHz, mono float32 costs 11.52 MB/minute before lossless compression. Optional ZIP archival preserves exact bytes; lossy voice packets are transport data, not the sole canonical training/replay payload. Do not add a codec dependency just to archive local recordings.

Finalization stops emission, copies accepted remaining capture data, closes/hashes stream files and then publishes take completion. Disk overflow/failure marks the saved input incomplete; it cannot silently produce a valid replay. Offline treats saved stream blocks as a bounded file-backed source and never opens the microphone or network. Reconstruct compact movement once into an ordinary numeric take, then reuse that immutable result and exact source samples for all quality variants.

Single-player replay from zero remains sequential: no checkpoints, seek cache, geometry snapshot or duration cap is required. Multiplayer additionally needs the accepted multi-actor timeline and authoritative corrections; exact client acoustic replay records the interpolated state it consumed, as section 12 specifies.

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

Author material curves at arbitrary ordered frequency knots, with interpolation/out-of-range conventions. The current energy curves interpolate linearly in Hz and clamp outside their knots. Transport bands have ordered nonoverlapping edges within [0, Fs/2] and a declared coefficient evaluation rule. Validate coefficients and grid bounds; interpolate once into the requested material-by-band table. More sampled bands do not create new measured material detail. During migration, keep eight-band packing only in the legacy adapter; remove it after its scenes/materials are migrated.

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
| State | float4 position/rotation/velocity/angularVelocity at 0/16/32/48; uint flags/padding at 64; 80 bytes. |
| Source/listener bindings | int32 object indices, 4 bytes each. |
| Response | float4 direction at 0 (xyz world listener-to-arrival-origin, w=delay rate in seconds/second); float32 delay/gain at 16/20; reserved zero at 24/28; 32 bytes. v3 disk rows are separately packed to six floats. Spectral paths use the final two uint lanes for frequency bin and topology hash. |
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
| Live | Captured Unity observations, effective source events and available clip/stream PCM. | Bounded work against output deadlines; select one monitored receiver. |
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

The direct and Spectral paths use velocity-derived continuous source-time resampling with optional response crossfades and linear/windowed-sinc interpolation. IR holds frozen scene responses. The motion approximation and remaining physical limits are defined above; no duplicate pitch shift is applied.

Energy synthesis uses a documented filter bank, normalized band overlap and reproducible phase/noise whose response energy matches the target. Maintain temporal continuity; independent random synthesis each frame causes audible changes unrelated to the geometry. Pressure/complex responses use compatible convolution. Bounded old/new filter transitions preserve history; cuts/seeks obey their declared reset/pre-roll policy.

**Implemented direct binaural rendering:** each receiver is one head pose with two ear channels. Direct paths retain world listener-to-source direction, delay and pressure gain. Inverse head rotation converts direction into +X right, +Y up, +Z forward. The shared live/offline mixer interpolates the bundled MIT KEMAR compact HRIRs on their measured azimuth/elevation grid, composes the result with the fractional-delay FIR once per response and crossfades old/new filters under the existing transition policy. Source gain is applied once; pressure/occlusion gain is applied once. No additional native Unity spatialization is applied to the output driver. Mono remains the default for old settings; the station explicitly selects Binaural.

The measured dataset contains 710 mirrored directions, 128 taps/ear at 44.1 kHz, with paired ear gain/timing preserved. Filters resample once at preparation for other rates. Both outputs sum all source paths; there are no independently traced ear receivers. Source history includes the HRIR support, and offline tail settings must cover the desired propagated/filter tail. Geometric delay is added once. The dataset's bulk measurement delay was cropped by its authors; its remaining timing is not a calibrated absolute eardrum arrival. Do not add an ear-offset ITD on top of measured ITD. `dspLatencySamples` reports fractional-delay FIR latency separately from the HRIR's time response. Coincident endpoints use the frontal filter convention; unmeasured elevations clamp to the measured elevation range. Interpolation wraps azimuth continuously across the rear seam. The spectral mixer uses the same angular interpolation and crossfades old/new convolved output sample by sample over `transitionSamples`; a cut switches the response immediately. It does not infer Doppler from path-slot correspondence.

This is a generic far-field mannequin model for headphones, not personalized HRTFs, near-field head scattering, headphone equalization or complete wave acoustics. The raytraced preview preserves each arrival direction through per-ear synthesis; validated directional reverberation and directional dataset export remain future work. Scalar band energy cannot recover directional reverberation without extra arrival data or an explicit diffuse assumption. Keep mono reference checks. Binaural tests verify ear timing/level, head rotation, stereo routing/capture and live/offline sample agreement on CPU and Metal.

Use the active Editor for authoring-time jobs, or batch mode when that project is closed. A second Editor must not share the open project. Existing CPU batch jobs can use -nographics; HLSL jobs need a graphics device. A built worker is an option when simultaneous authoring/jobs justify it, not required infrastructure. Interrupted runs restart initially; resumability later must restore DSP history and verify all identities.

### Microphone and streamed source audio (proposed)

The new source path supports a local microphone, decoded remote voice and saved streamed PCM. These are producers of the same source samples; they do not introduce a third renderer or a new shader family. Existing clips remain the simplest source. A microphone emitter follows the player's mouth transform; each client has one local listening head, whose two ears remain the existing binaural output.

```text
local mic → capture cursor → mono/rate/clock conversion ─┐
network voice → jitter/decoder → mono/rate/clock conversion ├→ timestamped source history
saved stream → bounded file reads ──────────────────────┘
asset clip → existing event-driven source reads ───────────→ CPU reference or GPU DSP
                                                        → listener PCM queue
canonical stream input + timing/status → recording writer
```

**Source boundary.** Extend `TakeRender.Signal` with Clip/Stream storage and retain its existing sample reads for DSP. A stream has a bounded circular sample history plus absolute int64 positions and a committed availability frontier. Add only append/range-readiness operations; no audio-provider plugin hierarchy is needed.

```text
append(source, firstSample, pcm, availableAtSample, status):
    validate finite data, ordering and retained capacity
    copy into owned source storage; publish only after the copy completes
    enqueue the same canonical bytes + timing/status for recording
prepareOutput(first, count):
    derive conservative source ranges from declared delay/resampling/filter support
    if any required stream range is still pending: defer this bounded job
    upload newly committed samples once; run existing CPU or GPU synthesis
    retire history only after every reader's required range has advanced
```

A missing future block is not silence and must never be uploaded as zeros that become permanent source history. Before capture begins/after a known stop, explicitly defined ranges are silence. At a real live deadline, apply the declared gap policy once, record its status and count it; late data cannot replace samples already used. Offline reads the recorded decision, or reports missing inputs, rather than obtaining better data for High than Low. Range-readiness checks use conservative DSP bounds without GPU path readback; test those bounds for every supported Doppler/filter configuration. Retention covers maximum acoustic delay plus actual DSP/resampling support; capture-buffer capacity and output playback lead are separate quantities. GPU submission/readback and file writing stay outside the device callback.

**Unity capture.** `MicrophoneInput` owns one selected device and an ordinary looping capture clip. Drain only newly committed samples using its cursor into reusable buffers; never reread/resample the full capture clip every frame. Unity provides the looped clip and current recording position ([Start](https://docs.unity3d.com/6000.4/Documentation/ScriptReference/Microphone.Start.html), [GetPosition](https://docs.unity3d.com/6000.4/Documentation/ScriptReference/Microphone.GetPosition.html)). Its loop length bounds memory, not session duration. Enable through the platform microphone permission flow; handle refusal, absent/disconnected devices and shutdown explicitly. Merely opening a scene or replay never enables the microphone.

Convert the actual captured channel layout to mono and resample to the fixed session rate with a stateful, anti-aliased resampler. Preserve phase/history across chunks; validate 44.1/48 kHz and varying chunk sizes against an independent oracle. Do not use the current whole-clip linear resampler as a high-fidelity stream converter. The renderer consumes and the recorder stores the exact same post-conversion samples, before source gain and virtual propagation. Device/OS processing, optional gain control/noise suppression and conversion versions are provenance; processing must not differ between paired runs.

**Clocks and gaps.** Unwrap the device cursor into a monotonic frame count. Map capture-frame positions to scene emission samples using an explicit origin/rate mapping; track clock drift without discontinuously jumping the source timeline. `GetPosition` is not a calibrated hardware ADC timestamp: record estimated offset/uncertainty and measure capture-to-output latency with a loopback fixture. A long poll gap can hide whole buffer wraps; detect ambiguous/overwritten data and mark loss rather than treating the cursor as proof of continuity. If exact lost-frame positions cannot be established, close the capture epoch and record the uncertain interval; do not invent sample-accurate recovery. Device/rate/clock resets end a capture epoch (or the take when its fixed-rate contract changes). Resampling drift correction has a saved version; saved canonical PCM is authoritative on replay.

Capture timestamps, canonical-sample availability, render publication and device-consumption timestamps are distinct. In live synthesis, block waiting/resampler lookahead adds measured latency; it never shifts the recorded emission time or duplicates physical propagation delay. The current ~372 ms playback lead is a known obstacle to comfortable voice monitoring. Report capture, resampling, network/jitter, renderer scheduling, DSP and device buffering separately from simulated path travel time. Reduce lead only after device-deadline tests pass with scene graphics; no new latency guarantee follows from adding a mic selector.

**Controls and output.** Mic starts disabled. Enabling it is an explicit action with a small active/recording indicator; mute stops retaining/transmitting speech and commits intentional silence while earlier echoes drain. Expose device and level controls in the existing released-cursor panel. Local processed self-monitoring is opt-in for acoustic experiments, with a headphone-first workflow to avoid speaker feedback. Save its audibility mask/timing in render provenance so a live/offline comparison uses the same sources. Remote listeners still hear the source when local monitoring is off: apply a local source/listener audibility mask, not a global source mute. Do not also play the microphone clip through Unity's native spatializer.

The first implementation does not invent acoustic echo cancellation. A later voice backend may supply AEC using the actual speaker-output reference; ordinary room reflections from our solver are not microphone feedback cancellation. For live usability, microphone failure should disable that source and give a concise reason without stopping walking/other audio. Saved gaps remain visible in diagnostics and training eligibility.

**Recording and training.** Section 4 defines saved stream bytes and availability/status. 'Dry' here means before virtual acoustics; a real microphone may already contain room reverberation, noise and device processing. Label that origin. Use generated impulses/sweeps or suitable dry recordings for controlled physics checks. Save manual mic takes for playback and reuse; do not automatically enqueue them for training. When selected for an acoustic comparison, Low/High receive identical canonical speech, motion, preprocessing and gap decisions. Windows affected by capture loss remain identifiable and are excluded from clean-source teacher claims. Response reconstruction can still train using synthetic source signals; mic speech is an additional end-to-end workload.

Implementation touches `RaytracedAudioSource`, scene source export/guard validation, manager session setup, `Signal`/Live scheduling and the take reader/writer. Stream sources do not need dummy AudioClips; fixed source identity and input kind replace the current mandatory-clip guard. Add one capture component and keep bounded stream storage alongside existing source types. GPU filters, transport and output callback remain the same except for availability-aware submission and local audibility selection.

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

A resolved recipe records scene/layout family, dimension, geometry/material variation, endpoint placements, motion curves, source-event schedule, capture/tick schedule, length and independent scene/motion/signal seeds. Resolve ranges before capture using authored traversable bounds. Manual and scripted captures use the same take format. Include environment family (station/church/outdoor), source origin (asset/generated/mic/decoded voice), source PCM/preprocessing hashes and gap/availability metadata. Group speaker and original utterance/session ancestry before windowing; recording does not automatically enroll microphone speech into training.

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

Direct scalar addressing dispatches on the render manifest version (independent of take-input version):

```text
C = {1: 2, 2: 5, 3: 6}[formatVersion]
frame r, pair p, channel c:
offset = r*(8 + 4*C*P) + 8 + 4*C*p + 4*c
c=0 delaySeconds; c=1 pressureGain; c=2,3,4 direction XYZ; c=5 delayRate
[R,P,C] strides = [8+4*C*P, 4*C, 4]
initial offset = 8
nonempty addressed span = (R-1)*(8+4*C*P) + 4*C*P
full file length = R*(8+4*C*P)
```

Use an unaligned structured dtype/explicit strides, not float32 reshape of timestamps. A mono temporal experiment may select delay/gain, but its descriptor must state that selection; retain the other stored channels. Delay rate has units seconds/second. Preserve directions and head rotations for binaural experiments; unit vectors use the coincident-zero convention and require directional validity/normalization rather than generic scalar interpolation. Missing channels in old formats must not be misread as stored zeros.

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

The current roster is 53 objects, two sources and one binaural listener; older rosters need their matching scene for pose replay, while offline audio remains self-contained. Capture records actual late-frame poses, not a separately simulated trajectory. Movement and gravity use one CharacterController sweep per simulation tick, including sub-millimetre steps. Wheels align to the rails; emitters lie outside the opaque train proxy. The passenger carriage uses separate floor/side/roof geometry. Mouse sensitivity is independent of render-frame duration. Autonomous passengers remain future work.

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

For the direct baseline, worst-case traversal scales O(R*S*L*G) and CPU mixing about O(N*S*L*C*Ktap), where C is one or two output channels and Ktap includes the HRIR. These formulas do not account for spectral rays/bounces/group FFT work. Current version-3 direct diagnostic bytes are R*(8+24*S*L), output about 4*N*L*C excluding headers. Binaural filter preparation also searches the measured directions and composes the FIR per response; profile that cost when increasing pair count or update rate. These guide sweeps, not arbitrary capacity guarantees. Tiling/pair batching preserves identity and normalization.

Record hardware, Unity/backend/build, device sample rate, audio buffer, visual workload, seeds and deadlines before a realtime acceptance claim. Use same take/physical model/output range across comparisons. Include static/open/blocked paths, moving doors/trains, fast motion, empty geometry, large counts and cuts. Score absolute level, arrival time, decay, phase where meaningful, clicks, staleness and deadline misses. No automatic time shifting or separate amplitude normalization.

### Recordings entry point and replay scope (planned)

Extend the existing comparison window with Open recording/folder and available recordings; retain Inspector replay/render actions. No separate main-menu framework is required for church, outdoor or mic work. Discovery reads manifests, not whole payloads. Show supported completed runs and concise reasons for corrupt/incomplete ones. Do not build automatic scene snapshots, a database, checkpoint system or playback cache in the first delivery.

Opening behavior depends on the actual saved inputs:

| Recording | Playback/render contract |
|---|---|
| Compact scene inputs | Load the matching authored scene/build and replay from initialization; stream mic PCM from its saved payload. Missing revision/assets is an actionable error, not silent substitution. |
| Full/generated numeric take | Use saved state/geometry/PCM for offline rendering without re-running gameplay. Current listener/top-down visual previews need matching scene assets; a geometry-only fallback can be added in the same viewer when needed. |
| Completed offline audio | Play the selected listener's compatible Rendered/Low/High/Difference output. Keep its original sample mapping and declared DSP/propagation latency. |
| Captured monitored output | Available only when explicitly retained; call it Recorded and keep its callback/lead mapping. Regenerated output is never labeled Recorded. |

Opening a take never activates a live microphone or joins a server. Select the recorded listener initially; another listener needs its own render. Missing monitored output does not invalidate valid source inputs. Audio and views share an int64 take-sample clock; measured output follows actual consumed audio, not frame deltaTime. For compact replay, no seeking/checkpoints are promised. Existing numeric comparison range/scrub behavior remains; continuous bounded audition is separate work, not a prerequisite for these scenes.

Use the existing Save → finalize/drain → replay/render lifecycle. Keep walking controls usable during capture and restore the user's scene/controls after preview. Source mute stops new emission; pending reflections remain. Handle failed saves explicitly and never silently drop required microphone payloads.

Batch generation stays a small saved JSON job list around `TakeBatch` and existing render/comparison methods. Resolve a compact run once, then reuse its immutable numeric inputs for every budget. Jobs bind scene/source hashes, code, settings, seeds and target acceptance. Publish a complete dataset only after its referenced files and gates pass. The open Editor runs jobs sequentially; do not start another Editor on that same project. Optional later lossless archive export preserves bytes; portable decorative snapshots/standalone browser/seek caches are deferred until required.

## 12. Multiplayer

Multiplayer is a mode for the same Church, LakeForest and TrainStation assets, not a copied environment or a second acoustic engine. The first network acceptance run is two separate processes, then two machines over LAN/direct IP. Test independent walking/head direction and a shared door/speaker before adding voice. Player count is configured and benchmarked; two is the validation starting point, not a hardcoded maximum.

Use **Netcode for GameObjects + Unity Transport** for session/object replication, with compatible versions pinned when implementation begins. Host owns shared actors and accepted interaction/sound events; clients own local input/camera and receive authoritative state. Start with package-supported transform replication and measure control latency. Do not implement competitive rollback, host migration, matchmaking or internet Relay until requested. Physics is simulated by the authoritative owner; peers do not infer identical outcomes from input alone ([Unity network physics](https://docs.unity3d.com/Packages/com.unity.netcode.gameobjects@2.7/manual/advanced-topics/physics.html)).

### World, sound and listener ownership

- Assign session-stable IDs to players, scene actors and acoustic sources; verify scene/clip/build identities at join. Map IDs to the existing numeric source/receiver tables.
- Replicate accepted events once with event ID, source/target, action, gain, clip offset/loop state and server tick/emission time. The host must not apply an event once locally and again through its client path.
- Join state includes current actor transforms/settings and playing voice phase/start time. Replaying old broadcasts is not a substitute for current state.
- Each client renders all audible emitters for **its local head only** in ordinary gameplay; it need not synthesize every other player's ears. Multiple receivers remain available for explicit offline experiments.
- Only the local player's camera/input and one output AudioListener are active. Remote player heads/mouths provide pose/source data, not extra native audio outputs.
- Local acoustic quality is independent of network tick/packet rates. No response tensors, ray buffers, wet PCM or HRTF outputs are replicated as gameplay state.

Use one declared presentation clock for replicated remote actors, voice and scene events, mapped to the client's fixed acoustic sample rate. Record server/client clock corrections and input availability. Align voice emission and remote mouth/occluder presentation on that clock; do not synchronize by packet-arrival time. Track local-head pose/application latency separately. Late commands advance phase or expire according to their timestamp, never rewind consumed audio or restart a continuing clip from zero. Explicitly benchmark the extra delay from transform interpolation as well as audio queues.

### Microphone voice transport

NGO does not itself supply the required voice codec/jitter pipeline. Use a maintained Unity-compatible **Opus** implementation for compressed mono speech; verify its package/platform/license support before selecting/pinning it. A voice SDK is acceptable only if decoded PCM can enter our source pipeline and automatic spatialized playback can be disabled. Do not write a codec. Codec packet duration/bitrate/loss handling are saved settings and are independent of our STFT grid ([Opus specification](https://www.rfc-editor.org/rfc/rfc6716)).

```text
local canonical mic → encoder → timestamped packets → remote jitter buffer
  → decoder / declared loss concealment → fixed-rate canonical stream
  → existing local raytracing + binaural rendering
```

Packet contract: session/source ID, capture epoch, sequence number, sender frame/sample timestamp, frame count, codec configuration identity and payload. Session handshake supplies rate/channel/clock details; codec changes begin a declared epoch. Use bounded unreliable delivery for voice frames so retransmission cannot hold later speech behind a lost packet; reliable session/control messages remain separate. The transport adapter chooses a tested packet size/framing within its limits, not a large reliable RPC stream. A LAN PCM fixture may isolate timing during tests but is not the shipping bandwidth plan.

The jitter buffer orders frames and finalizes each playout range once. Duplicates are ignored; late packets cannot replace committed speech. Record packet gaps and the actual decoder concealment/silence used. Adaptive jitter delay/clock drift must preserve a declared monotonic scene-time mapping, with corrections recorded. Starting a remote voice waits for its prebuffer without blocking the audio callback. Network decode creates no Unity scene objects on its worker thread.

Each receiving client can hear a different acoustic result because its head and occlusion differ. It can also receive different packets; do not hide those differences behind a shared host recording. Local self-monitoring uses the pre-encode source only when enabled, avoiding a second looped-back copy. Muting/disconnect stops new emission and drains retained paths; a disappearing source's data slot remains valid until no pending renderer work references it.

### Recording and replay

A host recording keeps accepted inputs/events for all actors plus authoritative corrections needed to reproduce the host world. A **client acoustic recording** keeps the actual interpolated scene states/events consumed by that renderer and the canonical decoded pre-acoustic PCM, including concealment. Local input alone cannot reconstruct remote interpolation, packet loss or authoritative physics corrections.

Keep this distinction explicit: compact deterministic single-player takes remain compact. Exact network-client acoustic capture opts into the necessary dynamic-state observations and streamed source payloads; static geometry still references its matching scene/build until offline reconstruction creates a portable numeric take. Initially reuse the existing full `State[]` observation layout (including static poses); sparse pose encoding is a later measured storage optimization, not a prerequisite. Geometry/assets and optional monitored output are still not mandatory snapshots for every run. Never advertise an input-only network recording as exact client replay.

Low/High comparisons for a client reuse that client's exact scene/source timeline, with identical network artifacts. Separate experiments can evaluate codec/network quality, but cannot count those errors as acoustic reconstruction error. All host/client views of one session and their voice-speaker ancestry stay in the same dataset split.

Keep the existing fixed-roster acoustic segment invariant initially. Join/leave closes the current segment before rebinding, then starts another with current actor/source state and the dry pre-roll needed for pending propagation/filter history. A small session index records segment offsets, gaps and warmup; do not truncate old echoes silently. If overlap/pre-roll is not yet implemented, declare the transition discontinuity and exclude affected comparison windows. A disconnect finalizes valid data; host departure ends the first implementation's session visibly.

### Turning multiplayer captures into training examples (planned)

Capture at the **source-history boundary**, immediately after receive-side decoding, concealment and rate/clock conversion, before source gain, propagation or HRTF. Commit those same canonical samples once to live source history and the recording queue. A separate recording of the microphone or speaker output is not an exact copy of what the acoustic renderer consumed. Keep one mono stream per source; do not premix speakers or store a stereo file for every source/listener pair. Local mic uses its pre-encode canonical stream; each remote source uses that client's post-decode canonical stream.

```text
session.json                         session ID, capture consent/provenance, segment index
  client ID → ordered segment paths, scene-time offsets, warmup/gap status
client-<id>/segment-<id>/
  take.json                          scene/build/material/source IDs, local receiver
  states.bin                         actual presented poses at acoustic observations
  events.bin                         accepted emission/gain/mute changes
  source-<index>.f32                  canonical pre-acoustic mono samples
  source-blocks.bin                  firstSample/count/availableAt/status (section 4)
  clock mapping                      sender epochs → this segment's sample clock
```

This is the proposed streamed-take extension, not files the current clip recorder already writes. Keep clock mappings in the segment manifest initially; use sample-indexed records only if updates become numerous. Matching scene assets supply fixed geometry and clips. Finalization binds all payload hashes/counts and marks completion last. A bounded writer queue moves disk IO off the audio callback; an overrun makes the capture incomplete instead of silently dropping speech. Runtime playback can continue even if recording fails.

| Recorded identity/state | Why training needs it |
|---|---|
| Session/client/segment IDs; stable source, receiver and pseudonymous speaker IDs | Join the right views, preserve source ownership and avoid split leakage. No dependence on transient network object IDs. |
| Scene/build/material/clip hashes, solver version and settings | Reconstruct exactly the acoustic problem; distinguish asset changes from quality changes. |
| Local head pose and presented remote mouth/occluder poses at observed samples | Reproduce this client's spatial input, including interpolation and authoritative corrections. Do not substitute the host's different timeline. |
| Canonical source samples, absolute emission sample, block availability, loss/concealment/discontinuity status | Preserve what was audible and what was known at each deadline. A muted sample, an unavailable sample and a concealed sample are different facts. |
| Sender epoch/rate and clock mapping changes; codec/resampler/jitter settings | Explain alignment and degradation without requiring the decoder or network to run identically later. |
| Local monitoring/source-listener mask; HRTF and ear-layout identity | Reproduce self-voice routing and binaural labels without treating two ears as two players. |

A raw packet log and sender-side clean mic copy are **optional diagnostics** for a separate codec/network experiment. They cannot replace receive-side canonical PCM. Do not store both by default. Likewise, wet monitor WAV is an optional audit output, not the training master. Speech is a real recorded asset: participants explicitly opt into dataset collection, with a visible recording state; training selection is a separate choice. Retain permitted-use provenance and source ancestry so excluded speech can be removed from derived examples.

Offline generation uses the existing reconstruction/render path:

1. Validate/finalize a client segment and resolve its exact scene/build. Compile fixed geometry once and open saved source streams with bounded reads. Input-only solo runs first reconstruct their motion; network client runs read their observed poses directly.
2. Select the target: directional acoustic response/IR for transport reconstruction, or the existing complex STFT/waveform target for audio reconstruction. Store the exact target axes, units, time origin, FFT/window/hop and directional representation. Both ears can be derived from one reconstructed directional response; an ear-output model declares the HRTF explicitly.
3. Render Low and converged High against **identical** source samples, scene trajectory, event times, interpolation rules and source/receiver masks. Vary the declared ray/bounce/response-time budgets only. A finer acoustic solve samples the saved trajectory; it does not invent unobserved authoritative motion. Network tick density and pose interpolation error remain separate limitations.
4. Produce causally ordered training windows containing Low features, available motion/material/direction metadata, persistent-history reset flags and valid/loss masks, with High targets. Model inputs stop at the deployment deadline; an offline teacher may use additional computation, but must not leak future speech, future poses or later packet arrivals into the student's features. The already-finalized received timeline defines the label problem.
5. Preserve pre-roll long enough for the selected propagation horizon and DSP/filter state, not just one STFT window. At startup/join/leave/clock-reset boundaries either restore valid history or mask the affected windows. Record actual tail truncation; missing tails are not target silence. A noncausal lookahead experiment must declare and benchmark its added latency separately.
6. Split by related session, speaker and scene/layout ancestry **before** windowing. All clients, ears, quality variants and overlapping windows from a session stay together. Keep held-out layouts/materials, speakers, source overlap, rapid turns, moving occlusion and network-loss conditions. Save seeds and low/high/reference settings with each generated run.

Acoustic reconstruction learns missing acoustic detail with the same decoded speech in both budgets. High must not use clean sender speech while Low uses lost/concealed receiver speech: that would silently turn the task into codec enhancement or missing-speech synthesis. Keep concealment windows labeled; report their results separately, and exclude ambiguous timing/capture failure windows. Test single-source and overlapping talkers; per-pair response labels are reusable across dry clips, while mixed audio targets must include all contributing sources.

Do not label today's maximum budget “ground truth” automatically. Church higher-order reflection/diffuse limits still need physical checks and convergence. Approve the teacher per target, then compare learned reconstruction to unmodified Low, simple temporal interpolation and spending the same total GPU time on more simulation. Report held-out quality, head/voice latency, deadline misses, memory and total render-plus-inference cost. Actual microphone, network transport, streaming persistence and model training remain later implementation stages.

The implementation can use three small adapters: session/clock ownership, network player/actor binding, and network source/voice delivery. Keep Unity package replication for ordinary transforms; avoid a networking class per prop. Reuse the microphone stream/history, recording writer and local DSP.

## 13. Implementation sequence and acceptance

The shared SceneSession, generic scene input recording/reconstruction, shared player/speaker prefabs and editable Church blockout are now implemented. Church currently uses the existing diagnostic chime and provisional material/transport settings. Legacy room/material migration, microphone streams, LakeForest, networking and training remain planned. Keep each change reviewable and reuse the independent CPU/GPU oracles.

| Stage | Smallest concrete change | Completion gate |
|---|---|---|
| 1. Shared session + cleanup | Extract common input/capture/replay from StationDemo while creating the Church blockout. Migrate the two old room fixtures, then delete unused legacy renderer/UI/helpers. Complete material-knot migration before relying on new material presets. | Station walking/boarding/save/replay still works; a scene with no train records/reconstructs identically; old supported takes remain readable; no missing scene/prefab scripts. |
| 2. Church | Editable architecture, common player/speaker/door, listener/top-down views and impulse/sweep/dry speech. | Correct simple early-arrival fixtures and full tail reporting; repeated capture/render comparisons; measure ray/bounce convergence before teacher eligibility. |
| 3. Local microphone | One capture component, canonical streaming source/history, timing/gaps and versioned saved source files. | Permission/device lifecycle, bounded wrap/drift/resampling, clip-stream CPU/GPU parity and exact saved-input replay. Actual device test reports latency and deadlines with church graphics. |
| 4. LakeForest | Mesh terrain, shoreline/cliff, acoustic tree/rock proxies and the same session/mic tools. | No ground/collider failures; independently explainable distant arrivals; static BVH agreement and measured primitive/edge/volume/DSP costs as geometry scales. |
| 5. Network world, then voice | Host/join, ownership/state/clock mapping, local receiver, then Opus/jitter feeding the same stream path. | Two processes and two machines; independent control/occlusion, one event application, late join, rate mismatch, loss/reorder/disconnect and replay of actual client inputs. |
| 6. Dataset/reconstruction | Validate selected teacher/target, export per-pair data, then existing cadence/ray/temporal training plan. | Identical low/high scene and source inputs, loss/status/availability masks, source/session/layout split isolation, reference convergence and held-out quality/cost benefit. |

A recordings picker can extend the existing comparison window when needed; portable visual snapshots, new menu frameworks, checkpoints and automatic caches are not dependencies of these stages. No new compiler or render pipeline is required. BLAS/TLAS, filter reuse and scheduling changes follow measured bottlenecks; compare whole-job memory/latency/deadlines, not CPU dispatch time alone.

Atomic tests should cover one contract and use synthetic data; physical microphone/network/hardware timing is a separate category. Reuse the existing test files/fixtures instead of adding another runner.

| Test responsibility | Independent check |
|---|---|
| Scene initialization/steps | Same accepted inputs produce matching observations for supported actors; one owner steps each actor/physics scene. |
| Stream storage | Absolute positions survive wrap; pending differs from silence; late overwrite is rejected; retention/overflow and source end preserve needed history. |
| Stream conversion/clock | Chunk-size-invariant anti-aliased conversion, known tone/impulse timing, sample-rate mismatch, drift, reset and ambiguous capture-wrap loss. |
| Stream files | Exact PCM/count/hash round trip, explicit gaps/status, interrupted save rejection and old clip-only readers. |
| DSP boundary | Identical finite clip and chunked stream yield matching IR/Spectral CPU/GPU outputs, including delayed tail and two sources/heads. |
| Voice delivery | Synthetic packet sequence proves reorder/dedup/deadline/concealment behavior; accepted event IDs apply exactly once. |
| Materials/transport | Knot migration and energy bounds; analytic wall/door/medium fixtures, BVH/brute-force agreement and declared decay/convergence tests. |

Keep a few end-to-end scenarios: the existing station lifecycle; a church walk/door/saved-stream Low/High replay; a two-client scene with independent audible occlusion and voice loss/rejoin; and the explicit real-device deadline test with graphics. A smaller outdoor route is a scene smoke/benchmark fixture, not a duplicate of every unit test. Missing hardware is an explicit skip, not simulated success. There is no zero-bug guarantee; bounded waits and resource cleanup prevent ordinary failures from contaminating later tests.

Benchmark each scene with the same saved input, warmup policy and declared CPU/GPU/quality settings. Save code/scene/material/source hashes, wall time, stage scopes, device deadlines, source-to-output and head-to-audio latency, peak buffers, upload/readback bytes and geometry/source/listener counts. Report repetitions and failures; preserve previous implementations in Git. Microphone and network recordings are deterministic *inputs for replay*, not proof that real devices or distributed physics repeat identically.

Before expensive labels, approve the selected teacher's physics and convergence. Before calling voice usable or reconstruction faster, set workload/hardware-specific latency, memory and quality targets from the prototype measurements, then enforce them. Unchosen Opus package/version, measured material curves and final latency targets are explicit implementation decisions; do not silently fill them with arbitrary constants.
