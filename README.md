# Acoustic ray tracing

Unity **2022.3.62f3**, C# and HLSL. The existing realtime renderer is a 2D prototype. The new recording path captures native 2D/3D scene geometry, motion and source events for repeatable offline experiments.

**Implemented now:** editable 3D walkthrough creation, strict take capture, live direct-sound monitoring, pose replay/scrubbing, CPU/HLSL direct solvers for 2D/3D, offline WAV/response export, bounded asynchronous GPU batches, captured output PCM/timing, configurable fractional-delay FIR, and temporal comparisons with an optional finer-reference quality check. Live and offline use the same interpolation, source timeline and audio mixer. CPU and GPU implement the same direct/opaque model and are checked against each other.

**Still planned:** general reflection transport, configurable FFT/STFT convolution, acceleration structures, converged ray-budget training pairs and learned reconstruction. The old realtime prototype remains available when **Use Take Renderer** is disabled. Its physics differs from the shared direct model; neither is a validated high-fidelity reflection teacher. See [the design contract](docs/simulation-contracts.md) for the full plan.

## Record a scene

For a working starting scene, choose **Tools → Acoustics → Create 3D walkthrough**. It creates native walls, a moving occluder, a walking listener and an emitter using the existing audio/material assets. Save the scene, enter Play, select the simulation manager, record, and start the emitter through its context menu. WASD/mouse move the listener. The scene stays editable through ordinary Unity components.

1. Add/use a `RayTraceManager` on a separate, always-active GameObject. Set **Take Root** to the root of the objects you want to capture. Enable **Use Take Renderer** for the shared direct path and assign an `AudioManager`. **Direct Shader** can be empty for CPU, or `Assets/Shaders/Direct.compute` for GPU. The legacy Raytrace Shader field is separate. Leave Use Take Renderer disabled for silent capture; legacy playback must be stopped before starting a take.
2. Create ordinary Unity objects. For 3D, add `AcousticSurface` to acoustic geometry: Auto uses `MeshFilter.sharedMesh` first, then a supported collider. For 2D, choose **Planar2D**, use Collider2D shapes in the XY plane, and add `AcousticSurface`. Existing entries in **Obstacle Objects** are also registered.
3. Assign an `AudioMaterial` on each surface, or **Take Default Material** as a fallback. Optional submesh overrides share one material table. `AudioMaterial.energyCurve` accepts any number of ordered frequency knots; an empty curve preserves the existing material. Coefficients use linear interpolation in Hz and clamp outside the authored range. The direct baseline records these curves but treats all geometry as opaque.
4. Add `RaytracedAudioSource` components with clips and a manager reference under Take Root, or list them explicitly in **Take Sources**. Clips must be readable/preloaded (Decompress On Load). Preparation for the new path happens once when capture/preview starts. Assign the walking camera/player transform to **Listener**, or list receiver transforms in **Take Listeners**.
5. In Edit mode, click **Register scene objects / persist IDs**, then save the scene. Review the resulting object/source lists. IDs live on the manager; don't register again during a take.
6. Enter Play mode, click **Record take**, and choose an output parent outside `Assets` (for example `Recordings`, already gitignored). Use existing controllers, animation or physics to move objects. Start/stop sounds through `RaytracedAudioSource.Play()` / `.Stop()`; the component context menu exposes both. Start voices after recording begins.
7. Click **Stop and save take**, then wait for **Idle** before leaving Play mode so the monitored output finishes recording. **Choose saved take** reopens it later. **Replay recorded poses** applies recorded world poses; **Pause replay** lets you edit **Replay Seconds** to scrub. Stop restores the original scene poses, activity and suspended movement components.

**Start live preview** runs the same pipeline without writing a take. Stop preview before starting a recording. All registered sources can play through the new renderer; the selected **Monitor Listener** feeds the output device. Offline output includes every listener. Live monitoring requires the device sample rate. Settings are fixed when the session starts.

Replay suspends the existing player/look/oscillator scripts, Animators and rigid bodies on recorded objects and their ancestors. Add other movement scripts to **Replay Motion Writers**. Pose replay does not play the rendered WAVs or rerun gameplay. Keep the manager outside any hierarchy that the take can deactivate.

Visual replay requires the original scene hierarchy/assets: object IDs are checked, geometric equivalence is not. Offline rendering uses the geometry saved inside the take and does not depend on that live scene.

3D export supports readable triangle meshes, BoxCollider, MeshCollider, and uniformly scaled SphereCollider/CapsuleCollider. Native Unity cylinders use their mesh, avoiding their capsule collider approximation. 2D export supports Box, Polygon, Edge, Circle and Capsule colliders, including offsets. Curved collider tessellation uses **Take Curve Tolerance** in world metres. A selected collider object must have one supported collider; use a mesh/proxy for compound or unsupported shapes.

## What a take means

Object membership, shape, material and clip assets are fixed for the take. Rigid movement and activation are recorded. Every accepted observation and session finalization verifies authored geometry/material values, source/listener bindings, root membership, parent relationships and scale/shear. In-place mesh/collider/material edits abort capture and leave the take incomplete. Clip reference and audio clock changes are checked too. Prepared PCM is frozen at session start. Non-rigid deformation and changing hierarchy require a new take. Validation compares captured authoring inputs using reusable buffers. Mesh vertices/indices are checked once per shared mesh per observation, without tessellating colliders or rebuilding material tables. Work remains O(mesh data); its time is reported separately.

Observations run in `LateUpdate`, timestamped against `AudioSettings.dspTime` as integer samples at the take's sample rate. The file preserves float32 positions/quaternions exactly at stored timestamps. Between observations, position is linear and rotation uses shortest-path slerp. This is exact replay of the recorded observations, not recovery of unobserved motion between frames. The manager accepts the first observation at each audio sample and keeps that knot immutable; additional observations at that sample leave it unchanged. This prevents offline replay from changing an interval already rendered live. The manifest records this policy, the maximum observation gap and how many observations shared a sample. Duplicate polls only service pending rendering and update counters; scene validation runs on accepted observations and at finalization.

Velocity and acceleration are estimated from successive observed world poses, with validity flags. Call `MarkTakeDiscontinuity(transform)` when teleporting; playback holds the prior pose until that sample instead of sweeping through the jump. Physics is not rerun to reconstruct a take.

The first voice model is **one voice per source**. Play restarts it; Stop and Gain change it at their recorded sample. Source PCM is prepared once per unique clip using the existing channel-average mono conversion and linear resampler. All clips are stored at the take's sample rate. This is not a production resampling filter. Native `AudioSource.Play()` is not intercepted. The shared renderer supports all recorded emitters. The old realtime renderer still plays only its selected emitter. Recorded events describe accepted source commands, not the legacy output device's exact playback timing.

For scripted capture, the public entry points on `RayTraceManager` are:

```csharp
manager.BeginTakeRecording(newTakeDirectory);
manager.RecordSourceEvent(emitter, AcousticTake.VoiceAction.Play, gain: 1, repeat: true);
// Existing gameplay moves the scene; the manager records in LateUpdate.
manager.RecordSourceEvent(emitter, AcousticTake.VoiceAction.Gain, gain: .5f);
manager.EndTakeRecording();
// In a coroutine, wait for the queued output and its capture to finish.
yield return new WaitUntil(() => manager.Mode == RayTraceManager.SessionMode.Idle);
manager.BeginTakeReplay(newTakeDirectory, play: false);
```

## Live scheduling and diagnostics

Live rendering waits for the next observed pose before interpolating the interval, then renders only audio whose inputs are known. It does not extrapolate future movement. **Monitor Lead Samples** gives that producer time to run before playback; **Monitor Capacity Samples** bounds the PCM queue. Missed deadlines produce silence at the correct timestamp and late PCM is dropped rather than shifted later. Increase lead only if the extra latency is acceptable; use the counters to identify missed deadlines.

The new audio callback copies PCM through a single-producer/single-consumer queue, without locks, logging, file I/O or managed allocations. Simulation, validation and mixing stay on the control thread. The legacy path still has its old callback implementation. `live.json` records validation/solve/DSP costs, observation gaps, queue underruns/overflow/late samples, callback overruns and maximum callback duration. These counters are sampled when the producer stops; a queued tail may still play afterward. Recording also writes `monitor/output.wav`, `callbacks.bin`, `applications.bin` and `output.json`. This preserves the component output, including mute/underrun silence, before downstream Unity mixing or device processing. Callback records carry sample positions, channel counts, monotonic timing and the DSP clock; the output manifest records the take clock origin. Application records carry scheduled output samples, the consumed-sample cursor and response values. A bounded callback-to-main-thread queue moves all file work outside the callback. Overflow or interrupted shutdown leaves output capture incomplete; input-take validity is separate.

Unity Profiler markers `Acoustics.ValidateScene`, `Acoustics.SolveDirect` and `Acoustics.MixPCM` expose those stages for closer inspection.

Live GPU rendering batches the response samples within each known motion interval and uses [Unity asynchronous readback](https://docs.unity3d.com/2022.3/Documentation/ScriptReference/Rendering.AsyncGPUReadback.html). One in-flight batch owns its pose/response buffers until completion. **Maximum Responses Per Batch** bounds that storage; **Maximum Pending Observations** bounds queued motion intervals. A full queue stops the session visibly. Finishing/cancelling waits for owned requests before releasing their buffers. Offline GPU runs use measured synchronous readback. Completion wall time includes scheduling and readback, not isolated GPU execution time.

The manager exposes **Idle**, **Live**, **Recording**, **Replay** and **Draining** states. Legacy playback/precomputation cannot enter an active take session. Direct audio initializes only its bounded queues and output driver; legacy buffers/debug textures are allocated when their callers need them.

**Maximum Delay Seconds** declares the source-history horizon; a response beyond it fails explicitly. Completed intervals discard older voice/activity history while retaining the boundary state and original loop phase. **Maximum Signal Events** bounds retained history per source and catches excessive event density. `live.json` includes retained-event counts, maximum pending observations, completion latency and a source/listener-pair × primitive test upper bound. The upper bound is workload accounting, not a measured hit count. CPU remains useful for small workloads.

## Render offline

On the manager Inspector, choose a saved take and click **Render saved take to responses + WAV**. **Response Step** controls how often recorded geometry/poses are evaluated. Smaller steps cost more and reduce motion quantization. This is a time-resolution control, not a ray count or FFT size. **Tail** adds output samples after capture stops so already-emitted sound can arrive; no new source audio is emitted past the end of the take.

The baseline computes source-major/listener-minor direct delays and pressure gains, with 1/r spreading above the configured minimum distance and two-sided opaque intersections. Planar mode uses 2D visibility with the same point-source spreading law. Responses are held for each update interval. **Transition Samples**, from zero through the response step, optionally crossfades the old and new delayed signal taps. Teleports reset that transition at the next response update. Live and offline use the same mixer, including events within a response interval. Source events and source activation gate emission at their recorded sample, including events inside an interval. Geometry and listener activity follow the response grid. Moving boundaries, continuous Doppler, reflections, diffraction, phase-dependent materials and binaural output are not modeled.

**Delay Filter** selects linear interpolation or a normalized windowed-sinc fractional-delay FIR. **Sinc Radius** controls the FIR half-width and adds that many samples of causal DSP latency, reported as `dspLatencySamples`. Coefficients are computed per response, and live/offline share the same taps. This improves fractional-delay reconstruction; it does not change transport physics. Tail length must cover the propagation delay and desired filter tail. FFT/STFT convolution is still a separate future implementation.

For a batch job, save this JSON next to your data directory; paths are relative to the job file (absolute paths also work):

```json
{
  "take": "take-001",
  "output": "direct-001",
  "settings": {
    "responseStepSamples": 256,
    "speedOfSound": 343,
    "referenceDistance": 1,
    "minimumDistance": 0.1,
    "tailSamples": 48000,
    "transitionSamples": 64
  }
}
```

```sh
"$UNITY_EDITOR" -batchmode -nographics -projectPath "$PWD" \
  -executeMethod TakeBatch.Render -takeJob /absolute/path/job.json \
  -logFile /absolute/path/render.log
```

Use an output directory that does not exist. The process exits 0 on completion and 1 on failure. CPU direct rendering works with `-nographics`; future GPU methods will require a graphics device. A shell/Python loop can invoke jobs sequentially; there is no separate orchestration service. To use the GPU, add `"shader": "Assets/Shaders/Direct.compute"` to the job and omit `-nographics`. A missing or unsupported shader fails explicitly.

To compare budgets, click **Compare two response budgets** in the Inspector, or add a `comparison` settings object to the job. The main `settings` object describes the coarser run; `comparison.responseStepSamples` must be smaller, with all other settings identical. For the example above:

```json
"comparison": {
  "responseStepSamples": 64,
  "speedOfSound": 343,
  "referenceDistance": 1,
  "minimumDistance": 0.1,
  "tailSamples": 48000,
  "transitionSamples": 64
}
```

The output contains `low/`, `high/` and `comparison.json`. Both runs use the same scene, source PCM, event timeline, sample count and physical model. The report includes per-listener waveform RMS error, maximum error, reference RMS and SNR when defined, together with both run timings and hashes. The backend is warmed before measured solves. These are single-run measurements; repeat representative workloads before drawing performance conclusions.

Enable **Check finer reference** or add a `referenceCheck` object with `responseStepSamples`, `maximumRmsError` and `maximumAbsoluteError`. Its response step must be smaller than the high run and at least `transitionSamples`; tolerances must be explicit finite nonnegative values (zero requires exact agreement). The third run writes `reference/`, errors against the high run, and `referenceAccepted`. Only a passing check sets `eligibleForTemporalTraining`; a completed render alone does not. This is agreement with one finer direct-model reference, not proof of physical convergence.

Runs record the compiled C# module ID, Unity/build identity, shader dependency hash in the Editor, and a hash of all settings. Player builds use the build GUID and shader name alongside the C# identity. Old takes remain readable and have no capture provenance when it was not recorded.

This experiment changes **temporal resolution**, not ray count. The finer run is a comparison reference for this direct model, not reflection ground truth and not the planned low/high-ray training dataset.

## Files and ownership

Unity scenes/prefabs remain the authoring source. Recorded inputs and render outputs live outside `Assets`:

```text
take-001/
  take.json           version, completion, scene tables, sample rate, hashes
  geometry.bin        local baked triangles or planar segments
  states.bin          observed poses, motion estimates, activity, cut flags
  events.bin          source commands at integer samples
  clip-0.f32          prepared mono PCM; one file per unique clip
  monitor/            present when recording monitored output
    output.json       completion, scope, timing layouts, settings, provenance, hashes
    output.wav        mono component output, including deadline/mute silence
    callbacks.bin     output sample ranges, channels and callback timing
    applications.bin  response values and scheduled/consumed sample positions
direct-001/
  run.json            method/settings, input hashes, output hashes, timing
  responses.bin       sampled unit-input direct delays and pressure gains
  listener-0.wav      mono float32 audio; one file per listener
```

Binary data is little-endian. Geometry rows contain `float3 a,b,c; int32 objectIndex,materialIndex` (44 bytes). State frames contain an `int64 sample`, then one row per object: `float3 position; float4 rotation(x,y,z,w); float3 velocity,angularVelocity,acceleration,angularAcceleration; uint32 flags` (80 bytes/object). Event rows contain `int64 sample,clipOffset; int32 sourceIndex,action; float32 gain; byte loop` (29 bytes). Table order and units are part of version 1; world distance is metres and angular velocity is radians/second. Response row layout is declared in `run.json`.

Writers mark the manifest complete only after payloads are closed and hashed. Readers reject incomplete files, mismatched hashes and unsupported versions. Failed/cancelled jobs remain incomplete. Render reports separate preparation, state evaluation, solve, DSP and file-writing costs, maximum solve duration, PCM/core-buffer accounting and Unity/CPU/GPU identity. The buffer figures exclude managed object overhead and some scratch/history storage; they are not process memory measurements. Render wall time excludes opening/validating the take before the render call. These are diagnostic measurements, not a claim of realtime feasibility on arbitrary workloads. No database, networking service, compiler migration or new package dependency is introduced.

## Repository map

```text
Assets/
  Scripts/
    Scene/        AcousticSurface, AudioMaterial, SceneHelper
    Simulation/   RayTraceManager + Take partial, Take, TakeRender + Live partial,
                  DirectSolver, TakeComparison, ComputeHelper
    Audio/        AudioManager, RaytracedAudioSource
    Demo/         PlayerController, MouseLook, Oscillator, TrainVisualLoader,
                  RaytraceControlPanel, FFTTesting
  Shaders/        Direct.compute, Raytrace2D.compute, AudioConvolve.compute, Common.hlsl
  Audio/          source clips and DefaultMixer
  Materials/      acoustic material assets
  Editor/         TakeEditor (Inspector/batch jobs), TakeDemo (scene creation)
  Tests/Editor/   TakeChecks
  Scenes/         editable scenes and their baked lighting
  Prefabs/        project prefabs
  External/       imported train assets
  TextMesh Pro/   imported UI assets
Packages/         Unity dependency manifest
ProjectSettings/  Unity project configuration
docs/             simulation contracts
```

Folders group existing files; they introduce no new assemblies or dependencies. Asset `.meta` GUIDs are preserved so scenes and prefabs keep their references. `FFTTesting` stays a runtime component because the existing FFT scene uses it. Relative shader includes stay together in `Shaders`.

## Cleanup implemented

| Files | Result |
|---|---|
| [SceneHelper.cs](Assets/Scripts/Scene/SceneHelper.cs) | Reuses authoring snapshots and comparison storage, deduplicates shared meshes/materials, and detects in-place changes without per-observation scene export. |
| [RayTraceManager.cs](Assets/Scripts/Simulation/RayTraceManager.cs), [RayTraceManager.Take.cs](Assets/Scripts/Simulation/RayTraceManager.Take.cs) | Defines exclusive session states and retains ownership while output drains. Legacy precomputation/playback is excluded during a take session. |
| [AudioManager.cs](Assets/Scripts/Audio/AudioManager.cs) | Defers legacy allocation and captures actual component PCM, callback timing and response applications through bounded storage. |
| [DirectSolver.cs](Assets/Scripts/Simulation/DirectSolver.cs), [Direct.compute](Assets/Shaders/Direct.compute) | Batches live response frames with asynchronous completion and explicit resource ownership. Reports completion latency and traversal workload bounds. |
| [TakeRender.cs](Assets/Scripts/Simulation/TakeRender.cs), [TakeRender.Live.cs](Assets/Scripts/Simulation/TakeRender.Live.cs) | Bounds event/cut/observation history and shares configurable linear/windowed-sinc delay filtering between live and offline rendering. |
| [Take.cs](Assets/Scripts/Simulation/Take.cs), [TakeComparison.cs](Assets/Scripts/Simulation/TakeComparison.cs) | Records implementation/settings provenance and optionally gates temporal training pairs against a third, finer direct-model run. |

The next transport work is validated 2D/3D indirect responses, followed by configurable FFT/STFT convolution and converged ray-budget comparisons. [Raytrace2D.compute](Assets/Shaders/Raytrace2D.compute), [Common.hlsl](Assets/Shaders/Common.hlsl) and [AudioConvolve.compute](Assets/Shaders/AudioConvolve.compute) remain legacy migration candidates. General reflection training and learned reconstruction are not implemented by these cleanups.

## Checks

Run **Tools → Acoustics → Run take checks** for file round trips, invalid captures, motion/cuts, native geometry, variable material knots, direct-path fixtures, sample-timed events, multiple sources, capture mutations, live/offline sample equality, callback allocation and queue deadline behavior, authoring snapshot allocation/mutations, long-lived source history, analytic FIR accuracy, exact captured component PCM and reference acceptance/rejection. **Check legacy GPU FFT** also checks the new 2D/3D direct kernels against CPU fixtures on the active GPU, plus batched async completion/disposal and GPU live/offline equality for both delay filters.

Both checks also support `-executeMethod TakeChecks.Run` or `TakeChecks.RunGpu` in batch mode; omit `-nographics` for the GPU check. The legacy FFT remains fixed at 1024; its out-of-bounds butterfly indexing was repaired. This does not validate the rest of the old reflection/DSP model.

Local verification used an isolated Unity **6000.4.1f1** project and Metal because Unity 2022 is not installed on this machine. The repository's Unity version and package manifest were not upgraded. Unity 2022 and DirectX validation remain outstanding.

Measured results and reproduction details are in [cleanup-results.md](docs/cleanup-results.md).
