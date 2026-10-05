# Acoustic ray tracing

Unity **6000.4.1f1 (Unity 6.4)**, C# and HLSL. The existing realtime renderer is a 2D prototype. The new recording path captures native 2D/3D scene geometry, motion and source events for repeatable offline experiments.

**Implemented now:** editable 3D station with two sources and one binaural listener, train service with station stops and walking aboard, sliding door and walkthrough controls, strict take capture, live direct-sound monitoring, pose replay/scrubbing, CPU/HLSL direct solvers for 2D/3D, offline WAV/response export, bounded asynchronous GPU batches, captured output PCM/timing, configurable fractional-delay FIR, binaural headphone rendering, and a native comparison window with synchronized audio, plots and recorded scene views. Live and offline use the same interpolation, source timeline and audio mixer. CPU and GPU implement the same direct/opaque model and are checked against each other.

**Still planned:** general reflection transport, configurable FFT/STFT convolution, acceleration structures, multiplayer, converged training pairs and learned reconstruction, including denser response updates from sparse solves. The old realtime prototype remains available when **Use Take Renderer** is disabled. Its physics differs from the shared direct model; neither is a validated high-fidelity reflection teacher.

[The consolidated design contract](docs/simulation-contracts.md) covers scene/assets, Unity/HLSL data, live/offline execution, IR/complex/spectrogram reconstruction, temporal upscaling, training, multiplayer and acceptance gates. This README covers running the implemented code; [cleanup results](docs/cleanup-results.md) records historical measurements.

The proposed [recordings browser and portable playback](docs/simulation-contracts.md#planned-recordings-browser-and-portable-playback) specifies the Record/Recordings entry point, loading old takes without their original scene, continuous audio/pose replay, shared visual snapshots and batch example generation. It is a design spec; the controls below remain the implemented workflow.

## Compact recording experiment

TrainStation records automatically from scene initialization, before its first movement or sound. The train, door, walking and looking use one fixed simulation clock (50 Hz by default). The input log stores movement changes, look deltas, jump and interact; held axes persist, while deltas/presses apply once. New recordings do not store a mid-session controller/world snapshot. The full take remains alongside the log for accuracy checks.

Press **R / Save run** to finish recording. Wait for **Saved** to finish draining monitored audio. Stopping Play also finalizes the inputs/reference automatically; its monitored WAV may end before the buffered tail. Movement and live audio can continue after saving, but this run is finished; re-enter Play to record another run from initialization. There is no duration cap, seek or checkpoint flow.

To replay, select **Acoustic simulation → Replay saved inputs from beginning** in the Inspector after saving. **Choose saved take** selects another run. Replay reloads the matching saved scene, disables live input, applies recorded inputs from tick zero and plays the recorded source events. Its tick/audio mapping drives motion and acoustic observations on the recorded timeline, so the current render frame rate does not replace the original observation times. Playback stops at the recorded end. The current scene, code, motion settings, Unity version, platform and sample rate must match. Each new Editor recording also identifies its Git commit and whether scene/code/project files had uncommitted changes. Restore the matching project revision before reopening historical inputs; automatic Git checkout is not implemented.

**Export input-only run** on the same Inspector creates a portable five-file folder: `input.json`, `input.bin`, `input-observations.bin`, `events.bin` and `take.json`. It contains version/configuration checks, consumed controls, their audio timing and source events. It needs no `states.bin`, geometry payload, prepared PCM or captured WAV to replay: the matching authored scene supplies those assets. The ordinary full recording stays as the reference; export does not delete it or overwrite an existing folder.

The station adapter supports its train, door and walking controller; arbitrary rigid bodies and unrelated runtime edits are not reconstructed. Motion-setting changes invalidate the input log while full capture continues. Older mid-session input logs require their old code; ordinary full takes still work with the offline tools.

**Tools → Acoustics → Verify recorded station inputs** reconstructs an ordinary take from either the full recording or the input-only export. Run it after saving and before starting playback. Without reference files, it rebuilds geometry and prepared audio from the matching scene and checks their identity. When reference poses are available it also compares every reconstructed observation. It writes `input-replay-<UTC time>/verification.json` and a take usable by the offline renderer/comparison window; `comparedReference` distinguishes reconstruction from an accuracy comparison. Reference poses never correct the simulation. This diagnostic/offline output is not needed by normal playback.

**Tools → Acoustics → Run take checks** covers the file formats and core audio/simulation behavior. **Test station recording and replay**, run from Edit mode in the saved station, exercises initialization, boarding, jumping, saving, compact export, reconstruction, audio comparison and actual input-only replay. It writes its result under `Recordings/station-lifecycle-result.txt`. This test avoids compiling automation scripts during recording, which can stall the Editor and cause live-audio dropouts.

The end-to-end check on Unity 6000.4.1f1 / Apple M5 passed **382 core checks** and recorded 645 ticks covering startup, boarding, departure, jumping and walking. The complete five-file export was **25,894 bytes**, including identity tables and sound events. All **53 objects** had byte-identical reconstructed states/events; the two offline stereo renders matched exactly across **560,129 frames**. Normal scene-reload input playback also finished with zero position error. Capture and replay each reported **zero underrun, late or overflow samples**, using the existing 24,000-frame audio lead. GPU batching now includes the latest known movement interval before dispatch, removing an avoidable frame of producer latency. These same-build/platform results do not establish general physics determinism or guarantee audio deadlines under arbitrary Editor stalls. Artifacts: `Recordings/station-20261005-084143-977/` and `Recordings/station-lifecycle-result.txt`.

Enter Play mode in TrainStation, then **Tools → Acoustics → Test compact input replay**. Save any active recording first. This opt-in test pauses live monitoring, runs isolated station copies, restores live monitoring, and writes a report under `Recordings/compact-experiment-<UTC time>/`. Automatic station capture still retains full observed states for validation.

This earlier controlled experiment stores a scene revision, code/driver identities, initial state, a fixed 50 Hz clock and 12 movement/look/jump/door/gain commands in `input.json`. It runs a scripted 46-second walk onto the train, a jump during departure, a ride and walk off, plus a falling rigid body. Two fresh physics scenes reconstruct the motion from the saved commands. Full reference/reconstructed takes remain beside the log for validation and use with the existing offline renderer; their duplicated data is not part of the compact-storage claim. Its scripted recipe is distinct from the live input format above.

On Unity 6000.4.1f1 / Apple M5, the 54-object test produced **byte-identical states and sound events on both replays**, zero per-pair delay/gain/occlusion differences, and exactly matching non-silent binaural WAVs across 2,028,601 frames. A negative control using a 1% incorrect simulation step produced up to 5.56 m of positional drift and 27 occlusion mismatches. Reproducing the simulation clock matters.

| Movement representation | Bytes in this test |
|---|---:|
| Full state observations | 9,958,728 |
| Full observations, lossless GZIP | 452,915 |
| Input/event recipe JSON | 3,375 |
| Input/event recipe, lossless GZIP | 689 |

These figures exclude shared scene/assets and rendered audio. Prepared source PCM plus acoustic geometry cost another 1,459,124 bytes in this fixture; the source Unity scene and its visual assets are also required by the experimental driver. Both GZIP payloads were round-trip verified. Results and expanded takes: `Recordings/compact-experiment-20261005-053633-994/`. The implementation is [CompactReplayChecks.cs](Assets/Tests/Editor/CompactReplayChecks.cs). This demonstrates repeatability on the tested build/platform; it does not establish cross-platform determinism or historical replay after code/scene changes.

## Record a scene

Open `Assets/Scenes/TrainStation.unity` and enter Play. Recording, the train and scene audio start automatically. Right-click in Game view to walk; **WASD + mouse** moves/looks, **Space** jumps, and **Esc** releases the cursor and reveals mouse sensitivity.

- **E** opens/closes the waiting-room door when nearby. Walk onto and off the train normally.
- **R**, or **Save run**, finishes the recording that began at scene initialization. Walking and the train continue while audio drains; live preview resumes afterward. Re-enter Play for another recording.

Takes are saved under `Recordings/station-<UTC time>`. Wait until **Save run** becomes **Saved** for a fully drained monitored output; stopping Play still finalizes the input/reference data automatically. Replay, source diagnostics and render settings stay on the simulation/source Inspectors. Use **Tools → Acoustics → Compare audio** for saved Low/High audio and recorded scene views. Each new Play session starts recording from the same authored initialization.

The scene uses the GPU direct renderer, two independent sources (the original train recording and room chime), and one walking receiver. That receiver represents one head and produces left/right ear channels. Extra sources/listeners remain supported through the manager, without adding unused ones to this demo. Use headphones and turn your head to hear the direction change. Select `Acoustic simulation` to edit the manager and `StationDemo` settings; **Direct Settings → Listener Output** switches between Binaural and the Mono baseline. Select it in Scene view to inspect the box proxies. `Assets/Prefabs/Station/Platform.prefab` and `Train.prefab` are ordinary editable assemblies; the FBX train remains visible in Edit mode. Acoustic geometry explicitly uses colliders; small visual details are excluded. Moving locomotive, carriage and player roots are registered along with their acoustic children so visual replay follows the same motion.

The train follows a **215.67 m closed oval around the entire station**. Its two straights lie outside the platform and waiting room, with the end turns beyond the station buildings. Select **Train loop** to edit its position/yaw, straight length and turn radius at unit scale, then **Rebuild rails / align train** before Play. Both the baked rail mesh and vehicle motion use the same distance-based curve. The carriage follows 7.2 m behind along the route. `StationDemo` exposes cruise speed (12 m/s), acceleration/braking (3 m/s²), a 10-second station dwell and starting/stop distance. It waits 10 seconds at startup, departs automatically, brakes to rest at the same stop each lap and repeats after the station dwell. The loop seam has continuous position/heading; there is no teleport or instantaneous reversal. Input playback reruns the matching scene and code from initialization; offline comparisons continue to use the retained trajectory.

The yellow steps lead to a platform level with the carriage floor. When the train stops, walk across the narrow gap through the open side; walk back out at the station. WASD/mouse and jumping work aboard, with normal collisions against the benches, walls and roof. The CharacterController follows the floor’s translation and turns without changing the player's parent. There is no boarding key, teleport, snap point or separate boarding state. Movement is captured continuously, including entering and leaving the carriage. Autonomous passenger AI is not implemented. Record a new take after changing scene geometry.

Mouse sensitivity defaults to **180**, up from 100 (1.8× at the old 60 Hz calibration). **Esc → Mouse sensitivity** adjusts it during Play. Mouse deltas no longer multiply by frame duration, keeping the same physical turn consistent across render rates. Inspector changes persist; the Play-mode slider follows normal Unity Play-mode lifetime.

The original `train_horn.mp3` recording and waiting-room chime loop automatically after capture opens at initialization. Saving finishes that audio session; the clips restart when live preview resumes. Input replay uses the recorded playback events. Source clips/gain/loop settings are authored on ordinary `RaytracedAudioSource` components; their context-menu Play/Stop and `SetGain` API still produce recorded source events. Detailed performance counters remain on the manager Inspector.

`LoopTrack.asset` bends the existing Kenney rail/tie mesh along the route and retains its material; track decoration remains outside acoustic geometry.

The scene audit removed Platform 2, obsolete crossing/edge markings, redundant signs and route dashes, the reference receiver/marker, the unused synthesized engine source/asset, and the boarding teleport markers/sign. The saved scene has **121 GameObjects**, down from 211; the builder and prefabs use the same reduced layout. The boarding steps, shelter, waiting room and sliding door remain useful walking/occlusion geometry. Record fresh takes after the roster change.

The current station has **53 registered objects, 564 triangles, two sources and one listener**. Older recordings still render audio offline; a different roster requires its original scene version for visual replay.

**Tools → Acoustics → Build train station** regenerates the station scene, generated materials and prefabs; preserve custom edits before rebuilding. It adds no packages. The four-second room chime is generated by `TakeDemo.Station.cs`; the existing horn remains a preview asset whose provenance must be checked before dataset distribution.

**Measured preview constraint:** an earlier background Editor run was roughly 10 FPS. Monitor Lead Samples is 24000 (0.544 seconds at 44.1 kHz). A six-second Metal preview with both sources and the moving train had zero underrun, late, overflow or over-budget callback samples; the 8192-sample lead failed. This is buffered preview, not a validated low-latency configuration. Check counters when reducing the buffer or increasing scene complexity.

The earlier 270-check suite included collider-only proxy capture and component removal/replacement. That station verification captured a ten-second walk with 36 registered objects, 360 triangles, two sources and two listeners. Door closed/open gains were 0 / 0.0909983; CPU/Metal path disagreement was below 8e-9, and replay position error below 6e-8 metres. Observations were about 10 Hz, with a maximum 6144-sample gap: finer offline evaluations interpolate recorded motion, not new physical observations. Coarse/fine offline exports (2048/256 response steps) completed for both listeners from the same take. These are direct-model comparison outputs, not approved training targets. Local inputs, reports and screenshots are under `.context/station-build/` and `Recordings/`.

The binaural milestone passed 292 CPU checks plus Metal fixtures: ear timing/level, head rotation, resampling, channel routing, capture and shared live/offline output. An eight-second foreground Editor station recording on Apple M5 used two sources/listeners, a moving train and rotating head. At 44.1 kHz with a 512-sample response step, it had no underrun, late, overflow, channel-mismatch or over-budget callback samples; maximum observation gap was 1025 samples, GPU completion 7.83 ms and callback 0.194 ms. Captured stereo matched offline output exactly across 353281 frames after removing the 24000-frame lead. Offline export with a 2048-frame tail took 1.02 seconds, including 0.56 seconds of DSP. These are one-run direct-model measurements, with reports under `.context/hrtf/`; they do not establish low-latency or reflection performance.

The loop/source-control milestone passed **343 checks**. A full-lap recording with temporarily increased speed exercised native UI gain, one-shot/looped Play and Stop; all five engine events were retained exactly. Carriage route error stayed below 4.3 µm, and replay position error below 0.5 µm with no observed rotation error. Sampling the full route every 0.5 m produced 37,315 collider comparisons with no train/carriage/scenery penetration. That ~11-second Metal run had zero underrun/late/overflow/callback-budget misses at the existing 24000-frame lead; this does not resolve the earlier workload-dependent stalls. Those measurements used the earlier 6 m/s and 1.5 m/s² scene defaults. Captured stereo also matched the same-settings offline render exactly across 488,449 frames after removing the declared lead, including the source-control events. The engine-only opening was non-silent. Low/High exports are available for both receivers. Reports are under `.context/station-loop/`.

The station-boarding milestone passed **359 checks**. A 26.5-second recording included walking up the steps, boarding, a full lap, an exact platform stop and alighting. All 57 objects replayed within 3 µm (zero observed angular error); player/head each retained exactly two pose cuts. Route sampling made 221,184 collider comparisons with no intersections. The Metal capture had zero underrun/late/overflow/callback-budget misses at the existing 24000-frame lead. Captured stereo matched offline output exactly across 1,169,408 frames after removing that lead. Low/High renders for both receivers completed in 2.95/4.68 seconds. Reports and scripts are under `.context/station-boarding/`; these direct-model outputs are comparison data, not approved reflection-training targets.

The walking-carriage check passed the existing **359 checks**, plus native collision checks through a full route with 10–120 Hz movement steps. Standing drift was below 0.11 mm; the player could traverse the aisle, stop against benches/end walls and jump against the roof. A new eight-second recorded walk during departure reached 12 m/s and replayed within 2 µm with no observed angular error. These are controller/replay checks, not audio performance measurements. Scripts and screenshots are under `.context/train-walking/`.

The official `com.unity.pipeline` editor bridge starts with this project. `unity status --json` checks the connection; `unity command list_open_scenes --project-path "$PWD" --json` reads the open scene.

For a working starting scene, choose **Tools → Acoustics → Create 3D walkthrough**. It creates native walls, a moving occluder, a walking listener and an emitter using the existing audio/material assets. Save the scene, enter Play, select the simulation manager, record, and start the emitter through its context menu. WASD/mouse move the listener. The scene stays editable through ordinary Unity components.

1. Add/use a `RayTraceManager` on a separate, always-active GameObject. Set **Take Root** to the root of the objects you want to capture. Enable **Use Take Renderer** for the shared direct path and assign an `AudioManager`. **Direct Shader** can be empty for CPU, or `Assets/Shaders/Direct.compute` for GPU. The legacy Raytrace Shader field is separate. Leave Use Take Renderer disabled for silent capture; legacy playback must be stopped before starting a take.
2. Create ordinary Unity objects. For 3D, add `AcousticSurface` to acoustic geometry: Auto uses `MeshFilter.sharedMesh` first, then a supported collider. For 2D, choose **Planar2D**, use Collider2D shapes in the XY plane, and add `AcousticSurface`. Existing entries in **Obstacle Objects** are also registered.
3. Assign an `AudioMaterial` on each surface, or **Take Default Material** as a fallback. Optional submesh overrides share one material table. `AudioMaterial.energyCurve` accepts any number of ordered frequency knots; an empty curve preserves the existing material. Coefficients use linear interpolation in Hz and clamp outside the authored range. The direct baseline records these curves but treats all geometry as opaque.
4. Add `RaytracedAudioSource` components with clips and a manager reference under Take Root, or list them explicitly in **Take Sources**. Clips must be readable/preloaded (Decompress On Load). Preparation for the new path happens once when capture/preview starts. Assign the walking camera/player transform to **Listener**, or list receiver transforms in **Take Listeners**.
5. In Edit mode, click **Register scene objects / persist IDs**, then save the scene. Review the resulting object/source lists. IDs live on the manager; don't register again during a take.
6. Enter Play mode, click **Record take**, and choose an output parent outside `Assets` (for example `Recordings`, already gitignored). Use existing controllers, animation or physics to move objects. Start/stop sounds through `RaytracedAudioSource.Play()` / `.Stop()`; the component context menu exposes both. Start voices after recording begins.
7. Click **Stop and save take**, then wait for **Idle** before leaving Play mode so the monitored output finishes recording. **Choose saved take** reopens it later. **Replay recorded poses** applies recorded world poses; **Pause replay** lets you edit **Replay Seconds** to scrub. Stop restores the original scene poses, activity and suspended movement components.

**Start live preview** runs the same pipeline without writing a take. Stop preview before starting a recording. All registered sources can play through the new renderer; the selected **Monitor Listener** feeds the output device. Offline output includes every listener. Live monitoring automatically selects the actual device sample rate when a session starts; silent capture keeps its configured rate. Settings are fixed when the session starts.

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

Live GPU rendering batches response samples across the queued, known motion intervals and uses [Unity asynchronous readback](https://docs.unity3d.com/2022.3/Documentation/ScriptReference/Rendering.AsyncGPUReadback.html). One in-flight batch owns its pose/response buffers until completion. **Maximum Responses Per Batch** bounds that storage; **Maximum Pending Observations** bounds queued motion intervals. A full queue stops the session visibly. Finishing/cancelling waits for owned requests before releasing their buffers. Offline GPU runs use measured synchronous readback. Completion wall time includes scheduling and readback, not isolated GPU execution time.

The manager exposes **Idle**, **Live**, **Recording**, **Replay** and **Draining** states. Legacy playback/precomputation cannot enter an active take session. Direct audio initializes only its bounded queues and output driver; legacy buffers/debug textures are allocated when their callers need them.

**Maximum Delay Seconds** declares the source-history horizon; a response beyond it fails explicitly. Completed intervals discard older voice/activity history while retaining the boundary state and original loop phase. **Maximum Signal Events** bounds retained history per source and catches excessive event density. `live.json` includes retained-event counts, maximum pending observations, completion latency and a source/listener-pair × primitive test upper bound. The upper bound is workload accounting, not a measured hit count. CPU remains useful for small workloads.

## Render offline

On the manager Inspector, choose a saved take and click **Render saved take to responses + WAV**. **Response Step** controls how often recorded geometry/poses are evaluated. Smaller steps cost more and reduce motion quantization. This is a time-resolution control, not a ray count or FFT size. **Tail** adds output samples after capture stops so already-emitted sound can arrive; no new source audio is emitted past the end of the take.

The baseline computes source-major/listener-minor direct delays, pressure gains and arrival directions, with 1/r spreading above the configured minimum distance and two-sided opaque intersections. Planar mode uses 2D visibility with the same point-source spreading law. Responses are held for each update interval. **Transition Samples**, from zero through the response step, optionally crossfades the old and new delayed signal taps. Teleports reset that transition at the next response update. Live and offline use the same mixer, including events within a response interval. Source events and source activation gate emission at their recorded sample, including events inside an interval. Geometry and listener activity follow the response grid. Retarded-time propagation through moving boundaries, continuous Doppler, reflections, diffraction and phase-dependent materials are not modeled.

**Delay Filter** selects linear interpolation or a normalized windowed-sinc fractional-delay FIR. **Sinc Radius** controls the FIR half-width and adds that many samples of causal DSP latency, reported as `dspLatencySamples`. Coefficients are computed per response, and live/offline share the same taps. This improves fractional-delay reconstruction; it does not change transport physics. Tail length must cover the propagation delay and desired filter tail. FFT/STFT convolution is still a separate future implementation.

**Listener Output → Binaural** applies a measured left/right HRIR to each direct path, using arrival direction in the recorded head's local frame (+X right, +Y up, +Z forward). The bundled MIT KEMAR compact dataset has 710 directions and 128 taps/ear at 44.1 kHz; other rates resample the filters once during preparation. Nearest measured directions use the existing response crossfade. The FIR preserves interaural timing and gain; geometric delay is added once, without a second ear-offset delay. Tail support must also include the HRIR. `dspLatencySamples` describes the fractional-delay stage, not direction-dependent HRIR arrival timing. This is a generic far-field mannequin model with limited elevation coverage, not a calibrated physical eardrum simulation. See [data attribution and limitations](Assets/Resources/Acoustics/ATTRIBUTION.md).

One listener stays one scene object and yields one WAV: mono or interleaved left/right. Source clips remain mono. Live binaural output requires Unity's stereo speaker configuration. Queue capacities, lead and counters always count sample frames, regardless of channel count. Old take files can render either output mode. Batch settings use `"listenerOutput": 1` for binaural, or omit it for mono; comparisons must use the same mode. Run/output manifests identify channel count, HRTF hash/taps and DSP contract. Direct response and output-capture manifests now use format version 2; their response records retain world arrival directions for later reconstruction work.

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

## Compare audio and watch the take

Open **Tools → Acoustics → Compare audio**, or the manager's **Open audio comparison window** button. Choose a completed `comparison.json`, or expand **Render a saved take at two response budgets**, choose a take and render Low/High. The Inspector's existing comparison button now opens this window when rendering completes. Stop Play mode for rendering and audition. Existing exports remain usable; new reports include stable listener IDs/names and a relative link to the input take.

Select a listener, set the time range and click **Load range**. **Play/Pause**, **Low**, **High**, **Difference** and the shared cursor operate on the same sample-frame timeline. Clicking either waveform/spectrogram seeks both sides. A/B changes restart preview at the preserved sample; they are not guaranteed gapless. One gain applies to both ears/sides. Audition adds a 5 ms edge fade; original WAVs and metrics remain unchanged. Difference is Low minus High. Preview uses Unity's internal Editor audio API behind a signature check; WAV inspection remains available if that API changes.

The selected listener/range is bounded to 30 seconds and 2,000,000 sample frames; full recordings/listeners are not loaded into RAM together. Waveforms use min/max envelopes and a common amplitude scale. Spectrograms use a configurable power-of-two FFT (64–16384), periodic Hann window, one-sided peak-amplitude dBFS, a common color range and linear frequency. The actual displayed hop is shown: time columns are decimated to at most 512 and frequency rows retain the strongest bin within each row. Final windows zero-pad. These are display plots, not training tensors or a new runtime STFT/convolver.

**Recorded scene views** adds listener-perspective and top-down playback at the shared audio cursor. Open the original scene with matching registered IDs; **Reload visuals** copies its current static meshes, text, lights and solid 3D colliders into an isolated preview scene. Recorded poses drive that copy; gameplay, rigidbody simulation and audio never run there. Visual assets and lighting are not captured video: later scene edits can change appearance, and unrecorded deformation/animation is not reproduced. The view holds the final pose during the audio tail. Camera rendering is capped at 30 updates/second; the device's eventual sound/display latency is not calibrated.

- In the listener view, hold **right mouse + WASD**, **Q/E** for vertical movement and **Shift** for faster free-camera inspection. The inspection camera collides with authored ground/walls using a small spherical CharacterController. **Follow listener** restores the exact recorded head pose.
- In the top view, **scroll** zooms and **middle-drag** pans. **Frame top view** fits registered object positions. Markers identify sources (`S`) and receivers (`L`).
- Free-camera movement changes the view; the compared audio continues to use the saved receiver trajectory. Record a new live walk to test a different listening route. This is interactive pose replay, not MP4 export.

The window shows whole-take RMS/peak/SNR, budgets and wall time. Expand **Timings, recorded deadlines and provenance** for preparation/evaluation/solve/DSP/write costs, working-buffer accounting and input/settings/HRTF identities. Recorded underrun/late/overflow and callback counters come from the matching input take's original live run, not the two offline budgets. Selected WAV hashes, headers, dimensions and run compatibility are checked before audition. Missing scene assets or live diagnostics do not prevent audio comparison.

A station comparison at response steps 2048 / 256 rendered 8.10 seconds of audio in 0.77 / 1.28 seconds on Apple M5/Metal. DSP took 0.476 / 0.558 seconds, while synchronous solve/readback took 0.172 / 0.508 seconds. The coarse run is primarily DSP-bound; finer update cadence increases solve/readback overhead. These are single-run wall costs, not isolated GPU timings. Detailed results are under `.context/comparison/`.

Three subsequent six-second live captures (two sources/listeners, moving train, rotating head, 44.1 kHz, 512-sample response step, 1024-frame device buffer) all completed without underrun, late, overflow, channel-mismatch or over-budget callback samples:

| Lead frames | Lead time | Max GPU completion | Max callback | DSP time over ~6 s |
|---:|---:|---:|---:|---:|
| 4096 | 92.9 ms | 7.71 ms | 0.185 ms | 0.467 s |
| 8192 | 185.8 ms | 20.26 ms | 0.490 ms | 0.502 s |
| 24000 | 544.2 ms | 7.89 ms | 0.099 ms | 0.530 s |

Each run's maximum accepted observation gap was 1025 samples (~23.2 ms). The 4096-frame lead is a candidate for further active-Editor testing, not a general latency guarantee. Earlier background runs missed deadlines with 8192; the saved scene therefore retains 24000. Preparation, observation availability, callback/device buffering and physical propagation are separate contributors. The helper reports sampled `Time.frameCount` intervals, which are not display presentation FPS. Raw captures and profiles are retained for inspection.

The comparison/control milestone passed **319 checks**, including an independent FFT oracle, WAV/report integrity, stereo audition/range handling, isolated pose replay, and camera ground collision/sliding. Native Editor A/B/Difference playback and listener switching were also exercised. A subsequent ten-second keyboard-driven station recording covered 12.43 m, a platform descent/ascent and a jump; recording retained cursor/keyboard control and the measured minimum ground clearance was zero. The train wheels now sit on the rail heads, and player gravity/movement share one CharacterController sweep with no minimum movement threshold. Old takes retain their original recorded placements; record a fresh take to capture scene corrections.

That interactive walking run **did miss audio deadlines** even with the saved 24000-frame lead: 3648 underrun and 3648 late frames (~82.7 ms each, not two independent gaps), maximum accepted observation gap 9215 frames (~209 ms), and maximum GPU completion 287 ms. There were no overflow or callback-budget misses. Earlier quiet profiling runs are not a guarantee under interactive Editor load. Investigate observation/producer stalls and async completion before claiming stable realtime latency. The saved walk, new Low/High comparison, keyboard test and diagnostics are under `Recordings/station-20261003-210252-086/` and `.context/comparison/`.

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
    output.wav        mono/stereo component output, including deadline/mute silence
    callbacks.bin     output sample ranges, channels and callback timing
    applications.bin  response values and scheduled/consumed sample positions
direct-001/
  run.json            method/settings, input hashes, output hashes, timing
  responses.bin       sampled direct delay, pressure gain and arrival direction
  listener-0.wav      mono/stereo float32 audio; one file per listener
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
    Audio/        AudioManager, RaytracedAudioSource, Hrtf
    Demo/         PlayerController, MouseLook, Oscillator, TrainVisualLoader,
                  RaytraceControlPanel, FFTTesting
  Shaders/        Direct.compute, Raytrace2D.compute, AudioConvolve.compute, Common.hlsl
  Audio/          source clips and DefaultMixer
  Materials/      acoustic material assets
  Resources/      bundled KEMAR filters and attribution
  Editor/         TakeEditor, TakeDemo, TakeComparisonWindow + audio/view helpers
  Tests/Editor/   TakeChecks
  Scenes/         editable scenes and their baked lighting
  Prefabs/        project prefabs
  External/       imported train assets
  TextMesh Pro/   imported UI assets
Packages/         Unity dependency manifest
ProjectSettings/  Unity project configuration
docs/             consolidated design contract and historical cleanup results
tools/            reproducible KEMAR importer (Python standard library)
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

The project has migrated to Unity **6000.4.1f1**. The earlier cleanup measurements used an isolated project on the same Unity version with Metal; see the linked report for those measurements. DirectX validation remains outstanding.

Measured results and reproduction details are in [cleanup-results.md](docs/cleanup-results.md).
