# Acoustic ray tracing

Unity **6000.4.1f1 (Unity 6.4)**, C# and HLSL. The shared renderer runs directional acoustic transport in 2D and 3D, live and offline. Compact station recordings retain inputs/events and the matching scene/code identity.

**Implemented now:** editable 3D station and church with asset/voice sources and a binaural listener, train service with station stops and walking aboard, sliding door and walkthrough controls, strict take capture, live directional spectral monitoring, pose replay/scrubbing, CPU/HLSL direct solvers for 2D/3D, offline WAV/response export, bounded asynchronous GPU batches, captured output PCM/timing, configurable fractional-delay FIR, binaural headphone rendering, and a native comparison window with synchronized audio, plots and recorded scene views. Live and offline share interpolation, the source timeline and scheduling. Raytraced DSP has a GPU implementation and a selectable CPU reference. The direct/opaque baseline still has matching CPU/GPU implementations. The GPU spectral preview adds reflections, scattering, transmission/refraction, material/air attenuation and deterministic sampling of signed edge diffraction, with per-arrival head orientation.

**Still planned:** LakeForest, validated/converged indirect transport, training-example curation and learned reconstruction. Church, microphone sources, LAN/Relay multiplayer and offline client replay are implemented. All five scenes use the current renderer; the obsolete 2D renderer is retained in Git history. The geometric preview is not a validated high-fidelity training teacher.

**Audit repairs:** [current implementation and remaining limits](docs/simulation-contracts.md#renderer-repairs-2026-10-05). CPU IR synthesis uses partitioned linear convolution; GPU IR evaluates equivalent delayed short filters; the optional spectral synthesis uses continuous path clocks for Doppler and short spectral/ear filters. Both retain sample-by-sample response fades, complete per-frequency ray budgets and first-order image-source reflections. Compact recording failures invalidate the save. Independent convolution/reflection/replay checks cover the repaired errors; diffuse transport and diffraction still need physical validation before training.

[The consolidated design contract](docs/simulation-contracts.md) covers scene/assets, Unity/HLSL data, live/offline execution, IR/complex/spectrogram reconstruction, temporal upscaling, training, multiplayer and acceptance gates. This README covers running the implemented code; [cleanup results](docs/cleanup-results.md) records current station benchmarks and historical measurements.

The [next-work specification](docs/simulation-contracts.md#runtime-cleanup-and-shared-scene-session) covers runtime cleanup, shared scene sessions, Church/LakeForest, microphone recording and multiplayer voice. Clips, microphones and decoded remote voices share the source timeline and independent CPU/GPU DSP implementations. Shared `SceneSession`, compact scene replay, player/speaker prefabs and the editable Church scene are implemented. LakeForest and model training remain planned. [Replay scope](docs/simulation-contracts.md#recordings-entry-point-and-replay-scope-planned) keeps compact runs dependent on their matching scene/build and defers snapshots/checkpoints/caches.

Open **Assets/Scenes/Church.unity** and press Play. Walk with WASD/mouse, jump with Space, use **E** near the two entrance doors, **R** to save and **Esc** to release the cursor and expose microphone / Host / Join controls. Click the Game view to resume walking. The imported medieval church has relief-textured stonework, stained glass, carved pews, a detailed altar and organ façade, candle-ring chandeliers and cool window lighting in a darker dusk setting. Walkable ground extends beyond the courtyard. Tower bells and an original sampled-organ passage play automatically on separate 30-second loops. Walk into the courtyard to hear the bells; the nave is screened by masonry. The scene shares its player and recording/replay path with TrainStation. **Tools → Acoustics → Build church** rebuilds the scene and replaces edits, so use the existing scene for normal work.

Church uses separate illustrative stone/wood/glass/ground acoustic materials, **2 rays per frequency bin, 32 bounces, GPU spectral synthesis and mono output** for speakers. Choose Binaural for headphones. Experimental diffraction remains off. Architecture and doors use a simplified collision/acoustic mesh; tiny decorative details are visual only. This is an editable acoustic demo, not measured church acoustics or an approved training reference. The recorded bell includes its original outdoor ambience. The organ uses samples from an organ emulator with no additional reverb. The dry diagnostic clap remains available separately. [Asset sources, CC0 licenses and processing](Assets/External/Church/README.md). No new runtime scripts or packages were added.

The [multiplayer capture/training plan](docs/simulation-contracts.md#turning-multiplayer-captures-into-training-examples) saves each client's actual presented poses plus separate decoded, pre-acoustic voice streams and their availability/loss timing. Low/High will use identical received speech; recording a mixed output alone cannot supply those training inputs. The streamed-take format now saves those inputs; it does not automatically enroll speech in a training dataset.

## Diffraction

The current experimental renderer samples surface and edge interactions inside the same GPU path loop. Paths can visit multiple edges within the bounce budget. Each accepted edge sample carries signed pressure, travel time and arrival direction; summing its delayed contribution creates the complex frequency response in the existing audio pipeline. Moving interactions use the incoming and outgoing path directions for Doppler.

**Diffraction Factor** scales edge pressure: **1** uses the experimental rigid-wedge model, **0** disables edge sampling. Edge and surface events share **Rays Per Bin** and **Max Bounces**; there is no separate shortest-route pass or extra edge-arrival block. Frequency bins share random samples. The report contract is `integrated-edge-preview-v7`. The earlier dominant-edge renderer is preserved at `checkpoint/pre-integrated-diffraction-2026-10-07`.

This remains an unvalidated training teacher. Mixed diffuse/surface weights are approximate; exact specular–edge connections, general impedance wedges and vertex corrections are incomplete. More rays reduce sampling noise but do not establish physical accuracy. Planar geometry represents an infinite extrusion perpendicular to XY with spherical point sources. [Model and limitations](docs/simulation-contracts.md#integrated-edge-trial).

For independent pressure maps behind walls, use the [silent wave-reference comparison](docs/cleanup-results.md#independent-wave-pressure-maps--2026-10-18). It compares the actual GPU transport with a 3D Helmholtz boundary-element solution, checks the reference against an analytic sphere, and exports shared-scale maps, route plots, complex pressures and mesh-refinement measurements.

The optional [planar wave lab](docs/cleanup-results.md#planar-wave-reference--2026-10-18) adds cheap 2D analytic controls and a 2.5D FEM reference matched to `Planar2D`: free space, a barrier and staggered walls. The [contracts and next steps](docs/simulation-contracts.md#wave-reference-suite-research-and-implementation-plan) distinguish implemented checks from future material, scene and training coverage.

The [research experiment protocol](docs/simulation-contracts.md#ray-transport-and-learned-reconstruction-experiment) specifies kernel candidates, staged parameter searches and a 25-scene benchmark. `compare_wave_field.py suite` generates its fixtures; `score` reports balanced complex-pressure error with equal scene-family weights.

The [material/room/motion extension](docs/simulation-contracts.md#materials-enclosed-spaces-and-motion) specifies the next physical cases. A [silent early-reflection control](docs/cleanup-results.md#material-enclosed-room-and-motion-controls--2026-10-18) already checks synthetic absorption and receiver movement in enclosed 2D/3D boxes (`renderer --room-control`).

## Microphone and multiplayer

Use **Church** or **TrainStation**, press Play, then **Esc**. The controls include:

- **Enable / Disable microphone**: opt-in device capture. Set `MicrophoneInput.device` in the Inspector to select a device; an empty name uses the default. Self-monitoring is off; enable `monitorSelf` before Play when needed. Remote players hear your voice through their own local acoustic renderer.
- **Online / join code → Host**: save the current run and start a fresh hosted session through Unity Relay. Use **Copy join code** to share the code; another player opens the same scene/build, enters it and chooses **Join**. Unity Services must be linked and Relay available for that project.
- **LAN / IP address**: the same Host/Join controls use direct UDP instead. Enter the host's LAN IP, or `127.0.0.1` for two processes on one machine. The default port is **7777**, configured on `MultiplayerSession`.
- **Save run [R]**: finalize this client's take and drain pending acoustic output. Choose a completed take under **Replay**, then **Replay selected run** for the current scene to reload it from the beginning, without a server or microphone. Numeric replay fills the existing bounded output queue ahead of playback; it creates no additional cache. **New run** starts a fresh solo capture; the same button becomes **Leave session** while connected and **Return to live** during replay.

The controls hide while walking; a small status line shows recording and microphone activity. Join setup hides after connecting, while the session code and player count remain visible in the controls. Saved-run timestamps use local time in the chooser; recording folder names remain UTC. The R shortcut only operates while walking, so typing a join code cannot save a run.

Both scenes have four authored player slots and separate spawn markers. The host drives the train and accepts nearby door interactions; each player controls their own movement/head. `MultiplayerSession.remoteVoices`, `sharedObjects` and `spawnPoints` are ordinary scene references. The scene builders create these bindings too. Online hosting uses the Relay API in **Multiplayer Services 2.2.4**, encrypted DTLS, and anonymous player authentication. The host forwards dry Opus frames; each receiving client raytraces them for its own ears. Host departure ends the session; matchmaking, host migration and competitive movement validation are not implemented. More players need a measured scene/network budget; two separate processes are the initial acceptance target.

Voice uses pinned **Netcode for GameObjects 2.7.0**, **Unity Transport 2.6.0**, and managed **Concentus 2.2.2** (Opus). Speech is mono 48 kHz, 20 ms packets at 32 kbit/s. Packets carry the speaker/epoch, a 64-bit source sample index and its estimated session capture time. Unordered unreliable delivery preserves reorderable voice frames. The receiver reorders them, ignores duplicates/late replacements, removes codec delay, preserves the final valid samples, performs Opus loss concealment and smoothly resamples into the acoustic device clock. Microphone restarts retain pending speech from the previous epoch. The default remote-pose/voice presentation delay is 100 ms, in addition to the renderer/device latency. Late voice starts receive a fresh prebuffer; the take records the adjusted delay. The Mac Editor microphone capture/lifecycle test passes. Microphone timing uses captured cursor/sample observations; pose snapshots preserve their original session timestamps. Physical ADC latency is unmeasured, and echo cancellation is not implemented. The station retains its 16,384-sample output lead: lower-buffer deadline tests failed, so responsive conversation remains a performance goal. Two-player LAN and actual Relay routing/capture/replay pass on this Mac. Four-player same-Mac stress tests overload renderer/recording queues; four independent machines and longer conversations still need hardware validation. See [measured results](docs/cleanup-results.md#timestamped-voice-and-relay--2026-10-13). Use headphones.

Solo runs retain compact movement inputs. Enabling the microphone additionally saves its canonical source samples. Network runs save each client's actual presented numeric poses, source events, geometry/clip inputs and separate **pre-acoustic** source streams. Monitored output WAVs are optional (`Retain Full Reference`). Slots are retained after a participant leaves and are not reused inside that session, so old voice history cannot become another speaker's input.

Take format **v3** adds `source-<index>.f32` (mono float32 at the take rate), `source-blocks.bin` (source, absolute first sample, count, availability sample, data/silence/missing/concealed flags), plus session/source-epoch metadata and hashes in `take.json`. Silent ranges use sparse files. The saved bytes are exactly those committed to source history: local pre-encode microphone audio or that client's decoded remote voice. Opus compresses the network traffic; saved PCM is lossless and seekable. Older clip-only take versions remain readable.

Both scenes use a 1,024-sample acoustic response step (~23 ms at 44.1 kHz) and FFT 1,024 / hop 256. Station uses five bounces; the Church echo preset uses 32. Two-process Church and TrainStation capture/replay passed at this cadence with the earlier five-bounce preset after fixing repeated GPU source-buffer uploads and replay backpressure. The imported Church has offline audio and walking checks; its multiplayer realtime deadlines have not been remeasured. Earlier queue failures remain documented in the benchmark results. Playback lead stays 16,384 samples (~372 ms); all settings remain editable on the manager. This short test does not establish low-latency production voice.

The existing offline Low/High render/comparison tools consume these same source files and poses. Keep all client views of one session in the same training split, preserve concealment/gap flags, and select recordings explicitly for training. Physical microphone audio can already contain real-room reverb and device processing; it is not automatically clean speech or a physics reference.

[Current audio/multiplayer checks](docs/cleanup-results.md#audio-continuity-and-multiplayer-verification--2026-10-07) verify four-player voice routing and replay, but also reproduce output gaps and GPU overload when several clients share one Mac. Functional connectivity does not yet establish artifact-free realtime voice.

## Compact recording and replay

TrainStation and Church record automatically from scene initialization, before movement or sound. **R / Save run** finishes the recording; stopping Play also finalizes the inputs. Re-enter Play to begin another run. There is no duration cap, seek, checkpoint or playback cache.

Compact scene recordings contain `input.json`, `input.bin`, `input-observations.bin`, `events.bin`, `take.json` and small `live.json` diagnostics. Inputs store held movement changes and one-shot look/jump/interaction values at fixed ticks. The observation map preserves the original relationship between simulation ticks and audio samples. Geometry and asset-clip PCM are hashed during preparation but are not written to compact recordings. Streamed microphone/voice PCM and gap metadata are always retained when present. Full poses and monitored WAVs are also omitted.

The saved scene supplies static geometry, initial state and source assets. Editor recordings hash the contents and relative paths under `Assets`, `Packages` and `ProjectSettings`, so matching source can be rebuilt in another checkout. They also identify the Git commit and uncommitted changes. Engine/platform, fixed step, motion settings and audio rate remain checked; Player recordings and older recordings use their original runtime module identity. Preserve the matching repository revision; automatic Git checkout is not implemented. This adapter supports the shared walking controller, sliding doors and station train, not arbitrary runtime edits or rigid-body simulations. Authored source/surface enabled states are preserved. Unsupported runtime activation changes fail the recording explicitly. A save is successful only after both acoustic metadata and the required input recorder finalize.

Select **Acoustic simulation → Choose saved take → Replay saved run from beginning** after saving. Playback reloads the matching scene and runs recorded inputs and sound events from zero. Live playback requires a matching output sample rate; offline reconstruction uses the recorded rate without an audio device. **Export input-only run** copies compact replay files and required source streams to a separate folder, including from older full solo recordings. Network recordings use their numeric poses.

Enable **SceneSession → Retain Full Reference** before Play only when comparing against original poses or monitored audio. This adds geometry, source PCM, full observations and captured output to that run. Existing recordings are left intact. Use Save and wait for audio to drain before stopping Play when validating monitored output.

**Tools → Acoustics → Reconstruct recorded scene inputs** works in Edit mode. It creates an ordinary acoustic take in an isolated preview physics scene and reports exact state/event agreement when a full reference is available. It does not modify the open scene or enter Play mode. The Inspector's offline render/comparison actions and `TakeBatch.Render` also accept compact recordings directly; they reconstruct into a sibling `<output>-inputs` directory, then render the requested output. These generated files are offline outputs, not mandatory recording or playback data.

Open **Window → General → Test Runner → EditMode → Acoustics.Tests**. Checks are independently selectable; the station end-to-end tests enter Play mode themselves. See [Checks](#checks) for categories, automation and the explicit realtime deadline test.

Current and historical [validation results](docs/cleanup-results.md) cover three-lap replay, different measured frame cadences, 1,417 core checks plus GPU checks, reconstruction in a fresh Editor from a clean Git checkout, and a standalone Mac recording. They establish repeatability on the tested build/platform, not general physics determinism.

## Record a scene

Open `Assets/Scenes/TrainStation.unity` and enter Play. Recording, walking controls, the train and scene audio start automatically. **WASD + mouse** moves/looks, **Space** jumps, and **Esc** releases the cursor and reveals mouse sensitivity. Click inside the scene to resume walking; clicking the UI leaves the cursor available for its controls.

- **E** opens/closes the waiting-room door when nearby. Walk onto and off the train normally.
- **R** while walking, or **Save run** in the controls, finishes the recording that began at scene initialization. Walking and the train continue while audio drains; live preview resumes afterward. Use **New run** for another recording.

Editor takes are saved under `Recordings/<scene>-<UTC time>`. Wait until the status says **Run saved** before starting replay. Monitored output is only saved with Retain Full Reference enabled; stopping Play still finalizes the input data automatically. Replay, source diagnostics and render settings stay on the simulation/source Inspectors. Use **Tools → Acoustics → Compare audio** for saved Low/High audio and recorded scene views. Each new Play session starts recording from the same authored initialization.

The station uses GPU raytraced transport with IR synthesis, two asset sources (the original train recording and room chime), four reserved voice slots, and one walking receiver. That receiver represents one head and produces left/right ear channels. Extra sources/listeners remain supported through the manager. Never-used voice slots and muted self-voice pairs skip acoustic tracing. Use headphones and turn your head to hear the direction change. Select `Acoustic simulation` to edit the manager and `StationDemo` settings; **Direct Settings → Listener Output** switches between Binaural and the Mono baseline. Select it in Scene view to inspect the box proxies. `Assets/Prefabs/Station/Platform.prefab` and `Train.prefab` are ordinary editable assemblies; the FBX train remains visible in Edit mode. Acoustic geometry explicitly uses colliders; small visual details are excluded. Moving locomotive, carriage and player roots are registered along with their acoustic children so visual replay follows the same motion.

The train follows a **215.67 m closed oval around the entire station**. Its two straights lie outside the platform and waiting room, with the end turns beyond the station buildings. Select **Train loop** to edit its position/yaw, straight length and turn radius at unit scale, then **Rebuild rails / align train** before Play. Both the baked rail mesh and vehicle motion use the same distance-based curve. The carriage follows 7.2 m behind along the route. `StationDemo` exposes cruise speed (12 m/s), acceleration/braking (3 m/s²), a 10-second station dwell and starting/stop distance. It waits 10 seconds at startup, departs automatically, brakes to rest at the same stop each lap and repeats after the station dwell. The loop seam has continuous position/heading; there is no teleport or instantaneous reversal. Input playback reruns the matching scene and code from initialization; offline comparisons reconstruct the trajectory from those inputs.

The yellow steps lead to a platform level with the carriage floor. When the train stops, walk across the narrow gap through the open side; walk back out at the station. WASD/mouse and jumping work aboard, with normal collisions against the benches, walls and roof. The CharacterController follows the floor’s translation and turns without changing the player's parent. There is no boarding key, teleport, snap point or separate boarding state. Movement is captured continuously, including entering and leaving the carriage. Autonomous passenger AI is not implemented. Record a new take after changing scene geometry.

Mouse sensitivity defaults to **180**, up from 100 (1.8× at the old 60 Hz calibration). **Esc → Mouse sensitivity** adjusts it during Play. Mouse deltas no longer multiply by frame duration, keeping the same physical turn consistent across render rates. Inspector changes persist; the Play-mode slider follows normal Unity Play-mode lifetime.

The original `train_horn.mp3` recording and waiting-room chime loop automatically after capture opens at initialization. Saving finishes that audio session; the clips restart when live preview resumes. Input replay uses the recorded playback events. Source clips/gain/loop settings are authored on ordinary `RaytracedAudioSource` components; their context-menu Play/Stop and `SetGain` API still produce recorded source events. Detailed performance counters remain on the manager Inspector.

`LoopTrack.asset` bends the existing Kenney rail/tie mesh along the route and retains its material; track decoration remains outside acoustic geometry.

The scene audit removed Platform 2, obsolete crossing/edge markings, redundant signs and route dashes, the reference receiver/marker, the unused synthesized engine source/asset, and the boarding teleport markers/sign. That audit reduced the scene from 211 to 121 GameObjects, before the voice/player slots were added; the builder and prefabs use the same reduced layout. The boarding steps, shelter, waiting room and sliding door remain useful walking/occlusion geometry. Record fresh takes after the roster change.

With voice slots, the current station has **60 registered objects, 564 triangles, six sources and one listener**. Church has **52 objects, 12,761 triangles, six sources and one listener** (bell, organ, local microphone and three remote voices). Older recordings still render audio offline; a different roster requires its original scene version for visual replay.

**Tools → Acoustics → Build train station** regenerates the station scene, generated materials and prefabs; preserve custom edits before rebuilding. It adds no packages. The four-second room chime is generated by `TakeDemo.Station.cs`; the existing horn remains a preview asset whose provenance must be checked before dataset distribution.

The station currently uses **GPU spectral transport and GPU audio synthesis**, a **16,384-sample lead**, FFT 1024 and hop 256. At 44.1 kHz the lead alone is 371.5 ms. Earlier CPU defaults and 8,192/24,000-sample measurements below are historical direct-model results. Inspect `live.json` for current deadline misses; a large lead is not proof of low-latency realtime operation.

The earlier 270-check suite included collider-only proxy capture and component removal/replacement. That station verification captured a ten-second walk with 36 registered objects, 360 triangles, two sources and two listeners. Door closed/open gains were 0 / 0.0909983; CPU/Metal path disagreement was below 8e-9, and replay position error below 6e-8 metres. Observations were about 10 Hz, with a maximum 6144-sample gap: finer offline evaluations interpolate recorded motion, not new physical observations. Coarse/fine offline exports (2048/256 response steps) completed for both listeners from the same take. These are direct-model comparison outputs, not approved training targets. Local inputs, reports and screenshots are under `.context/station-build/` and `Recordings/`.

The binaural milestone passed 292 CPU checks plus Metal fixtures: ear timing/level, head rotation, resampling, channel routing, capture and shared live/offline output. An eight-second foreground Editor station recording on Apple M5 used two sources/listeners, a moving train and rotating head. At 44.1 kHz with a 512-sample response step, it had no underrun, late, overflow, channel-mismatch or over-budget callback samples; maximum observation gap was 1025 samples, GPU completion 7.83 ms and callback 0.194 ms. Captured stereo matched offline output exactly across 353281 frames after removing the 24000-frame lead. Offline export with a 2048-frame tail took 1.02 seconds, including 0.56 seconds of DSP. These are one-run direct-model measurements, with reports under `.context/hrtf/`; they do not establish low-latency or reflection performance.

The loop/source-control milestone passed **343 checks**. A full-lap recording with temporarily increased speed exercised native UI gain, one-shot/looped Play and Stop; all five engine events were retained exactly. Carriage route error stayed below 4.3 µm, and replay position error below 0.5 µm with no observed rotation error. Sampling the full route every 0.5 m produced 37,315 collider comparisons with no train/carriage/scenery penetration. That ~11-second Metal run had zero underrun/late/overflow/callback-budget misses at the existing 24000-frame lead; this does not resolve the earlier workload-dependent stalls. Those measurements used the earlier 6 m/s and 1.5 m/s² scene defaults. Captured stereo also matched the same-settings offline render exactly across 488,449 frames after removing the declared lead, including the source-control events. The engine-only opening was non-silent. Low/High exports are available for both receivers. Reports are under `.context/station-loop/`.

The station-boarding milestone passed **359 checks**. A 26.5-second recording included walking up the steps, boarding, a full lap, an exact platform stop and alighting. All 57 objects replayed within 3 µm (zero observed angular error); player/head each retained exactly two pose cuts. Route sampling made 221,184 collider comparisons with no intersections. The Metal capture had zero underrun/late/overflow/callback-budget misses at the existing 24000-frame lead. Captured stereo matched offline output exactly across 1,169,408 frames after removing that lead. Low/High renders for both receivers completed in 2.95/4.68 seconds. Reports and scripts are under `.context/station-boarding/`; these direct-model outputs are comparison data, not approved reflection-training targets.

The walking-carriage check passed the existing **359 checks**, plus native collision checks through a full route with 10–120 Hz movement steps. Standing drift was below 0.11 mm; the player could traverse the aisle, stop against benches/end walls and jump against the roof. A new eight-second recorded walk during departure reached 12 m/s and replayed within 2 µm with no observed angular error. These are controller/replay checks, not audio performance measurements. Scripts and screenshots are under `.context/train-walking/`.

The official `com.unity.pipeline` editor bridge starts with this project. `unity status --json` checks the connection; `unity command list_open_scenes --project-path "$PWD" --json` reads the open scene.

For a working starting scene, choose **Tools → Acoustics → Create 3D walkthrough**. It creates native walls, a moving occluder, a walking listener and an emitter using the existing audio/material assets. Save the scene, enter Play, select the simulation manager, record, and start the emitter through its context menu. WASD/mouse move the listener. The scene stays editable through ordinary Unity components.

1. Add/use a `RayTraceManager` on a separate, always-active GameObject. Set **Take Root** to the root of the objects you want to capture. Enable **Monitor Live** and assign an `AudioManager` for audible output. **Direct Shader** can be empty for CPU, or `Assets/Shaders/Transport.compute` for GPU. Disable Monitor Live for silent capture.
2. Create ordinary Unity objects. For 3D, add `AcousticSurface` to acoustic geometry: Auto uses `MeshFilter.sharedMesh` first, then a supported collider. For 2D, choose **Planar2D**, use Collider2D shapes in the XY plane, and add `AcousticSurface`. List geometry outside Take Root explicitly in **Take Objects**.
3. Assign an `AudioMaterial` on each surface, or **Take Default Material** as a fallback. Optional submesh overrides share one material table. `AudioMaterial.energyCurve` accepts any number of ordered frequency knots; an empty curve preserves the existing material. Coefficients use linear interpolation in Hz and clamp outside the authored range. The direct baseline records these curves but treats all geometry as opaque.
4. Add `RaytracedAudioSource` components with clips and a manager reference under Take Root, or list them explicitly in **Take Sources**. Clips must be readable/preloaded (Decompress On Load). Preparation for the new path happens once when capture/preview starts. List walking camera/player receiver transforms in **Take Listeners**.
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

The first voice model is **one voice per source**. Play restarts it; Stop and Gain change it at their recorded sample. Source PCM is prepared once per unique clip using the existing channel-average mono conversion and linear resampler. All clips are stored at the take's sample rate. This is not a production resampling filter. Native `AudioSource.Play()` is not intercepted. The shared renderer supports all recorded emitters. Recorded events describe accepted source commands; monitored output diagnostics capture their delivered timing.

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

## Inspect 2D ray propagation

Open `Assets/Scenes/SmollRoom.unity` (or Big Room / SampleScene), then **Acoustics → 2D Ray Debugger**. Choose the scene manager, source/listener pair, frequency bin and probe ray count, and click **Capture**. Use **Play** or **Travel ms** to follow propagation. Select a path or click a segment for its material, delay and signed weight; scroll to zoom and drag to pan. The field to the right of the time slider sets the displayed time range in milliseconds.

This captures a frozen geometry probe from the actual GPU transport, including direct, mirror, surface and edge paths. Scene motion is paused for the probe; no audio playback is required. Capturing again picks up your scene edits. Rejected proposals are optional dotted lines. Ordinary playback has no debug buffers or readback. [Capture layout and limitations](docs/simulation-contracts.md#2d-propagation-debugger).

## Live scheduling and diagnostics

Live rendering waits for the next observed pose before interpolating the interval, then renders only audio whose inputs are known. It does not extrapolate future movement. **Monitor Lead Samples** gives that producer time to run before playback; **Monitor Capacity Samples** bounds the PCM queue. Missed deadlines produce silence at the correct timestamp and late PCM is dropped rather than shifted later. Increase lead only if the extra latency is acceptable; use the counters to identify missed deadlines.

The audio callback copies PCM through a single-producer/single-consumer queue, without locks, logging, file I/O or managed allocations. Scene observation, validation and renderer scheduling stay on the control thread. `live.json` records validation/solve/DSP costs, observation gaps, queue underruns/overflow/late samples, callback overruns and maximum callback duration. These counters are sampled when the producer stops; a queued tail may still play afterward. Full diagnostic recording also writes `monitor/output.wav`, `callbacks.bin`, `applications.bin` and `output.json`. Compact station recording omits these. This preserves the component output, including mute/underrun silence, before downstream Unity mixing or device processing. Callback records carry sample positions, channel counts, monotonic timing and the DSP clock; the output manifest records the take clock origin. Application records carry scheduled output samples, the consumed-sample cursor and response values. A bounded callback-to-main-thread queue moves all file work outside the callback. Overflow or interrupted shutdown leaves output capture incomplete; input-take validity is separate.

Unity Profiler markers `Acoustics.ValidateScene`, `Acoustics.SolveTransport` and `Acoustics.MixPCM` label scene validation, transport submission and CPU-reference mixing. These are CPU scopes, not GPU execution timers.

GPU command markers `Acoustics.GPU.Direct`, `Trace`, `Compare`, `Group`, `Filters`, `InputFFT`, `Mix` and `PCM` separate the transport and audio stages. `InputFFT` includes source-time resampling; `Mix` includes per-ear reduction and inverse FFT. Use the GPU Profiler on supported backends, or Metal tools on macOS. Metal may combine several labeled stages in one encoder: its duration is not the duration of its first label. [Unity GPU profiling support](https://docs.unity.com/en-us/engine/6000.3/manual/analysis/graphics-performance-profiling/profile-rendering/profiler-gpu).

Live and offline reports include `gpuAudio`: response updates/reuse, current/peak grouped paths, reserved path capacity and material-filter bytes. Counts arrive with PCM, without another readback. The comparison window shows them under **Timings, recorded deadlines and provenance**. Existing underrun/late/overflow and callback counters remain the live deadline evidence; offline timings cannot establish those deadlines.

Live GPU rendering batches response samples across the queued, known motion intervals and uses [Unity asynchronous readback](https://docs.unity3d.com/2022.3/Documentation/ScriptReference/Rendering.AsyncGPUReadback.html). One in-flight batch owns its pose/response buffers until completion. **Maximum Responses Per Batch** bounds that storage; **Maximum Pending Observations** bounds queued motion intervals. A full queue stops the session visibly. Finishing/cancelling waits for owned requests before releasing their buffers. Offline GPU renders use the same batch capacity; GPU DSP consumes transport buffers directly and the CPU waits for completed PCM before writing it. The CPU reference reads transport paths explicitly. Completion wall time includes scheduling and readback, not isolated GPU execution time.

**Static geometry BVH** optionally accelerates CPU/HLSL direct intersections in both 2D and 3D. Geometry starts in the tree; when an object's pose first changes, its primitives leave the tree and use direct tests for the rest of that solver session, including when the object stops. Only that membership change rebuilds the tree; ongoing motion needs no refits. GPU batches share one static tree. No new scene tags or recording fields are required. The station enables it for multi-bounce tracing; new generic settings retain the direct baseline with BVH off. Toggle it in offline render settings or **Direct Settings → Use Bvh**. [Measured Low/High offline totals, preserved Git revisions and benchmark instructions](docs/cleanup-results.md) compare this split with the earlier moving-geometry BVH and brute force.

The default station lead is 16,384 sample frames (371.5 ms at 44.1 kHz), plus device latency. The station assigns `Transport.compute`. Spectral synthesis adds 1,024 samples (23.22 ms at 44.1 kHz) of declared DSP latency, plus path/HRIR delay. Measure completion time and deadline counters when selecting a lead.

The manager exposes **Idle**, **Live**, **Recording**, **Replay** and **Draining** states. One session owns the bounded PCM queue and output driver. **Monitor Live** controls audible monitoring during capture; disabling it records silently.

**Maximum Delay Seconds** declares the source-history horizon. Direct responses beyond it fail explicitly; raytraced paths are truncated at this horizon. Completed intervals discard older voice/activity history while retaining the boundary state and original loop phase. **Maximum Signal Events** bounds retained history per source and catches excessive event density. `live.json` includes retained-event counts, maximum pending observations, completion latency and a source/listener-pair × primitive test upper bound. That upper bound describes the direct diagnostic queries only, not the added multi-bounce/edge work. It is not a measured hit count. CPU remains useful for small workloads.

## Render offline

On the manager Inspector, choose a saved take and click **Render saved take to responses + WAV**. **Response Step** controls how often recorded geometry/poses are evaluated. Smaller steps cost more and reduce motion quantization. This is a time-resolution control, not a ray count or FFT size. **Tail** adds output samples after capture stops so already-emitted sound can arrive; no new source audio is emitted past the end of the take.

**Transport → Direct** computes source-major/listener-minor direct delays, pressure gains and arrival directions, with 1/r spreading above the configured minimum distance and two-sided opaque intersections. Planar mode uses 2D visibility with the same point-source spreading law. Responses are held for each update interval. **Transition Samples**, from zero through the response step, optionally crossfades the old and new delayed signal taps. Teleports reset that transition at the next response update. Live and offline use the same mixer, including events within a response interval. Source events and source activation gate emission at their recorded sample, including events inside an interval. Geometry and listener activity follow the response grid. Source emission position and delay rate use locally constant source/listener velocities. Continuous source-time reads produce Doppler without a separate pitch shift. Occluders remain at their solve-time pose; acceleration history, reflections, diffraction and phase-dependent materials are outside this direct baseline.

**Delay Filter** selects linear interpolation or a normalized windowed-sinc fractional-delay FIR. **Sinc Radius** controls the FIR half-width and adds that many samples of causal DSP latency, reported as `dspLatencySamples`. Static taps are computed per response; moving paths resample source history continuously. Windowed sinc also low-passes pitch-up to reduce aliasing; Linear remains the cheaper approximation. Live and offline share this implementation. This improves fractional-delay reconstruction; it does not change transport physics. Tail length must cover the propagation delay and desired filter tail. Both direct and spectral rendering use the selected fractional-delay filter.

**Listener Output → Binaural** applies a measured left/right HRIR to each direct path, using arrival direction in the recorded head's local frame (+X right, +Y up, +Z forward). The bundled MIT KEMAR compact dataset has 710 directions and 128 taps/ear at 44.1 kHz; other rates resample the filters once during preparation. Measured directions are interpolated across the dataset’s azimuth/elevation grid, including the rear seam, and use the existing response crossfade. The FIR preserves interaural timing and gain; geometric delay is added once, without a second ear-offset delay. Tail support must also include the HRIR. `dspLatencySamples` describes the fractional-delay stage, not direction-dependent HRIR arrival timing. This is a generic far-field mannequin model with limited elevation coverage, not a calibrated physical eardrum simulation. See [data attribution and limitations](Assets/Resources/Acoustics/ATTRIBUTION.md).

One listener stays one scene object and yields one WAV: mono or interleaved left/right. Source clips remain mono. Live binaural output requires Unity's stereo speaker configuration. Queue capacities, lead and counters always count sample frames, regardless of channel count. Old take files can render either output mode. Batch settings use `"listenerOutput": 1` for binaural, or omit it for mono; comparisons must use the same mode. Run/output manifests identify channel count, HRTF hash/taps and DSP contract. Direct response and output-capture manifests now use format version 3: each pair stores six floats (delay, gain, direction XYZ, delay rate in seconds/second). Comparison still reads v1/v2 reports. Take/input recording formats are unchanged.

**Transport → Raytraced** requires `Transport.compute` and uses the same scene/state and source-event pipeline. It supports segments in XY and full 3D triangles, including moving objects. Every arrival retains delay, frequency and world direction until its left/right KEMAR filters are applied. Sources/listeners use the existing source-major/listener-minor indexing; ears are output channels, not additional scene listeners.

The station starts at **one ray per frequency bin** (513 total per source/listener pair), five bounces, acoustic FFT size 1,024, convolution block 256, and a 1,024-sample response step. Rays per bin, sampling seed, bounces, frequency grid and block length are configurable. Each bin receives the same integer budget and weight `1/raysPerBin`; common random samples across frequencies keep broadband paths coherent. Old serialized total-ray budgets migrate by rounding up to complete per-bin budgets. Acoustic FFT size must be a power of two; the block divides it and is at most half its size; response steps are whole blocks. CPU IR synthesis uses separate zero-padded transforms of twice the block length. Spectral synthesis pads for the short acoustic filter plus the full HRIR and block; neither truncates the HRIR. Frequency bins follow the acoustic FFT, without a new fixed-band representation. Existing material controls supply absorption, scattering, transmission, IOR and damping curves sampled onto that grid. Take format v2 authenticates the additional curves. V1 takes remain readable, with absent transmission/damping zero and IOR one.

Raytraced travel time sums `distance * IOR / airSpeed` separately for every medium traversed; `IOR = airSpeed / materialSpeed`. Interior pressure attenuation is `exp(-damping * distance)` and air attenuation applies only in air. Closed, consistently wound triangles/2D loops supply thickness; open surfaces remain zero-thickness sheets. Sources may start inside a solid, and nested solids restore the enclosing medium on exit. Each closed object must share IOR/damping across its surface materials. Among overlapping solids the smaller enclosed volume takes priority (ties follow object order); this is a deterministic authoring rule, not a mixed-material physical model.

**Transition Samples** crossfades the old and new convolved output sample by sample. The station uses 512 samples (11.6 ms at 44.1 kHz); zero switches responses immediately, as do cuts. Disabling a listener fades its output to silence once. Source Stop ends emission, preserving sound already in flight and the filter tail. No delay derivative is inferred from reused indirect-path slots. In Spectral mode, delay rates come from endpoint and surface velocities; source-time resampling produces Doppler even with crossfades disabled. IR mode crossfades frozen filters and does not implement continuous Doppler. Neither mode is a general retarded-time moving-boundary solver.

The preview combines deterministic direct/transmitted frequency bins, sampled **first-order image-source mirror paths**, and seeded indirect rays. The mirror path uses the unscattered reflection coefficient; diffuse next-event connections use the scattering coefficient. Sampling primitives at low budgets can miss mirror paths or produce large weighted samples; more rays and independent seeds are needed to assess convergence. Experimental diffraction now samples signed rigid-wedge contributions inside the same bounce loop, including successive edge events. It uses the existing arrival buffers and DSP; no separate shortest-edge route remains. Higher-order perfect-mirror/edge connections, diffuse pressure/PDF normalization, impedance materials and multi-edge physical accuracy remain unvalidated. Low ray budgets can miss edge sound; this preview is not a training teacher. [Model, checkpoint and limitations](docs/simulation-contracts.md#integrated-edge-trial). The transmitted direct connection stays straight; traced rays refract at interfaces.

**Synthesis → Impulse Response** uses frozen responses. The CPU reference retains the partitioned scene-IR convolver with overlap-add and circular source-spectrum history; the GPU evaluates the equivalent sum of delayed path filters with overlap-save. This remains the default for existing settings. Propagation, fractional delay, material response and the complete per-arrival ear filter are composed into a scene FIR. It is suitable for stationary response rendering; crossfading snapshots does not make moving Doppler exact.

**Synthesis → Spectral** enables continuous Doppler on either audio backend. Each physical path reads the source at a continuously varying emission time, then applies short material and ear spectra with overlap-save. It does not build a propagation-length scene IR. Those short filters are still convolution mathematically. Direct paths and first planar mirrors account for emission/interaction time under locally constant velocity; sampled indirect geometry and other occluders remain frozen at solve time. Topology hashes separate path groups, and each solve refreshes the physical emission-time anchor. Accelerating/rotating general scenes still need validation. CPU per-path synthesis is expensive; the GPU version processes path FFTs in parallel. Neither offline throughput nor numerical agreement establishes realtime deadlines.

Both routes declare the acoustic FFT size as DSP latency, plus sinc radius when selected. `responseSupportSamples` reports observed filter/delay support; it is not an exact bound for arbitrary continued motion. The output tail holds the final scene pose with zero velocity and emits no new source audio. The one-second station tail is configurable and does not cover every possible ten-second path. [Motion contract and limitations](docs/simulation-contracts.md#continuous-doppler-and-synthesis-methods-2026-10-06).

**Audio Backend → Gpu** is the raytraced default; **Cpu Reference** keeps the original DSP for comparisons. Scene data, source events, the sample timeline and PCM queue are shared. GPU grouping, material/head filters, Doppler resampling, FFTs and mixing stay on the GPU; only dry source PCM/head poses go up and finished PCM comes back, alongside small direct diagnostic probes. Live batches known intervals into one asynchronous PCM readback. No source spectra or path arrays cross back for routine GPU synthesis.

`backendScope` identifies the selected transport/DSP combination. `audioShaderDependencyHash` records GPU DSP provenance; `audioUploadBytes` and `audioReadbackBytes` count streaming audio/pose uploads and PCM/metadata reads, excluding initialization and transport data. `traversalCounterScope` identifies the direct-only diagnostic bound; it is not a count of spectral ray/bounce/edge work.

`responses.bin` and optional monitor response records still contain **direct opaque diagnostic probes**. Raytraced WAVs include indirect/transmitted sound; the probes do not reconstruct those WAVs. Reports explicitly identify this scope and the spectral solver/DSP contract. `responsePreparationSeconds` is the CPU time spent preparing/submitting responses, included in `audioSeconds`. For GPU DSP, shader execution is asynchronous: subtracting it does not isolate GPU synthesis time. Offline `audioSeconds` also includes waiting for final PCM. Full directional-ray dataset export and model training are not implemented. Temporal A/B comparison works for either transport; both runs must share other settings, and spectral runs are not accepted as training references.

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

Use an output directory that does not exist. The process exits 0 on completion and 1 on failure. CPU direct rendering works with `-nographics`; raytraced transport requires a graphics device. A shell/Python loop can invoke jobs sequentially; there is no separate orchestration service. To use the GPU, add `"shader": "Assets/Shaders/Transport.compute"` to the job and omit `-nographics`. A missing or unsupported shader fails explicitly.

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

Enable **Check finer reference** or add a `referenceCheck` object with `responseStepSamples`, `maximumRmsError` and `maximumAbsoluteError`. Its response step must be smaller than the high run and at least `transitionSamples`; tolerances must be explicit finite nonnegative values (zero requires exact agreement). The third run writes `reference/`, errors against the high run, and `referenceAccepted`. Only a passing direct-model check sets `eligibleForTemporalTraining`; spectral preview comparisons always keep it false; a completed render alone does not. This is agreement with one finer direct-model reference, not proof of physical convergence.

Runs record the compiled C# module ID, Unity/build identity, shader dependency hash in the Editor, and a hash of all settings. Player builds use the build GUID and shader name alongside the C# identity. Old takes remain readable and have no capture provenance when it was not recorded.

This experiment changes **temporal resolution**, not ray count. The finer run is a comparison reference for this direct model, not reflection ground truth and not the planned low/high-ray training dataset.

## Compare audio and watch the take

For silent inspection without Unity, use `tools/analyze_audio.py` (Python 3.11+, NumPy, SciPy 1.12+, Matplotlib). It writes one waveform/spectrogram PNG per channel and `analysis.json`, including file hashes, exact sample ranges, peak/RMS, silence spans, largest sample steps and aligned differences. It never plays audio or opens a window. For example:

```sh
python tools/analyze_audio.py monitor/output.wav render/listener-0.wav \
  --labels Live Offline --trim 16384 0 --start 7.3 --end 8.8 \
  --fft 4096 --hop 128 --min-hz 300 --max-hz 2000 \
  --ceiling-db -40 --grid-samples 1024 --output .context/audio-inspection
```

Use the capture's actual `leadSamples` for the live trim; the example's 16,384 is not inferred. Omit the second WAV for single-file inspection. Waveform scales and spectral dBFS scales are shared; channels remain separate, with no normalization, resampling or guessed alignment. The difference panel has its own labeled scale. Zoom using `--start/--end`; choose a longer FFT for narrow pitch/sideband inspection or a shorter one for transients. Spectra use [SciPy ShortTimeFFT](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.ShortTimeFFT.html), preserving full surrounding windows when zooming. Only the common selected interval is compared, with excluded leading/trailing samples recorded in JSON. Sharp steps, zeros and spectral changes are diagnostic observations, not proof of an audible defect. Use a fresh output directory for each view. The tool loads the WAVs into memory; choose exported excerpts for very large recordings.

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

## Transport flow

```text
CPU: compile scene once; upload object states for each response batch
GPU: evaluate direct/mirror sound → sample surface and edge paths in one bounce loop
GPU: synthesize IR or spectral audio → read back PCM
```

Normal GPU playback uses PCM completion to fence the response batch. Diagnostic probes are dispatched/read back only when requested; offline reports retain them. `Geometry.hlsl` owns intersection and BVH traversal, `Transport.hlsl` owns acoustic calculations, and `Transport.compute` holds the entry points. The CPU reference provides direct transport and audio synthesis; it is not a second full raytracer. Frequency-dependent transport still evaluates each configured FFT bin. No new runtime dependencies were added.

## Repository map

```text
Assets/
  Scripts/
    Scene/        AcousticSurface, AudioMaterial, SceneToTake
    Simulation/   RayTraceManager, Take, TakeRender + Live partial,
                  TransportSolver, DirectBvh, RayTransport, GpuAudio, Doppler, SpectralAudio, ImpulseResponseAudio,
                  TakeComparison
    Audio/        AudioManager, RaytracedAudioSource, Hrtf
    Demo/         SceneSession, SceneRecording, StationDemo, StationLoop, SlidingDoor, PlayerController,
                  MouseLook, Oscillator, RaytraceControlPanel
  Shaders/        Transport.compute, Geometry.hlsl, Transport.hlsl, WorldText.shader
  Audio/          source clips and DefaultMixer
  Materials/      acoustic material assets
  Resources/      Acoustics/GpuAudio.compute; bundled KEMAR filters and attribution
  Editor/         SceneReplay, TakeEditor, TakeDemo, TakeComparisonWindow + helpers,
                  DirectBenchmark
  Tests/Editor/   TakeChecks + numerical/transport checks, GpuAudioChecks, DirectBvhChecks
  Scenes/         editable scenes and their baked lighting
  Prefabs/        project prefabs
  External/       imported train assets
  TextMesh Pro/   imported UI assets
Packages/         Unity dependency manifest
ProjectSettings/  Unity project configuration
docs/             consolidated design contract, validation results and benchmark samples
tools/            reproducible KEMAR importer (Python standard library)
```

Folders group responsibilities; the runtime, editor and test assembly definitions only separate compilation. This scene/session change adds no package dependencies. Asset `.meta` GUIDs are preserved so scenes and prefabs keep their references. The unused train loader/convolution shader and duplicate FFT scene/component have been removed; automated GPU tests retain the FFT coverage. Relative shader includes stay together in `Shaders`.

## Cleanup implemented

| Files | Result |
|---|---|
| [SceneToTake.cs](Assets/Scripts/Scene/SceneToTake.cs) | Reuses authoring snapshots and comparison storage, deduplicates shared meshes/materials, and detects in-place changes without per-observation scene export. |
| [RayTraceManager.cs](Assets/Scripts/Simulation/RayTraceManager.cs) | Defines exclusive session states and retains ownership while output drains. One renderer serves live monitoring, capture and replay. |
| [AudioManager.cs](Assets/Scripts/Audio/AudioManager.cs) | Consumes the bounded PCM queue and optionally captures output and callback diagnostics. |
| [TransportSolver.cs](Assets/Scripts/Simulation/TransportSolver.cs), [Transport.compute](Assets/Shaders/Transport.compute), [DirectBvh.cs](Assets/Scripts/Simulation/DirectBvh.cs) | Batches live/offline response frames with explicit resource ownership; optional static CPU/HLSL BVH plus direct tests for moving geometry. Reports completion latency, classification/rebuild costs and traversal workload bounds. |
| [RayTransport.cs](Assets/Scripts/Simulation/RayTransport.cs), [Transport.hlsl](Assets/Shaders/Transport.hlsl), [SpectralAudio.cs](Assets/Scripts/Simulation/SpectralAudio.cs) | Directional 2D/3D spectral transport, bounded GPU arrival buffers, material curves, image-source first reflections and selectable IR/spectral binaural synthesis. |
| [GpuAudio.cs](Assets/Scripts/Simulation/GpuAudio.cs), [GpuAudio.compute](Assets/Resources/Acoustics/GpuAudio.compute) | Own GPU source history, path filters/FFTs, per-ear synthesis and asynchronous PCM batches; CPU DSP remains the reference. |
| [TakeRender.cs](Assets/Scripts/Simulation/TakeRender.cs), [TakeRender.Live.cs](Assets/Scripts/Simulation/TakeRender.Live.cs) | Bounds event/cut/observation history and shares configurable linear/windowed-sinc delay filtering between live and offline rendering. |
| [Take.cs](Assets/Scripts/Simulation/Take.cs), [TakeComparison.cs](Assets/Scripts/Simulation/TakeComparison.cs) | Records implementation/settings provenance and optionally gates temporal training pairs against a third, finer direct-model run. |

The next transport work is physical validation, reducing GPU path/filter cost and converged ray-budget comparisons. The separate prototype renderer is removed; its previous implementation remains in Git. All five scenes use the current renderer. General reflection training and learned reconstruction are not implemented by these cleanups.

The current GPU stores one material spectrum per path group and response bank, shared by both ears. Each bank retains its own head blend for crossfades. Unchanged filters skip both small and large FFT construction. The station benchmark reserves **564 MB versus 916 MB** before this change; matched layout controls have similar wall times. A new `DirectBenchmark.MeasureReciprocity(planar, rays, scattering, seeds)` probe checks source/receiver reversal on a finite wall. The diffuse preview fails this necessary physical check even with 65,536 rays/bin, while the mirror control agrees. More rays are not a fix for this bias. [Measurements and limitations](docs/cleanup-results.md#gpu-response-storage-and-diagnostics--2026-10-07).

## Checks

Open **Window → General → Test Runner → EditMode → Acoustics.Tests**. Tests use fresh scenes and temporary directories; Unity restores the original scene setup. Unexpected Unity errors fail. GPU cases skip unsupported hardware and never substitute CPU results.

| Category | Coverage |
| --- | --- |
| `Core` | Formats, materials, geometry, independent DSP oracles and queue contracts |
| `GPU` | Shader execution, CPU/GPU agreement, transport and readback ownership |
| `Integration` | Streaming/offline agreement, response batching and audio files |
| `Voice` | Source persistence, resampling, Opus order/loss, self-monitor masks and tail draining |
| `EndToEnd` | Offline comparison; church/station recording/reconstruction/replay; explicit realtime delivery |
| `Realtime` | Short silent callback/recording smoke; explicit station deadline benchmark |
| `Device` | Explicit microphone lifecycle test; no microphone PCM saved or transmitted |

Keep Unity audio enabled for these checks. `AudioManager.muteOutput` clears device samples **after** copying PCM to the recorder; batch workers enable it automatically. Pausing still records silence. Scene and multiplayer probes use the same output mute, and tests no longer open the Game window. This captures our selected listener before Unity mixer/device processing; it is not an OS loopback recording.

`AudioOutputTests.SilentOutputRecordsRealCallbacks` uses an empty scene and two PCM blocks to verify the real audio callback, advancing DSP timestamps and exact nonzero recorded samples. It runs silently and needs no GPU workload, microphone, network or virtual audio driver. The atomic mono/stereo capture cases separately check zero device buffers, channel mapping, finalization and zero callback allocation. Queue edge cases and concurrent publication retain their own focused checks.

Categories can overlap; each selected test runs once. FIR cases use pairwise FFT/hop/channel/delay/filter/method combinations against a time-domain oracle, with separate long-HRIR, transition and larger-transform regressions. Numerical agreement does not establish physical accuracy of scattering/diffraction approximations.

`RecordingReconstructsAndReplays` and `ChurchWalksThroughDoorAndReplays` are **Explicit** scene checks. The station case walks aboard, jumps and turns the head, then checks exact reconstructed state/event hashes and final replay transforms. `StationDeliversAudioBeforeItsDeadline` is **Explicit**: select it on an idle target machine with stereo output. After two seconds of warmup it measures 20 seconds of device-clock movement, failing on any new underrun, late, overflow, channel mismatch or over-budget callback. Run All excludes it; exclusion is not a pass. The configured playback lead is unchanged. Test watchdogs do not limit recording duration.

For the open Editor:

```sh
unity command run_tests --mode editor --filter TakeChecks --async_tests true
unity command test_status
```

The optional hardware microphone check is `MicrophoneDeviceTests.MicrophoneCapturesAndClosesItsOpenedDevice`. It requires an available, already-permitted microphone and captures one second after saving/disabling recording, without connecting to peers. Select it explicitly in Test Runner or pass its full name with `--include_explicit true` to the Pipeline CLI. Run All never opens a microphone.

For station tests, select the exact test in Test Runner. Pipeline CLI status can remain `running` across Play-mode reloads; Unity's result/XML is authoritative. Station recordings are deleted after testing, including on failure.

The opt-in `MultiplayerSmoke` probe runs in Development Players. Put Church or TrainStation first in the build scene list, then launch two copies with `--acoustics-smoke=host` and `--acoustics-smoke=client`, each with its own absolute `--acoustics-result=/path/result.json` and `-logFile` path. It sends **synthetic** tones through the production Opus/transport/source-history path, moves both players and a host-owned actor, saves each client view, then replays it offline and checks the final poses. It never opens a microphone. Results include packet concealment and device deadline counters; a functional pass is not a zero-underrun claim. `--acoustics-smoke-step=<samples>` overrides acoustic cadence and `--acoustics-smoke-lead=<samples>` overrides output lead for budget measurements. `--acoustics-players=N` checks the intended roster. Add `--acoustics-network=relay` for internet transport; the host writes its `joinCode` to the result JSON once ready, and clients receive it through `--acoustics-join=CODE`. Generated takes use the normal Player recordings directory; retain them only as test data.

For CI with the project **closed**:

```sh
"$UNITY_EDITOR" -batchmode -projectPath "$PROJECT_PATH" \
  -runTests -testPlatform EditMode -assemblyNames Acoustics.Tests \
  -testFilter 'TakeChecks|AudioOutputTests' -testResults "$RESULTS_DIR/tests.xml" \
  -logFile "$RESULTS_DIR/unity.log"
```

Use a closed project or a disposable snapshot when the working Editor is open. Keep audio and graphics enabled; run one worker at reduced priority (`nice -n 15` on macOS), adding `-job-worker-count 2` to limit its CPU workers. The command selects atomic checks, a small offline comparison and the short silent callback check. Heavy scene/deadline tests and microphone tests require explicit selection. Require expected cases to execute; reject failures, unexpected skips and missing/incomplete XML. Device-clock tests need active audio and graphics; offline renders cannot replace them. An external process watchdog is still needed for native/driver hangs.

No new test package: Unity Test Framework handles selection, coroutines and reports. Three assembly definitions isolate runtime, editor and tests; friend declarations expose internal DSP to tests without expanding the public API. `DirectBenchmark` remains the offline Low/High timing harness. [Measured results](docs/cleanup-results.md) exclude cold import/build costs; DirectX and standalone Player test runs remain separate validation work.

## Standalone entry point

Build settings start `Assets/Scenes/TrainStation.unity`. Player recordings go to `Application.persistentDataPath/Recordings`; Editor recordings stay in the project's `Recordings` folder. The Escape controls list completed takes for the current scene in Editor and standalone. Editor tools also provide folder selection and offline comparisons.
