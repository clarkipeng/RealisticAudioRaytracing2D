# Acoustic simulation: design and implementation contract

Design contract, 2026-09-28. Reviewed against repository commit `c74c5d7`. This is the authoritative design document; it incorporates the earlier architecture proposal. Type definitions and entry points below describe the target design in pseudocode. Implementation has started; [README.md](../README.md) documents the actual APIs, supported workflow and current limitations.

The current implementation provides native shape export with strict geometry/material validation, recorded scene/state/event files, pose replay/scrubbing, shared live/offline direct audio, CPU and 2D/3D HLSL direct solvers, bounded monitoring and output-capture queues, asynchronous live GPU batches, configurable fractional-delay FIR, code/settings provenance, and Editor/batch temporal comparisons with optional finer-reference error gates. It also repairs and checks the legacy FFT. Indirect transport, configurable FFT/STFT, the ray-budget dataset and learning contracts below remain unimplemented. The temporal-budget comparison is a separate experiment, not a substitute for the first stochastic training pair. This document must not be read as a claim that all correctness gates have passed.

The intended workflow is: **create a 2D/3D scene → move through it while objects move and sounds play → record a take → replay at different compute budgets → collect aligned data and timings → train and evaluate acoustic reconstruction.**

The invariant is: **a scene and a recorded take determine the inputs; a render configuration determines the computation.** Replaying at a different quality never changes the take.

Read in order, or jump to [review findings](#0-critical-review), [layout and dependencies](#1-four-representations-each-with-one-job), [Unity integration](#2-what-goes-on-a-unity-gameobject), [scene data](#3-cpu-scenestate-contracts), [recording](#4-recordplayback-contract), [GPU layouts](#5-exact-c--hlsl-layouts), [resource ownership](#6-geometry-updates-and-resource-ownership), [methods and configuration](#7-solver-input-and-output), [execution flow](#8-one-path-through-unity-recording-shaders-and-replay), [correctness checks](#9-checks-required-before-calling-this-contract-implemented), [saved data and jobs](#10-saved-data-and-offline-jobs), [benchmarking](#11-diagnostics-and-benchmarks), [learning](#12-learned-acoustic-reconstruction), or [implementation order](#13-implementation-order-and-readiness).

## 0. Critical review

The small architecture is sufficient. The main risks are incorrect acoustic quantities, replay timing, and reference-data validity, not a lack of modules. The following changes are part of this revision:

| Finding | Decision |
|---|---|
| Fixed spectral layouts constrain experiments without justification. | Frequency grids are explicit arrays. No fixed band count or FFT size in scene/material/recording contracts. Backend limits are validated capabilities. |
| IR, FFT, and STFT were being treated as competing physical methods. | Separate transport physics, response quantity, numerical budget, and audio processing. FFT convolution can render an IR from any compatible solver. |
| Recording transforms alone cannot reproduce what a realtime run actually played. | Keep the immutable take separate from its render log. Log response application samples, buffering and discontinuities; capture final application PCM for exact playback. |
| A dense arrival-by-band buffer can dominate memory. | Bound temporary arrivals and process frequency/ray batches. Saved response shape does not dictate temporary GPU allocation. |
| A single material per object makes ordinary imported meshes awkward. | Use a default material plus optional submesh bindings; resolve each primitive to a shared material-table index. |
| More rays do not make an incorrect estimator or missing wave physics correct. | Validate normalization and reference convergence before dataset promotion. Tag physical approximations. |
| Shader time alone does not establish realtime performance. | Measure response age, readback, DSP, inference, audio deadline misses and total latency under the same workload. |

**Findings from the pre-implementation review:**

| Current implementation | Consequence for the work |
|---|---|
| [RayTraceManager.cs](../Assets/Scripts/Simulation/RayTraceManager.cs) owns one source/listener, forces 1024 samples, uses frame-based randomness, and performs synchronous hit-count/waveform readbacks. | Extract scene/state and deterministic sampling contracts before attempting performance comparisons. |
| [SceneHelper.cs](../Assets/Scripts/Scene/SceneHelper.cs) repeats material data in every segment and reduces 3D objects to their longest projected XY axis. | The existing path is a 2D prototype. Real triangle geometry and material indices are new work. |
| [Common.hlsl](../Assets/Shaders/Common.hlsl) and the spectrogram IFFT in [Raytrace2D.compute](../Assets/Shaders/Raytrace2D.compute) launch `N` threads for a mapping with only `N/2` butterflies. `j = idx % m2; if (j < m2)` does not exclude any thread. | Index evaluation at `N=1024` shows 512 threads access beyond the shared array at each stage. Replace/fix and verify against an independent CPU result before reusing it. This is a source-level finding, not a GPU test result. |
| [Raytrace2D.compute](../Assets/Shaders/Raytrace2D.compute) samples frequencies from input audio and multiplies arrivals by that input spectrum. | This does not expose the reusable unit-input response the new contract requires. Source excitation belongs in audio rendering. |
| [AudioManager.cs](../Assets/Scripts/Audio/AudioManager.cs) locks its ring buffer and can grow a recording list or log underruns inside `OnAudioFilterRead`. | Replace callback work with bounded copies/counters and move capture/logging off the audio thread. |
| [RaytracedAudioSource.cs](../Assets/Scripts/Audio/RaytracedAudioSource.cs) assigns each source into global manager fields. | Multiple sources currently overwrite each other; explicit indexed sources are required. |

At the time of this review there was one 2D transport implementation and eleven declared kernel names across the shader sources, including FFTs and debug drawing. There was no implemented 3D transport or reconstruction model. The subsequent direct-sound milestone is described above. Kernel declarations are not a count of physical methods or a verified list of successfully compiled kernels.

This review does not establish realtime feasibility or accurate acoustic reference output. Those claims require the measurements and correctness gates below. It does establish enough boundaries to implement the foundations without choosing a neural network or inventing quality presets.

## 1. Four representations, each with one job

```text
Unity components              Inspector references and live transforms
        ↓ compile/capture
SceneData + SceneState        Indexed numeric data used by every solver
        ↓ pack/upload
GPU buffers                   Fixed binary layouts, counts, and explicit ownership
        ↓ solve/read back
ResponseFrame                 Tagged acoustic result, ready for rendering/capture

Recording writes SceneData identity + successive SceneStates + audio events.
Replay produces the same SceneState contract without reading live Transforms.
```

No GameObject, Transform, AudioClip, Unity instance ID, or material asset reference crosses into a shader or a saved take. Assets are resolved to data/content hashes; objects are resolved to stable IDs and numeric indices.

### Repository layout and dependencies

Implementation rule: reuse/refactor existing code first. The contracts below describe responsibilities and data, not a checklist of new classes, files or abstractions. Add code only when the next working experiment needs it; preserve correctness checks when reusing a defective implementation.

```text
Assets/
  Scripts/
    Scene/        object/material authoring and numeric geometry export
    Simulation/   manager, take data, shared live/offline rendering and comparison
    Audio/        source events and audio output
    Demo/         walkthrough controls, scene helpers and the legacy FFT component
  Shaders/        HLSL compute shaders and their shared includes
  Audio/          source clips and mixer
  Materials/      acoustic material assets
  Editor/         Inspector, scene creation and offline batch entry points
  Tests/Editor/   checks for the implemented contracts
  Scenes/        editable scenes and baked lighting
  Prefabs/       project prefabs
  External/      imported train assets
  TextMesh Pro/  imported UI assets
Packages/        existing Unity dependencies
ProjectSettings/ Unity configuration
docs/            this design
```

The folders group the existing implementation without introducing assembly boundaries or replacement components. The [repository map and implemented cleanup](../README.md#repository-map) identify the current files. Extract a helper/file when it removes real coupling or duplication. Keep existing scenes and `.meta` references working during migration. Unity checks belong under `Assets/Tests`; add tools/learning directories only when that work begins. No separate package distribution, storage service, backend framework or orchestration service is needed.

Dependencies flow from Unity adapters to numeric data to solver/DSP to saved results. The solver never searches GameObjects or reads AudioClips. Recording does not depend on ML. Offline jobs call the same scene, solver and DSP code; they do not implement their own acoustics.

Keep the repository's Unity `2022.3.62f3`, C# and HLSL initially. Unity supports HLSL compute across its DirectX/Metal targets; portability still requires explicit counts and valid buffer access. We will verify the same layouts and fixtures on macOS/Metal and Windows/DirectX. A GPU capability query selects supported execution settings, not different scene formats. [Unity compute shaders](https://docs.unity3d.com/2022.3/Documentation/Manual/class-ComputeShader.html)

Use the existing Unity Test Framework for runtime tests, a small CPU reference for numerical fixtures, and Python for local batch control/data analysis. NumPy/SciPy can provide independent offline DSP checks when that work begins; a training framework and realtime inference runtime are selected only when there is a validated dataset and measured model. The simulator has no training-framework dependency. A Mojo/CUDA/other backend is a later implementation option behind these numeric contracts, justified by a demonstrated compiler or performance limitation.

## 2. What goes on a Unity GameObject

The names in this section describe component roles. Implement the object/source roles by extending `AcousticSurface` and `RaytracedAudioSource`, and the session role by refactoring `RayTraceManager`; do not create parallel replacements just to rename them. Listener registration can remain a serialized session reference until a separate component has a concrete benefit. Examples use the conceptual names to make responsibilities clear.

| Component/asset | Inspector fields | Responsibility |
|---|---|---|
| `AcousticObject` | Serialized stable ID; geometry source/reference; default material and optional submesh overrides | Registers one transform and, optionally, one geometry instance. A geometry-free object can carry a source/listener. |
| `AcousticSource` | Clip, gain, loop; requires `AcousticObject` | Issues sample-timed Play/Stop events to the session. It does not create or configure a global ray tracer. |
| `AcousticListener` | Requires `AcousticObject` | Marks a receiver; usually attached to the player's camera/head. Initial output is mono per listener. |
| `SimulationSession` | Scene root, dimension/plane, settings | Owns compilation, recording/replay mode, GPU resources, and output. One session owns each registered object. |
| Existing `AudioMaterial` asset | Acoustic coefficients at authored frequencies | Shared authoring asset. Each render samples these curves into a material-by-band coefficient table. |

These are small adapters around the data model, not five managers. Source/listener orientation is recorded even while the initial mono model is omnidirectional.

`AcousticListener` is a simulation receiver, not an additional Unity `AudioListener`. Realtime playback selects one simulated listener's output for the existing audio sink; offline capture can write every listener. Several playing voices on one source are mixed into its dry signal before propagation rendering.

```text
SimulationRoot                        [SimulationSession]
  Room
    Wall                              [AcousticObject: BoxCollider + Concrete]
    ImportedModel                     [AcousticObject: MeshFilter + Wood]
  Train                               [AcousticObject: mesh + Metal]
                                      [AcousticSource: engine.wav]
  Player                              [existing movement controller]
    Camera                            [AcousticObject: no geometry]
                                      [AcousticListener]
```

Membership is explicit through components below the session root. Compile once at session start; do not search the entire scene every Update. Freeze object, material and clip membership for a take in v1. Geometry/material edits or unsupported deformation require a new take; fail capture visibly rather than silently changing its scene. Rigid movement, source playback, and active/inactive state are recorded.

The compile step enumerates registered components including inactive participants. Once running, enabling/disabling an object changes its recorded active flag; it does not remove/reinsert it in the index table. Session end retires resources after pending work completes. `OnValidate` checks authoring fields/IDs and never starts a simulation or uploads GPU data.

IDs are serialized per scene instance. Duplicating a prefab/object must yield a distinct object ID before recording; scene validation rejects duplicates. Never use `GetInstanceID()` or hierarchy paths as persistent identity. At compile time, sort IDs once into an object table; its `uint` indices remain fixed throughout the take. Disabling an object never renumbers the table. A source and listener on the same object reference the same object index.

Preserve existing scene references and `.meta` GUIDs while adapting `AcousticSurface` and `RaytracedAudioSource`. In particular, remove the current source behavior that overwrites `RayTraceManager.source/inputClip` for every source.

The editor workflow needs only a session Inspector and a small recording/render panel: validate registered objects, record/stop, choose a take, scrub/replay, choose render configurations, and queue a comparison. Existing controllers, animation and Timeline can drive objects during capture. Exporting the compiled scene makes offline replay independent of those controllers and the original `.unity` scene. A moving door is an ordinary rigid acoustic object; an imported room can use its mesh submaterials or a simpler explicit acoustic proxy.

### Native shapes and normal scene editing

Create Cube, Sphere, Capsule, Cylinder, Plane and Quad objects through Unity's normal GameObject menu. Move, rotate, scale, duplicate, parent and edit them using the existing Scene view/Inspector. Add `AcousticObject` and choose an acoustic material; add `AcousticSource` only if the object should emit audio. A Rigidbody is optional and concerns motion/collisions, not eligibility for acoustic tracing. [Unity primitive objects](https://docs.unity3d.com/2022.3/Documentation/Manual/PrimitiveObjects.html)

`AcousticObject.geometrySource` selects `Auto`, `Mesh`, `Collider`, `Proxy`, or `None`. Auto resolves once during authoring validation: use the object's MeshFilter mesh when present, otherwise its single supported collider. Multiple candidate colliders require an explicit selection. Display and save the resolved source; do not register the mesh and collider as two coincident acoustic surfaces. Camera-dependent visual LOD changes do not silently change the selected acoustic mesh.

| Authored object/source | Compiled acoustic geometry |
|---|---|
| Unity Cube, Sphere, Capsule, Cylinder, Plane or Quad in Mesh mode | Triangles from the selected native mesh, with authoring scale baked. |
| `BoxCollider` | Twelve triangles from its center and size. |
| `SphereCollider`, `CapsuleCollider` | Generated triangle approximation of the selected collider, including center, radius, capsule height and axis. Curvature tolerance is an export setting, not a hardcoded segment count. |
| Selected mesh or MeshCollider mesh asset | Its triangle/submesh geometry. Do not claim that the asset reproduces a separately cooked convex collision hull. |
| `BoxCollider2D`, `CircleCollider2D`, `CapsuleCollider2D`, `PolygonCollider2D`, `EdgeCollider2D` | Closed boundaries or explicit open edges as appropriate, converted to segments in the simulation plane. Include offsets and declared curve tolerance. |

Mesh and collider modes intentionally may differ. Unity's default cylinder has a capsule collider, so selecting its mesh preserves the visible cylinder while selecting the collider chooses the capsule approximation. A collider provider must match that collider's effective shape under scale; blindly applying nonuniform scale to a mathematical sphere/capsule can produce a different surface. Validate scaled collider fixtures and require an explicit mesh/proxy for unsupported transforms. Capsule center, total height and lengthwise axis are native authoring parameters. [Unity primitives](https://docs.unity3d.com/2022.3/Documentation/Manual/PrimitiveObjects.html), [CapsuleCollider properties](https://docs.unity3d.com/2022.3/Documentation/Manual/class-CapsuleCollider.html)

Child parts with different materials or motion use normal child GameObjects and explicit registrations. Each selected geometry source has one acoustic owner. Record resulting world-space poses, so parent motion and rotations are included without reconstructing Unity hierarchy behavior during offline replay.

Before recording, any native shape/scale/collider edit invalidates the cached acoustic geometry; the next validation/preview rebuilds it. During a take, moving/rotating objects is supported continuously. Changing radius, height, scale or mesh topology is a shape change, not rigid motion: the initial take format ends/requires a new take for that change. This restriction applies to capture, not ordinary scene authoring. Later live shape editing needs timestamped geometry revisions; it must not silently mutate an existing take.

### Moving, accelerating and rotating objects

Choose one Unity motion writer for each object; the acoustic adapter observes its result. Use an existing controller/Transform script for gameplay, Animation/Timeline for authored paths, or Rigidbody/Rigidbody2D for collision-driven motion. A kinematic Rigidbody can follow scripted motion through its physics movement APIs. Dynamic bodies can receive forces or acceleration; the actual resulting motion also includes collisions and other physics. [Rigidbody.MovePosition](https://docs.unity3d.com/2022.3/Documentation/ScriptReference/Rigidbody.MovePosition.html), [Rigidbody.AddForce](https://docs.unity3d.com/2022.3/Documentation/ScriptReference/Rigidbody.AddForce.html)

Use existing controllers and Animation/Timeline for prescribed paths/rotation. Constant-acceleration translation follows `p(t) = p0 + v0*t + 0.5*a*t*t`; add a small fixture/helper only if an actual experiment needs that control and existing authoring does not provide it. There is no new general motion system in the initial implementation. A force-driven experiment applies its commanded acceleration to the Rigidbody instead; do not simultaneously overwrite its Transform with a prescribed path. These are motion-authoring inputs, not acoustic solver settings.

```text
MovingCapsule
  Transform / MeshFilter / MeshRenderer    ordinary editable Unity capsule
  CapsuleCollider                         optional physical collision shape
  Rigidbody                               optional physics-driven movement
  AcousticObject                          selected geometry + acoustic material
  AcousticSource                          optional engine/voice/other sound
  existing controller / Animation         one motion owner
```

Acoustic recording captures the actual movement, not just a command such as "accelerate at 2 m/s²". A collision can make the observed acceleration very different from the requested value. Playback applies captured state and disables the original writers; it does not rerun forces or integrate saved acceleration and hope to reproduce the same path.

## 3. CPU scene/state contracts

The following is C#-style pseudocode. These are host types; their in-memory layout is not a disk or GPU ABI.

```csharp
sealed class SceneData {
    int schemaVersion;
    Hash sceneHash;                    // Hash of canonical compiled inputs
    GeometryDimension dimension;       // Planar2D or Spatial3D
    PlaneFrame plane;                  // 2D: origin + orthonormal U,V,N
    ObjectDef[] objects;               // Canonical index -> stable ID + geometry/material
    GeometryData[] geometry;           // Local segment/triangle arrays; bounds
    MaterialData[] materials;          // Coefficient curves + model/interpolation version
    SourceDef[] sources;               // Source index -> object index
    ListenerDef[] listeners;           // Listener index -> object index
    ClipData[] clips;                  // Content hash -> canonical dry PCM
}

struct ObjectDef {
    StableId id;
    int geometryIndex;                 // -1 means no acoustic geometry
    uint[] materialIndices;            // Local material slot -> canonical material index
}

sealed class SceneState {
    ulong snapshotId;
    long sample;                       // Session audio time, never Time.frameCount
    ObjectState[] objects;             // Same indexing as SceneData.objects
}

struct ObjectState {
    Vector3 worldPosition;
    Quaternion worldRotation;
    MotionSample motion;               // Observed/estimated kinematics at this sample
    bool active;                       // Host-only bool; explicitly packed for GPU
    uint roleFlags;                    // Effective geometry/source/listener enable bits
    bool discontinuity;                // Teleport/cut: do not interpolate through it
}

struct MotionSample {                  // CPU/take data; not an unconditional GPU upload
    Vector3 linearVelocity;            // m/s, world axes, at recorded Transform origin
    Vector3 angularVelocity;           // rad/s, world axes
    Vector3 linearAcceleration;        // m/s^2, same origin and axes
    Vector3 angularAcceleration;       // rad/s^2, world axes
    uint validFields;                  // Missing derivatives are not fabricated zeros
}
```

Units: meters, seconds, Hz; Unity XYZ world coordinates. The session has one fixed integer sample rate. The take stores that rate, its origin, and the scene hash. Asset hashes identify actual geometry/material/PCM data, not merely their Unity GUIDs. Saved data uses versioned fields and explicit little-endian array formats; never dump the raw memory of a C# object graph.

Take metadata is JSON; pose/response chunks use explicitly described fixed-width arrays. Store position/quaternion values as float32 without decimal rounding and sample indices as int64. The schema defines quaternion order xyzw, array shape, endianness and offsets. The same recorded values feed both the live-solver snapshot log and replay; the take is not serialized from packed GPU buffers.

Motion measurements use the same timestamp, coordinate frame and reference point as the recorded pose. The general adapter estimates velocity from successive observed world poses and acceleration from successive velocity estimates, with elapsed time from their sample indices. Angular estimates use relative quaternions, not Euler-angle subtraction. Record the derivative method/window and whether values were estimated or supplied by a timestamp-matched motion driver. Initial samples and teleports invalidate derivatives until sufficient continuous history exists; a teleport is not an enormous physical velocity.

Native Rigidbody velocities are usable only with the matching physics pose/time. An interpolated visual Transform can represent another time; do not combine it with an unsynchronized physics velocity. Prefer the common observed-pose estimator when that synchronization is unavailable. Attached source/listener points include rotational motion: rigid-body point velocity is `v_origin + omega × (point - origin)`. Unity also exposes a point-velocity API that accounts for angular velocity. [Rigidbody interpolation](https://docs.unity3d.com/2022.3/Documentation/Manual/rigidbody-interpolation.html), [Rigidbody.GetPointVelocity](https://docs.unity3d.com/2022.3/Documentation/ScriptReference/Rigidbody.GetPointVelocity.html)

Stored samples restore their captured motion values along with poses. Between samples, kinematics remain estimates under a recorded interpolation policy; they cannot recover unobserved collision impulses. The initial linear-pose replay uses piecewise path velocity for delay-based audio; recorded acceleration remains a measurement/learning feature, not a second trajectory integrator. A smoother trajectory interpolant must be explicit and validated before a velocity-based acoustic method relies on its derivatives. Recording frequency is independent of acoustic update frequency, so capturing motion need not trace sound at every observation.

Geometry primitives refer to local material slots; an object's bindings resolve these to canonical material indices during upload. Colliders normally use slot zero; mesh submeshes can use distinct slots. Source/listener definitions record their object binding; effective role flags in the state record component enablement independently of object activity. Disabling emission mutes new dry samples while voice cursors continue on the take clock; explicit Stop events end voices. Preserve already emitted sound and tails. Disabling geometry removes it from subsequent solves. These changes take effect at recorded sample positions.

**Rigid-motion rule:** bake fixed authoring scale into local geometry when compiling. During the take only translation and rotation may change. Compilation/capture checks that applying the recorded rigid pose to the baked geometry reproduces the object's Unity transform. Changing stretch/shear/deformation invalidates this v1 contract. This allows ordinary scaled room objects without implementing deformable geometry recording.

Geometry extraction is explicit:

- 3D: native primitive meshes/readable mesh geometry become triangles; selected primitive colliders use the providers above. Visual meshes and acoustic proxies may differ, but the selected acoustic geometry and tessellation settings are what get hashed and replayed.
- 2D: supported collider boundaries/open edges or authored segments, transformed into the chosen plane. Include collider offsets. Out-of-plane motion is rejected within a recorded tolerance; no silent longest-axis projection. Automatic slicing of arbitrary moving 3D meshes is outside v1.
- Degenerate primitives, invalid coefficients, unreadable geometry and unknown references fail compilation with object IDs. Empty acoustic geometry is valid for a free-field scene.

`Planar2D` initially means paths constrained to a plane with a declared pressure-spreading convention. For the comparison fixtures, choose direct delay `r / c` and unit-input pressure gain `referenceDistance / max(r, minimumDistance)`, with positive distances and sound speed recorded in the physics config. Its squared magnitude is the corresponding direct energy gain. Use this same direct convention in 2D and 3D; derive indirect receiver/sampling normalization separately in each dimension. This is a planar approximation, not a cylindrical 2D wave equation. Different physical conventions cannot share unlabeled training targets.

Surface sidedness, intersection epsilon policy, homogeneous-air attenuation and scattering law are versioned physical settings. The initial opaque model uses two-sided reflecting acoustic surfaces with normals oriented against incidence. It does not infer solid interior media from triangle winding. Frequency-dependent air loss, if enabled, uses declared units and coefficients. Geometric rays do not automatically supply diffraction, low-frequency room modes or full moving-boundary wave propagation; increasing rays/bands cannot introduce these missing effects.

## 4. Record/playback contract

```csharp
sealed class Take {
    int schemaVersion;
    Hash takeHash;                     // Final content hash after capture closes
    Hash sceneHash;
    int sampleRate;
    long durationSamples;
    PoseTrack[] poses;                  // Object index, sample, pose, motion, validity/role flags
    VoiceEvent[] audioEvents;           // Sorted by (sample, sequence)
    Hash[] dryPcmAssets;
    SnapshotRecord[] submittedStates; // Exact state/hash consumed by each live solve
    CaptureReport report;              // Clock policy, gaps, unsupported changes
}

struct VoiceEvent {
    long sample;                       // Effective emission time
    uint sequence;                     // Stable ordering within this sample
    uint voiceId;                      // Identifies one playback instance
    uint sourceIndex;
    VoiceAction action;                // Play, Stop, SetGain
    uint clipIndex;                    // Play only
    long clipStartSample;              // Play only
    float gain;
    bool loop;
}

SceneState Evaluate(Take take, long sample);
VoiceId Play(uint source, uint clip, long startSample, long clipOffset, float gain, bool loop);
void Stop(VoiceId voice, long sample);
```

The source component's convenience `Play()` schedules the earliest writable sample; capture saves the effective accepted time, including any scheduling delay. Explicit scheduling at a past/unwritable sample returns an error. An offline block is `[startSample, startSample + count)`; events split processing inside the block, so a play event is not rounded to the next block boundary. Looping and source sample cursors follow the recorded clip offset and loop rule.

Record a complete initial state at sample zero, representing any already active voices with Play events and their actual clip offsets/gains. Start with empty propagation history, or capture an explicit pre-roll; do not silently reuse sound from before the take. Define the final render range and tail policy, including whether geometry holds at the final state or is captured through the tail. Random access/segment rendering replays enough prior dry audio and response history to reconstruct the chosen range; starting a convolver empty in the middle of a take is a different result.

Prepare clips into canonical mono float32 PCM at the session rate before recording; hash those samples. Resampling/downmixing is a preparation step with recorded settings. Procedural/live audio must supply recorded dry PCM. Unity `AudioSource` playback is not the authoritative source signal: our renderer applies the take's events to PCM. The final output may use a Unity audio sink, but must not also play a duplicate spatialized source.

Capture live state after movement/physics/animation updates at an explicit session sampling point. Map the observation to the session audio clock using an origin anchored to `AudioSettings.dspTime`; preserve this mapping and the sample rate. This clock is based on processed audio samples rather than frame time. [Unity DSP clock](https://docs.unity3d.com/2022.3/Documentation/ScriptReference/AudioSettings-dspTime.html)

Capture timestamped control observations as well as the exact snapshots consumed by the live solver. Recording more reads of the same game-frame pose does not provide finer motion information. Require monotonic samples per pose track; same-sample observations are coalesced to the final value and counted in the capture report. Output-device rate changes or clock discontinuities end the take/segment; they cannot silently retime it.

At stored samples, return the recorded values exactly. Between observations, use linear position interpolation and shortest-arc quaternion interpolation, with an explicit hold across teleport/discontinuity boundaries. This interpolation policy is part of the take version. Dense output requests interpolate recorded motion; they do not rerun gameplay or invent ground-truth intermediate motion.

For replay, disable the live movement/animation/physics writers for participating objects. The simulator evaluates take data directly; applying those poses to GameObjects is optional visualization. Validate matching scene hashes before starting. A take remains valid if a quality setting changes, but not if a referenced asset has changed.

There are two deliberate replay operations:

- **Render the take again:** use the recorded state/audio, a declared snapshot schedule and a new render configuration. This creates low/high compute comparisons independent of the original machine's stalls.
- **Review the original run:** use its submitted states and render application log. Record `(responseId, stateSample, appliedSample, transitionSamples)` plus underruns/drops and renderer configuration. Saving the actual output PCM gives exact playback of the application stream, including gaps. This does not measure the physical DAC/headphone output or promise identical re-simulation across GPUs.

Submitted state logs alone cannot reproduce the audible effect of late responses. Keep state time, GPU completion time and actual DSP application time separate. A slow capture writer uses a bounded queue; an overflow creates an explicit incomplete interval or stops capture. It never blocks the audio callback or silently labels a take complete.

## 5. Exact C# ↔ HLSL layouts

GPU structs use complete 16-byte lanes: `Vector4`/`float4` and a four-`uint` struct/`uint4`. All fields are 32-bit. GPU indices are integers, not integers encoded as floats. Absolute sample time remains a 64-bit CPU value; kernels receive short relative times and a derived seed.

Unity requires C#/HLSL structured-buffer strides to match and recommends multiples of 16 for portable layouts. The following is our explicit ABI, to be verified on Metal and DirectX, not assumed from type names. [Unity structured buffers](https://docs.unity3d.com/2022.3/Documentation/ScriptReference/ComputeBufferType.Structured.html)

```csharp
[StructLayout(LayoutKind.Sequential, Pack = 4)]
struct UInt4 { public uint x, y, z, w; } // 16 bytes

[StructLayout(LayoutKind.Sequential, Pack = 4)]
struct GpuTriangle {                    // 64 bytes
    public Vector4 a;                  // offset  0: world xyz, w = 0
    public Vector4 b;                  // offset 16: world xyz, w = 0
    public Vector4 c;                  // offset 32: world xyz, w = 0
    public UInt4 ids;                  // offset 48: object, material, primitive, flags
}

[StructLayout(LayoutKind.Sequential, Pack = 4)]
struct GpuSegment {                     // 48 bytes
    public Vector4 a;                   // offset  0: plane u,v, z=w=0
    public Vector4 b;                   // offset 16: plane u,v, z=w=0
    public UInt4 ids;                   // offset 32: object, material, primitive, flags
}

[StructLayout(LayoutKind.Sequential, Pack = 4)]
struct GpuMaterialBand {                // 16 bytes; one material at one transport band
    public Vector4 coefficients;       // x: absorption; y: scattering; z=w=0
}
```

```hlsl
struct GpuTriangle {
    float4 a;
    float4 b;
    float4 c;
    uint4 ids;
};
struct GpuSegment {
    float4 a;
    float4 b;
    uint4 ids;
};
struct GpuMaterialBand {
    float4 coefficients;
};

#if SCENE_DIMENSION == 3
StructuredBuffer<GpuTriangle> Triangles;
#else
StructuredBuffer<GpuSegment> Segments;
#endif
StructuredBuffer<GpuMaterialBand> MaterialBands; // [materialIndex * BandCount + bandIndex]
StructuredBuffer<float4> Bands;                 // lower Hz, center Hz, upper Hz, 0
StructuredBuffer<uint> PrimitiveIndices;
```

`ids`: x = canonical object index, y = material-table index, z = stable primitive index within this compiled scene, w = flags (`bit 0 = active`; remaining bits zero). Reordering BVH indices does not change object/primitive identity. Normals are derived from triangle winding/segment direction and oriented by the hit-side rule. Do not upload a separate normal that can disagree with updated vertices.

Primitive material indices resolve default/submesh bindings into a shared table; they do not duplicate material coefficients. The existing material's physical sliders can resolve coefficient curves before compilation, but record the resulting values and derivation version. Heuristic values are not automatically measured material data. Missing material bindings fail validation unless the author explicitly chooses a named fallback material.

`BandCount` is a render setting, not a struct size. Store material coefficients at their authored frequency knots in `SceneData`; preserve the interpolation and out-of-range policy. Build/cache `MaterialBands` from those curves and the requested transport grid. A grid change affects the render hash, not scene identity. The existing eight authored bands can supply a curve, but interpolating it onto more bands does not create measured detail. Each transport band declares its edges and representative frequency; its coefficient evaluation/averaging rule is part of the model. Band ranges must be ordered, nonoverlapping and within `[0, sampleRate / 2]`. Scalar index/energy buffers use explicit 4-byte elements; record structs retain the 16-byte-lane convention.

The v1 material ABI models absorption and scattering. Transmission/IOR/damping from the legacy shader are not silently copied into this smaller model: configurations requesting unsupported physics fail validation. A legacy comparison run identifies its separate physics model. For opaque surfaces, reflected energy is `1 - absorption`; scattering partitions that reflected energy. A coherent specular pressure coefficient has a different meaning from either energy fraction; its phase convention must be declared.

Other buffers use the same packing rule:

| Struct | Lanes and offsets | Bytes |
|---|---|---:|
| `GpuBvhNode` | 0: `float4 min`; 16: `float4 max`; 32: `uint4 links` | 48 |
| `GpuEndpoint` | 0: `float4 positionRadius`; 16: `float4 rotation`; 32: `uint4 ids` | 48 |
| `GpuEnergyArrival` | 0: `float4 directionDelay`; 16: `uint4 ids` | 32 |

For a BVH node, `links = (leftChild, rightChild, firstPrimitiveIndex, primitiveCount)`. A nonzero primitive count identifies a leaf; children are `UINT_MAX`. An internal node has two valid child indices and zero primitive count. Empty geometry uses node count zero and performs no traversal. A flat `PrimitiveIndices` array provides leaf ordering, preserving contiguous geometry ranges per object for updates.

For an endpoint, `positionRadius.xyz` is world xyz in 3D or `(u,v,0)` in 2D; w is the receiver capture radius for listeners, zero for point sources. `rotation` is normalized quaternion xyzw. `ids = (objectIndex, flags, 0, 0)`. Sources and listeners are separate arrays, so their array indices identify their roles. Receiver radius is an estimator parameter recorded in render settings; it must not introduce an unexplained output-level change.

For an energy arrival, `directionDelay.xyz` is the final propagation direction toward the receiver, expressed in world xyz or planar `(u,v,0)`; w is delay seconds. `ids = (sourceIndex, listenerIndex, reflectionOrder, flags)`. For the current dispatch, a separate float32 buffer stores `ArrivalEnergy[arrivalIndex * ActiveBandCount + localBandIndex]`; the global band is `BandBase + localBandIndex`. `ActiveBandCount` is a bounded batch within the full `BandCount`, and can be one. Changing resolution does not change the record ABI or require an all-arrivals/all-bands allocation. Initialize every energy element belonging to a valid arrival. Contributions include sampling-PDF compensation and normalization for the complete estimator budget; frequency sampling, if used, also needs its own probability correction. Chunking must preserve these weights, rather than renormalizing each chunk independently. Coherent pressure arrivals require a different quantity/ABI; never reinterpret this energy buffer as complex pressure.

`reflectionOrder >= 1` for these indirect arrivals; direct sound has its own deterministic result. A valid arrival has flag bit 0 set. Unwritten/invalid slots are zero and never contribute to reduction. Define each band value as the incoherent sum/integral of squared unit-input pressure path gains within its delay bin, evaluated using the band's declared frequency model. It is not a per-second density, calibrated SPL, or energy integrated across that band's width. Do not sum bands as physical broadband energy without a declared source spectrum and synthesis/filter weighting. The estimator must account for its receiver acceptance measure to approach this target. Its exact sampling/receiver correction must be derived and verified against analytic fixtures before that transport implementation is accepted.

GPU globals include explicit logical counts (including `BandCount`), pair indices, total ray budget, global ray offset, seed, bounce limit, speed of sound, response limits, and buffer capacity. Validate array-size products and device memory limits before allocation. Bind only buffers declared by that kernel; bind valid initialized storage even when its logical count is zero. Each kernel guards its dispatch index before any buffer access. The portability baseline uses buffer operations and does not rely on texture atomics or implicit buffer-size discovery. [Unity compute portability](https://docs.unity3d.com/2022.3/Documentation/Manual/class-ComputeShader.html)

Estimate memory before submitting: response energy storage is `4 * pairCount * bandCount * delayBinCount` bytes; arrival scratch is `capacity * (32 + 4 * ActiveBandCount)` bytes, plus counters/reduction storage. For illustration, one million arrivals with 4097 bands would require 16.388 GB for energy alone. That is a reason to tile, not a proposed quality preset. Bounded batches reduce into response tiles, which offline capture can stream to disk. Realtime requests must fit their resident response/DSP budget; the planner reports an error or an explicitly selected smaller configuration instead of silently reducing fidelity.

## 6. Geometry updates and resource ownership

V1 compiles local geometry once, assigns each object contiguous primitive ranges, expands it into world-space GPU primitives, and builds one BVH. A changed rigid pose updates that object's ranges and refits affected BVH bounds. Static ranges remain untouched. Keep primitive ordering separate from BVH ordering. A full BVH rebuild changes the acceleration structure only, not primitive IDs. Treat 2D similarly in plane coordinates.

Moving a capsule or box does not regenerate its local shape each frame. Unity produces its new pose; the simulator updates its world-space primitives/bounds at the requested acoustic snapshot. Kernels consume only the motion channels their physical model needs. The initial frozen-geometry tracer needs geometry at a time; it does not advance objects using acceleration. Future moving-boundary kernels require an explicit motion buffer/physical contract, while CPU motion records already support diagnostics and learning features.

This deliberately starts with flat world-space geometry. It avoids a second level of instance traversal and shader matrix rules. Record geometry-upload and refit costs; use those measurements to decide whether per-mesh BVHs and instance traversal are needed later. Do not introduce that system before it earns its cost.

The session owns every buffer. Submitted work leases immutable geometry, endpoints, parameters and output slots until GPU work and required readback complete. A later snapshot cannot overwrite memory still used by an earlier dispatch. Use a small bounded set of reusable slots; when all slots are busy, offline waits and realtime records a missed update while retaining the last compatible response.

Within a session/configuration, completed responses publish in increasing state-sample order; an older result finishing after a newer applied result is discarded and counted. A seek, new take or incompatible configuration creates a new generation and invalidates previous publications/history. First playback has an explicit policy: prepare a valid response before starting, or use a declared direct-only fallback. Never play uninitialized response storage.

Arrival output is bounded. If a counter-based writer fills capacity, it increments an overflow counter and never writes out of bounds. An overflow marks that solve invalid; offline retries with a fitting dispatch partition/capacity, and realtime retains the previous valid response. Histogram/reduction passes consume the GPU-produced valid count; do not synchronously read hit counts to the CPU merely to launch the next pass. Response accumulation uses a defined race-free reduction, not unsynchronized float additions.

Read back the response needed by the audio renderer and small diagnostics, not all paths by default. Detailed path capture is a diagnostic mode whose cost is reported. Copy readback data into owned memory before releasing its request/slot; retain resources until completion, including on cancellation. Unity provides asynchronous GPU readback, which fits this ownership rule. [Unity async readback](https://docs.unity3d.com/2022.3/Documentation/ScriptReference/Rendering.AsyncGPUReadback.Request.html)

## 7. Solver input and output

### Methods, representations and numerical controls

The planned transport methods are a geometric energy estimator and, after separate validation, a phase-aware geometric pressure estimator. Both share 2D/3D scene geometry and deterministic direct visibility/delay. Frequency resolution and ray budget configure a method; they do not create a new method for each setting.

| Axis | Contract |
|---|---|
| Geometry | `Planar2D` segments or `Spatial3D` triangles, selected by the compiled scene. |
| Transport | `GeometricEnergy` first; `GeometricPressure` after pressure/material-phase and path-sampling validation. Direct-only operation is a supported baseline. |
| Material/transport frequencies | Explicit ordered band edges/centers, coefficient interpolation and integration rules. Log, linear or custom spacing; no fixed count. |
| Response | `BandEnergy`, `PressureIR`, or `ComplexTransfer`, with required origin/provenance and axes. |
| Sampling | Initially `raysPerPairPerBand`, reflection-order limit, sampler version and seed. Record actual traced counts. |
| Time | Exact requested scene samples, response support in samples, and delay-bin width for energy. Independent of STFT hop. |
| Audio processing | Direct fractional delays, a declared energy-to-IR synthesis where needed, and time-domain or partitioned FFT convolution. STFT is a separate analysis/model option. |
| Execution limits | In-flight slots, memory, batch sizes and work deadlines. These bound execution without redefining normalization. |

The first energy implementation can process one frequency band at a time, with independent sampling per pair/band. Shared paths across bands or listeners are optimizations only when their estimator/PDF rules remain correct. Scattering that varies with frequency cannot be assumed to give every band the same unweighted path distribution. Rays estimate a unit-input response; do not tie the estimator to the currently playing clip's spectrum. Any later importance sampling of excitation must preserve full support and correct weights.

Ray count primarily controls variance; reflection-order limits and response duration can truncate energy and introduce bias. The first low/high pair changes only ray count, keeping bounce limit, physical model, grids and cadence equal. Subsequent experiments can vary these other controls explicitly. No universal low/high numbers or fixed realism presets are selected here.

The dispatch pipeline has distinct responsibilities: initialize output, update geometry/BVH, solve direct paths, trace indirect batches, reduce/finalize responses, then synthesize/render audio as required. Shader entry-point count is an implementation result: kernels can specialize or fuse stages after profiling. A software BVH is the common compute baseline; this design does not require hardware ray-tracing APIs.

```csharp
sealed class SolveRequest {
    ulong generation;                 // Session/seek/configuration epoch
    Hash requestHash;                 // Canonical scene/state/config/seed identity
    Hash sceneHash;
    SceneState state;                  // Immutable for the duration of the solve
    Pair[] pairs;                      // Each pair is (sourceIndex, listenerIndex)
    PhysicsConfig physics;             // Dimension, effects, transport bands, estimator version
    ResponseFormat format;             // Quantity + shape + time/frequency axes
    QualityBudget quality;             // Rays per pair/band, bounces, execution limits
    uint seed;
}

sealed class ResponseFrame {
    ulong generation;
    Hash requestHash;
    Hash sceneHash;
    ulong snapshotId;
    long sample;                       // Which scene time this result describes
    Hash configHash;
    ResponseOrigin origin;            // Energy estimate, synthesized pressure, or traced pressure
    Pair[] pairs;
    uint[] pairFlags;                  // Fixed pair order; activity/validity mask
    DirectPath[] direct;               // Deterministic direct pressure paths
    ResponsePayload indirect;          // Explicit quantity and array layout
    SolveStats stats;                  // Validity, counts, timings, overflow/truncation
}

JobHandle Submit(SolveRequest request);
bool TryComplete(JobHandle job, out ResponseFrame frame);
void Cancel(JobHandle job);            // Prevent consumption; retire GPU resources safely
```

This can be one concrete C# solver class initially. The function boundary makes input/output testable without a generic backend/plugin framework.

`requestHash` includes the evaluated state bytes/hash, not just its timestamp. `configHash` covers the transport method/version, frequency and delay grids, medium/receiver parameters, approximation policies, numerical budgets and response format. Run metadata additionally records implementation/shader hashes and device/backend. The seed is explicit in request identity. Do not let a cache treat two renders as equivalent merely because they have the same take and requested sample.

Keep the requested pair order fixed across a take. Inactive pairs produce masked zero output; they do not disappear and shift later pair indices. The solver validates supported physics/quantity combinations before allocating/submitting. Unsupported modes return a configuration error, not a silent fallback. A cancelled result or one tagged with another scene/session cannot be published to the renderer.

Direct paths carry visibility, delay, pressure gain, and arrival direction per pair. With frequency-dependent direct attenuation, pressure gain is a curve on a declared grid, not one broadband scalar. The indirect payload excludes those direct contributions, so rendering/reconstruction cannot accidentally double them. A saved total IR/spectrum has an explicit `includesDirect` tag. Its quantity is mandatory:

All responses describe a unit input; voice gain is applied once to the dry source signal in DSP. `PressureIR` uses a unit discrete impulse (`x[0]=1`, other samples zero). These are relative response units, not calibrated sound-pressure levels.

| Quantity | Array layout | Meaning / consumer |
|---|---|---|
| `BandEnergy` | float32 `[pair][band][timeBin]` | Nonnegative indirect energy; metadata defines bin width in samples and the normalization/model. Response synthesis or reconstruction consumes it. |
| `PressureIR` | float32 `[pair][sample]` | Signed unit-input pressure response; audio convolution consumes it. Record the synthesis method if derived from energy. |
| `ComplexTransfer` | float32 `[pair][bin][real,imag]` | Complex pressure frequency response; includes FFT size, phase/time origin and grid. Origin distinguishes physically traced phase from synthesized phase. |

Flattening is row-major in the stated order; dimensions, axes, units and numeric format accompany every saved payload. A complex spectrum is the nonnegative half of a real-signal transform; DC/Nyquist imaginary components are zero and the negative half is conjugate-symmetric. Specify forward sign `exp(-i2πkn/N)` and inverse normalization `1/N`. IR support, transform size and convolution padding must avoid unintended circular wrap.

Broadband (one band) and configurable multiband transport share geometry and an energy-arrival ABI; broadband coefficients/aggregation are an explicitly named approximation. Band count is not a new solver method or a requirement for one shader per band. A coherent spectral transport variant shares scene buffers but must declare its own physically consistent pressure/phase estimator, including complex reflection coefficients or an explicit material-phase approximation. An FFT of an energy histogram is not such an estimator. Taking `sqrt(energy)` also does not recover path signs or phase. A pressure IR may be rendered using FFT convolution without changing its physical model.

### Independent frequency and time settings

Transport bands describe the frequency dependence of propagation/materials. STFT bins describe windowed pressure audio used by DSP or a learning model. They need not have the same count or spacing. Energy-to-pressure synthesis and any band-to-bin mapping remain explicit operations.

```csharp
struct StftConfig {                    // DSP/learning setting, not geometry or take data
    int windowSamples;                // W: observed samples per analysis frame
    int fftSize;                      // N >= W; real audio has N/2 + 1 stored bins
    int hopSamples;                   // H: distance between successive frame starts
    WindowSpec analysisWindow;        // Kind/parameters or explicit coefficient asset
    WindowSpec synthesisWindow;
    OverlapNormalization overlap;     // Explicit reconstruction normalization
    FrameAlignment alignment;         // Causal or centered; exact sample origin recorded
    BoundaryPadding padding;          // Explicit behavior at take boundaries
}
```

The take fixes `sampleRate`. For even `N`, bin `k` is `k * sampleRate / N`; adjacent bins are spaced by `sampleRate / N` Hz. Window duration is `W / sampleRate` seconds and frame spacing is `H / sampleRate`. Actual frequency resolving power depends on the window length and shape: increasing `N` with fixed `W` only zero-pads and samples the same spectrum more densely. Hop size is neither acoustic snapshot cadence nor total output latency. Record actual buffering/lookahead separately; centered analysis needs future samples.

For reconstructable STFTs, declare the synthesis window and overlap normalization, including padded boundaries. Initially support validated power-of-two FFT sizes with `0 < H <= W <= N`; reject configurations without a valid reconstruction normalization or enough device memory. Supported sizes are backend capabilities, not an architectural limit of 1024. FFT size must not equal thread-group size by assumption: larger transforms need multiple values per thread and/or passes. FFT implementations may use size specializations without becoming separate acoustic solvers. Partitioned convolution has its own block/FFT settings; it is not forced to use the learning STFT grid. SciPy's explicit window/hop/FFT/dual-window conventions provide an independent offline comparison, not a runtime dependency. [SciPy ShortTimeFFT](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.ShortTimeFFT.html)

`AudioSTFT[source or listener][frame][frequencyBin][real,imag]` includes frame-start samples and the full STFT configuration. A `ComplexTransfer` response is the FFT of a pressure IR at one scene snapshot; it is not automatically an audio STFT. Keep propagation delay, scene snapshot time and STFT frame time distinct. The transport grid, response delay-bin width/tail, acoustic update cadence, STFT settings, and ray/bounce budget are independently declared render settings, subject to model/backend validation.

The first stochastic learning pair should use the same `BandEnergy` model/shape with low versus high ray budgets. Retain deterministic direct sound. Synthesize corresponding wet audio with the same documented method/seed when comparing audible effects. Choosing a coherent complex-response target later requires its own correctness gate; do not label the current prototype's spectral output as ground truth merely because it uses FFTs.

Response duration, time-bin width, band centers, receiver model, scene state, source PCM and physical effects are fixed for that first pair. A changed snapshot cadence or physics model is a different experiment. Record truncation at response limits and exclude invalid solves. Reference budgets need convergence checks across independent seeds; preserve uncertainty rather than treating one noisy target as exact.

Later experiments may explicitly pair different band/STFT resolutions. Save both grids, frame origins, target quantity and alignment/mapping rules; do not silently reshape or interpolate labels. A trained model declares the grids it accepts (or its explicit conditioning/remapping scheme), so simulator configurability does not imply that one model accepts every configuration.

## 8. One path through Unity, recording, shaders and replay

```csharp
// Main/control thread: Unity calls and GPU submission only.
BeginSession() {
    scene = CompileRegisteredObjects(root, sampleRate);
    ValidateSceneAndGpuLayouts(scene);
    clock = StartSessionClock(sampleRate);
    recorder.Begin(scene.sceneHash, sampleRate);
    solver = new Simulator(scene);
    pending = new List<JobHandle>();
}

AfterSceneWriters() {
    state = mode == Live
        ? CaptureRegisteredTransforms(clock.SampleNow)
        : take.Evaluate(requestedSample);

    recorder?.AppendObservedState(state);
    if (AcousticUpdateDue(state.sample) && solver.HasFreeSlot) {
        recorder?.AppendSubmittedSnapshot(state);
        request = BuildSolveRequest(scene, state, physics, format, quality, seed, generation);
        pending.Add(solver.Submit(request));
    }
    foreach (job in pending) {
        if (solver.TryComplete(job, out response)) {
            ValidateIdentityAndQuantity(response);
            capture?.Append(response);
            dspWorker.PublishIfCurrentAndNewer(response);
            MarkJobForRemoval(job);
        }
    }
    RemoveCompletedJobs(pending);
}

// Offline: same state/solver contract, fixed requested sample sequence.
foreach (sample in jobSpec.responseSamples) {
    state = take.Evaluate(sample);
    low  = await Solve(scene, state, commonPhysics, commonFormat, lowBudget, seed);
    high = await Solve(scene, state, commonPhysics, commonFormat, highBudget, seed);
    capture.AppendPair(take.takeHash, state, low, high);
}
foreach (render in jobSpec.renders)
    RenderAudio(take, capturedResponses[render.name], render.dsp, jobSpec.outputRange);
capture.CompleteOnlyAfterAllOutputsAndChecksSucceed();
```

The random key derives from experiment seed, snapshot identity, stable source/listener identity, band identity, global ray index and bounce—not `Time.frameCount`, current buffer order, or dispatch chunk size. Record the sampler version. For a matched low/high subset, the generated ray must not depend on the final ray count (the existing angle stratification does). Use budget-independent sample keys and separate reference seeds to estimate convergence. Cross-GPU results use explicit numerical/statistical tolerances rather than a promise of bitwise acoustic identity. Independent offline submissions for the same evaluated state must use the same snapshot identity.

The DSP worker applies the take's voice events and direct delays, synthesizes/receives pressure responses, and performs convolution. Its source-history buffers include propagation/IR tails. Moving direct-path delay is interpolated at audio rate; that resampling handles the chosen Doppler approximation without an additional duplicate pitch shift. Replacing an indirect response crossfades convolution outputs without resetting source history. Snapshot-based indirect geometry remains a recorded quasi-static approximation.

For the initial moving-scene model, each response freezes geometry at its recorded state sample and the renderer applies it according to the declared schedule. Direct delay from instantaneous separation is also a motion approximation, not a solved retarded-time boundary problem. Changing old propagation paths when an object moves has a declared crossfade/tail policy; this is not exact wave transport through moving surfaces. These limitations accompany moving-scene training data.

The DSP worker owns convolution state and produces preallocated PCM blocks. Begin with CPU DSP so the audio callback never waits for GPU completion; profile the cost before moving convolution or inference onto a competing GPU queue. Use partitioned convolution when IR lengths justify it, with a short CPU convolution as a correctness reference. Energy-to-pressure synthesis needs calibrated per-band filter normalization, reproducible random phase, and temporal continuity; do not choose independent synthesis noise every snapshot. Log the chosen synthesis seed/version and response transition length. Audible validation is required before using synthesized pressure as an audio target.

The audio callback only copies from preallocated output storage, fills an underrun with silence, and increments counters. It never calls Unity scene APIs, submits/waits for GPU work, allocates capture lists, or writes files. The GPU host remains active for offline HLSL computation; batch mode does not mean compute can run without a graphics device/context.

## 9. Checks required before calling this contract implemented

1. Assert every C# GPU stride and field offset. A GPU echo kernel returns distinctive per-field values correctly on Metal and DirectX, including integer indices. Verify material/arrival indexing at several band counts, including one and a count not divisible by four.
2. Native box/sphere/capsule/cylinder and 2D shape fixtures compile from their selected mesh/collider geometry, including offsets, scale and parent transforms. They can move, disable, record and replay without changing object/primitive/material identity. Constant velocity/acceleration, rotation, collisions and teleport fixtures verify motion timestamps, units and validity. Several sources do not overwrite each other's clips or endpoints.
3. BVH hits match brute force after geometry updates, including empty scenes and dispatch sizes not divisible by thread-group size.
4. Saved scene/state/event data round-trips; event samples inside a block remain exact; replay poses at stored timestamps match and gaps/discontinuities stay explicit. Original-run capture preserves applied-response samples and output PCM; seeking respects declared pre-roll/tails.
5. Direct delay/gain and analytic reflection fixtures pass. Increasing ray count lowers variance without an unexplained mean-level change; receiver normalization is checked independently.
6. IR/spectrum conversions obey quantity, phase, normalization and padding conventions. STFT round trips reconstruct at supported FFT/window/hop settings, including padded boundaries; unsupported settings fail validation. Validate the existing FFT indexing issue before reusing its implementation.
7. A full output queue, GPU error, cancellation or arrival overflow never produces a falsely complete training pair or premature buffer reuse. Out-of-order completions cannot replace newer responses; a seek/configuration generation rejects stale work.
8. Two renders of one take preserve scene hash, state timestamps, PCM and pair identities; only declared render settings differ. Collect stage timings without synchronous diagnostic readbacks in the realtime path.

Additional numerical fixtures cover full absorption, specular reflection from an analytic plane, receiver-radius convergence, several frequency grids, batch-size invariance and dense-versus-tiled reduction. Energy and pressure tests use different expected quantities. Statistical comparisons use repeated seeds and stated confidence/tolerances; a single noisy run is not an assertion of correctness. Test absolute levels as well as normalized shapes so an incorrect gain cannot hide in normalization.

## 10. Saved data and offline jobs

Unity authoring stays in ordinary `.unity` scenes, prefabs, meshes and material assets. The exported scene/take/run files serve computation and replay; they do not replace Unity's editable scene format. Export local geometry and material curves once, then record timestamped object state/audio events separately. Do not serialize an entire Unity scene or duplicate unchanged geometry at every observation.

In memory, both execution modes use the same indexed `SceneData`, evaluated `SceneState`, `SolveRequest` and `ResponseFrame` types. HLSL receives packed numeric buffers derived from them. Files have an explicit schema independent of GPU padding and Unity object serialization, so the Python dataset reader does not need to launch Unity.

| Execution | State/clock source | Scheduling and output |
|---|---|---|
| Live | Observe Unity objects against the session audio clock. | Submit bounded asynchronous work; render current compatible responses to the audio device. Record missed deadlines and optional take/run data without waiting in the callback. |
| Offline replay | Evaluate the take at requested integer sample positions. | Use the same solver/DSP, wait for complete results, stream responses/PCM/diagnostics to disk. Wall-clock speed does not advance scene time. |
| Offline take generation, when needed | Drive authored motion/physics with an explicit simulation schedule and virtual sample clock. | Record state/audio events into a take first; low/high acoustic renders consume that take. This is a later extension, not a prerequisite for offline replay. |

The initial offline path consumes a recorded take. If automated take generation is added, it does not require audio playback or a live DSP clock: record the simulation-to-sample-time mapping and tick schedule instead of using wall-clock frame duration. Reproducibility of a newly generated physics trajectory across machines is separate from exact replay of the stored take.

Keep metadata readable and arrays simple: versioned JSON manifests plus little-endian binary chunks. Store schemas in code beside their serializers and document them here; avoid an additional schema-generation toolchain. Every array descriptor includes quantity, dtype, shape, axes, units, byte offset/length and content hash. Validate descriptors and chunk checksums on load. Waveform exports may additionally be float WAV; they do not replace response/provenance data.

Artifacts live under a configurable data root outside `Assets`, so Unity does not import recordings or datasets. Add the chosen local output root to ignore rules when implementation begins; commit experiment definitions and small fixtures, not bulk captures. The logical layout is:

```text
<data-root>/
  scenes/<scene-hash>/       manifest, compiled geometry/materials, asset references
  assets/<content-hash>/    canonical dry PCM and other shared payloads
  takes/<take-id>/           manifest, state/event chunks, exact submitted snapshots
  runs/<run-id>/             resolved config, responses, optional PCM, timing/validity log
  datasets/<dataset-id>/     manifest of aligned run/frame references and split assignment
```

Run/take IDs name in-progress work; content hashes identify finalized immutable data. Chunk paths are relative to the selected root, not machine-specific absolute paths. A dataset references original artifacts so one take does not get copied for every sweep. A portable export includes all referenced scene/PCM chunks and verifies their hashes before execution.

| Artifact | Required metadata |
|---|---|
| Scene | Schema version, dimension/plane, stable object/role tables, geometry/material curves and hashes, unit convention, authoring/export versions. |
| Take | Scene identity, sample rate and origin, sample range, initial state/voices, interpolation and discontinuity policy, state/audio event chunks, dry asset hashes, capture gaps/completeness. |
| Run | Take and evaluated-state identities, resolved transport/response/DSP settings, requested samples/range, seeds, code/shader hashes, Unity/backend/device versions, build mode, output descriptors and status. |
| Response | Pair order/masks, scene sample, request/config identity, direct-inclusion flag, quantity/origin, frequency/delay axes, estimator normalization, validity, truncation and overflow diagnostics. |
| Dataset | Exact low/high run/frame references, allowed differences, alignment, preprocessing version, reference uncertainty and split membership. |

A local job file describes inputs and requested work. Start with one take and one render configuration per invocation; a comparison can invoke it twice. The same resolved configuration is available from the Unity panel and the command line. The multi-render pseudocode below describes a later convenience loop over the same operation, not a requirement for a queue or job framework.

```csharp
sealed class RenderConfig {
    string name;                       // Unique within this job
    PhysicsConfig physics;
    ResponseFormat format;
    QualityBudget quality;
    DspConfig dsp;                     // Synthesis, convolution, optional STFT and timing
    uint seed;
}

sealed class RenderJob {
    int schemaVersion;
    AssetRef take;
    SampleRange outputRange;
    long[] responseSamples;             // Or a schedule expanded to this exact list
    RenderConfig[] renders;             // Concrete resolved configurations; no magic presets
    PairingRule pairing;                // Optional: same quantity/grid or explicit cross-grid task
    string outputRoot;
}

RunOffline(job) {
    ValidateInputsCapabilitiesAndMemory(job);
    foreach (render in job.renders) {
        foreach (sample in job.responseSamples) {
            state = take.Evaluate(sample);
            response = SolveAndWaitForCompletion(scene, state, render);
            AppendCheckedResponse(response);
        }
        RenderRequestedAudioWithHistory(take, render);
        FinalizeRunAfterFlushAndValidation();
    }
    ValidateAndWritePairManifestIfRequested();
}
```

The initial offline entry point runs one render in Unity with a graphics device available. Batch mode is appropriate; `-nographics` is not used for GPU simulation. Keep the player/editor loop alive until dispatch, readback, DSP and capture finish, then return a meaningful exit status. Do not assume a static `-executeMethod` returning has completed asynchronous work. Unity documents these launch and exit controls. [Unity Editor command-line arguments](https://docs.unity3d.com/2022.3/Documentation/Manual/EditorCommandLineArguments.html)

A GPU-less machine may validate/read artifacts but cannot silently claim to run the HLSL solver. An Editor batch run requires that project not already be open in another Editor. A built worker player is an option if running alongside authoring becomes necessary; do not add worker management or GPU job concurrency before that need exists. [Unity batch-mode constraints](https://docs.unity3d.com/2022.3/Documentation/Manual/EditorCommandLineArguments.html)

Write chunks to temporary files, close/check them, then publish the final manifest with `complete` status. Failed/cancelled runs keep diagnostics and cannot enter a training dataset. Initially restart an interrupted render; add resume only when run cost warrants it. If added, resume must verify input/config/code identities and reconstruct audio history; partial tiles are not completed frames. Capacity retries preserve sample identities and normalization, and stop with a resource error if even the minimum batch cannot fit.

## 11. Diagnostics and benchmarks

Every run emits a compact machine-readable report. Separate measurement from debugging: detailed rays, visual overlays and full path readback are opt-in modes with their costs reported. CPU submission time is not GPU execution time. Use asynchronous GPU timing where supported; if a backend cannot provide it, mark it unavailable instead of relabeling CPU timings.

| Area | Measurements |
|---|---|
| Scene preparation | Extraction, compilation, BVH build/refit and upload time; primitive/material counts; changed bytes per snapshot. |
| Transport | GPU direct/trace/reduce durations when available; traced rays and bounce distribution; accepted arrivals; overflow; truncation; per-band/pair cost. |
| Scheduling | Queue depth, submission-to-completion and completion-to-application time; response age in samples; skipped/stale updates. |
| Audio | Synthesis/convolution time, output queue depth, callback deadline misses/underrun samples, allocations and captured output discontinuities. |
| Learning | Feature preparation, transfer, inference, postprocessing, memory and added buffering/lookahead, measured with graphics and acoustics active. |
| Capture | Readback bytes/time, writer queue depth, serialization/write throughput, incomplete intervals and total data size. |
| Overall | Wall time, realtime factor, CPU/GPU memory peaks, steady-state p50/p95/p99 timings, maximum response age and total measured pipeline latency. |

For audio blocks of `B` samples at rate `Fs`, the callback interval is `B/Fs`; this does not mean the simulation must trace every block. Each benchmark specifies an acoustic update deadline, acceptable response age, audio buffering, and quality/error thresholds. Until those numerical targets and test hardware are attached to an experiment, report results rather than claiming a realtime pass.

Use the same recorded take, requested samples, physical model and output range for comparisons. Separate startup/import/shader compilation from a warm steady-state interval. Record repetitions, seeds, build settings, device and visualization/capture modes. Compare standalone runs as well as Editor iteration results; run under representative graphics load for the intended application. Include cold-start cost separately when it matters to playback startup.

Start with controlled fixtures and then the actual walkthrough: free field, a single reflector, a room, a moving door/occlusion transition, multiple sources, and denser imported geometry. Cover 2D and 3D using their declared physical conventions. Sweep one axis at a time before combined sweeps. Report quality versus total cost so a faster trace that causes worse audio deadlines does not appear to be a win.

Correctness metrics depend on the quantity: arrival delay/absolute gain for analytic paths; energy error and decay curves for `BandEnergy`; waveform and complex-response errors only for comparable pressure targets; audible transition artifacts and listening checks for moving rendered audio. Reference convergence includes independent seeds and increasing budgets. A normalized spectrogram alone cannot establish correct level, phase, decay or latency.

## 12. Learned acoustic reconstruction

“Audio DLSS” means a learned reconstruction stage built for these acoustic quantities. This design does not depend on NVIDIA's image-rendering DLSS SDK. The first experiment denoises a low-ray `BandEnergy` response toward a converged response from the same model. That tests whether learning buys useful quality per unit of compute before adding simultaneous frequency/time upscaling or changing physics.

```text
recorded state -> low-cost solver -> optional reconstruction -> response renderer -> wet audio
                      |                      ^
                      +-- aligned dataset ---+ <- offline converged reference

deterministic direct paths ----------------------------------> direct audio
```

The model boundary takes the low-cost response, its grid/sample/validity metadata, available motion/context features and bounded history; it returns the same declared response quantity on a declared target grid. Keep direct sound outside the first reconstruction target. Invalid predictions or missed inference deadlines use a compatible current low-cost response and record the fallback. A model is optional; normal simulation and rendering continue without one.

The model manifest fixes tensor names, dtype/shape/layout, units, frequency/time grids, normalization and inverse transform, context length, causal/lookahead policy, history-reset rules, supported scene dimensions, training dataset/preprocessing hashes and exported model version. Reset temporal state on seeks, teleports, pair changes or incompatible grids. Realtime features must be available by the measured inference deadline; future high-quality responses or centered future context cannot leak into a supposedly causal input.

Training/validation/test splits are grouped by scene/take family, so adjacent frames, alternate ray seeds and low/high versions of the same walkthrough do not leak across splits. Include unseen rooms, materials, trajectories and, for audio-conditioned models, dry clips. Preserve masks and uncertainty rather than treating missing/overflowed targets as zero. Start with direct low-cost output and a simple causal temporal filter/interpolator as baselines; compare a learned model at matched latency and total runtime cost.

Frequency/time upscaling is a separate declared experiment with both grids and their alignment saved. Interpolated material curves and high-ray geometric targets remain approximations; they do not teach phase/wave effects absent from the teacher. An energy target cannot uniquely specify the original pressure waveform. A later pressure/complex model therefore needs a validated pressure teacher or explicitly labeled synthesis target, and its own phase/temporal consistency evaluation.

The network architecture, training framework and inference runtime are intentionally not chosen in this contract. Select them after measuring dataset statistics, the required input/output shape and backend deployment cost. A model is successful only if quality improves over the simple baselines within the declared end-to-end latency/memory budget on the actual target device.

## 13. Implementation order and readiness

The next coding work is staged so each change produces something inspectable and reusable. These are work steps within the same codebase, not new services or permanent subsystems.

| Step | Concrete deliverable | Completion evidence |
|---|---|---|
| 1. Record and replay | Native mesh/collider providers, numeric scene/material/motion types, Unity adapters, stable IDs, sample-timed voices and immutable take serialization. A 2D and a 3D scene with moving native shapes can be recorded and scrubbed. | Stored poses/motion/events round-trip exactly; multi-source and submesh bindings survive reload; primitive and acceleration fixtures pass; unsupported shape changes produce explicit errors. |
| 2. Establish correct computation | CPU analytic fixtures, verified FFT/STFT/convolution, matching GPU layouts, 2D/3D intersections/BVH and direct paths. Derive and document the first energy estimator's receiver/PDF normalization before implementing its reference path. | Independent numerical comparisons; geometry updates match brute force; Metal and DirectX ABI/fixture results recorded. |
| 3. Complete the simulator | Configurable-band indirect tracing/reduction with bounded batches, asynchronous ownership, CPU DSP and deterministic offline schedules. | One recorded moving scene renders at two ray budgets; absolute levels, seed/chunk invariance, tails, transitions and failure handling pass. |
| 4. Collect and diagnose | Thin offline entry point, completed artifacts, paired dataset references and benchmark reporting. Add batch automation only when repeated runs need it. | Repeatable low/high captures with stage costs, validity masks, reference uncertainty and aligned sample ranges. |
| 5. Prove reconstruction value | Dataset reader and simple baselines, then a small trained model and measured realtime integration. | Held-out quality improvement at matched total cost/latency; fallback and temporal reset behavior verified. |
| 6. Extend fidelity from evidence | Phase-aware transport, cross-resolution models or other effects whose experiments justify them. | Separate physics/phase correctness gates and comparisons against the earlier methods. |

Step 1 is underway, with a first direct-path fixture/runner from steps 2 and 4 to exercise saved takes end to end. See the README for implemented scope and remaining gaps. Exact stochastic receiver normalization is a required mathematical deliverable in step 2, not something this architectural review has proven. Realistic material phase, full moving-boundary propagation, diffraction/room modes and a deployable model remain separate capabilities; the existing prototype cannot serve as their unquestioned reference.

Initial implementation scope is explicit: fixed scene membership/assets within a take, rigid motion, point sources and mono receivers, homogeneous medium and opaque geometric reflection. These choices keep the first captured experiment tractable. They do not hardcode spectral resolution or prevent later recorded topology/material changes, directional sources, binaural rendering or other physical models. Add those when their contracts and measurements are ready; do not invent empty modules for them now.

Migration is incremental: keep the legacy scenes runnable, add controlled fixtures, preserve asset GUIDs when moving scripts, and switch a scene only after its replacement capture/render path passes. Do not make the archived implementation draft the starting point without reviewing it against this contract. Remove dead legacy/debug paths and unused dependencies only after confirming scene references and replacement coverage.

Before a performance claim, the benchmark definition must name actual hardware, sound-device settings, graphics load and numerical deadlines. Before training-label promotion, the chosen estimator/synthesis must pass its correctness and convergence gates. No further architecture framework, compiler migration, fixed band count or quality preset is needed to start the foundational work.
