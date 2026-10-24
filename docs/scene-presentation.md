# Church visual styles and silent footage

Open `Assets/Scenes/Church.unity`. The scene supports both styles through **Church Visual Preset** on **Acoustic simulation**: select **Polished Low Poly** or **Realistic** before Play, or use **F7 / F8** while playing. Changing the Inspector selection during Play also applies it. Polished Low Poly is the saved default. **F6** shows/hides audio-source markers in the Game view; markers start enabled in Church.

| Style | Treatment |
|---|---|
| Polished Low Poly | Faceted architectural additions, patterned stone floors, colored glass, warm chapel lamps and flat-color Standard materials. The original church shell remains; this is not a whole-scene decimation pass. |
| Realistic | Original textured Standard/PBR materials, dark ambient fill, warm candles, cool window projections and soft shadows. A realistic treatment of the existing art, not photorealistic scanned assets. |

Both use the same collision proxies, acoustic objects, input, movement and networking. Visual materials are separate from acoustic coefficients. The same small image effect supplies exposure in stops, contrast, saturation and vignette. Material pairs are serialized in the scene, so replacing a style does not destroy the other style's references on reload. No render-pipeline migration or package dependency was added.

## Seeded rain bursts — 24 October 2026

The renderer now supports seeded finite bursts on the existing masked plane. Each burst gets a new weighted-mask position and clip offset, then keeps its world position through its delayed tail. CPU/GPU spectral synthesis and live/offline scheduling share the same event clocks. Transport and DSP now use per-source/listener ranges, so point sources no longer reserve the largest plane's direct-site capacity.

Enable this on `AudioSourcePlane` with a positive **Burst Period Samples** and **Burst Duration Samples** of at least four, measured at the capture rate. **Samples** bounds retained bursts, including propagation/filter history; **Rays Per Bin** must cover every retained burst. Zero period keeps the original fixed-site behavior. Source gain is normalized against clip mean-square power, so set the level for this source model rather than assuming the old value has identical loudness. The existing Church scene retains its fixed-site source and current profile; this is an opt-in model change, not an automatic conversion of recorded input.

[Implementation and lifetime contract](simulation-contracts.md#seeded-emission-sampling-and-batching) · [Validation evidence](benchmarks/seeded-emission-2026-10-24/README.md). Thunder remains the existing point recording; a coherent channel source needs its own signal/site model, while reusing the same transport.


## Masked rain plane — 23 October 2026

The section below records the 23 October fixed-site plane checkpoint. The optional finite-burst path and packed storage are described above; lightning-channel emission remains future work.

Church now has one **Rain ground** object under **Church world / Weather**. Its `AudioSourcePlane` component distributes the existing looping rain clip over a 58 × 86 metre rectangle at `(0, 1.06, -7)`. The authored `RainMask.png` excludes covered ground, sampled from the existing static colliders. Thunder remains the ordinary elevated point source. The scene has eight registered sources, including four voice slots, and four autostart clips. Both visual styles use the same sound sources.

Edit **Size**, **Mask**, **Samples**, and **Seed** on `AudioSourcePlane`; keep its `RaytracedAudioSource` looping. White mask pixels emit, black pixels emit nothing, and gray values reduce power density. Masks must be readable, linear grayscale textures; empty means fully white. Masks are sampled without filtering across shelter boundaries. The selected object's gizmo outlines its plane. F6 labels its logical centre, not the individual sampled positions. Paint the mask again if you change the architecture: it is authored data, not a runtime roof detector.

The saved budget is eight stable spatial sites, independent of rays per frequency bin. Increasing the spatial budget retains existing site positions and adds more; it also changes their integration weights, so change it between takes. Source position and rotation can move during a take; dimensions, scale, mask, seed and sampling budget are fixed. The same component uses local XZ in 3D and XY in 2D.

Each site reads the same stored clip at its own deterministic cyclic offset. Offsets stay fixed across bins, listeners and response updates, so changing ray quality does not make the rain texture jump. A reflected path uses the same site's waveform as its direct path. Physical propagation delay, Doppler and source Play/Stop/Gain timing remain separate from these clip offsets. This is suitable for stationary noise-like ambience; shifting a tonal clip does not make independent rain, and this is not a general extended speech/microphone source.

For plane area `A`, mask density `m` and `N` spatial samples, a site's pressure weight is `sqrt(A*m/N)`. This targets approximately constant power density for decorrelated rain rather than coherently duplicating one waveform. Gain therefore behaves as an uncalibrated per-area control; Church uses 0.04. Larger planes emit more total power. A small budget can leave spatial hotspots or miss narrow mask regions; it is a sampled ambience approximation, not uniform coverage or roof-impact/structural sound simulation.

Direct paths evaluate every site. Indirect rays select among those same sites and keep the existing material/reflection/scatter/diffraction transport, compensating for site selection in their Monte Carlo weights. The indirect ray budget is shared by the plane: increase it when more spatial sites need reflected-path coverage. More spatial samples do not multiply the indirect ray count automatically.

The GPU receives one immutable PCM copy per distinct plane clip and one sample-clock/gain stream per logical plane source, not a separate PCM upload for every site. Spatial points and mask weights are prepared once; the GPU applies sample-timed poses. CPU reference synthesis follows the same clip-offset contract. The shared path ABI remains 48 bytes by using a former padding field. There is one new runtime component and no new weather manager, source hierarchy, network stream or package dependency.

Planes currently require **Raytraced + Spectral**, with GPU or CPU-reference synthesis. Direct-only and IR modes reject them explicitly. The single-origin ray-debug capture also rejects multi-site planes; ordinary point-source debug remains available. Its centre-based direct diagnostic is not an area-field measurement. Dynamic shelter masks, rain particles, roof vibration and calibrated emission patterns remain outside this implementation.

Take version 5 authenticates the plane axes, density mask, sample budget and seed alongside existing source/clip data. Compact input reconstruction preserves the component and source contract. Older take versions still use their original scene-hash layouts; replaying an older compact recording still requires its original scene/code revision. Multiplayer continues to share ordinary scene playback timing; no extra weather audio stream is sent. This change has not been tested with another computer.

At the 23 October checkpoint, GPU capacity used the largest plane budget across source/listener pairs: `bins * (maxSpatialSamples + raysPerBin * (maxBounces + 1))` slots per pair. Point pairs leave additional direct slots empty. At Church's unchanged live profile, source count falls from nine to eight while slots per pair rise from `67*bins` to `74*bins`; this does not imply a measured speedup. Plane PCM and clock buffers also consume memory. The scene's expensive live quality settings are unchanged.

The real-scene check preserved all 58 serialized colliders (59 loaded including the player controller) and all 17,321 acoustic triangles. It verified compact replay scene equivalence, rejected changed plane configuration/mask contents, and opened the previous version-4 weather take. Five of the eight sites survive this mask. A two-second scripted outdoor receiver move plus tail rendered finite, non-silent PCM with peak **0.02035**, RMS **0.003751**, and **24.30 MiB** reported numeric working storage. This used rain only, 256 FFT, one ray per bin and four bounces. The measured run took **142.29 s total**, including **141.79 s preparation** and **0.50 s after preparation**; these are one-run offline timings, not a full-profile real-time or warm-start benchmark. No artifact-free listening or physical-accuracy claim follows from this smoke test.

All **204 selected Core/GPU/Integration cases in `TakeChecks` passed**, with zero failures or skips, in 349.82 seconds. The ten plane cases cover weighted masks and stable sample prefixes, clip offsets/event gating, authenticated take metadata, rotated sources/multiple receivers, 2D/3D mirror gain and delay, and moving-source CPU/GPU synthesis agreement with mono, binaural, global-FFT and windowed-sinc paths. Realtime device, Play-mode multiplayer and wave-field measurement suites were not selected. Metal still reports the existing families of potential-initialization/integer-division shader warnings; they are retained in the log, not represented as a warning-free build.

[Plane/mask map](benchmarks/rain-plane-2026-10-23/rain-plane-map.png) · [Waveform/spectrogram](benchmarks/rain-plane-2026-10-23/waveform-spectrogram.png) · [Offline audio](benchmarks/rain-plane-2026-10-23/rain-smoke.wav) · [Scene/render checks](benchmarks/rain-plane-2026-10-23/verification.json) · [Regression results](benchmarks/rain-plane-2026-10-23/regression.json) · [Source/artifact hashes](benchmarks/rain-plane-2026-10-23/manifest.json). The checks ran in an isolated low-priority Unity batch with audio disabled; no microphone, speaker playback, Play-mode scene or foreground window was used. The audio file was written directly by the offline renderer.

## Earlier point rain and thunder checkpoint — 22 October 2026

The following records checkpoint `0659ec5`, before the masked plane above replaced its two point rain sources. Counts, costs and audio below describe that earlier checkpoint.

The saved Church scene now starts rain and occasional thunder alongside its organ and bells. Edit the ordinary `RaytracedAudioSource` components under **Church world / Weather**; F6 shows their positions. Both visual styles share these sounds. This adds three clips and three emitters, **zero production C# lines**, and no package, timer, weather UI, particle effect or new recording format.

| Emitter | Position (metres) | Gain | Clip |
|---|---|---|---|
| Rain courtyard | `(-7, 1.3, -40)` | 0.6 | 11-second rain loop |
| Rain east garden | `(23, 1.3, 4)` | 0.8 | Independent 13-second rain excerpt |
| Thunder outside | `(-32, 45, -16)` | 2 | Recorded roll at second 8 of a repeating 42-second timeline |

Rain boundaries have one-second constant-power crossfades. The thunder clip contains its quiet intervals, so no per-frame random scheduler is needed. `SceneSession.startSounds` starts the existing sample-timed clip events; takes retain those events and clip identity. Multiplayer's existing `SceneSoundOffset` selects the corresponding loop phase for a joining client. Weather uses installed scene assets, not a new network PCM stream. This reuses the existing synchronization; no new multi-client timing/latency test was performed.

Each emitter goes through the existing transport and material response. These are **point-source approximations** of two local rain patches and an outdoor rumble. They do not cover the whole exterior uniformly, model roof vibration or simulate the length of a lightning channel. There is no separate indoor volume switch. The rain/thunder recordings already contain environmental acoustics and uncalibrated levels; use them for ambience/source diversity, not as an acoustic ground-truth target. No listening-quality claim follows from the silent tests. [Asset provenance and exact preparation](../Assets/External/Church/README.md).

Cost matters: the three 44.1 kHz mono PCM16 assets total 5.55 MiB on disk and 11.10 MiB as decoded float PCM. There are now nine registered sources (five clip emitters and four existing voice slots). The current GPU implementation reserves filter/FFT buffers per registered source, even for quiet intervals. At the unchanged **1024 FFT / two rays per bin / 32 bounces** live profile, these three extra sources add approximately **2.36 GiB of filter and FFT scratch capacity alone**, calculated from `GpuAudio`'s existing allocation formulas. This is a capacity estimate, not measured residency or a live performance result. The source-count allocation needs attention before treating a richer ambient scene as cheap; silent thunder samples do not eliminate its reserved buffers. The implementation and saved quality settings are unchanged by this asset pass.

Verification ran in one isolated, low-priority Unity batch job with `-disable-audio`, without Play mode, microphone/network startup, speakers or foreground windows. Clip import/readback, recorded event PCM, wraparound/late-join phase, all five overlay labels and emitter clearance from solid geometry passed. All 58 existing colliders and 17,321 acoustic triangles are preserved, as are original source/DSP/network settings. A three-second fixed outdoor weather-only render used the real GPU transport and synthesis with a deliberately cheaper **256 FFT / one ray per bin / four bounces** profile; output is finite, non-silent and unclipped (peak 0.0296). It took 0.398 s including preparation and reported 35.76 MiB numeric working storage. This smoke result does **not** establish full-profile runtime, perceived balance, indoor coverage or artifact-free movement.

[Source view](benchmarks/church-weather-2026-10-22/weather-sources.png) · [Waveform/spectrogram](benchmarks/church-weather-2026-10-22/waveform-spectrogram.png) · [Silent-generated audio file](benchmarks/church-weather-2026-10-22/outdoor-smoke.wav) · [Verification](benchmarks/church-weather-2026-10-22/verification.json) · [Render settings/timing](benchmarks/church-weather-2026-10-22/render.json) · [Clip checks](benchmarks/church-weather-2026-10-22/clip-checks.json) · [Hashes](benchmarks/church-weather-2026-10-22/manifest.json). The smoke excerpt starts at loop phase 7 seconds to include thunder onset; it is not a recording of the first three seconds of normal Play.

## Source markers and architectural polish — 22 October 2026

[Latest views](benchmarks/church-sources-2026-10-22/overview.png) · [Source overlay](benchmarks/church-sources-2026-10-22/PolishedLowPoly-sources.png) · [Behind-camera markers](benchmarks/church-sources-2026-10-22/sources-behind.png).

Both visual styles now include finer diagonal nave ribs, candle sconces, a rose-pattern medallion above the organ, an organ surround, a reading-chapel wall crest and a warm sanctuary spotlight. These four mesh parts share the existing palette in one 129 KiB `CathedralAccents.fbx`; no new materials or packages are needed. The rose is decorative geometry, not a new opening through the acoustic shell. The additions remain visual detail; existing physical/acoustic objects are preserved.

`AudioSourceOverlay` is attached to the listener camera and explicitly references the scene's manager. **F6** toggles it. Amber labels identify clip emitters; cyan labels identify voice emitters. Each label shows the actual registered source position and rounded straight-line distance from that camera. Screen-edge arrows indicate offscreen sources; sources behind the camera are also labeled `behind`. The layout separates overlapping labels vertically, hiding a label if it cannot find room. Labels can be displaced for readability and remain visible through walls; they are annotations, not an occlusion or propagation visualization.

The overlay respects each receiver's existing self-voice exclusion and hides inactive/disabled sources. The preallocated remote voice slots therefore appear when their actors become active. Source-list size changes rebuild the small UI. No raycast, GPU readback, audio-buffer read or new playback state is involved. These labels **do not indicate current playback, speaking activity, received loudness, or the acoustic sound field**. In Scene view, enable Gizmos to see source names and small wire markers independently of F6. Their fixed visual size is not a physical emitter radius.

The component uses Unity's existing camera-space Canvas and source registry; it adds no capture format or DSP contract. To use it in another scene, add `AudioSourceOverlay` to a listener camera and assign its `manager`. Its public `Refresh()` also supports silent editor captures. UI graphics do not receive pointer events. The built-in camera projection follows Unity's [WorldToViewportPoint](https://docs.unity3d.com/6000.0/Documentation/ScriptReference/Camera.WorldToViewportPoint.html) convention; the [camera-space Canvas](https://docs.unity3d.com/6000.0/Documentation/ScriptReference/RenderMode.ScreenSpaceCamera.html) lets the same annotations appear in offscreen renders.

Verified in the isolated Unity batch runner, with audio disabled: moving-source distances/positions, behind-camera and camera-plane projections, screen-edge containment, toggling, disabled sources, local-voice exclusion, an activated remote slot and source-registry rebuilding. The cyan-voice screenshot activates a synthetic remote actor for presentation testing; it is not evidence of a new network/voice call. Source and manager settings remain unchanged after the UI checks. Both 723-tick chapel walks passed again with all ticks grounded and zero positional difference between styles. All 58 existing colliders and 71 existing behaviour definitions (except the expanded visual-preset bindings) are unchanged. Acoustic compilation remains at 17,321 triangles. Final run: no script/shader errors or exceptions; existing unrelated deprecation warnings remain.

[Verification](benchmarks/church-sources-2026-10-22/verification.json), [asset/source hashes](benchmarks/church-sources-2026-10-22/manifest.json), and [authoring/check scripts, logs and first-pass renders](benchmarks/church-sources-2026-10-22/authoring.zip). Initial renders caught overlapping offscreen labels and a rose rim intersecting masonry; the final images include those corrections. This pass uses still renders and reuses the existing traversal check rather than recording another four full videos.

## Authored rooms — 22 October 2026

[Room-authoring checkpoint views](benchmarks/church-rooms-2026-10-22/rooms.png) · Gameplay: [low-poly](benchmarks/church-rooms-2026-10-22/PolishedLowPoly-first-person-gameplay.mp4), [realistic](benchmarks/church-rooms-2026-10-22/Realistic-first-person-gameplay.mp4).

The low-poly preset now has modeled detail instead of relying on the wall textures for its identity. Both styles share these additions to the existing church:

- **Nave:** five pointed arcades, octagonal columns, capitals, hanging banners, timber wall panels and a stone aisle inlay.
- **Crossing:** a compass-pattern floor and framed openings into the existing side wings. The central route to the altar stays clear.
- **West baptistery:** an octagonal open font, circular stone inlay, faceted jewel panels, candle stands and stone ribs.
- **East reading chapel:** book-lined shelves, a writing table with an open book, benches, timber roof trusses and warm local lighting.

All additions live under `Church world / Cathedral rooms`. The table/benches and shelves have named **ensemble** parents: move those parents to keep their visuals and collision/acoustic proxies together. Meshes remain imported asset references; materials and transforms are ordinary Inspector-editable scene data. There are 13 mesh parts in one original 373 KiB FBX, 12 palette materials and three textured material variants. The asset authoring adds **zero production C# lines**, packages or runtime geometry generation. The one-off Blender/Unity authoring and checks are preserved in [authoring.zip](benchmarks/church-rooms-2026-10-22/authoring.zip).

Arches, font and timber supports use their low-poly mesh as collision/acoustic geometry. Shelves, table and benches use simple wooden box proxies. Flat inlays, banners, surface panels, water decoration and candle detail do not add acoustic surfaces. These are illustrative material assignments, not measured room properties. There are 58 colliders after adding 14; all 44 existing collider definitions remain unchanged. The complete scene compiles to 17,321 acoustic triangles. The added physical geometry changes the simulated room response; this is not an acoustic-quality benchmark. Replays of an earlier scene revision still require that revision.

Both presets passed a 723-tick route through both chapels with every sampled tick grounded and **zero positional difference**. Separate approaches confirmed the font and table stop the player. All new physical object types appear in acoustic compilation. Existing transform poses, mesh references, collider definitions and 56 existing behaviour definitions are preserved; the manager's object registry and the preset's material bindings intentionally expand. No DSP or networking settings changed.

The four silent gameplay/animation videos each contain 192 frames at 1280×720, 24 fps, with no audio stream; the two full 397-tick nave routes are identical. After grouping furniture for editing, the chapel route and collision/compilation checks passed again. A 16-frame smoke comparison has nine pixel-identical frames; the others differ by at most two 8-bit channel levels, with a worst mean absolute difference of 0.000605 on the 0–255 scale. Full videos precede that hierarchy-only grouping; the final smoke metadata records the grouped scene. Final Unity runs have no script/shader errors or exceptions.

[Verification](benchmarks/church-rooms-2026-10-22/verification.json), [full capture](benchmarks/church-rooms-2026-10-22/capture.json), [final grouping smoke](benchmarks/church-rooms-2026-10-22/grouped-smoke.json), [hashes](benchmarks/church-rooms-2026-10-22/manifest.json). The archive also retains the early dark-emission renders and failed temporary-check compilation as process history. Checks and rendering ran in the isolated, low-priority Unity batch runner with audio disabled.

## Movement and animation

The existing `PlayerController` / `MouseLook` / `SceneSession.Step` remain authoritative. Unity's [CharacterController.Move](https://docs.unity3d.com/6000.0/Documentation/ScriptReference/CharacterController.Move.html) consumes a displacement and resolves collisions; gravity is implemented by our existing controller. This API is not a cross-platform determinism guarantee. Replacing it with Starter Assets or a Rigidbody controller would change the replay/movement contract without addressing this presentation task.

A visual child uses the [Kenney Animated Characters Protagonists](https://kenney.nl/assets/animated-characters-protagonists) CC0 model with looping idle/run clips. The FBX import [preserves its root hierarchy](https://docs.unity3d.com/6000.0/Documentation/ScriptReference/ModelImporter-preserveHierarchy.html) so animation paths match the model, and selects the named Idle/Run takes rather than the targeting pose. Measured movement drives the Animator; root motion is disabled. First-person views hide the body while retaining its shadow. Third-person capture shows the rig. This is a local presentation avatar, not a replacement for multiplayer avatars or new gameplay movement mechanics.

The project uses Unity's Built-in render pipeline. [OnRenderImage](https://docs.unity3d.com/6000.0/Documentation/ScriptReference/MonoBehaviour.OnRenderImage.html) supports the small full-screen grade on this pipeline; it is not an SRP effect. Existing Standard materials, native lights, ambient probe and fog are retained. The ambient fill is an artistic approximation, not baked indirect lighting. The character remains stylized in both presets.

## Capture and reproducibility

`ChurchPresentationCapture.CaptureBoth` runs **only in batch mode**. It opens a fresh scene for each preset, steps the existing controller with identical authored inputs at 50 Hz, advances the same Animator with [Animator.Update](https://docs.unity3d.com/6000.0/Documentation/ScriptReference/Animator.Update.html), and captures the actual first-person camera plus a separate third-person animation camera. These are scripted gameplay inputs, not a recording of a human playing. The cameras render offscreen; scene audio and network startup do not run.

Each capture has a fresh output directory, PNG frames, a route CSV, a JSON summary and optional silent MP4s. The CSV records the sampled inputs, actual poses, grounding, animation time and a bone-pose fingerprint. Hashes in the summary cover collider definitions and acoustic component bindings; they are checks against accidental presentation changes, not acoustic-output parity or a full mesh-content checksum. Two styles must reproduce the same route within 1 mm on the tested machine.

Run against an isolated Unity project copy, not a checkout open in the editor:

```sh
nice -n 15 /Applications/Unity/Hub/Editor/6000.4.1f1/Unity.app/Contents/MacOS/Unity \
  -batchmode -quit -force-metal -disable-audio -job-worker-count 1 \
  -projectPath "$RUNNER" \
  -executeMethod ChurchPresentationCapture.CaptureBoth \
  -scenePresentationOutput "$OUTPUT" -logFile "$OUTPUT/capture.log"
```

Create `$OUTPUT` first. `RUNNER` contains the current `Assets`, `Packages` and `ProjectSettings`. `ffmpeg` must be on PATH to encode videos; encoding uses one thread. Omit `-nographics` because these renders need Metal. Add `-scenePresentationSmoke 1` for four frames per view before a full capture. A full run produces 8-second, 1280×720, 24 fps clips for each style/view. Raw frame sequences stay local; selected frames, videos and verification metadata are retained with the experiment history.

The materials, rig, Animator controller and preset bindings are ordinary authored Unity assets. Edit those assets or the scene Inspector directly. The one-off setup code is preserved in checkpoint `a5aabce`; it was removed after authoring to keep the codebase smaller and prevent capture from overwriting scene customization. The reusable capture tool only reads the saved scene and changes its isolated in-memory copy.

Asset provenance: [church assets](../Assets/External/Church/README.md), [character subset and license](../Assets/External/Kenney/AnimatedCharactersProtagonists/README.md).

## Verified capture — 22 October 2026

[Visual comparison](benchmarks/church-presentation-2026-10-22/comparison.png) · First-person videos: [low-poly](benchmarks/church-presentation-2026-10-22/PolishedLowPoly-first-person-gameplay.mp4), [realistic](benchmarks/church-presentation-2026-10-22/Realistic-first-person-gameplay.mp4) · Animation views: [low-poly](benchmarks/church-presentation-2026-10-22/PolishedLowPoly-third-person-animation.mp4), [realistic](benchmarks/church-presentation-2026-10-22/Realistic-third-person-animation.mp4).

Unity 6000.4.1f1 / Metal, silent isolated batch. Four videos passed `ffprobe`: 192 frames each, 1280×720, 24 fps, 8 seconds, zero audio streams. Both 397-tick CSVs are identical, all sampled ticks are grounded, and animation produced 396 distinct bone-pose fingerprints over multiple loops. Material toggles were checked against serialized references after scene reload. Final capture has no C# errors, shader errors, exceptions or Animator initialization warnings.

Scene comparison against `86d475a` preserves all 44 authored collider definitions, 45 mesh filters, 41 acoustic surfaces, session/network/renderer settings, and the existing transform definitions. This excludes the new presentation child and visual lighting/material settings. No acoustic renderer benchmark or audio-quality claim follows from these presentation checks.

[Verification](benchmarks/church-presentation-2026-10-22/verification.json), [capture metadata](benchmarks/church-presentation-2026-10-22/summary.json), [source/artifact hashes](benchmarks/church-presentation-2026-10-22/manifest.json), and [investigation logs / early failed frames / verification script](benchmarks/church-presentation-2026-10-22/investigation.zip) preserve the evidence. Early checks caught an empty Animator controller after a failed install, selection of the targeting pose instead of the real clips, stripped FBX animation paths, and incorrect avatar fitting. They are retained as process history, not counted as passing proof.

The follow-up cleanup removes 232 lines of one-off asset setup. Its 16 smoke frames are pixel-identical to the corresponding full-capture frames, and the sampled input/pose/animation rows match. The full setup remains available at `a5aabce`; normal use and capture need only the authored assets.
