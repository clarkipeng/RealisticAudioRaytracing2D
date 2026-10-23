# Church visual styles and silent footage

Open `Assets/Scenes/Church.unity`. The scene supports both styles through **Church Visual Preset** on **Acoustic simulation**: select **Polished Low Poly** or **Realistic** before Play, or use **F7 / F8** while playing. Changing the Inspector selection during Play also applies it. Polished Low Poly is the saved default. **F6** shows/hides audio-source markers in the Game view; markers start enabled in Church.

| Style | Treatment |
|---|---|
| Polished Low Poly | Faceted architectural additions, patterned stone floors, colored glass, warm chapel lamps and flat-color Standard materials. The original church shell remains; this is not a whole-scene decimation pass. |
| Realistic | Original textured Standard/PBR materials, dark ambient fill, warm candles, cool window projections and soft shadows. A realistic treatment of the existing art, not photorealistic scanned assets. |

Both use the same collision proxies, acoustic objects, input, movement and networking. Visual materials are separate from acoustic coefficients. The same small image effect supplies exposure in stops, contrast, saturation and vignette. Material pairs are serialized in the scene, so replacing a style does not destroy the other style's references on reload. No render-pipeline migration or package dependency was added.

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
