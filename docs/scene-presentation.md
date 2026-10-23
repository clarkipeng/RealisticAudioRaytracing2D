# Church visual styles and silent footage

Open `Assets/Scenes/Church.unity`. The scene supports both styles through **Church Visual Preset** on **Acoustic simulation**: select **Polished Low Poly** or **Realistic** before Play, or use **F7 / F8** while playing. Changing the Inspector selection during Play also applies it. Realistic is the saved default.

| Style | Treatment |
|---|---|
| Polished Low Poly | Flat-color Standard materials, brighter cool fill, simple surfaces. Uses the existing meshes; this does not decimate geometry. |
| Realistic | Original textured Standard/PBR materials, dark ambient fill, warm candles, cool window projections and soft shadows. A realistic treatment of the existing art, not photorealistic scanned assets. |

Both use the same collision proxies, acoustic objects, input, movement and networking. Visual materials are separate from acoustic coefficients. The same small image effect supplies exposure in stops, contrast, saturation and vignette. Material pairs are serialized in the scene, so replacing a style does not destroy the other style's references on reload. No render-pipeline migration or package dependency was added.

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
