# Church assets

All bundled third-party assets below are published under **CC0 1.0**: https://creativecommons.org/publicdomain/zero/1.0/

- **Church model and textures:** Daniel Andersson; explorable-interior adaptation by AnyRPG. https://opengameart.org/content/medieval-church-interior — derived from https://opengameart.org/content/medieval-church . Download: https://opengameart.org/sites/default/files/church.blend . Exported to FBX with separate editable parts and centered local origins. Packed textures extracted unchanged. `ChurchCollision.fbx` keeps architectural surfaces and doors, dissolving coplanar detail within 2 degrees; duplicate vertices and degenerate faces are removed. Decorative statue, window bars and small exterior trim are omitted from this approximation. The scene supplies its floor and furniture.
- **Bell recording:** “Church Bell” (0135), Joseph SARDIN / BigSoundBank. https://bigsoundbank.com/church-bell-s0135.html . Original WAV: https://bigsoundbank.com/UPLOAD/bwf-en/0135.wav . `Audio/Bell.wav` preserves the recording, with 60 ms boundary fades, a 0.55 peak and silence to make a 30-second loop. The original contains outdoor ambience and room/environment decay; it is not an anechoic training source.
- **Organ samples:** “Church organ emulation”, FreePats / Roberto, version 2019-09-24, recorded from Fons Adriaensen's Aeolus emulator. https://freepats.zenvoid.org/Organ/pipe-organ.html . Source archive: https://freepats.zenvoid.org/Organ/ChurchOrganEmulation/ChurchOrganEmulation-SFZ-20190924.tar.xz . `Audio/Organ.wav` is an original C / A minor / F / G passage rendered from the nearest supplied note samples, downmixed, resampled and faded, with a 0.45 peak and silence in a 30-second loop. No additional reverb was applied. This is sampled organ emulation, not a recording of a physical church organ or a calibrated acoustic reference.

Textures import as ordinary repeating 3D textures (not the project’s default clamped sprites), capped at 1024 pixels. Both meshes are readable for acoustic compilation and replay. Audio imports as preloaded PCM.

The two authored sources play through the existing raytracer. The tower emitter is at the front belfry opening `(0, 52, -4)`, representing radiation out of the tower; it does not model the enclosed bell-to-aperture field. Masonry screens the nave and much of the front courtyard. The east courtyard near `(26, 1, -40)` has a clear path. Speech remains a separate microphone/remote voice source. Keep the dry `Assets/Audio/church_clap.wav` for isolated acoustic diagnostics; the new scene does not automatically play it. Geometry and material coefficients are illustrative and not measured acoustic data.

`Details.fbx` contains original reusable pew, altar, organ, chandelier, window-pane and surrounding-ground meshes made for this scene. `LeadedGlass.png` and `WindowLight.png` are original geometric patterns; `Paving.png` and the wall normal maps derive visual variation from the bundled CC0 stone textures. The normal maps are illustrative relief inferred from texture brightness, not scanned surface measurements. These assets use Unity's existing Standard shader and native lights; there is no new runtime script, shader, post-processing package or lightmap bake. The darker lighting preset uses four candle lights, two window projections and a cool directional light. Flame/glass materials retain their emission keywords after reload.

Pew seat/back collisions use metre-sized boxes on a unit-scale parent, independent of the imported visual node's FBX axis/unit transform. The larger organ has simple wooden housing proxies. Thin closed panes fill selected nave/apse openings and participate in acoustic transport. The surrounding ground has a matching visual/collision/acoustic mesh and joins the courtyard at the same height; its distant edge is hidden by scene fog. Glass/ground acoustic coefficients are illustrative, like the existing stone/wood presets.

Original download SHA-256 hashes:

```text
church.blend  bcc2f59154473dcb46d580d1a60f1358fe6a0339d2d602c9f8cd3d31ef03e393
0135.wav      8b9ee0a7de39fb599dde1cf06ac2f1a030c2361986526f38eb575536c027abba
organ.tar.xz  1f0b70a2c2af14cdf051ef2d256b3bb652e7d64c0fdc8bed06642ad2af916db9
```

Import/export scripts and diagnostic evidence are retained locally under `.context/church-assets/`. No Blender or audio-synthesis dependency is required at runtime.

## Cathedral room kit

`CathedralRooms.fbx` is original scene geometry authored for this project: pointed arcades and octagonal columns, floor inlays, wall panels, banners, an open baptismal font, jewel panels, reading shelves/books/table/benches, candle stands and timber trusses. Its 13 named parts use the palette in `Assets/Materials/ChurchPresentation/Rooms-*`; the three `Rooms-PBR-*` variants reuse the existing CC0 marble/wood textures. No additional third-party asset license is introduced.

The saved Church scene includes these assets in both visual styles, with physical furniture/structural collision and acoustic proxies. Inlays and small decoration remain visual-only. The table and shelves have ensemble parents so their proxies move with their visuals. The source authoring scripts and validation evidence are archived in `docs/benchmarks/church-rooms-2026-10-22/authoring.zip`; Blender is not a runtime dependency. See `docs/scene-presentation.md` for room views, controls and the scope of verification.

`CathedralAccents.fbx` adds four original decorative mesh parts: nave vault ribs, the sanctuary rose/organ surround, nave sconces, and the reading-chapel crest. It shares the room kit's materials and does not add acoustic/collision geometry. The rose is decorative art, not an additional acoustic window. The original Blender authoring and Unity placement/verification scripts are retained in `docs/benchmarks/church-sources-2026-10-22/authoring.zip`.
