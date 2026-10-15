# Church assets

All bundled third-party assets below are published under **CC0 1.0**: https://creativecommons.org/publicdomain/zero/1.0/

- **Church model and textures:** Daniel Andersson; explorable-interior adaptation by AnyRPG. https://opengameart.org/content/medieval-church-interior — derived from https://opengameart.org/content/medieval-church . Download: https://opengameart.org/sites/default/files/church.blend . Exported to FBX with separate editable parts and centered local origins. Packed textures extracted unchanged. `ChurchCollision.fbx` keeps architectural surfaces and doors, dissolving coplanar detail within 2 degrees; duplicate vertices and degenerate faces are removed. Decorative statue, window bars and small exterior trim are omitted from this approximation. The scene supplies its floor and furniture.
- **Bell recording:** “Church Bell” (0135), Joseph SARDIN / BigSoundBank. https://bigsoundbank.com/church-bell-s0135.html . Original WAV: https://bigsoundbank.com/UPLOAD/bwf-en/0135.wav . `Audio/Bell.wav` preserves the recording, with 60 ms boundary fades, a 0.55 peak and silence to make a 30-second loop. The original contains outdoor ambience and room/environment decay; it is not an anechoic training source.
- **Organ samples:** “Church organ emulation”, FreePats / Roberto, version 2019-09-24, recorded from Fons Adriaensen's Aeolus emulator. https://freepats.zenvoid.org/Organ/pipe-organ.html . Source archive: https://freepats.zenvoid.org/Organ/ChurchOrganEmulation/ChurchOrganEmulation-SFZ-20190924.tar.xz . `Audio/Organ.wav` is an original C / A minor / F / G passage rendered from the nearest supplied note samples, downmixed, resampled and faded, with a 0.45 peak and silence in a 30-second loop. No additional reverb was applied. This is sampled organ emulation, not a recording of a physical church organ or a calibrated acoustic reference.

Textures import as ordinary repeating 3D textures (not the project’s default clamped sprites), capped at 1024 pixels. Both meshes are readable for acoustic compilation and replay. Audio imports as preloaded PCM.

The two authored sources play through the existing raytracer. The tower emitter is at the front belfry opening `(0, 52, -4)`, representing radiation out of the tower; it does not model the enclosed bell-to-aperture field. Masonry screens the nave and much of the front courtyard. The east courtyard near `(26, 1, -40)` has a clear path. Speech remains a separate microphone/remote voice source. Keep the dry `Assets/Audio/church_clap.wav` for isolated acoustic diagnostics; the new scene does not automatically play it. Geometry and material coefficients are illustrative and not measured acoustic data.

Original download SHA-256 hashes:

```text
church.blend  bcc2f59154473dcb46d580d1a60f1358fe6a0339d2d602c9f8cd3d31ef03e393
0135.wav      8b9ee0a7de39fb599dde1cf06ac2f1a030c2361986526f38eb575536c027abba
organ.tar.xz  1f0b70a2c2af14cdf051ef2d256b3bb652e7d64c0fdc8bed06642ad2af916db9
```

Import/export scripts and diagnostic evidence are retained locally under `.context/church-assets/`. No Blender or audio-synthesis dependency is required at runtime.
