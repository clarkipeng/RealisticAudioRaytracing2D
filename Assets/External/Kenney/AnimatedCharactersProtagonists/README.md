# Kenney animated characters: protagonists

Subset imported for the Church presentation pass:

- `Model/characterMedium.fbx`
- `Animations/idle.fbx`
- `Animations/run.fbx`
- `Skins/skaterMaleA.png`
- `License.txt`

Source: https://kenney.nl/assets/animated-characters-protagonists

License: CC0 1.0 Universal, as bundled in `License.txt`.

The avatar is used only as a visual child of the existing player transform for third-person presentation proof. It does not replace the existing controller, input, collision, acoustic listener, networking, or replay contracts.

Downloaded archive SHA-256: `ec3787de70fa2200256848d74201b10f6b6c3126594e9857bf989753312c2b84`.
The imported clips loop; the model creates a Generic avatar. The generated Animator controller switches between idle and run from measured controller speed, with root motion disabled. `Kenney-skater.mat` uses the original skin texture with Unity's Standard shader.

Animation import preserves the FBX root so clip bone paths match the model. The named `Root|Idle` / `Root|Run` takes are used, not `Root|0.Targeting Pose`. The model is fitted to the existing player capsule using mesh bounds; its collider and listener remain unchanged.
