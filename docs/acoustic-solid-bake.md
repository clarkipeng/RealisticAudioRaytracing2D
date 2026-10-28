# Acoustic solid boundaries

Implemented 28 October 2026; generation became automatic the same day.
`AcousticSolid` combines fixed, closed child meshes/colliders into one exterior boundary before scene capture.
Internal joining faces disappear; surviving surfaces retain their acoustic material assets.
Visual meshes are unchanged.
The same compiled primitives reach live transport, saved takes and offline audio.

A physical wall should not sound different because it was authored as several touching boxes.
The boundary fixes that representation problem; it does not supply missing higher-order acoustic paths or establish full-wave fidelity.
[Measured room comparison and controls](benchmarks/solid-boundary-2026-10-28/README.md).

## Authoring

1. Create a parent with `AcousticSolid` (it adds `AcousticSurface`).
   Use the parent's ordinary geometry selection, not `None`.
2. Put its fixed child `AcousticSurface` components into **Members**.
   Use closed manifold meshes or solid collider proxies.
   Omit moving doors, leaves and other independently moving objects.
3. On each machine that edits solids, install the backend once and select its Python in **Preferences > Acoustics > Solid Boundary Python**.
   The `ACOUSTIC_SOLID_PYTHON` environment variable overrides that preference.
   For example:

   ```sh
   python3 -m venv .context/solid-python
   .context/solid-python/bin/python -m pip install -r tools/solid-boundary-requirements.txt
   ```

4. Edit and save the scene normally; there is no bake step.
   The inspector shows whether the exterior is current, building or failed, with **Retry** after a failure.
   The root and members must all be registered geometry bindings; the manager's normal registration discovers them.

Earlier members own coincident exposed material overlaps.
This is an explicit coating choice, not a blend.
Distinct exposed material regions remain distinct.
Surface absorption, scattering, transmission and impedance coefficients may differ.
All members must share the same interior sound-speed ratio and damping curves because the current take format represents one interior medium per closed owner.
Keep different interior media in separate groups.

The entire group may translate and rotate.
Relative member transforms, visibility, geometry and material assignments are fixed during a capture; a running capture rejects changes to them.
Between takes, editing them regenerates the exterior.
Root scale follows the existing capture rule: applied at compilation, fixed during recording.
Turning off the component, or excluding its root with `Geometry Source = None`, restores the ordinary member geometry and permits independent member motion.
An unlisted door remains independent.

Existing takes keep their captured geometry; regeneration does not rewrite them.
Takes contain the ordinary triangle/material/object records and need no Python or Boolean library for playback.

## Regeneration and cache

The cache key is the SHA-256 of `AcousticSolid.ExportInput`.
It covers member geometry, curve tolerance, per-face material slots, membership order and member transforms relative to the root.
Moving or rotating the whole group, listener or source motion, unlisted doors and rain drops leave it unchanged.
Replacing a member's material asset while faces keep distinct owners gives the same exterior; only the slot-to-asset mapping, which is recomputed on every compile, changes.
Giving two members the same material, or splitting a shared one, changes face ownership and therefore the key.

The current boundary is serialized on the component, inside the scene or prefab.
A fresh clone, a teammate without Python and player builds capture from it directly.
`Library/AcousticSolidBoundaries/<hash>.json` keeps the 256 most recently used backend results per checkout.
Undo, duplicates and reverting an edit are therefore cache hits.
A component or Library hit starts no Python process.

Regeneration happens at these points:

- After any undoable Editor change, undo/redo, scene open or script reload, a check runs once edits have settled for 0.5 seconds.
  It recomputes the key for every loaded solid, including the open prefab stage.
  A miss queues one background worker; a newer edit cancels a worker for the old key.
- Saving a scene or prefab, and entering Play mode, finish stale solids synchronously with a cancelable progress bar.
  Saved files therefore carry current boundaries for teammates.
- Every capture compile calls `AcousticSolid.Resolve`.
  In the Editor, including batch mode, a stale or missing boundary is regenerated before compilation.
  A player cannot regenerate, and capture fails with a message instead of using stale geometry.
  Disabling the component or setting the root `Geometry Source = None` is the explicit fallback to separate member geometry.

Scripts that edit members without Undo are caught at the next save, Play entry or capture.
Workers run one at a time at `nice 19` (idle priority on Windows) with one numeric-library thread, a 120-second timeout and a 1 GiB memory limit.
Background workers can be cancelled from the Background Tasks window.
The Editor stops its worker before script reloads and on quit.

A missing Python executable, a missing backend module or a `manifold3d` version other than the pinned one fails with the install command.
Those environment failures retry on the next edit or **Retry**.
Invalid geometry, timeout, memory and cancellation failures are remembered for that key until the input changes, the Python preference changes or **Retry** is clicked.
A failed solid keeps its previous boundary, which is still stale and is never returned for capture.

## Implementation boundary

- `AcousticSolid.ExportInput` uses the existing scene compiler's geometry/material conversion, then puts source geometry in the group's local frame.
  Root motion cannot change its source hash.
- `AcousticSolidBuilder` (Editor) schedules checks, owns the Library cache and the single backend worker, and installs the synchronous `AcousticSolid.Regenerate` hook used by capture.
- `tools/solid_boundary.py` exactly welds duplicate positions, validates closed solids, performs ordered Boolean union and removes redundant coplanar seams at backend numerical tolerance.
  It does not voxelize, fill openings or approximate a visual mesh.
  Separated components and real gaps remain separate.
  Exit code 3 reports a missing or mismatched backend; exit code 4 reports invalid input.
- [Manifold](https://github.com/elalish/manifold) supplies the robust Boolean operation; `manifold3d==3.5.4` is pinned and enforced so equal keys give equal boundaries on every machine.
  Its [mesh provenance API](https://manifoldcad.org/docs/html/classmanifold_1_1_manifold.html) carries surface material identity through the Boolean.
  Source hash, backend, triangle count, area and volume are retained in the output.
- `SceneToTake` emits the boundary under the root's existing owner ID, with per-triangle materials.
  Member bindings remain for stable object/source indices but emit no duplicate geometry.
  DSP, arrival layout, shaders and take versions do not change.
- Guards reuse existing mesh/collider/material snapshots and additionally verify membership, local frames and boundary identity without managed allocations during validation.

Open sheets are ordinary acoustic surfaces, not solid input.
Bad topology is rejected.
This stage does not compute vibration, modal resonance, a simplification proxy or propagation caches.
Boolean triangle diagonals can differ between equivalent inputs; physical exterior geometry is the invariant.
The measured cube's field and rendered audio match exactly across the tested partitions, seeds and rigid motion, not a claim about every triangulation or sampling budget.

The worker limits are authoring safeguards, not a large-scene throughput guarantee.
Mac/Metal is validated; other platforms still need their own checks.
