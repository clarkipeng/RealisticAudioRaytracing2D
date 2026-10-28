# Acoustic solid boundaries

Implemented 28 October 2026. `AcousticSolid` combines fixed, closed child meshes/colliders into one exterior boundary before scene capture. Internal joining faces disappear; surviving surfaces retain their acoustic material assets. Visual meshes are unchanged. The same compiled primitives reach live transport, saved takes and offline audio.

A physical wall should not sound different because it was authored as several touching boxes. The bake fixes that representation problem; it does not supply missing higher-order acoustic paths or establish full-wave fidelity. [Measured room comparison and controls](benchmarks/solid-boundary-2026-10-28/README.md).

## Authoring

1. Create a parent with `AcousticSolid` (it adds `AcousticSurface`). Use the parent's ordinary geometry selection, not `None`.
2. Put its fixed child `AcousticSurface` components into **Members**. Use closed manifold meshes or solid collider proxies. Omit moving doors, leaves and other independently moving objects.
3. Install `tools/solid-boundary-requirements.txt` into a Python environment, and set its executable in **Bake Python**. For example:

   ```sh
   python3 -m venv .context/solid-python
   .context/solid-python/bin/python -m pip install -r tools/solid-boundary-requirements.txt
   ```

4. Click **Bake acoustic exterior** and save a new JSON asset under `Assets/`. Then register/capture the scene normally. The root and source members must all be registered geometry bindings; the manager's normal registration discovers them.

Earlier members own coincident exposed material overlaps. This is an explicit coating choice, not a blend. Distinct exposed material regions remain distinct. Surface absorption, scattering, transmission and impedance coefficients may differ; all members must share the same interior sound-speed ratio and damping curves because the current take format represents one interior medium per closed owner. Keep different interior media in separate groups.

The entire group may translate and rotate. Relative member transforms, visibility, geometry and material assignments are fixed during a capture. Editing them requires rebaking; capture guards reject stale changes. Root scale follows the existing capture rule: baked at compilation, fixed during recording. Turning off the component, or excluding its root with `Geometry Source = None`, restores the ordinary member geometry. An unlisted door remains independent.

Existing takes keep their captured geometry; authoring a bake does not rewrite them. New takes contain the ordinary triangle/material/object records and need no Python or Boolean library for playback. Numerical material tuning may reuse the boundary provided the interior-medium constraint and asset assignments still hold.

## Implementation boundary

- `AcousticSolid.ExportInput` uses the existing scene compiler's geometry/material conversion, then puts source geometry in the group's local frame. Root motion cannot change its source hash.
- `tools/solid_boundary.py` exactly welds duplicate positions, validates closed solids, performs ordered Boolean union and removes redundant coplanar seams at backend numerical tolerance. It does not voxelize, fill openings or approximate a visual mesh. Separated components and real gaps remain separate.
- [Manifold](https://github.com/elalish/manifold) supplies the offline robust Boolean operation; `manifold3d==3.5.4` is pinned. Its [mesh provenance API](https://manifoldcad.org/docs/html/classmanifold_1_1_manifold.html) carries surface material identity through the Boolean. Original geometry, source hash, backend, triangle count, area and volume are retained in the output.
- `SceneToTake` emits the baked boundary under the root's existing owner ID, with per-triangle materials. Member bindings remain for stable object/source indices but emit no duplicate geometry. DSP, arrival layout, shaders and take versions do not change.
- Guards reuse existing mesh/collider/material snapshots and additionally verify membership, local frames and boundary bytes without managed allocations during validation.

Open sheets are ordinary acoustic surfaces, not solid-bake input. Bad topology is rejected. This stage does not compute vibration, modal resonance, a simplification proxy, or propagation caches. Boolean triangle diagonals can differ between equivalent inputs; physical exterior geometry is the invariant. The measured cube's field and rendered audio match exactly across the tested partitions, seeds and rigid motion, not a claim about every triangulation or sampling budget.

Baking is explicit, outside Play mode, with cancellation, a 120-second timeout, a 1 GiB worker-memory limit, low process priority and one numeric-library thread. These are authoring safeguards, not a large-scene throughput guarantee. Mac/Metal is validated; other platforms still need their own checks.
