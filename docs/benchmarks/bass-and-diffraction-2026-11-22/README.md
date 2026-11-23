# Bass and diffraction: room modes below the Schroeder frequency, and diffraction as a walk event — 22 November 2026

Follow-up to the [path tracer benchmark](../path-denoise-2026-11-12/README.md).
Two tracks: the low-frequency room modes that geometric acoustics cannot produce (part A), and part 24's open problem, diffraction as an event of the walk (part B).
All numbers are measured on this Mac (Apple M5, 16 GB) under `tools/heavy_guard.py` (one heavy job machine-wide, nice 19), against the FDTD ground truth of the [method comparison](../method-compare/README.md) and its scorer.

**Verdicts.**
- **Bass: room modes are adopted as an opt-in part of the path method.** A per-scene bank of the room's rigid modes, damped by its walls to first order and read at the source and listener every update, takes over below the Schroeder frequency measured from the modes themselves. On the wave suite's closed rooms the 125 Hz error falls from 1.98 dB / 21.6% EDT / 32.4% T30 to 0.08 dB / 1.3% / 1.2%, and the 250 Hz error from 2.41 dB / 14.3% / 42.9% to 1.07 dB / 10.0% / 34.3% (the band's share above the crossover keeps the path method's errors). It costs 1–3.5 ms of GPU per update. In the Church the Schroeder frequency is 22.7 Hz, so its 125 and 250 Hz bands do not change: the modes add only what lies below 32 Hz.
- **Diffraction as a walk event: derived, verified, not adopted.** The normalisation for scoring at the listener (UTD's D evaluated at the listener, not smoothed over the beam; the full incident kernel; Kouyoumjian and Pathak's grazing factor) reproduces UTD on the half-plane and the 2 m wall. But in rooms an incoherent UTD added at every edge a ray passes is not energy-conserving (+18 dB in the furnished room), and in the Church the edge search alone costs three times the update's budget. The switch stays off.

## Part A: room modes below the Schroeder frequency

### The gap

On the wave suite's three closed rooms the path method's errors below 250 Hz are its largest anywhere in rooms (2,048 rays, two seeds, RMS over receivers and seeds):

| Band | level dB (bias) | EDT % | T30 % |
|---|---|---:|---:|
| 125 Hz | 1.98 (+1.09) | 21.6 | 32.4 |
| 250 Hz | 2.41 (+1.97) | 14.3 | 42.9 |

The rooms are small and live: their Schroeder frequencies, measured from their own modes below, are 241 Hz (wood shoebox, 41 m³), 272 Hz (furnished room, 69 m³) and 518 Hz (stone shoebox, 56 m³).
Below that frequency fewer than three modes share a modal bandwidth, and the response at a point is its modes, which a ray method has no means to represent.

### Choosing the method, with numbers

Three candidates were weighed on accuracy against the FDTD below 250 Hz, cost per update, memory, and moving sources, listeners and doors.

**A modal bake** (built; the method chosen).
The room's rigid modes are baked once per scene; walls damp them to first order in their admittance; each update couples them to the current source and listener positions.
Accuracy on the closed rooms (modes alone, the bank cut at the crossover's top; below):

| Band | level dB (bias) | EDT % | T30 % |
|---|---|---:|---:|
| 125 Hz | 0.08 (+0.02) | 1.4 | 1.1 |
| 250 Hz | 0.26 (+0.01) | 6.7 | 3.3 |

(At 250 Hz the bank stops below the band's top, 500 Hz, in two rooms; the band's skirt above it is missing.)
With every mode to 1.1 kHz (the shoeboxes' analytic modes) the 250 Hz band is within 0.10–0.18 dB and 2.6–5.1% EDT: the model reproduces the FDTD, because both solve the same wave equation with the same walls, and the walls' admittances are small (0.003–0.018), where first-order damping is exact to second order.
Cost per update and memory: see "Runtime" below; per update it is a lookup of each mode at two points and one decimated impulse response.
Moving sources and listeners are exact: the modes do not move, only where they are read.
Moving doors are not: a door changes the room, so the bank holds the room in its baked state and the modal part is switched off when a door leaves its baked pose.

**A low-frequency wave solver at runtime** (the FDTD itself, coarse, on the GPU).
Its accuracy is the ground truth's by construction, and it would handle doors and openings; its cost decides.
At the ground truth's own resolution rule (10 points per wavelength at the top frequency) up to the crossover's top, timed at the ground-truth runs' measured median of 2.06 billion cell updates per GPU second (`modes.py candidates`, `candidates.json`; 1,024-sample updates at 44.1 kHz):

| Room | Schroeder frequency | crossover top | modes to the top | bank (fp16 shapes) | FDTD cells | FDTD GPU per update |
|---|---:|---:|---:|---:|---:|---:|
| wood shoebox | 241 Hz | 341 Hz | 229 | 9.4 MiB | 40 k | 2.7 ms |
| furnished room | 272 Hz | 385 Hz | 569 | 51 MiB | 98 k | 7.4 ms |
| stone shoebox | 518 Hz | 732 Hz | 2,762 | 151 MiB | 544 k | 78 ms |
| Church | 22.7 Hz | 32 Hz | 190 | 25 MiB | 24 k | 0.15 ms |

The FDTD's cost grows as V f_S⁴ ∝ T²/V: cheap in the Church, 13 times over the 6 ms budget in the small stone room, and it is a second, streaming pipeline (the field driven by the source's signal) beside the per-update responses everything else uses.
A bank's per-update cost grows with its modes, about T^1.5/√V, and it adds one term to the existing response.

**A learned correction** of the path method's band envelope.
Its ceiling, measured: an oracle that removes, per room and band, the path method's mean level error and mean relative EDT and T30 errors exactly (`modes.py oracle`) still leaves 1.29 / 1.22 dB level, 12.4 / 11.6 % EDT and 15.2 / 19.3 % T30 at 125 / 250 Hz.
What remains is the modes' variation from receiver to receiver, which a correction would have to learn per room, i.e. learn the room's modes; and it would need wave-solved training rooms, offer no guarantee, and add a training pipeline.

**Decision.**
The modal bake: it is the wave equation's own solution below the Schroeder frequency (accurate to 0.1 dB and about 1% in EDT and T30 at 125 Hz), it fits every room tried within the budget (runtime below), and it adds one term to the existing per-update response instead of a pipeline.
The FDTD would be its better partner only for large rooms with moving doors, where it is cheap; that is noted, not built.

### The method

`tools/room_modes.py` (tests: `tools/test_room_modes.py`).

- **Modes.**
  The rigid room's modes are the eigenpairs of the Neumann Laplacian, discretised by finite volumes on a rectilinear grid whose lines include every wall face (the FDTD's face-conforming cells): K ψ = k² M ψ, M the cell volumes, Σ M ψ² = 1.
  They are found by shift-invert Lanczos in slices of about 160 modes; a slice is accepted only once the returned eigenvalues reach past both of its ends, so none inside is missed, and each slice is written to disk at once (the three rooms' bakes peak at 1.8 GiB; 13–255 s each).
- **Fourth-order eigenvalues.**
  The stencil underestimates k² by (h²/12) Σₐ (∂ₐ²ψ)², its truncation error; each eigenvalue gets that term back, measured on its own eigenvector with the stencil's second differences (first-order perturbation theory).
  On the wood shoebox at 12.5 cm this takes the median frequency error from −1.2% to −0.05%; with the analytic modes' frequencies shifted by the stencil's error instead, the 250 Hz EDT error would be 9.6% instead of 2.5%.
- **Reading a mode at a point.**
  Trilinear interpolation of ψ over the eight cell centres around the point, minus its own leading error, Σₐ tₐ(1 − tₐ) hₐ²/2 times the interpolated second difference along a.
  Across a wall the neighbour is the cell's mirror image (a rigid wall's zero normal gradient makes ψ even about it).
  Plain trilinear readout of the analytic modes sampled on the 12.5 cm cells biases the wood shoebox's 250 Hz level by −0.76 dB; with the correction, by −0.21 dB, the exact modes' own −0.19 dB.
  The whole bake against the analytic modes (`modes.py numeric`, both cut at the crossover's top, both against the FDTD): wood shoebox 125 Hz 0.08 dB / 1.35% EDT / 0.78% T30 against 0.07 / 0.66 / 0.54; stone shoebox 250 Hz 0.27 dB / 5.2% EDT against 0.18 / 5.0.
- **Damping.**
  A locally reacting wall of real normalised admittance β damps mode n, to first order, at δₙ = (c/2) Σₘ βₘ(fₙ) Sₙₘ, Sₙₘ the integral of ψₙ² over material m's faces (Morse and Ingard).
  βₘ is taken at each mode's own frequency (linear in log₂ f between the band centres), from the material's random-incidence absorption by Paris's formula, as the ground truth does; the energy that leaves through a wall is its absorption plus its transmission.
  The bank stores Sₙₘ, so a material can change without another solve.
- **Coupling.**
  A monopole at s heard at r: H(ω) = Σₙ 4πc² ψₙ(s) ψₙ(r) / (ωₙ² − ω² + 2iδₙω), in the suite's convention (free field e^{−ikr}/r).
- **The crossover, at the Schroeder frequency measured from the bank.**
  The modal overlap is the modes per hertz times their mean half-power bandwidth δ/π, both over a window an octave wide centred on f; the crossover f_S is the lowest frequency at which it reaches 3 (Schroeder's criterion; with a uniform f² density and one decay time this gives 2,095 √(T/V), the textbook's 2,000 √(T/V) to the rounding of its constant).
  The bake sizes itself from Sabine's estimate and then extends itself, slice by slice, until it holds every mode up to √2 f_S.
- **Energy consistency.**
  Modes carry the response below f_S/√2, the path method above √2 f_S, and between them the weight is the renderer's own band shape, cos² in log₂ f, W_m(f) = cos(π/2 · (log₂(f/f_S) + ½)).
  Parts that are coherent in both models, the exact direct sound, are blended in amplitude, W_m + (1 − W_m) = 1; the path method's incoherent rest is blended in power, W_m² + W_g² = 1.
  So the response is the path method's, with its non-direct band energies scaled by their band's share above the crossover, g_b, plus Δh = W_m ∗ (h_modal − h_direct): no part is counted twice and none loses energy.
  Modes past the crossover's top are not needed: cutting the bank there instead of at 1.1 kHz changes band levels by 0.002–0.10 dB (`modes.py truncation`).

### The wave suite's closed rooms

The path method's own runs (2,048 rays, seeds 0 and 1) crossed over with the baked modes at 12.5 cm (`tools/room_modes.py suite`; scored by `tools/method_compare.py`; `modes.py score`, `modes.json`):

| Room, band | path: level (bias) / EDT / T30 | path + modes: level (bias) / EDT / T30 |
|---|---|---|
| wood shoebox, 125 Hz | 1.31 (−0.13) / 21.8 / 14.9 | 0.07 (+0.04) / 1.1 / 0.7 |
| wood shoebox, 250 Hz | 1.81 (+1.13) / 16.8 / 6.8 | 1.38 (+0.92) / 12.5 / 4.1 |
| stone shoebox, 125 Hz | 2.55 (+2.27) / 22.5 / – | 0.05 (+0.05) / 0.9 / – |
| stone shoebox, 250 Hz | 2.59 (+2.22) / 13.3 / – | 0.28 (+0.24) / 5.2 / – |
| furnished room, 125 Hz | 1.74 (+0.99) / 20.0 / 46.5 | 0.12 (−0.02) / 1.9 / 1.5 |
| furnished room, 250 Hz | 2.80 (+2.73) / 12.1 / 65.1 | 1.31 (+1.22) / 11.3 / 52.2 |
| **closed rooms, 125 Hz** | **1.98 (+1.09) / 21.6 / 32.4** | **0.08 (+0.03) / 1.3 / 1.2** |
| **closed rooms, 250 Hz** | **2.41 (+1.97) / 14.3 / 42.9** | **1.07 (+0.74) / 10.0 / 34.3** |

(The stone shoebox's T30 is undefined: its decay does not fall 35 dB within the ground truth's 2 s.)
At 125 Hz every room is now within 0.12 dB, 2% EDT and 1.5% T30 of the wave solution, from 1.3–2.6 dB, 20–22% and 15–47%.
At 250 Hz the improvement is as large as the band's share below the crossover: complete in the stone shoebox (f_S 518 Hz), partial in the other two, whose crossovers (171–341 and 193–385 Hz) leave the band's upper half to the path method.
That upper half carries the path method's own errors above its Schroeder frequency, most of all the furnished room's slow late decay (T30 +53%), which part A of the [energy-audio benchmark](../energy-audio-2026-11-09/README.md) traced to perfectly specular boxes that trap grazing ray families; the modes have no such error there (modes alone: 250 Hz T30 3.4% in that room).
The 500 Hz band is untouched except in the stone shoebox (1.17 → 0.42 dB).
Against the suite's IR baseline (`baseline-static-ir`, closed rooms: 125 Hz 1.62 dB / 16.2% EDT / 4.1% T30, 250 Hz 2.18 dB / 12.6% / 7.7%) the path method with modes is better at 125 Hz in all three, and at 250 Hz in level and EDT but not in T30 (34%, 19% with the crossover at 354 Hz): the furnished room's slow late decay above the crossover is the path method's, and the modes do not reach it at the Schroeder crossover.

**The crossover higher, as a budget.**
The path method is valid only above f_S, the modes everywhere, so the crossover may sit higher than f_S when the modes are affordable; that is a cost choice, like a ray count, not a second rule.
With it centred at 354 Hz (the modes alone up to 250 Hz, the path method alone above 500 Hz; the stone shoebox keeps its own 516 Hz), `room_modes.py suite --crossover 353.5534`:

| Closed rooms | level (bias) | EDT % | T30 % | modes (wood / furnished) |
|---|---|---:|---:|---|
| 250 Hz, crossover at f_S | 1.07 (+0.74) | 10.0 | 34.3 | 229 / 569 |
| 250 Hz, crossover at 354 Hz | 0.45 (+0.36) | 6.3 | 19.0 | 659 / 1,166 |

The furnished room's T30 at 250 Hz falls from 52% to 29%; what is left is the band's skirt above 354 Hz, still the path method's.

### The Church

`church_modes.py bake 0.75` voxelises the take's acoustic triangles at their first pose (doors closed) at 0.75 m, separates the nave's air from the outside's with one dilation, and bakes its modes; `church-0.75.json` holds the numbers.
- **Geometry.** 69,334 air cells, 29,250 m³ (31,300 m³ at 0.2 m in the energy-audio benchmark: the voxel walls take 7%).
  Faces between the nave's air and the outside's are the openings (the unglazed gables and belfry, and the mesh's open seams): 170 m².
  An opening is not a locally reacting wall: sound leaves through it, absorption 1, as through an open window and as the path method's escaping rays.
  It enters the first-order damping through the admittance whose diffuse-field loss is that absorption: averaged over the modes δₙ = (c/2) Σ β Sₙₘ is Sabine's c Σ 8β A / (8V), so β = 1/8 (β = 1 would lose eight times too much at first order).
- **Wall integrals on the mesh.** A staircase of voxels has up to √3 times the area of a slanted wall, so Sₙₘ is integrated over the triangles themselves (ψ² read at points spread over each triangle at h/2, on the side that faces the nave, which a 0.2 m mask of the nave decides): 7,030 m² of mesh faces the nave.
- **The Schroeder frequency is 22.7 Hz** (modal overlap 3, measured from the 491 modes to 46 Hz); the modes' reverberation time there is 6.4 s.
  The crossover runs from 16 to 32 Hz and needs 190 modes (25 MiB of fp16 shapes).
- **Energy consistency at the crossover** (`church_modes.py consistency`): a source and 256 listeners at least a quarter wavelength from every surface, octave band centred on f_S.
  The modes give −0.13 dB against diffuse-field theory with the same surfaces and absorption once Waterhouse's interior correction 1/(1 + Sλ/8V) is applied (−1.43 dB without it; receiver-to-receiver spread 2.1 dB).
  So at the crossover the two models agree in energy, up to Waterhouse's term, a wave effect (the energy a rigid wall's interference holds near it) that the path method does not have.
  With the organ itself (1.5 m from its case and the floor) and listeners at head height the modes give +5.5 dB over diffuse theory: a source and listeners within a quarter wavelength of a rigid surface radiate and receive more, through their coherent images, at 23 Hz.
- **Below 250 Hz the Church does not change.**
  Its crossover tops out at 32 Hz, below the 125 Hz band's lowest frequency (62.5 Hz), so its 125 and 250 Hz level, EDT and T30 are the path method's, by construction: in a room this large the path method is already above its Schroeder frequency in every band it renders.
  What the modes add is below 32 Hz, where the renderer now plays the 125 Hz band's geometric envelope (its band filters are flat below 125 Hz): the organ's pedal fundamentals (16.4 Hz for 32′ C, 32.7 Hz for 16′ C).
- **The bake against the FDTD on the same voxel church** (`church_modes.py fdtd`: the ground truth's solver on the very cells and voxel walls the modes were baked on, one 2 s run, 0.3 GPU-seconds; octave band at 22.7 Hz): with first-order damping the modes are 1.24 dB low (RMS 1.32 dB, EDT 18%) with source and listeners clear of the walls, 1.49 dB low (1.66 dB, EDT 23%) with the organ and listeners at head height.
  That is first-order damping failing, not the modes: the openings (β = 1/8), the glass and the copper roof are not small admittances on patches metres wide, and near the Schroeder frequency the modes overlap, so the walls couple them.
  Projecting the admittance onto the rigid modes in full (the damping matrix C_nl = Σₘ βₘ ∮ ψₙψₗ dS instead of its diagonal) and solving the coupled modes exactly in that basis (`room_modes.coupled`) gives 0.14 dB RMS, −0.01 dB bias and 1.4% EDT (`church_coupled.py`, 32 listeners clear of the walls): the bake is right, and the coupled form is the method's exact limit.
  In the suite's rooms (β ≤ 0.018) it changes nothing measurable (wood shoebox and furnished room, `suite --coupled`: within 0.05 dB and 0.4% of the diagonal at 125 and 250 Hz), so the renderer keeps the diagonal; the coupled form is in the Python reference only (see "Not done").
- There is no wave reference for the Church as it is (with its real triangles rather than voxels), so the path method's own 125 and 250 Hz errors there are not measured.

### Runtime

`Assets/Scripts/Simulation/RoomModes.cs` loads a bank (`room_modes.py export`, format "RMB1"), probes it at the source and listener each update, damps the modes from the scene's own materials, measures f_S and synthesises Δh = W_m ∗ (h_modal − h_direct): the modes at a decimated rate (the smallest divisor of the output rate at least four times the bank's top), W_m by FFT, the direct pulse subtracted, a windowed-sinc upsampling.
`GpuAudio.Modes.cs` does the same on the GPU and adds Δh into the path response's filtered partitions; the path histograms keep the share g_b; `PathReverb` is the CPU reference.
It is off by default (`settings.roomModesBank`, `-roomModes PATH` for the live Church), and applies only while source and listener are in the bank's air and every object listed in the bank's sidecar is at its baked pose.

| Check (`RoomModeChecks.cs`) | Result |
|---|---|
| C# against the Python combination, wood shoebox, seed 0 | 0.0016 dB at most, 0.0002 dB RMS over 320 receiver-bands |
| The `path-modal` measurement job end to end | within 0.002 dB of the Python reference |
| Rendered modal correction against the exact transfer function | +0.016 dB at 125 Hz, −0.026 dB at 250 Hz |
| GPU against CPU audio with the modes on | 1.0e-3 (binaural), 7.6e-4 (mono) relative |

Cost per update (48 kHz, two ears, 1,024-sample updates; GPU kernels fenced, varying with the machine's load):

| Bank | modes / top | f_S | bank in RAM | GPU memory | CPU (probes) | GPU kernels |
|---|---|---:|---:|---:|---:|---:|
| wood shoebox (2 s) | 326 / 388 Hz | 241 Hz | 14 MiB | 0.15 MiB | 0.07 ms | 0.97 ms |
| furnished room (2 s) | 706 / 418 Hz | 272 Hz | 64 MiB | 0.16 MiB | 0.07 ms | 1.35 ms |
| stone shoebox (2 s) | 3,307 / 800 Hz | 518 Hz | 181 MiB | 0.36 MiB | 0.09 ms | 1.7–3.5 ms |
| Church (6 s) | 491 / 46 Hz | 22.7 Hz | 71 MiB | 0.11 MiB | 0.05 ms | 1.72 ms |

Every bank fits beside the path method's own update within the 6 ms allowance; the banks hold more modes than the crossover needs (their bakes reach √2 times Sabine's estimate), so cutting them at the crossover's top would take the stone shoebox from 3,307 to 2,762 modes and the Church from 491 to 190.
The CPU reference synthesis takes 13–118 ms per update: it is the reference, not the live path.

### Not done

- **Coupled damping in the renderer.** The renderer uses first-order (diagonal) damping, exact in the suite's rooms; in the Church, whose openings and glass are large admittances, it leaves the modal part 1.24 dB low at 23 Hz, which the coupled form (Python, `room_modes.coupled`) removes. Porting it means one complex eigen-decomposition per material set (2N states) and complex modes in the synthesis.
- **Doors.** A bank is one state of the room; the modal part is switched off while a door is away from its baked pose. Per-state banks (doors open: the opening as a radiating patch) are not baked; in a large room a coarse runtime FDTD would follow the doors natively and is cheap there (0.15 ms per update in the Church), in small rooms it is not.
- **Both ears hear the same modal signal.** Below 32 Hz in the Church that is right; in a small room whose crossover reaches 730 Hz (the stone shoebox) the interaural differences of the modal part are lost. Probing the modes at each ear instead of the head is the principled fix, at twice the probe cost.
- **The Church has no wave reference with its real triangles**, so the path method's own 125 and 250 Hz errors there are not measured; the modal part is checked against the FDTD on its own voxel model and against diffuse-field theory.
- **No listening check** of the live Church with the modes on; the feature is off by default (`settings.roomModesBank`, `-roomModes PATH`), and banks stay out of git (a build would ship one in StreamingAssets).
- **The opening's admittance** (β = 1/8, its energy-equivalent) is a convention for a radiating aperture, stated above, not a solved radiation impedance.

## Part B: diffraction as a walk event, with the derived normalisation

### B.1 What changed

**The switch** (`settings.pathWalkDiffraction`; `MethodPath.hlsl`, "Walk diffraction"; kernels `TracePath3DWalk` and `TracePath3DBvhWalk`).
With it the source's edge rays and every edge connection (D, RD, DD) are off, and the edge distribution is not built.
Diffraction is an event of the walk instead, with the normalisation derived for track B (`walk_edge.py`):

1. **Deposit.**
   A ray piece of flux Φ whose line passes an edge at offset ρ and angle γ deposits Φ K1(ρ) / sin γ at its closest point on the edge.
   K1(ρ) = sqrt(2 / (π w²)) exp(−2 ρ² / w²) is the beam kernel's 1-D marginal, w² = λ L / π, L the chain's unfolded length to that point.
   Summed over rays this is the incident intensity per unit edge length.
   The whole kernel counts: a ray that hits a plane containing one of the edge's faces before its closest approach deposits with its line extended past the hit.
2. **No double counting.**
   A ray whose latest event was on a plane that contains one of the edge's faces deposits nothing on that edge (UTD's reflection-boundary terms already diffract that field).
3. **The diffracted ray** leaves the closest point on Keller's cone of the incidence from the chain's origin (p − d · travelled: the source, a scattering point, an edge, or their image).
   Its azimuth is uniform over the exterior angle and it carries the deposit times the angle, with no D yet.
4. **D pointwise.**
   Its first segment is scored at the listener by the beam kernel (width from the edge, the clearance truncation as before) times |D|² sin²β at the listener's own azimuth, with the listener's distance in UTD's transition argument.
   A deposit of that segment on a second edge takes D of the first edge at the second's point.
   Only past its first hit does the diffracted ray take D at its own azimuth, with the distance to the hit.
5. **Grazing.**
   With the chain's origin on one of the edge's faces (edge 1 seen from edge 2 along a wall's top) the deposit counts twice and |D|² a quarter.
6. **Sampling: splitting instead of a choice.**
   The undiffracted ray goes on with its full weight, and every deposit is a candidate of a one-slot weighted reservoir per ray, as the transmitted branches are.
   When the ray ends (persistent rays: when the update ends) the kept candidate becomes a diffracted ray of its own, carrying the summed weight, after roulette on it.
   The brief's choice between diffracting (probability q) and going on (weight 1 / (1 − q)) is unbiased for any q, but it leaves the undiffracted ray a weight (1 − q)^−n with probability (1 − q)^n after n pieces that pass an edge.
   In the Church a ray passes an edge within reach on almost every piece (17 deposits per surface event on average, B.3), so with q = ½ the undiffracted late field would be carried by weights 2^n.
   Splitting removes that and keeps every candidate; the proposal weight is the deposit times the exterior angle and UTD's scale 1 / (8πk) at 125 Hz (once more per D not yet applied), so the slot's sum is an energy for the roulette.

**Two refinements the far-side check needed (both derived, neither tuned).**
- *Part 11's beam weights term by term.*
  UTD's four terms each have a transition at one boundary: the shadow boundaries at φ′ − π and φ′ + π and the reflection boundaries of face n at 2nπ − π − φ′ and of face 0 at π − φ′.
  The weight 1 − exp(−2X s′/s) stands for a traced beam that already smooths that boundary, so it now applies only where the boundary lies in the wedge's exterior: the shadow-boundary terms where the incident field is a reflected beam, the reflection-boundary terms where their face is lit.
  At grazing incidence the face's reflected field is the incident field itself (no reflected beam is traced), so its reflection term takes the coinciding shadow-boundary term's weight.
  Without this the 2 m wall read −4.0 / −2.1 dB at 125 / 250 Hz: at grazing the reflection-boundary term (the double of the incident term that Kouyoumjian and Pathak's ½ halves) was weighted down by 0.43 at 125 Hz, and D of the first edge toward the second lost a term whose boundary lies inside the solid.
  The current method's edge connections weight both reflection terms whatever φ′; that is the same flaw, left alone here because the default path must stay bit-identical.
- *The edge index* (`PathWalkEdges`, B.3).

**What else.**
- A persistent ray in its first diffracted segment keeps the edge's state in `PathRay.motion.yzw`; the slot is queued as a ray of its own at the end of an update, as the transmitted branch is, so a diffracted ray starts at most one update late (its contributions keep their own times).
- The walk kernels have no delay rates (no `PATH_DOPPLER` variant): the switch is for static scenes.
- The static walks are compiled from unchanged text: the walk code is under `PATH_WALK_EDGE`, and its per-term UTD is a function of its own.
  `MeasurePathHistogramHash` on the branch gives the base's hashes in both places it was measured: 407F2A3B… (full solve) and 1C01E5D5… (persistent) on part 44's code, and 6175A115… and 27FD8A63… after the rebase onto a112e54, whose own code gives the same.
  (A first version that generalised `PathUtdEnergy` for both kinds of walk gave 6175A115… and 27FD8A63… on part 44's code, the hashes a112e54 now gives by itself: these outputs move in their last bits with how the compiler lays out the static walk, as part 40 found, so the hash checks that the static walks compile alike, not that their arithmetic changed.)
- Rebased onto a112e54 (the energy and pressure methods removed): the walk keeps its own per-object plane grouping (`PathPrimitivePlane`), which the energy method's arrival enumeration used to provide; the checks below were rerun there and give the same numbers to every digit shown.
- `MethodChecks`: walk cases (shortcuts, beams only, persistent) of `MethodsMatchTheHalfPlaneReference` and `MethodsReachTheFarSideOfASolid`; `MeasurePathProfile` (now also for full solves), `MeasureChurchLevels` and the suite's jobs take the switch; profile bits 17 and 18 skip the candidates past a hit and the whole candidate search.
- `walk_edge.py` (the derivation's Monte Carlo) is copied here; `walk_church.py` and `walk_suite.py` drive the measurements below.

### B.2 The analytic checks

dB against the independent BTM band energies (`MethodsMatchTheHalfPlaneReference`, `MethodsReachTheFarSideOfASolid`), 125 / 250 / 500 / 1,000 / 2,000 Hz.
Walk diffraction: 4,194,304 rays (full solve) and 1,048,576 (persistent); the current method its checks' 65,536 and 131,072.
"Reference" is `walk_edge.py`'s Monte Carlo of the same estimator with this method's listener kernel (width from the distance to the listener image, as the beams use; 2–4 M rays).

| Fixture | UTD (grazing ½) | current method | walk, full solve | walk, persistent | reference |
|---|---|---|---|---|---|
| half-plane | +0.32 / +0.28 / +0.28 / +0.28 / +0.29 | – / +0.30 / +0.29 / +0.29 / +0.30 | +0.73 / +0.54 / +0.46 / +0.44 / +0.49 | +0.78 / +0.64 / +0.59 / +0.58 / +0.71 | +0.72 / +0.49 / +0.35 / +0.29 / +0.34 |
| 2 m wall (DD, grazing) | +0.17 / +0.21 / +0.24 / +0.27 / +0.32 | +1.79 / +1.50 / +1.19 / +0.91 / +0.70 | +1.03 / +0.72 / +0.45 / +0.62 / +0.79 | +0.56 / +0.26 / −1.23 / −3.01 / −4.23 | +0.65 / +0.55 / +0.59 / +0.67 / +0.77 |
| 0.25 m wall (measured only) | −3.24 / −2.50 / −1.66 / −0.84 / −0.21 | +8.29 / +7.84 / +7.13 / +6.09 / +4.76 | −3.48 / −2.69 / −1.88 / −1.05 / −0.62 | −3.14 / −2.35 / −1.50 / −0.14 / +2.40 | −2.80 / −2.05 / −1.16 / −0.19 / +0.40 |

- **The normalisation holds.**
  On the half-plane the walk is within 0.5 dB of BTM at 250 Hz–2 kHz and +0.73 dB at 125 Hz; on the 2 m wall within 1.03 dB in every band, and closer to BTM than the current method at 125–1,000 Hz.
  The GPU agrees with the reference within its noise everywhere but the 2 m wall at 125 Hz (+0.4 dB); the reference leaves out the box's other ten edges and truncates the listener kernel at 1 m where the listener's clearance is 2 m, both of which matter most for the widest kernel.
- **The excess over UTD at low frequency is the listener kernel's, not the event's.**
  The beams take the width at the distance to the listener image, lu = sqrt(L² + ρ²), not at the closest approach L; the kernel then integrates to 1.036 (125 Hz), 1.034, 1.019, 1.010 and 1.005 (2 kHz) over the half-plane listener's clearance instead of 1.
  With the width at L the reference reads +0.13 / +0.01 / −0.04 / −0.22 / −0.51 dB against UTD.
  The beams have used lu since part 8, for every reflected beam too; changing it changes the default path and is left for a separate decision.
- **The variance is the event's real cost.**
  D is scored through a beam kernel as narrow as the listener's Fresnel zone, so at 2 kHz the half-plane needs about 16 times the rays of 125 Hz for the same noise (the reference's standard error at 65,536 rays: 5 / 8 / 13 / 20 / 34 %).
  Double diffraction goes through two such kernels and is heavy-tailed above 500 Hz: the reference read −2.9 and −7.9 dB at 1 and 2 kHz with 1 M rays and +0.7 and +0.8 dB with 4 M; the persistent check, with 1 M rays, asserts only to 500 Hz.
- **The thin wall** is UTD's limit, not the event's: with its edges a quarter metre apart UTD itself is 3.2 dB under BTM at 125 Hz, and the walk follows it.
- **Brute force against the index**: the same checks with every edge tested (`useBvh` off) give the same energies to four digits and 770,553 against 770,546 diffracted rays (the reservoir draws in another order).

### B.3 The edge index and the time per update in the Church

**Index** (`PathWalkNear`, `PathWalkMesh`, `PathWalkEdges`).
- *Near a piece*: the scene's own trees (part 27), queried with the piece as a capsule: the top-level tree's world boxes and each mesh tree's own-frame boxes padded by three widths of the widest band at the piece's far end (a superset of the capsule), the piece first clipped to the scene's box padded likewise.
  A leaf's triangles give their edges (`EdgeRanges`), each taken once, from its first face (the edge lies in that triangle, so a leaf that could hold a candidate is always visited).
- *Past a hit*: only edges with a face in the hit's plane qualify, so they come from a list per mirror plane (the coplanar faces of one object, `MethodPrimitivePlane`); each passes the three-width test against its own closest approach, so no length bound is needed.
  The first version found them with a capsule along the extended line, bounded where an edge in the plane can no longer be within three widths (x² sin²θ = 9λ(L + x)/π); grazing hits made that capsule tens to hundreds of metres long, and it took two thirds of the walk's time.
- An edge counts within three widths of the widest band (exp(−18) of the kernel's peak; K1 is a Gaussian of standard deviation w/2, so this is six of them).

**Time per update in the Church** (persistent rays, population control, 23 ms updates, the walk render's settings with a 6 s horizon; GPU time of the walk with fences, mean of four poses, `MeasurePathProfile`):

| Configuration | N = 500 | N = 2,000 | N = 64,000 |
|---|---:|---:|---:|
| current method (shortcuts, edge rays, edge connections) | – | 3.9 ms | 22.0 ms |
| walk diffraction, extension by capsule (first index) | 202 ms | 268–336 ms | – |
| walk diffraction, plane lists (kept) | 67.5 ms | 91.9 ms | – |
| beams only with walk diffraction, plane lists | 76.5 ms | 98.7 ms | – |
| walk diffraction without the candidate search (profile bit 18) | – | 3.0 ms | – |

(Current method: 3.9, 7.1, 8.1, 13.2 and 22.0 ms at N = 2, 8, 16, 32 and 64 thousand, edge distribution included.)
- A ray in the Church meets 17 deposits per surface event on average (10.0 M per horizon of updates at N = 2,000): at 125 Hz three widths after 20 m of chain are 13 m, and after a second of reverberation 50 m, the whole church.
- The time is the slowest thread's: at N = 500 the walk still takes 67.5 ms, because a chain of 230–300 iterations, each now searching a capsule that holds most of the church's 14,075 edges, runs in series.
  So no population fits 23 ms: the floor is three times the budget, against the current method's 22 ms at N = 64,000.
- Narrowing the reach to two widths (K1 loses 6 × 10⁻⁵ of its mass) would shrink the capsules' area 2.25 times, not enough to reach the budget; it was not done.

### B.4 The Church's outside tail

Part 20's measurement (`MeasureChurchLevels`, `walk_church.py levels`): persistent rays under population control, 1,024-sample updates, a 2 s horizon of warm-up, then 64 updates; the late field 0.2–2 s at 125 / 500 / 2,000 Hz; outside at 16.5 s into the walk, inside at 3 s.
No population of the walk modes fits 23 ms (B.3), so they ran at N = 2,000, about four times the budget (92 ms per update), against the current method at its 23 ms point on this code (N = 64,000, 22 ms) and at the same N.

| Point | Configuration | time per update | level (125 / 500 / 2,000 Hz) | sd / mean | median / mean |
|---|---|---:|---|---|---|
| outside | current, N = 64,000 | 22 ms | 3.2e-4 / 2.4e-4 / 2.2e-4 | 2.6 / 3.0 / 4.1 | 0.11 / 0.10 / 0.08 |
| outside | current, N = 2,000 | 4 ms | 3.9e-5 / 2.4e-5 / 2.8e-5 | 3.2 / 2.1 / 3.1 | 0.09 / 0.10 / 0.04 |
| outside | walk, N = 2,000 | 92 ms | 2.3e-5 / 1e-8 / 1e-10 | 3.6 / 5.3 / 6.1 | 0.03 / 0.00 / 0.00 |
| outside | beams only, walk, N = 2,000 | 99 ms | 2.4e-5 / 1e-13 / 0 | 7.9 / 7.9 / – | 0.00 / 0.00 / – |
| inside | current, N = 64,000 | 22 ms | 8.9e-2 / 6.5e-2 / 5.1e-2 | 1.0 / 1.0 / 0.9 | 0.73 / 0.81 / 0.79 |
| inside | current, N = 2,000 | 4 ms | 7.6e-2 / 4.5e-2 / 3.8e-2 | 4.3 / 2.2 / 1.7 | 0.12 / 0.18 / 0.24 |
| inside | walk, N = 2,000 | 92 ms | 3.1e-1 / 3.1e-1 / 1.2e-2 | 3.2 / 7.6 / 5.9 | 0.12 / 0.01 / 0.05 |
| inside | beams only, walk, N = 2,000 | 99 ms | 2.4 / 3.0e-2 / 3.4e-2 | 7.7 / 7.2 / 7.7 | 0.00 / 0.00 / 0.00 |

- Outside, the walk's late field at 500 Hz and 2 kHz is four to six orders of magnitude under the current method's and its median is zero: at this population, almost no diffracted ray carries sound out of the doors in a given update.
  Its tail is not lighter than the current method's: sd / mean 3.6–6.1 against 2.6–4.1, at four times the time.
- Inside, its means are off by factors of 3.4 (125 Hz, high), 4.8 (500 Hz, high) and 4 (2 kHz, low), with sd / mean up to 7.6, so 64 updates do not separate a bias from a heavy tail; beams only reads 27 times the current method's level at 125 Hz with sd / mean 7.7.
  Both are what the event predicts in a room dense with edges.
  Every ray deposits on about 17 edges per surface event, and UTD's incoherent |D|² carries, per metre of a lit edge, the incident intensity over a width of order sqrt(λL): integrated over azimuth, 0.5 m at 125 Hz and 5 m (both boundaries beam-weighted), 0.13 m at 2 kHz.
  A diffracted ray goes on to deposit again, so in a reverberant room the walk adds UTD's energy at every edge a ray passes, generation after generation, where the current method adds it once per path (D) or twice (DD); the low bands grow, and the rare long chains of branches make the tail.
  In open scenes, which a ray leaves after a few events, the same event reproduces UTD (B.2).

### B.5 The wave suite at equal time

`walk_suite.py`: each scene's GPU time per solve at 2,048 rays (`MeasurePathProfile`, four receivers) sets the walk modes' rays for the current method's time, scene by scene; four seeds each (the current method's seeds 0 and 1 are the scoreboard's runs, 2 and 3 new); `beams.py`'s scorer, with the suite's static-IR baseline beside them.

| Scene | current, ms per solve | walk, ms | rays at equal time (walk / beams only) |
|---|---:|---:|---:|
| barriers (rigid / presets) | 0.8 / 0.8 | 1.4 / 1.4 | 1,119 / 1,227 · 1,084 / 1,184 |
| canopy (rigid / presets / BEM) | 1.0 / 1.0 / 2.6 | 2.0 / 2.0 / 6.5 | 1,028 / 1,019 / 821 · 1,005 / 1,039 / 803 |
| blocks (BEM) | 1.4 | 2.1 | 1,354 · 1,311 |
| shoebox (0020 / 0010) | 4.3 / 6.8 | 11.4 / 18.1 | 773 / 772 · 744 / 750 |
| doorway / corridor | 6.9 / 10.6 | 23.7 / 70.3 | 594 / 308 · 595 / 307 |
| furnished room | 19.7 | 223.9 | 181 · 181 |
| suite | 55.9 | 362.8 | (6.5 times the current method's time) |

RMS errors (level dB / EDT % / T30 % / ETC dB / level bias dB):

| Configuration | all | in view | shadow | closed rooms |
|---|---|---|---|---|
| current method, 2,048 rays | 3.38 / 195 / 70 / 13.8 / +1.31 | 1.81 / 239 / 65 / 12.1 / +0.93 | 4.91 / 93 / 79 / 16.1 / +1.92 | 1.95 / 19 / 31 / 5.6 / +1.38 |
| walk diffraction, equal time | 9.64 / 698 / 101 / 16.2 / +0.35 | 3.96 / 851 / 66 / 13.0 / +1.53 | 15.39 / 212 / 139 / 20.0 / −1.85 | 10.00 / 647 / 213 / 11.5 / +5.21 |
| beams only, walk diffraction, equal time | 9.65 / 702 / 101 / 16.2 / +0.33 | 3.96 / 856 / 66 / 13.0 / +1.53 | 15.41 / 216 / 139 / 20.1 / −1.90 | 9.99 / 647 / 213 / 11.5 / +5.21 |
| static-IR baseline | 8.85 / 420 / 38 / 10.1 / +5.45 | 3.69 / 113 / 41 / 7.4 / +1.74 | 13.36 / 657 / 35 / 13.1 / +11.20 | 1.73 / 15 / 6 / 4.9 / +0.85 |

Level error and bias by scene (dB; current / walk / static IR):

| Scene | level RMS | level bias |
|---|---|---|
| barriers rigid | 4.87 / 5.87 / 4.51 | +3.40 / +0.78 / +2.80 |
| barriers presets | 4.96 / 5.54 / 4.75 | +3.50 / +0.87 / +3.04 |
| canopy rigid | 2.92 / 4.85 / 8.62 | −0.53 / −1.25 / +5.60 |
| canopy presets | 2.80 / 4.58 / 8.84 | +0.18 / −0.63 / +5.98 |
| canopy BEM | 4.19 / 4.84 / 10.06 | +0.36 / +0.38 / +7.05 |
| blocks BEM | 3.17 / 17.68 / 11.69 | +1.20 / −5.93 / +7.98 |
| shoebox 0020 / 0010 | 1.55 / 1.54 / 1.84 · 2.23 / 2.28 / 1.66 | +0.80 / +0.78 / +1.12 · +1.85 / +1.90 / +1.05 |
| doorway | 3.11 / 3.33 / 12.65 | +2.25 / +2.63 / +9.80 |
| corridor | 1.96 / 3.83 / 11.08 | +1.31 / +3.09 / +7.87 |
| furnished room | 2.14 / 20.30 / 1.65 | +1.72 / +17.95 / +0.07 |

- Beams only and shortcuts give the same numbers here: the suite's surfaces do not scatter, so the two modes differ only in the scattering points' connections.
- **The furnished room diverges: +18 dB.**
  It is B.4's mechanism with nothing to hide it: furniture puts an edge within reach of nearly every piece of every ray, each deposit adds UTD's incoherent energy on top of a complete geometric field, and the diffracted rays deposit again; the room's level and decay (T30 error 355 %) are those of a field that gains energy at every reflection.
  The corridor (+1.8 dB over the current method's bias) shows the same effect, weaker.
- **Shadow scenes do not gain either.**
  At equal time the walk gets 800–1,350 rays against 2,048, and its D is scored through narrow kernels: the barriers' bias falls from +3.4 to +0.8 dB, but the RMS error rises (4.9 → 5.5–5.9 dB) with the noise; blocks reads −5.9 dB, the double diffraction between nearby blocks being both heavy-tailed (B.2) and at UTD's limit (the 0.25 m wall).
- Overall the walk modes are worse than the current method in every group and metric but the level bias, and worse than the static-IR baseline overall (9.6 against 8.9 dB), where the current method beats it by 5.5 dB.

### B.6 Decision

Not adopted: neither configuration is as accurate per unit of compute as the current method; both are far less accurate, and both cost more.
- The derivation is right where it applies: on the analytic checks the walk event reproduces UTD (half-plane within 0.5 dB of BTM above 125 Hz, the 2 m wall within 1.03 dB in every band, closer than the current method's edge connections at 125–1,000 Hz), and the grazing rules are what make double diffraction right.
- It does not apply in rooms: an energy UTD added at every edge a ray passes, generation after generation, is not conservative, and in an edge-dense room (the furnished room, the Church's inside) the response grows by up to 18 dB.
  The current method's edge connections add UTD once per path, which bounds the same overcount.
  A walk event would need its deposit taken out of the geometric field (or the lit-side half of UTD's transition left out wherever the geometric field already reaches) to be usable there; that is a different estimator, not a normalisation.
- It costs more: three widths of the widest band reach most of the Church's 14,075 edges in its late field, so the time per update has a floor of 68 ms (three times the 23 ms budget) at any population, and the suite takes 6.5 times the current method's time at equal rays.
- Its variance is high where it should help: D through the listener's Fresnel-zone kernel needs about 16 times the rays at 2 kHz as at 125 Hz, and double diffraction is heavy-tailed above 500 Hz.

The switch stays (`settings.pathWalkDiffraction`, off), with its checks, so the comparison can be repeated; the default path is unchanged and bit-identical.
Two findings stand apart from the decision: part 11's reflection-boundary weight in the current method's edge connections is applied whether or not the face is lit (B.1), and the beams' width taken at the listener image's distance instead of the closest approach makes the kernel integrate to up to 1.036 (B.2).

## Merge notes

Branch `research/bass-and-diffraction`, rebased on main at a112e54 (not pushed):

| Commit | What |
|---|---|
| 366018a | feat: room modes below the Schroeder frequency, crossed over with the path method (`RoomModes.cs`, `GpuAudio.Modes.cs`, `RoomModeChecks.cs`, `tools/room_modes.py`, `tools/room_modes_unity.py`, `docs/method.md` §9) |
| c5077a9 | fix: the probe's neighbours, the Schroeder search, the Church's openings in C# |
| 02ee3b5 | feat: diffraction as a walk event (`settings.pathWalkDiffraction`, off; separate walk kernels) |
| c570061 | docs: part B's write-up (folded into this README by the next commit) |
| (docs) | this README, `modes.json` |

- It applies to a112e54 as a fast-forward; if main has moved, rebase onto it (the conflicts seen so far were in `MethodMeasurement.cs`, `GpuAudio*`, `PathReverb.cs` and `tools/method_runs.py`).
- **Defaults are unchanged.** `settings.roomModesBank` is empty and `settings.pathWalkDiffraction` false; with both off the path method's output is main's (the static walks' histogram hashes equal a112e54's: 6175A115… / 27FD8A63…, not part 44's, which moved with main's compile; part B's note).
- **Banks are not in git** (14–181 MiB each). Reproduce: `$PY tools/room_modes.py suite` (the suite rooms), `$PY docs/benchmarks/bass-and-diffraction-2026-11-22/church_modes.py bake 0.75` then `$PY tools/room_modes_unity.py church` (the Church's bank and its pose sidecar), all under `tools/heavy_guard.py`. A build would ship a bank and its `.json` sidecar in StreamingAssets (`-roomModes NAME`).
- The `research/walk-diffraction` branch and the `realisticaudioraytracing2d-walkedge` worktree are superseded (their commits are cherry-picked here) and can be removed, with `realisticaudioraytracing2d-research` once merged.
- **Open decisions on the default path**, found here and not changed: the beam kernel's width is taken at the distance to the listener image rather than at the closest approach (it integrates to 1.036 at 125 Hz, B.2), and the current edge connections weight both of UTD's reflection-boundary terms whether or not the face is lit (B.1).
