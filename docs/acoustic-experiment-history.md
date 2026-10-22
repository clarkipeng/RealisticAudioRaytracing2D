# Acoustic experiments: history, 2D maps and GPU route generation

The latest offline experiment also moves **direct/reflected and mixed reflection–diffraction calculations onto the GPU** and reuses route geometry. At one million samples, the 9-box map improves from **32.0% to 22.3% complex-pressure error**, with measured component time falling from **30.95 to 1.51 seconds**. The 25-box map improves from **83.2% to 62.2%**, with **214.23 to 3.66 seconds**. The remaining dense-scene error is still unacceptable for a high-fidelity reference.

These are complete **single-frequency spatial-field component totals**, not live audio or full-process timings. The previous totals combine this session's GPU measurements with previously measured CPU common-field costs. Small scenes need far fewer samples; one million is slower without a useful accuracy gain there. No runtime files or dependencies changed. All new work ran silently in the isolated Unity Metal batch runner with audio disabled, one worker and reduced priority.

**Latest experiment: geometry reuse and GPU reflection/mixed paths — 21 October**

![Larger 2D pressure maps](benchmarks/gpu-reuse-large-2026-10-21.png)

![Original smaller 2D scenes](benchmarks/gpu-reuse-small-2026-10-21.png)

Each map uses seed 17, width 1, a two-interaction limit and 250 Hz. It represents a point source and infinitely extruded rigid boxes: an XY/2.5D benchmark, not finite 3D or strictly cylindrical 2D propagation. Columns share an absolute scale. “Error” is `100 × norm(complex pressure − wave reference) / norm(wave reference)` and includes phase; the ±3 dB metric measures amplitude agreement. Neither is a listening-artifact score. The independent wave solutions retain their earlier refinement qualifications; they are numerical references, not exact truth.

| Scene | Previous 65K error | Previous map time | All GPU 65K time | All GPU 1M error | All GPU 1M time |
|---|---:|---:|---:|---:|---:|
| 1 box | 1.85% | 0.619 s | 0.294 s | 1.83% | 1.652 s |
| 9 boxes | 32.02% | 30.954 s | 0.563 s | 22.29% | 1.510 s |
| 25 boxes | 83.23% | 214.227 s | 2.535 s | 62.23% | 3.664 s |
| Thick wall | 1.46% | 0.322 s | 0.165 s | 1.46% | 0.649 s |
| Staggered walls | 8.69% | 0.822 s | 0.154 s | 8.69% | 0.613 s |
| Slit | 2.37% | 0.808 s | 0.155 s | 2.42% | 0.630 s |
| Corridor | 13.51% | 0.868 s | 0.139 s | 13.41% | 0.582 s |

At the same 65K budget, all-GPU errors are 1.85/32.10/83.69/1.46/8.69/2.36/13.52% in table order: moving the mixed integral from CPU quadrature into the shared Monte Carlo budget slightly changes variance, and does **not** improve accuracy by itself. The time saved permits more samples. For nine boxes, **262K gives 24.03% in 0.754 s**; for 25 boxes, **262K gives 62.56% in 2.844 s**, close to the 1M result. These are useful operating points, not a claim that 1M should become a universal default.

What changed, measured separately:

1. **Geometry reuse:** prepare each pure route's distances and wedge terms once per receiver; every height sample reads them. Modes 0 and 1 produce bit-for-bit identical pressure in the measured cases. At 65K, extra dispatch cost offsets the saving: old/new median solve/readback speed ratios are 0.997–1.010× across the three larger maps. At 1M, the old/new median-time ratio is **1.17× / 1.20× / 1.22×**. Reuse is helpful when enough height samples amortize preparation.
2. **GPU reflections and mixed paths:** GPU image-source geometry enumerates direct, one- and two-face reflections, plus one-reflection/one-edge paths. Pure and mixed diffraction routes share the fixed height-sample budget. No cached CPU acoustic field is added to the new result; the CPU only adds the two returned complex component sums. This removes the large CPU common-field calculation. The final reduction/readback could be combined later.
3. **Spend the saving on samples:** compare 65,536 / 262,144 / 1,048,576 samples without changing proposal width, gain, horizon or covered path orders. Large-scene noise decreases, while several small-scene errors plateau. No amplitude fitting, smoothing, extra physical gains or rejection of inconvenient large contributions is used.

```text
GPU, scene/source setup:
    build edge visibility graph
GPU, each receiver:
    count visible one-edge and two-edge routes
    prepare each route's distances and wedge terms once
    find valid mirror→edge and edge→mirror routes; prepare and compact them
    find valid direct / one-mirror / two-mirror distances
    distribute fixed N height samples across pure + mixed edge routes
    for each sample, in parallel:
        read its prepared geometry
        sample heights with the existing complete-route Cauchy mixture
        add signed pressure × physical phase / sampling probability
    sum sampled pressure and analytic mirror pressure
CPU:
    read back the two component sums and add them
```

This still contains an explicit two-interaction route/face enumeration. It is not a general-depth GPU ray walk. With E edges/faces and W boxes, graph construction and brute-force mirror/mixed visibility cost **O(E²W)**, with **O(E²)** persistent storage. Pure-route preparation is O(E²) candidate/count work plus the valid descriptors; sampling is O(N) bounded-depth integration with prefix-search overhead. Mixed compaction is a serial O(E²) scan. Geometry reuse removes repeated per-sample distance/angle work; it does not remove the dense visibility cost. The 25-box all-GPU 65K solve is slower per receiver than the pure-edge stage because it now includes that work. A scene BVH, sparse visibility and general-depth sampling remain separate future experiments.

**Cost accounting and repeatability**

Totals sum Python input packing, C# file read/buffer allocation/initial upload submission, warmed scene-graph work, receiver input submission, all receiver dispatches, blocking readbacks and finite-value checks. Previous methods additionally include the CPU direct/mirror/mixed calculation: archived 0.3605/30.6905/213.9573 s for the three larger maps, freshly measured 0.1785/0.6891/0.6738/0.7500 s for the four small maps. The large CPU costs were reused instead of rerunning several minutes of identical quadrature; the totals are **sums of measured components from different runs**, not one stopwatch around the old process. Shared fixture allocations include buffers unused by modes 0/1. Results are host-observed solve/readback times, not GPU hardware timestamps.

Unity startup, initial shader compilation/warmup, reference solving, diagnostic exports, plots, broadband bins, time-domain synthesis and networking are excluded. The static graph is timed once per case; receiver-dependent geometry is rebuilt at every map position. Modes rotate execution order per receiver. This is one final map pass, not a statistical hardware speed guarantee. Final Unity test execution took 46 seconds for the whole suite, separate from per-method field costs.

Four seeds were compared on the same 25 evenly spaced receiver IDs in each scene, including seed 17 from the full maps. No proposal tuning was performed on these results. In the nine-box subset, all-GPU mean error changes **31.94% → 23.49%**, and seed-to-seed RMS variation **17.04% → 2.73%** of reference RMS. For 25 boxes, subset mean error changes **108.55% → 87.22%**, and variation **59.12% → 10.68%**. Subset error is higher than full-map error because these are different receiver distributions; it must not be substituted for the full-map score. The much larger remaining discrepancy suggests sampling noise alone is insufficient to explain it, but this experiment does not identify or prove the missing physical contribution. More complete path coverage needs its own controlled comparison.

**Verification and limits**

The final suite passes **70 Unity tests**: 21 full-map cases, 42 additional-seed cases, six geometry/sampling controls and the 115-value angle-control test. All 4,842 spatial cells are evaluated at all three budgets. Independent Python checks verify visibility, mirror lengths, mixed-route eligibility and Monte Carlo pressure at 22 control positions, including empty geometry, short delay support and more routes than samples. Maximum aggregate control discrepancy is **0.00125%** relative norm. Geometry reuse is bit-for-bit identical over the entire measured suite. Across the full maps, GPU direct/mirror pressure matches independent CPU image-source pressure within the declared tolerance (max absolute difference recorded in the ledger).

The controls caught two implementation problems before the final suite: too many read/write buffer bindings on Metal, fixed by declaring integration inputs read-only; and reflections numerically landing at a face endpoint, fixed by rejecting contacts within the precision of the intersection operands. This numerical endpoint margin is not qualified for arbitrary scales or near-degenerate geometry. Initial failed runs remain in the evidence but are not counted as passing. Existing compiler warnings remain; passing tests establish only the tested numerical cases.

Keep this as a reproducible **offline prototype**, not a production renderer or DLSS teacher. It covers static rigid rectangles, source/listener z=0, 250 Hz and at most two interactions; moving geometry, other frequencies/materials, finite 3D and audible temporal continuity are unqualified. Pilot-guided allocation and reuse of sampled paths across frequency bins were **not implemented or benchmarked** in this experiment. The result supports removing the CPU common-field bottleneck and amortizing geometry at larger budgets; it does not yet support claiming realistic dense-scene audio.

[Full metrics and provenance](benchmarks/gpu-route-reuse-2026-10-21.json) · [Source, raw results, controls and pressure arrays](benchmarks/gpu-route-reuse-2026-10-21.zip). The live renderer is unchanged from the previous checkpoint; the prior offline experiment is preserved at `cb0fef8`.

**The history, in order**

| Checkpoint | What we tried | What it established / decision |
|---|---|---|
| `05a5477`, 7 Oct | Explicit visible edge routes, replacing proximity-based diffraction | Better-defined geometry, but a separate route search was not the unified multi-edge ray walk we wanted. Preserved in Git; no comparable wave-reference map from this stage. |
| `343b797`, 7 Oct | Signed diffraction inside the ordinary GPU path loop | Removed four separate edge kernels and their route buffer; multi-edge paths shared the existing audio backend. Numerical controls passed, but later spatial wave comparisons exposed severe variance. |
| `d2f5d47`, 17 Oct | Source-centric, listener-centric, joined and bidirectional MIS sampling, with/without guidance | Joined sampling helped the box example; no universal winner on staggered walls. This was a discretized two-edge integral comparison, **not independent acoustic truth**, and not a complete bidirectional renderer. Tables and code remain at that revision. |
| `3d24edc`, 18 Oct | Shared-solid-face factor and grazing-geometry corrections; conditional next-edge sampling | Kept the physical factor and numerical fixes. The conditional sampler helped a known route but worsened some whole-walk outliers, so it was not shipped. |
| `62f0382`, `1b8fec8`, 18 Oct | Independent BEM and planar FEM wave references | Added a baseline that does not reuse our ray model. Free-space/cylinder controls and refinement checks qualify its numerical accuracy. These references are approximate solutions, not perfect truth. |
| `dcccabb`, 19 Oct | Deterministic edge integration; three local/reachable-edge proposal variants | Isolated sampling error from missing paths. Rejected the three proposal variants: improvements on one layout did not carry across layouts. Kept the integration diagnostic and exact previous-vertex storage. |
| `41faa1b`, 19 Oct | Choose a complete route first; sample heights using that route and Sobol points | Major practical improvement on small scenes. Retained signed pressure and exact probability weights. Its original budget was **per route**, so cost grew with the number of routes. Small-scene maps are below. |
| `1d847f9`, 20 Oct | More mixed mirror/edge paths, peak residual integration, roulette, resonance sums and local wave scattering | No universal speed/accuracy win for the route modifications. Resonance sums passed restricted axial controls; scattering operators passed isolated-cylinder controls. Neither became a general scene solver. |
| `ce17057`, 20 Oct | Analytic/grid SDF marching versus segments, BVH and analytic boxes | SDF was slower and failed some grazing/intersection checks. Kept exact intersections; specialized boxes were promising. This measured geometry queries, not acoustic pressure. |
| `bd33f91`, `7d42a08`, 20 Oct | Larger planar/finite-3D scenes and individual-path diagnosis | Larger scenes revealed enormous multi-edge outliers. Double-precision recomputation agreed with a captured extreme contribution: the problem was not merely float precision. Kept a finite-edge CDF numerical fix, without clipping amplitude. |
| `db295a7`, 21 Oct | Fixed total-budget complete-route Monte Carlo on CPU | Replaced the per-route sample budget with a per-receiver budget. Every route is represented when the budget permits; otherwise route selection is probability-corrected. CPU preparation/integration remained expensive. |
| `8ade051`, `5f25246`, 21 Oct | Give the old GPU walk many more samples | Failed as a remedy. The 9- and 25-box fields still contained enormous outliers at 65,536 rays/bin. These depth-8 runs are not compute-matched to the depth-2 complete-route prototype. |
| `e100724`, 21 Oct | Port complete-route height integration to GPU | Closely matched the CPU estimator and greatly reduced integration time. CPU route preparation and mirror/mixed fields remained. |
| `8e3391d`, 21 Oct | Tune height-proposal widths; stabilize inverse-CDF sampling | Quarter-width guidance improved some full-map results, but held-out gains were mixed. Kept width 1 as default; quarter width remains an experiment. Increasing to 1,048,576 samples improved these maps but did not remove model error. |
| `cb0fef8`, 21 Oct | Build visibility/counts and generate complete routes on GPU | Removed CPU route lists while retaining complete-route height guidance. Full-map equivalence passes; the measured diffraction stage is much faster. Restricted to one/two-edge rigid planar paths for this experiment. |
| Latest experiment, 21 Oct | Reuse route descriptors; compute mirror/mixed fields on GPU; sweep total sample budget | Removes the CPU common-field bottleneck. More samples improve larger maps, but errors plateau and dense-scene fidelity remains poor. Detailed comparisons above. |

The chronology is a record of accepted and rejected experiments, not a claim that each step improved the renderer. In particular, source/listener/bidirectional sampling was tested on a restricted integral before independent wave references existed. Its earlier “error” cannot be read as error against real-world acoustics. To inspect historical code without switching your Unity checkout, use `git show REVISION:path/to/file`.

**Reading the 2D maps**

Each coloured cell is an actually evaluated receiver position. Blue stars are sources; dark grey boxes are obstacles; light grey cells were not sampled. Every map below uses the same absolute pressure scale, from −40 to +5 dB. **Turquoise means above the colour scale**, not a good result. There is no interpolation, amplitude fitting, per-image normalization or smoothing.

These are XY maps of a 3D point source around infinitely extruded rigid walls—the existing planar acoustic convention. They are not line-source 2D acoustics or finite-height 3D buildings. All shown fields use 250 Hz and seed 17. The reference independently solves the wave equation. For the larger scenes, the last mesh refinement changed whole-field pressure by about 0.13–0.66%, and shadow pressure by at most about 1.21%; that is observed convergence, not a certified absolute error bound.

“Within ±3 dB” means the predicted pressure amplitude is between approximately 0.708 and 1.413 times the reference at that cell. It ignores phase. Complex relative L2 error also compares phase: take the root-sum-square pressure difference across cells and divide by the reference's root-sum-square pressure. Neither metric is a perceptual listening score.

**Earlier small scenes: wall, staggered walls, slit and corridor**

![Earlier four small-scene wave/reference maps](benchmarks/acoustic-history-small-2026-10-21.png)

This is the 19 October CPU prototype: 256 samples **per route**, two edge orders, the same supported mirror/mixed paths, and seed 17. Whole-map complex errors are 2.01%, 9.14%, 4.20% and 13.66%, respectively. These are single-seed map scores; the older report's four-seed means and 31-position walking scores are different measurements.

**Larger scenes: the progression**

Read each figure left-to-right, then top-to-bottom. The bottom-left CPU complete-route field was subsequently reproduced by the untuned GPU port. The middle-bottom panel includes both a larger budget and quarter-width tuning; the bottom-right panel tests route-generation speed at the original 65,536-sample budget and width 1. Its similarity to the bottom-left panel is intentional.

![One-box experiment history](benchmarks/acoustic-history-1-2026-10-21.png)

![Nine-box experiment history](benchmarks/acoustic-history-9-2026-10-21.png)

![Twenty-five-box experiment history](benchmarks/acoustic-history-25-2026-10-21.png)

| Complete-route field | 1 box: complex error | 9 boxes: complex error | 25 boxes: complex error |
|---|---:|---:|---:|
| CPU, original width, 65,536 samples | 1.85% | 32.02% | 83.23% |
| GPU integration, original width, 65,536 | 1.85% | 32.02% | 83.23% |
| GPU integration, quarter width, 65,536 | 1.85% | 27.90% | 78.76% |
| GPU integration, quarter width, 1,048,576 | 1.83% | 22.44% | 62.04% |
| **GPU-generated routes, original width, 65,536** | **1.85%** | **32.02%** | **83.23%** |

The old GPU walk uses eight interactions; the complete-route maps use two and include additional cached mirror/mixed classes. Historical comparisons therefore change more than sampling. The new route-generation comparison is narrower: same two-edge estimator, routes, height samples, frequency, seed, delay horizon and cached common field. It is the controlled speed experiment.

**Earlier experiment at cb0fef8: GPU route generation**

Previously:

```text
CPU, scene/source setup:
    find visible edge neighbours and source-visible edges
CPU, each receiver:
    count and choose complete routes
    calculate every chosen route's angular terms and segment lengths
    upload route records and integration steps
GPU:
    sample heights on each route; evaluate signed pressure; reduce
```

Now:

```text
CPU:
    upload box bounds, source, reusable sample coordinates
GPU, scene/source setup:
    build edge-to-edge and source-to-edge visibility
GPU, each receiver:
    find listener-visible edges
    count valid one-edge and two-edge routes; build prefix counts
    launch the configured total sample budget:
        select a complete route using those counts
        calculate its segment lengths and wedge terms
        sample heights conditioned on that complete route
        evaluate signed pressure with the full probability weight
    reduce pressure
```

When route count R is at most sample budget N, each route receives `floor(N/R)` samples and the remainder is distributed once, exactly as in the CPU control. When R exceeds N, each sample selects a route and carries its inverse-probability weight. Routes outside the delay horizon contribute zero; they are not retried or renormalized away. The same-face factor, signed wedge kernel, distance spreading, physical phase and mixture PDFs remain. We reuse the existing integration shader through the experiment's build script rather than writing another DSP/transport stack.

The experiment has a fixed **height-sample budget**, not constant total scene cost. With E edges and W boxes, its simple graph build costs O(E²W) visibility work, once per scene/source setup; receiver preparation costs O(EW + E²); route selection uses prefix searches. Graph storage is O(E²). Each selected route has one or two edges. This is suitable for the measured 4/36/100-edge scenes; it is not a proof of scalability to an entire city. General-depth counting, BVH-backed graph construction and efficient updates for moving geometry are not implemented here.

**Measured cost, including CPU preparation**

Fresh paired measurements on the complete 12 × 11 m maps: 65,536 height samples per receiver, width 1, depth 2, 250 Hz, 0.4 s support, seed 17. The two shader variants ran in the same silent Unity session. These are one-pass map measurements, not repeated statistical performance guarantees.

| Scene | Receiver cells | Previous diffraction stage | GPU-generated diffraction stage | Measured ratio |
|---|---:|---:|---:|---:|
| 1 box | 1,014 | 0.548 s | 0.259 s | 2.1× |
| 9 boxes | 960 | 8.413 s | 0.267 s | 31.5× |
| 25 boxes | 880 | 14.963 s | 0.260 s | 57.5× |

The previous totals include freshly measured CPU route preparation (0.147/8.107/14.571 s), route-input reading, upload submission, GPU integration and blocking result/flag readback. The new totals include input packing, receiver input/submission, warmed scene-graph construction, and receiver preparation/route generation/integration/reduction/readback. One-time buffer allocation, initial random/geometry uploads, first-use compilation, optional diagnostic exports and report generation are excluded. Receiver input/submission totals 0.21–0.25 ms per whole map and is included. These are measured stage totals rather than full-process wall times.

Warm median per-receiver solve/readback was **0.229 → 0.240 ms**, **0.232 → 0.264 ms**, and **0.238 → 0.281 ms**, respectively. The GPU does more work, while the much larger CPU preparation cost disappears. Map totals include outliers; they are not the median multiplied by cell count. Warm scene-graph dispatch/readback took 0.22/0.45/0.17 ms in these runs, near the small-job submission/synchronization floor; that ordering is not a geometry scaling result. No hardware GPU timestamps were collected.

Input files shrink from 1.71/38.67/59.18 MB to about **1.32 MB** each, most of which is the reusable Sobol sequence. The new GPU buffers occupy about 1.32/1.33/1.40 MB. The per-receiver upload is just XY coordinates instead of all route descriptors. The old arrays could have been cached for static receiver positions; this experiment matters when new receiver positions need evaluating.

**These are not full-renderer speedups.** Direct/mirror/mixed fields are still the identical cached CPU arrays. Their earlier whole-map calculation took approximately **0.36/30.69/213.96 seconds**. They are added to every complete-route plot but excluded from both timing columns. Broadband bins, temporal synthesis, HRTFs, moving boundaries and networking are also outside this experiment. The scene graph/counts can be reused across frequencies, but the current prototype still recalculates route descriptors inside each integration dispatch.

**Verification and what remains**

The final runs passed 25 Unity test executions across controls, maps and the added single-edge subset check, including repeated 115-value angle checks. Python checks independently compare GPU visibility, route counts, aggregate pressure and selected per-sample contributions against the CPU implementation. Controls include empty geometry, no physical delay support, first/second order, and more routes than samples for both orders. Full-map route counts match and pure-edge pressure differs from the prepared-route GPU control by **0.00020%, 0.00082% and 0.00071%** in relative norm. All 2,854 map positions are finite and pass the declared aggregate equivalence tolerances.

A small-budget control caught evaluation beyond the intended sample count when the compiler lowered a conditional expression. An explicit guarded sample call fixes it in the experimental shader. Early shader compilation failures and the failed small-budget run remain in the archive; they are not counted as passing validation. The fixture historically named `small-dc` is evaluated at **250 Hz** in this experiment and is not a new DC control.

Keep this as an offline experiment for now. It has not been qualified at near-degenerate float-precision shadow boundaries, other frequencies, arbitrary materials/meshes, higher orders or finite 3D. The new speed result does not repair the large multi-obstacle field error or demonstrate artifact-free moving audio. The next useful work is to remove the remaining CPU mirror/mixed bottleneck and test more complete path coverage against the independent wave maps, while preserving this estimator as a reproducible control. DLSS training still needs a qualified broadband/moving-scene teacher.

**Evidence and reproduction**

- [New timings, quality and hashes](benchmarks/gpu-route-generation-2026-10-21.json), [source, raw inputs/results, plots and controls](benchmarks/gpu-route-generation-2026-10-21.zip).
- Existing detailed history and rejected trials: [cleanup-results.md](cleanup-results.md).
- Prior ledgers: [bounded CPU routes](benchmarks/route-monte-carlo-2026-10-21.json), [GPU port](benchmarks/gpu-route-integration-2026-10-21.json), [GPU sample sweep](benchmarks/gpu-full-2d-maps-2026-10-21.json), [proposal tuning](benchmarks/gpu-sampling-tuning-2026-10-21.json), [transport candidates](benchmarks/transport-candidates-2026-10-20.json), [SDF/BVH](benchmarks/sdf-scaling-2026-10-20.json).

The archive stores paths relative to the project root. Its `README.md` gives the opt-in commands and identifies the existing scientific environment and isolated Unity runner. Source snapshots preserve the current experiment without adding experimental shader variants to the live project's Assets folder. Do not switch the active Unity checkout to inspect older code.
