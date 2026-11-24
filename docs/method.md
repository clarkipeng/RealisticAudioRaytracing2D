# The acoustic path tracer

This document describes how the tracer works today, from the scene to the ears.
It is written for a reader who knows geometric acoustics and Monte Carlo rendering, and it keeps the derivations.
The history of each decision, the experiments that were tried and rejected, and every measurement in full are in the lab notebook, [docs/benchmarks/path-denoise-2026-11-12/README.md](benchmarks/path-denoise-2026-11-12/README.md) (cited below as "notebook part N").
The earlier [path tracer benchmark](benchmarks/path-tracer-2026-11-12/README.md) introduced the method.

## Contents

1. [What is computed](#1-what-is-computed)
2. [The scene](#2-the-scene)
3. [The walk](#3-the-walk)
4. [Persistent rays: sound in flight](#4-persistent-rays-sound-in-flight)
5. [Population control](#5-population-control)
6. [Scoring at the listener: beams, connections and edges](#6-scoring-at-the-listener-beams-connections-and-edges)
7. [Discrete echoes, Doppler and ITD](#7-discrete-echoes-doppler-and-itd)
8. [The denoiser](#8-the-denoiser)
9. [The GPU reverb](#9-the-gpu-reverb)
10. [The auto-budget](#10-the-auto-budget)
11. [Validation](#11-validation)
12. [Cost](#12-cost)
13. [Known limitations](#13-known-limitations)
14. [Where the code is](#14-where-the-code-is)

## 1. What is computed

For every source–listener pair and every audio update (1,024 samples, 21.3 ms at 48 kHz), the tracer estimates the energy arriving at the listener as a histogram over three axes:

- **delay**, in bins of 0.25 ms up to T_f = 16 ms (64 fine bins), then with constant relative width: each later bin is 1/64 of its start time wide;
- **octave band**, eight bands centred on 125 Hz to 16 kHz, all carried by every ray as one vector;
- **direction of arrival in the head frame**, as six channels (±x, ±y, ±z): an arrival from unit direction u adds u_a² of its energy to the channel of axis a on the side of the sign of u_a, so the six shares sum to one and a moving arrival never jumps between bins.

The direct sound is not in the histogram.
It is rendered exactly, as a delayed path with its own retarded-time delay, Doppler and full HRTF.
Everything else, reflected, scattered, transmitted, refracted and diffracted sound, is the histogram.

Every quantity in the histogram is an unbiased Monte Carlo estimate of the energy transport defined by the scene's material fractions (the room acoustic rendering equation with measured absorption, scattering and transmission).
Where an estimator is only consistent, or where a physical approximation enters, this document says so.

Alongside the energies the histogram keeps the number of contributions per bin and, while anything moves, a rate channel: per bin the energy-weighted first two moments of the contributions' delay rates and their energy-weighted mean delay within the bin (section 7).

## 2. The scene

**Geometry.**
The scene is triangles grouped into objects, each with a pose per update.
Each object has an immutable bounding volume hierarchy built with a binned surface-area heuristic (12 bins on each of three axes, leaves of up to eight triangles, median splits below depth 18 so the traversal stack of 32 entries suffices).
Rays are brought into an object's frame once and test boxes with the inverse direction; the nearer child is visited first and the farther one is culled when popped if a closer hit has been found since.
Triangles are always tested in world space and ties go to the lower primitive index, so the hits do not depend on the tree.
A top-level tree over the objects' boxes is rebuilt, not refitted, whenever an object moves: a refitted tree keeps its first grouping, and an object that travels far (a train on its loop) then stretches every box above it across its path.

The scene's queries descend a tree of their own (notebook parts 48 and 49): every object that has not moved since the first update merged into one world-space tree of its triangles, built by the same heuristic, under a small instance tree of the objects that have moved.
Its leaves store their triangles in the node array, in world space (first vertex, both edges, the primitive's index), so a triangle test loads one node and rotates nothing.
An object that moves leaves the merged set for good; until the merged tree without it is rebuilt on a worker thread, queries descend the instance tree, so the hits never depend on which tree served them.
Point location (the medium at a point) keeps the instance tree, whose objects it looks up by their own trees.
In the Church the merged tree halves the node visits per walk segment's query against separate trees whose boxes all span the nave (77 → 55) and stored triangles double the traversal's rate.

Hardware ray tracing is not available to these kernels on Metal in Unity 6.4.
Measured outside Unity on the same triangles and rays, Metal's intersector answers 0.6–1.1 G queries per second on the M5, 5–13 times this software traversal (notebook part 46).

**Materials.**
Each material has per octave band an absorption coefficient α, a scattering coefficient s (the share of reflected energy that is scattered, ISO 17497-1), and a transmission τ, plus an acoustic index (the speed of sound in air over that in the material: 0.064 for stone, 0.095 for wood).
Thin leaves modelled as closed boxes are partitions: their transmission is applied once, at index one.
Solids with an index below one are media: a ray that enters one refracts by Snell's law, and beyond the critical angle (3.6° from the normal for stone) it is totally reflected.
The Church and train station materials are sourced from published tables; [the station's sources and derivations](benchmarks/path-denoise-2026-11-12/README.md#43-published-acoustic-materials-for-the-train-station) are in notebook part 43.

**Edges.**
A preprocessing pass lists the diffracting edges (wedges with their exterior angle); the Church has 14,075.

## 3. The walk

A ray starts at the source with the source's power split over the rays launched in that update.
It is traced segment by segment, with air absorption along every leg.
At every surface the material's energy fractions per band decide what happens:

- **absorbed**, share α;
- **transmitted**, share (1 − α) τ: through a partition straight on (the assembly's transmission applied once), into a medium with refraction;
- **scattered**, share (1 − α)(1 − τ) s, by Lambert's law;
- **specularly reflected**, share (1 − α)(1 − τ)(1 − s).

The continuation is drawn with the band-mean fractions, and each band's weight is multiplied by its own fraction over that probability.
A small transmitted share (a wall, a door, a window) is not drawn: it is split off into a one-slot weighted reservoir per ray, and the kept branch continues as its own ray, so the energy that leaks through stone (τ of 10⁻⁴ to 10⁻⁶) is carried with bounded weight rather than sent rarely with all of a ray's energy.
Russian roulette terminates rays with weight compensation.

Two rules come from the limits of the ray model rather than from tuning:

- A ray that bounces between two facing parallel faces closer than half the shortest band's wavelength (1.07 cm) ends: below that gap no transverse mode exists in any band, so the field in it is a plane wave along the slit, which rays bouncing across it do not represent.
  In practice such slits are defects of a mesh.
- A ray in air that meets a closed solid's face from inside has come through a leak in the mesh and ends.

A segment that leaves a surface starts RayEpsilon out along that surface's outgoing normal and is traced from zero distance, so it cannot skip a second surface within the tolerance; a partition crossing ends only once every other object entered on the way has been left too.

**The long-chain lottery.**
An update's GPU time is set by its longest serial chains of iterations (a dispatch lasts as long as its slowest thread), not by its total work.
Past 16 loop iterations of one ray within one update, every further reflection survives with probability 0.75 and the survivor's weight is divided by 0.75.
This is unbiased; at equal time it buys about 27% more rays, which outweighs its added variance (notebook part 28).

## 4. Persistent rays: sound in flight

The rays are the sound in flight.
Their state (160 bytes per ray: position and path time, direction, cohort age, flux and attenuation per band, the latest event's plane, the chain's unfolded length and origin time, the delay-rate accumulator, flags) stays in GPU buffers across updates.

Every update:

1. advances each live ray by one update period Δt of path time; a ray stops in mid-segment and resumes there at the next update;
2. launches the update's new rays (its cohort) from the source's position at that update;
3. scores everything the rays meet during the update against the listener's position at that update (section 6);
4. applies population control (section 5) and writes the survivors compactly to the other half of a double-buffered pool.

**Retarded time.**
A contribution made now is heard after its own travel time from the scoring event to the listener.
It therefore goes to the histogram of the update in which it is heard: each pair keeps a ring of future histograms, and after each update the ring's current slot becomes that update's histogram.
The ring needs to reach as far as the longest connection's travel time: RingSlots = ⌈3 D n / (c Δt)⌉ + 1, with D the diagonal of the scene's bounding box, n the largest acoustic index (no medium is slower than air, so n = 1) and the factor 3 for the longest connection (a double-edge connection is three lines of sight).
A contribution that would land beyond the ring is counted, and the checks assert there are none.

New rays advance half a period in their first update, so update n covers, for every cohort, the path times heard within half a period of the time of update n's state; the response is centred on the update's time rather than lagging it by half a period.

So what the listener hears at time t is the sound that left the source at t − T along each path, through the scene as it was where and when each event happened (to the resolution of one update).
Through a door that opens, nothing is heard until the first sound through the opening can arrive; a source passing at speed is heard where it was, not where it is.
The one remaining approximation is that a connection (a straight line to the listener, section 6) is checked against the scene at the time it is made, not as the sound passes along it, which matters only for geometry moving during a connection's flight of a few tens of milliseconds.

A full static solve is the same loop with an infinite period.

With several source–listener pairs, one dispatch walks all of a frame's pairs, one thread row per pair, each row with its own pools, ring and random sequence, so the outputs are those of walking the pairs one by one, bit for bit; a pair's walk alone left the GPU mostly idle while its longest rays finished (notebook part 56).
Cut into updates with the same random numbers, the persistent estimate equals the full solve to rounding (section 11), which is the strongest check that stopping, resuming, the ring and the compaction lose and duplicate nothing.

A transmitted branch is queued as a ray of its own when its parent stops in an update, so it starts at most one update late.
A scene cut empties the pools and the ring.

## 5. Population control

The pool holds a fixed number N of rays in flight (23,000 per source in the Church, 4,000 in the station).
Rays are grouped by cohort, the rays launched in one update; cohort k has age a_k = (k − ½) Δt.
After each update's walk, every cohort is brought to its share of N:

target(a) = n̄ S(a), S(a) = min(1, T_f / a),

with T_f = 16 ms, the end of the fine bins, and n̄ = N / (1 + Σ_k S(a_k)) new rays launched per update (17% of N in the Church, with 23 ms updates and a 6 s horizon).

**Why this profile.**
A histogram bin at delay τ has width b(τ): 0.25 ms in the fine region and τ/64 after it.
It receives contributions from the rays of age τ at a rate proportional to their number n(τ), so with a constant relative variance per contribution its relative variance is proportional to 1 / (n(τ) b(τ)).
The cost of an update is the number of rays in flight, N = ∫ n(τ) dτ / Δt.
Minimising the sum of the bins' relative variances, ∫ dτ / (n(τ) b(τ)²) (there are 1/b bins per unit delay), at fixed N gives by Cauchy–Schwarz

n(τ) ∝ 1 / b(τ), i.e. S(τ) = min(1, T_f / τ):

every bin then receives as many contributions as a fine bin.
The new rays are the age-zero end of the same profile, so their share follows; neither a survival time nor the share of new rays is chosen by hand.
If the relative variance per contribution varied with delay as κ(τ), the same argument would give S ∝ √κ / b; κ is not measured per update.

**How.**
A cohort of A rays is brought to its target by the ratio ρ = n̄ S(a) / A: each ray is split into ⌊ρ⌋ or ⌊ρ⌋ + 1 copies (the extra copy with probability ρ − ⌊ρ⌋), or for ρ < 1 kept with probability ρ, and every copy's weight is divided by ρ.
At most 16 copies are made per ray (a cost guard; any limit keeps the weights exact).

**Unbiasedness.**
Any rule that gives ray i an expected number of copies e_i > 0 fixed by the past, and divides its weight by e_i, is unbiased: E[copies_i · w_i / e_i] = w_i.
Here ρ depends only on the cohort's count after the walk, which is the past; each copy then continues with its own random sequence.
The histogram is divided by a fixed count when it is read, and each cohort's new rays carry Rays / n of a ray's share, n being the new-ray count fixed on the CPU before the cohort launches; so N may change from update to update (section 10) without bias.

**Coherence.**
Population control writes the kept copies in sorted order, by a counting sort in three passes (count, a single-group scan, scatter).
The key is the direction's octant (3 bits), then the Morton code of the position's cell (4 bits per axis).
The cells are the quantiles of the rays' own positions along each axis from the previous update (16 slabs of equal count per axis), because a box over the scene is useless when the scene includes a meadow 1.2 km across.
Only the order changes, so the estimates are the same; the next update's SIMD groups then trace neighbouring rays in similar directions through the same tree nodes, which takes about 0.6 ms off the Church's tracer (notebook part 31).

Rejected alternatives, measured: systematic resampling and weight-proportional selection of the copies (notebook part 32).
Resampling is a minor part of an update's noise, which is set by where the rays go; and in the Church a weight-proportional rule would thin hardest the transmitted branches that carry all the sound reaching a listener outside.

## 6. Scoring at the listener: beams, connections and edges

Three estimators score the walk at the listener, and each path is counted by exactly one of them.

### 6.1 Beams with Fresnel width

Geometric acoustics cannot resolve the field more finely than the width of a sound beam, so after a ray's first specular reflection every segment of it scores at the listener with a normalised two-dimensional Gaussian kernel of its distance ρ from the listener:

K(ρ) = 2 / (π w²) · exp(−2 ρ² / w²), with ∫ K dA = 1.

ρ is measured from the segment's line, extended, to the listener; in the unfolded space of the specular chain that is the distance from the straight ray to the listener's image.
Each image is scored once, by the segment that belongs to it, whether or not the closest approach falls inside the segment.
The deposit's time is the image's: the chain's origin time plus √(L_c² + ρ²) / c, with L_c the unfolded length to the closest approach.

**The width.**
A Gaussian beam of waist w₀ has radius w(z) = w₀ √(1 + (z / z_R)²), with z_R = π w₀² / λ.
The narrowest beam that spans a path of length L is the one whose largest radius over the path is smallest: its waist sits at mid-path, and minimising w(L/2)² = w₀² + (λL / 2πw₀)² over w₀ gives w₀² = λL / 2π and an end radius

w² = λ L / π,

with λ the band-centre wavelength and L the unfolded path length from the chain's origin (the source, a scattering point or an edge) to the listener's image.
For comparison the first Fresnel zone at mid-path has radius √(λL/4) = 0.89 √(λL/π), so the kernel is a Fresnel-zone-wide beam; no constant is tuned (K's 1/e² radius is w).
At 1 kHz, w is 0.8 m after 6 m and 1.5 m after 20 m; at 125 Hz, 2.3 m and 4.2 m.

**Two geometric rules.**
In the unfolded space a room is a lattice of copies, one per image.
A ray that passes an image closer than the listener's clearance R (its distance to the nearest surface) has necessarily crossed that image's copy, but a ray farther out may belong to a neighbouring image and is never scored for this one; a kernel wider than R would therefore lose energy.
The kernel is truncated at R and renormalised by 1 / (1 − exp(−2R²/w²)); R is measured once per update by 256 rays from the listener.
A kernel also reaches through walls, so each scored segment needs the listener visible from the segment's point nearest the listener, kept 4 RayEpsilon off the walls at both ends of the segment.

**Why beams.**
The beams replaced an exact connection of each specular chain to its image (detection regions, trials and an image retrace).
On the wave ground truth they are more accurate in level, early decay, T30 and the energy–time curve, overall and in shadow, and cheaper in rooms (notebook part 8).
In a controlled porch they give 0.74 of the exact connection's specular beams behind a small opening, at a Fresnel number a² / λd of about 0.1, where the wave field is indeed smooth across the geometric beams.
Outside the Church they closed a 4.6 dB gap between the exact connection and the energy method: the exact connection was unbiased but so heavy-tailed that a single solve almost never scored the outside specular chains.

### 6.2 Next-event connections

The source and every scattering point connect straight to the listener (next-event estimation), with the scattering lobe, distance and air absorption.
A connection blocked by a surface is traced through the walls only with probability q, the largest energy transmission of that surface over the bands, and weighted by 1/q; since the transmitted energy is at most q, the weight stays bounded, and through stone (q ≈ 10⁻⁶) the crossing loop almost never runs.
An unobstructed connection skips the crossing loop altogether.
The source's direct sound is the exact path of section 1, not a connection.

The beam kernel does not score the first segment after the source, a scattering point or an edge: these connections already count it.

### 6.3 Edge connections and edge rays

Diffraction uses the uniform theory of diffraction (UTD) as an energy kernel:

- **single-edge connections**: from the source or a scattering point, through one edge to the listener;
- **mirror-and-edge (RD) connections**: one specular reflection then an edge, the mirror drawn in proportion to its area;
- **double-edge (DD) connections** over two parallel edges;
- **edge rays**: with probability ½ a ray leaving the source is an edge ray, its edge and point importance-sampled by 1/m², its azimuth uniform on Keller's cone, carrying the UTD energy flux; it then walks like any ray.

The edge of a connection is drawn from a distribution over all edges, weighted by the edge's angle and by the integral of 1/m² along it from the source and from the listener, with a floor for ends that cannot see it.
Every edge keeps a positive weight, and a sample's probability is read from the same distribution it was drawn from, so any positive weights are unbiased; the distribution only steers the variance.
Each update refreshes 2,048 of the weights, and the live scenes rebuild the distribution every fourth update, which is unbiased for the same reason and within the noise of every update (notebook part 30).
Scattering points draw their edge half the time from this listener-aware distribution and half the time locally or uniformly, weighted by the mixture's probability.

**UTD and the beam width.**
UTD's two reflection-boundary terms (in φ + φ′) carry the weight 1 − exp(−2 X s′/s), X = kLa their transition argument and s, s′ the source- and listener-side distances.
This is one minus the beam kernel at the listener's offset x from the boundary: with w² = λ(s + s′)/π and, near the boundary, X ≈ π s x² / (λ s′(s + s′)), the kernel's exponent 2x²/w² equals 2X s′/s.
So where the reflected beam already reaches past the boundary, UTD's transition is not added a second time, and far from it UTD is unchanged; the incident-boundary terms are untouched, because the direct sound is exact and not a beam.

### 6.4 The ring

Every estimator deposits at the time the contribution is heard: an update's connections and beam deposits go to the ring slot of the update in which their sound arrives (section 4).
Without the ring, a connection through a door that has just opened would be heard before the sound could get there.

## 7. Discrete echoes, Doppler and ITD

### 7.1 Delay rates

A path's travel time T changes with the time t at which it is heard.
To first order,

dT/dt = −v_source · s_emit + v_listener · s_arrive + Σ_P v_P · (s_in − s_out),

with s a leg's slowness vector (its direction over its medium's speed) and v_P the velocity of an event's point on its surface.
At a specular reflection the tangential parts of s_in − s_out cancel by the reflection law, so only the normal velocity counts, 2 (v · n) cos θ; at a refraction Snell's law cancels them the same way; a scattering point or an edge point moves with its surface, so for scattering and diffraction the whole velocity counts.

Exactly, each leg from a point moving at v_a to one moving at v_b scales the frequency by (1 − v_b · s) / (1 − v_a · s).
Each ray therefore carries ρ = −ln of the product of these factors along its path, and a contribution's delay rate is r = 1 − exp(−ρ).
The product form is exact for constant velocities; the first-order sum is 8% off in the shift at 30 m/s, because the source's term is per emission time and the listener's per reception time.

Per bin, the rate channel holds Σ E r, Σ E r² and Σ E (τ − bin start), with E the contribution's energy summed over bands.
A persistent update gathers what is heard within half a period of its time, over which a moving echo's delay drifts, so the delay is referred to the update's own time: τ − r (heard − update time).

The rate code is compiled only into separate walk kernels, dispatched while anything moves and, with persistent rays, for a horizon afterwards (the rays in flight still carry the rates of what moved).
A static scene runs a walk without it.
The separation is not cosmetic: the walk is a long loop, and Metal's compiler changes its static outputs in their last bits when the loop grows; with the rate code out of the static kernels, static outputs are bit-identical with and without the feature.

### 7.2 Which bins are echoes

The early part of a response consists of discrete echoes; the late part is a dense field.
A bin is rendered as one echo while its contributions share one arrival, that is one slowness vector at the listener.
The synthesis tests this from the data of each update, in time order:

- at least two contributions (one cannot show that they agree);
- one delay rate: the spread no larger than the mean, Var(r) ≤ E[r]² (the rates are all zero while nothing moves, which passes);
- one direction, by the same rule on the arrival direction: with ū the energy-weighted mean of the arrival directions, the spread E|u − ū|² = 1 − |ū|² no larger than |ū|², i.e. |ū|² ≥ ½.

The early part ends at the first bin that holds energy and fails a test: it pools arrivals of different shifts or directions, which one impulse cannot represent, and it and every later bin stay in the convolution of section 9.
So where "early" ends is read from each update's data, not set by a constant.

The mean direction comes from the six direction channels.
For one arrival the tracer splits its energy by squared components, so u_a = √(E_{+a} / E) − √(E_{−a} / E) exactly, and |u| = 1.
For several arrivals the same expression is an energy-weighted mean: two arrivals on opposite sides of an axis shorten u (two equal ones cancel), two arrivals symmetric about an axis give exactly cos θ, the true mean resultant; arrivals within one octant cannot be told from one arrival at their mean direction, which is also what the ear hears (summing localisation).

### 7.3 How an echo is rendered

An echo leaves the convolution and becomes one impulse at its contributions' mean delay τ (from the rate channel's delay moment), with the bin's ear energies per band.
When nothing moved, the walk has no rate channel, and the echo sits at the bin's centre: the centroid of the uniform spread that the convolution would give the bin, and the position that minimises the worst-case error (half a bin: 0.125 ms in the fine bins, 0.4 ms at 50 ms).

Its delay slides, τ + r (t − t_ref), read from a ring of the band-filtered source signal with the direct path's interpolation filter (a windowed sinc, its cutoff lowered when the read runs faster than real time); t_ref is the end of the bank's crossfade in, where the update's state is.
The time-varying delay gives the echo its own pitch shift; with the tracer consistent from update to update, the old and new banks' echoes coincide through each crossfade.
An impulse rather than the convolution's noise slots, because a slot's random sequence restarts its phase whenever the echo moves to another bin (the first version measured 260.75 Hz against an expected 271.17 Hz for that reason).

### 7.4 Interaural time difference

An echo is one arrival, so it has one direction and its ears hear it at different times.
The ITD is the HRTF set's own: per measured direction of the KEMAR set, the lag of the peak of the cross-correlation of its left and right impulse responses (the interaural cross-correlation estimator), interpolated band-limited (a Lanczos-windowed sinc of radius 8) and maximised on a 1/64-sample grid.
It is laid on a grid of head-frame directions every 2° of azimuth and elevation, each point the blend of measured directions that the direct path itself uses there, so an echo from a direction gets the ITD the direct path has from it.
The echo's delay is split symmetrically: the left ear's is τ + ITD/2, the right's τ − ITD/2.
(Woodworth's spherical head, the usual alternative, needs a head radius and gives 656 µs at 90° for 8.75 cm against the set's 735 µs.)

This holds whether or not anything moves.
The dense late field gets no ITD of its own: its two ears' noise is mixed to the HRTF set's diffuse-field interaural coherence, which already holds the interaural delays of every direction, and a per-direction delay there would count them twice.

## 8. The denoiser

Each update's histogram is noisy: in the Church at 11,000 rays in flight one update's window error against a converged target is about 10 dB, and consecutive updates share their rays in flight.
A small network running on the GPU beside the tracer cleans it.

**Inputs.**
The last four updates' histograms and contribution counts, plus two exponential moving averages of the histogram (decays 0.7 and 0.9, bias-corrected), all scaled by the four-update mean's total; log energies of 6 directions × 8 bands per bin.
A persistent update's bin at delay t holds the cohort emitted t earlier, so the same delay in consecutive updates comes from different cohorts, and the same rays reappear one update later at t + Δt; a one-dimensional dilated convolution along the delay bins (eight layers, 64 channels, receptive field 66 bins, 144,280 parameters) sees both.

**Output and unbiased band levels.**
The network predicts non-negative weights on the four-update mean and the two averages, plus a non-negative multiple of a smoothed level so that it can fill cells no update reached.
Then each band's energy per time segment (0–50 ms, 50–200 ms, later) is rescaled to a convex combination of the three inputs' totals there, the mixing weights being a softmax of the network's output averaged over the segment.
Each input's totals are unbiased for a static listener, so the combination is too: the network only moves energy within a band and segment, and chooses how much history to trust.

**Training.**
Noise2Noise: the target of each update is the same retarded-time quantity traced at 8 N with independent seeds (2–8 times the rays, at most 2¹⁹), still noisy per bin but unbiased; with an L2 loss the minimiser is the conditional mean of the target, which is the clean answer because the target's noise has zero mean and is independent of the inputs.
The loss is L2 on energies summed into perceptual windows (1 ms up to 16 ms, then about t/8), each divided by an input-only normaliser.
Data: generated rooms of six families (shoeboxes, doorways, corridors, canopies, barriers, furnished rooms), two static listeners and one walking per room, plus the Church's walk; whole families are held out for validation and test.
Inputs are rendered at populations log-uniform on 4,000–23,000 and at each room's 23 ms population, so one model serves the auto-budget's whole range; the eight rotations and reflections about the vertical axis augment the data.

**Runtime.**
A hand-written compute kernel per step of the model (convert the tracer's fixed-point histogram, update the history and averages, features, normaliser, eight convolutions with exact GELU, prediction, band totals, rescaling), recorded into the audio's command buffer between the tracer and the reverb.
It matches the PyTorch model to a worst relative error of 1.75 × 10⁻⁵ and takes 0.64 ms per update at the Church's 447 bins.
With several source–listener pairs, one denoiser serves them all: each of its dispatches runs every pair, one thread group layer per pair, each in its own blocks, so every output is bit for bit what a denoiser per pair gives (notebook part 57).

**What it buys.**
Over the held-out families it beats the best equal-compute temporal filter, a bias-corrected moving average with its decay chosen on validation, by a small margin (window error 1.96 against 2.08 dB; notebook part 25), and in the Church it removes about 4 dB of window error at every population (section 11).
Its weakness is a walking listener in a corridor, where it trusts its history while the level changes with every step; it is told nothing about motion.

## 9. The GPU reverb

**The impulse response from a histogram.**
Time is cut into slots of the fine width (0.25 ms).
Each slot holds two fixed noise-like sequences, one per ear, scaled by the square root of the histogram energy that falls in the slot (a bin spreads its energy uniformly over its span).
They are Gaussian per sample, so a filled late field is Gaussian noise with a normalised echo density near one (the measured cathedrals have 0.79–0.91), while an isolated arrival is a 0.25 ms burst and an empty bin is silent.
The first sequence has area ±1 and the second area zero, both unit energy and orthogonal, so bands whose period is long against a slot see the slot's energy exactly, as from an impulse.

**Exact band energies at any coherence.**
Each band of the source is convolved with its own response, and the bands are summed; the band filters are linear-phase FIRs with amplitude-complementary cos² weights in log frequency between neighbouring band centres, applied with half a partition of look-ahead inside the DSP latency.
Per group of slots and per band, the two sequences are mixed by W = √target · G^(−1/2), with G the Gram matrix of the band-filtered pair (computed exactly from the spectra the convolution already uses) and the target what an impulse train of the same energy and the HRTF set's diffuse-field interaural coherence would give through that band.
So each band's energy is exact for every group, at any interaural coherence, and not only on average.
The ears' level per direction and band comes from diffuse-field equalised HRTF band energies.

**Crossfade.**
Updates crossfade sample by sample over one update between two kernel banks.
Two independent noisy estimates differ, and a linear crossfade between unequal amplitudes loses power (1.2 dB at 512 rays in the Church), so each block of the blend is scaled so that its power is the linear interpolation of the two outputs' powers, measured over the block with their cross term; equal responses keep a gain of one.

**On the GPU.**
Convolution is linear, so the same output comes from one kernel per ear: the sum over bands of each band's slot sequences convolved with its band filter.
That kernel is built on the GPU every update and convolved by uniformly partitioned overlap-save (partitions of 1,024 samples, 283 of them for the Church's 6 s horizon).
A partition is summed in the frequency domain from its impulse groups' fixed spectra when G groups × 8 bands × 2 ears complex multiply-adds per bin cost no more than eight forward band FFTs and their filter products, which gives G ≤ 6; the other partitions take eight band FFTs and one inverse.
The shared FFT is radix 2 with two stages per pass in registers, so a 2,048-point transform has 6 barriers instead of 11, and reads its twiddles from a table computed in double precision.
The echoes of section 7 are added after the inverse transform, from a ring of band-filtered input, before the crossfade's power correction.
The GPU matches the CPU reference to a relative 3.75 × 10⁻⁵ binaural and 6.1 × 10⁻⁵ with a source passing at 20 m/s.

The CPU cannot do this in real time: it spent 432 ms building one Church update's kernels against an update period of 21 ms.

**Room modes below the Schroeder frequency (opt-in, `settings.roomModesBank`).**
Below the Schroeder frequency a closed room's response is its modes, which an energy tracer cannot carry.
A per-scene bank holds the rigid room's eigenmodes on a rectilinear grid, their wavenumbers and their wall integrals per material, so each mode's damping follows the scene's materials at its own frequency.
Per update and pair inside the bank's room, the modal response is synthesised at a decimated rate, crossed over at the Schroeder frequency measured from the bank (amplitude-complementary against the exact direct sound, power-complementary against the histogram, whose band energies are scaled by their share above the crossover), and added to the reverb's kernel (`RoomModes.cs`, `GpuAudio.Modes.cs`).
The bank applies only while both ends are in its air and every object is at its baked pose; elsewhere the reverb is unchanged.
The method, its validation against the wave ground truth and its cost are in [docs/benchmarks/bass-and-diffraction-2026-11-22/README.md](benchmarks/bass-and-diffraction-2026-11-22/README.md).

## 10. The auto-budget

In a game the audio cannot have the whole GPU, so N follows a budget B in milliseconds per update (`-pathBudget MS`), within 4,000–23,000, the denoiser's trained range.

Update times are gathered in blocks of 16 updates at fixed N, leaving out the first update after a change, which still traces the old cohorts.
If B lies outside the block median's 97.9% confidence interval (order statistics 4 and 13 of 16), N becomes N · B / T, with T the block median; otherwise N stays.

The update's time is T = T₀ + cN with fixed costs T₀ ≥ 0 (audio, denoiser, dispatches).
After the step the time is T₀ + cNB/T, so its error is (T₀ / T)(T − B): the same sign as before, and a fraction T₀/T < 1 of it.
N therefore approaches its target from one side, without overshoot, and stops inside the median's noise.
The step assumes all of the time scales with N; because T₀ ≥ 0, that assumption can only make it cautious, so there is no gain to tune.

In the Church it held N within a few percent of its target and followed a GPU slowed midway down to the floor in ten blocks, without overshooting (notebook part 34).
It needs per-stage GPU times, which Unity reports only in development players and the editor, so it is off by default in the release showcase.

## 11. Validation

### 11.1 Against the wave ground truth

Eleven scenes (open canopies and barriers, a doorway, corridors, shoeboxes, a furnished room) solved by an FDTD wave solver serve as ground truth; errors are RMS over receivers and bands.
The path method at 2,048 rays, full solves (which persistent rays equal in expectation), two seeds, current code:

| Group | level dB | EDT % | T30 % | ETC dB |
|---|---:|---:|---:|---:|
| all | 3.37 | 199 | 74 | 13.8 |
| in view | 1.78 | 250 | 63 | 12.1 |
| in shadow | 4.91 | 66 | 88 | 16.1 |
| closed rooms | 1.93 | 19 | 40 | 5.9 |
| static-IR baseline (a game convolution reverb), all | 8.85 | 420 | 38 | 10.1 |

Four seeds give 3.38 dB in level, 4.91 dB in shadow and 1.95 dB in closed rooms.
The static-IR baseline uses one central receiver's ground truth at every receiver, with only the direct sound moved; the path method is 5.5 dB better in level overall and 8.5 dB better in shadow, and worse in T30 and the energy–time curve, which the baseline gets from the ground truth itself.
C50 and C80 are not meaningful on this suite for any ray method: their RMS error runs to thousands of dB where a receiver's early energy is nearly zero.
The [scoreboard](benchmarks/energy-audio-2026-11-09/scoreboard.md) has every group and the methods this one replaced.

### 11.2 Analytic checks

These run in the test tiers (`PathChecks.cs`, `PathPersistentChecks.cs`, `PathDopplerChecks.cs`); values are against the analytic reference.

| Check | Result |
|---|---|
| free field | exact |
| rigid plane: image / total | 1.013 / 1.005 |
| shoebox image lattice | within 1.5% in every 20 ms window |
| half-plane against BTM | +0.30 dB |
| far side of a solid (double diffraction) against BTM, 125 / 250 / 500 / 1,000 Hz | +1.79 / +1.50 / +1.19 / +0.91 dB |
| Eyring decay, T60 | 0.606 s against 0.585 s |
| diffuse-field leak through walls, s = 1 / s = 0.5 | 0.994 / 1.058 |
| diffuse room radiating through a hole | 1.151 (the field near a hole is not perfectly diffuse; an independent energy tracer agreed to 0.1%) |
| a specular porch in front of the hole, against an image reference good to 5% | 1.040 |
| persistent rays against the full solve, same seed, cut into 16 ms updates | sum of differences 1.7–3.8 × 10⁻⁵ of the total per band |
| a door that opens: nothing heard until the first sound can arrive, then the open room's level | 0 before arrival; last ten updates 1.001 |
| source at 100 m/s past a rigid plane: direct and image delays against retarded time | direct 39.2908 ms (retarded 39.2908); image 40.88 against 41.05 ms (frozen: 30.00) |
| population control, N alternating 9,000 / 17,000 every update, against the full solve | window ratios 0.97–1.21, within three standard errors |
| reflection Doppler at 30 m/s: rate channel / rendered tone | frequency ratio within 3.5 × 10⁻⁴; 271.25 Hz against the image's 271.17 Hz |
| echo ITD against the direct path's through the same HRTF blend, −90° to 90° | worst 3.9 µs moving, 2.0 µs static (limit 10 µs) |
| denoiser kernel against the PyTorch model | worst relative error 1.75 × 10⁻⁵ |
| GPU reverb against the CPU reference | 3.75 × 10⁻⁵ binaural, 6.1 × 10⁻⁵ moving |

### 11.3 The denoiser in the Church

Window / band level / decay-curve errors in dB against the 8 N targets, standing and walking, after a horizon of warm-up (the targets' own noise is 4.5–6 dB in window, so every row is inflated alike):

| N | noisy input | denoised |
|---:|---|---|
| 4,000 | 13.03 / 7.11 / 6.68 | 8.09 / 3.69 / 4.89 |
| 11,000 | 10.35 / 5.21 / 4.57 | 6.34 / 2.09 / 2.75 |
| 23,000 | 8.92 / 3.72 / 4.11 | 6.12 / 2.30 / 2.54 |

## 12. Cost

Measured on an Apple M5 (16 GB) under Unity 6.4 on Metal, GPU time per update, the Church at N = 11,000, medians:

| Phase | ms per update |
|---|---:|
| tracer: edge distribution, walk, population control and sort, histogram and ring | 3.0 (4.2 before notebook parts 47–49) |
| audio: path kernels (0.34), late convolution (0.27), direct path, four spectral hops, PCM and readback | 1.9 |
| denoiser | 0.45 (0.73 before notebook part 55) |
| total | about 5.4 |

Each phase includes one fence's round trip (about 0.2 ms).
The denoiser's eight convolutions are most of its time; each thread issues sixteen inputs' loads before their multiply-adds, in the same order, so its output is bit for bit what it was (notebook part 55).
Of the audio's floor, the four spectral hops' FFTs, sums and inverse FFTs take about 0.4 ms (0.1 per hop, the current and the old filter's passes together).
At 100,000 rays in flight the tracer takes 13.0–14.7 ms (21.8 ms before notebook parts 47–49).
Per thread and update the walk makes 3.4 scene queries (walk segments 46% of them, next-event and edge connections 19%, beam visibility 9%, transmissions 18%) and 172 node visits, at a SIMD utilisation of about 30% (notebook part 46's work profile); it realises 12–26 M queries per second against 83–124 M/s for its traversal alone in a kernel that does nothing else.
The histogram and ring writes cost nothing measurable; the audio side and the denoiser do not depend on N.
Memory: 160 bytes per ray in flight, and the ring of future histograms (44 MB in the Church, whose meadow makes the ring 262 slots deep).

In the live showcase (release player, 60 Hz, windowed smoke runs of a scripted walk), the Church at 23,000 rays had no underrun in four runs, frame GPU time 10.6 ms median, the scene's own rendering included (8.9 ms at 11,000; notebook part 51).
With the auto-budget at 6 ms per update for the tracer, the audio and the denoiser together, the Church settles at 12,500–16,000 rays (fixed populations put 6 ms at about 15,000: tracer about 3.7 ms, audio with the denoiser 2.3 ms; notebook parts 51 and 55).
The train station runs two sources at 4,000 rays each.
One update of both sources costs about 11 ms there at 8,000 rays each (tracer 6.5 ms, audio and denoiser 4.2–4.6 ms), which the GPU could sustain, but the live renderer keeps one batch in flight and its readback is quantised to frames: near the passing train a two-update batch takes longer than two frames, delivery falls below the device's rate and the queue drains (notebook part 45).
So 6,000 and 8,000 rays per source both underran in some runs, and the station stays at 4,000.

## 13. Known limitations

- **Diffraction outside the Church is heavy-tailed per update.** A listener who sees no surface inside receives all interior sound diffracted by the porch's edges, and the single-edge connections from interior points whose line to the listener grazes an edge carry the UTD peak rarely and with large weight. The level agrees with the energy method within its noise, but a single update plays close to the median, about 9 dB below the mean. Listener-side sampling was built and did not fix it; diffraction as an event of the walk is the open route.
- **Beams and UTD near reflection boundaries.** The beam's smoothing and UTD's transition still add just beyond a reflection boundary, by 3–4 times on a level 10–20 dB below the lit side, and the lit side reads 25–30% low; UTD's reflection-boundary terms are weighted by the beam (section 6.3), but there is no coherent reference to settle the lit side.
- **Energy, not pressure.** Interference, modes and the coherent sum at a reflection boundary are not modelled; low-frequency room modes come only from an opt-in room-mode bank (section 9), in closed rooms in their baked configuration.
- **UTD on short edges.** UTD assumes edges long against the wavelength; 71% of the Church's edges are shorter than a wavelength at 250 Hz, and the bias against BTM for such edges is not measured.
- **The early decay at realtime budgets.** EDT errors on the wave suite are large for every variant, and the denoiser's decay curves in the Church are worse than its window errors.
- **Connections are checked against the scene when made.** Geometry moving during a connection's flight (tens of milliseconds) is seen as it was when the connection was made.
- **Path guiding did not pay.** Guiding the scattering directions by a per-primitive distribution learned from the rays' own contributions (unbiased, one-sample MIS with Lambert's law) raised the noise inside by 1–2 dB at every population, because the importance it learned per primitive was dominated by rare large contributions (notebook part 50); it is kept as a patch, not in the code.
- **Echo selection is per update.** At realtime budgets the end of the early part moves from update to update with the noise; the crossfade hides the switch.
- **Echoes from static scenes sit at their bin's centre**, up to half a bin (0.4 ms at 50 ms) from their true delay, because the static walk has no delay moment.
- **The denoiser knows nothing about motion**, and loses to a moving average for a listener walking along a corridor.
- **One batch in flight.** The live renderer waits for each batch's readback before dispatching the next, so its throughput is quantised to frames; this, not the GPU's capacity, limits the station to 4,000 rays per source (section 12). Several batches in flight would remove it.
- **The audio's fixed cost per source.** Kernels, convolution and denoiser cost about 2 ms per source per update whatever the population, so a second source costs half the tracer's budget.
- **The first run of a new build stalls** for about 2.5 s when the Doppler walk is first dispatched (Metal compiles its pipeline on first use).
- **Hardware ray tracing did not pay, and is not in the code.** Unity's compute kernels cannot call Metal's intersector, so a native plugin answered the walk's queries between waves of the walk (notebook part 52).
  It was exact, but slower at every population and wave count, also after the walk was rebuilt as a state machine with one query per loop iteration (notebook part 54), which only DXC compiles and which made the megakernel itself slower.
  Both are kept as patches beside the notebook (`hardware-waves.patch`, `state-machine.patch`), not in the code.

## 14. Where the code is

| Part | Files |
|---|---|
| walk, persistent rays, population control, beams, connections, rate channel | `Assets/Shaders/MethodPath.hlsl`, `Assets/Resources/Acoustics/PathMethod.compute` |
| shared walk helpers: edge sampling, UTD, materials, media, Lambert directions | `Assets/Shaders/MethodCommon.hlsl`, `Assets/Shaders/Transport.hlsl` |
| intersection and trees | `Assets/Shaders/Geometry.hlsl`, `Assets/Scripts/Simulation/DirectBvh.cs`, `StaticTraceTree.cs`, `SahBuilder.cs` |
| work profile and traversal ceilings | `MethodTransport.WorkProfile` and `TraceBench`, `MeasurePathLiveBudget` (`profile=1`), `tools/metal_rt_ceiling.swift` |
| direct sound and its spectral synthesis | `PathDirectArrival` and `PathExpandDirect` in `MethodPath.hlsl`, `Assets/Scripts/Simulation/SpectralAudio.cs`, `Doppler.cs`, `GpuAudio.cs` |
| dispatch, pools, ring, Doppler kernels | `Assets/Scripts/Simulation/MethodTransport.cs`, `Assets/Scripts/Simulation/TransportSolver.cs` |
| histogram bins | `Assets/Scripts/Simulation/PathBins.cs` |
| synthesis: CPU reference and GPU | `Assets/Scripts/Simulation/PathReverb.cs`, `Assets/Scripts/Simulation/GpuAudio.Path.cs`, `Assets/Resources/Acoustics/GpuAudio.compute` |
| room modes (opt-in) | `Assets/Scripts/Simulation/RoomModes.cs`, `Assets/Scripts/Simulation/GpuAudio.Modes.cs`, `tools/room_modes.py`, `tools/room_modes_unity.py` |
| denoiser | `Assets/Scripts/Simulation/PathDenoiser.cs`, `Assets/Resources/Acoustics/PathDenoise.compute`, `tools/path_persistent_denoise.py` |
| auto-budget | `Assets/Scripts/Simulation/PathPopulationBudget.cs` |
| live showcase | `Assets/Scripts/Demo/PathLive.cs`, `tools/path_live_church.py` |
| checks | `Assets/Tests/Editor/PathChecks.cs`, `PathPersistentChecks.cs`, `PathDopplerChecks.cs`, `PathGpuAudioChecks.cs`, `PathDenoiseChecks.cs`, `RoomModeChecks.cs` |
| wave ground truth and scoring | `tools/fdtd_wave.py`, `tools/method_compare.py`, `docs/benchmarks/energy-audio-2026-11-09/scoreboard.py` |
