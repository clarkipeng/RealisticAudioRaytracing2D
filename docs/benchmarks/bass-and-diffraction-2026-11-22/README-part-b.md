## Part B. Diffraction as a walk event, with the derived normalisation

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
