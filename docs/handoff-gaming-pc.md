# Handoff: moving to the gaming PC (2026-11-24)

Development moves from the Mac (Apple M5) to the gaming PC (RTX 5070, 32 cores, 22 GB) so heavy runs stop overloading the Mac.
Every number below was measured on the Mac; none has been measured on the PC yet.
The method is in [method.md](method.md); the lab notebook is [benchmarks/path-denoise-2026-11-12/README.md](benchmarks/path-denoise-2026-11-12/README.md) (parts 1–59).

## State

- `main` holds everything.
- `offline-3d-simulation` is a snapshot of early 3D layout work from Sep 29 that `main` has moved past.
- `experiment/{bidirectional,guided,listener}-acoustics` hold the October sampler experiments.

| At about 6 ms of GPU per update | Rays in flight |
|---|---|
| Church | about 17,600 |
| Train station | about 7,000 per source |

At 100,000 rays the Church's tracer takes 14.66 ms per update and the audio about 2 ms, against an update period of 21.3 ms (1,024 samples at 48 kHz).
The memory is not the limit: 100,000 rays of 160 bytes are 16 MB, 32 MB with the second pool.
The time is: the walk runs at about 30% SIMD utilisation, a large kernel at low occupancy (part 51).

## First: the failing smoke run

The showcase rebuilt from `5c204e1` failed its smoke run on the Mac: in the Church the live observation queue filled 6.8 s in, with GPU work pending, and the station never ran.
The Mac was paging (7.8 GB of swap), so memory pressure may be the cause rather than the code.
Run the smoke on the PC first; if it fails there too, suspect part 58 (the spectral hops in batches of four, about 80 MB of new buffers), then part 56 (every pair in one dispatch).
The editor tests for parts 56–58 pass; only the player is untested.

The tools assume macOS (Metal, `.app` builds, `.context/fdtd-teacher/venv/bin/python`, `ioreg` for GPU load) and need porting to Windows and D3D12.

## The road to 100,000 rays

In order; each step must be measured, exact or within float rounding against the wave ground truth, and kept only if faster (TASTE.md).

1. **Measure 100,000 rays on the RTX 5070 as the code stands.**
   It may already fit the budget there.
2. **Inline hardware ray queries.**
   On D3D12 a compute kernel can call the hardware ray query in place (`RayQuery`, DXR 1.1; Unity's `inlineraytracing`), so the megakernel keeps its loop and its state in registers.
   The Metal attempt lost only because its queries had to leave the walk, saving and restoring 608 bytes per ray per query (parts 52, 54); that cost does not exist here.
   The software tree stays as the reference, and a check compares the hits.
3. **Fewer registers per ray, for occupancy.**
   The ray's state lives in registers during the walk, and the walk's low occupancy is what leaves the SIMD units idle.
   Pack it from 160 bytes towards 64: the direction as a 32-bit octahedral code, the band energies as one float plus half-precision ratios (plain half floats cannot reach the roulette floor of 10⁻⁶), and depth, source index and flags in one word.
   Bandwidth is not the reason: the megakernel reads and writes a ray once per update, about 32 MB at 100,000 rays, about 0.3 ms.
4. **A compressed 8-wide tree** (Ylitie, Karras and Laine 2017, quantised child boxes), only if the software tree is still the path after step 2.
   The 4-wide tree that lost (part 53) was not compressed.
5. **Fewer global atomics.**
   Contributions are added into the ring with global atomics; summing per thread group in group-shared memory and flushing once would cut the contention.
   Profile it first; its share of the walk is not measured.

## Removing rays: keep the compaction

Rays are never removed in place, and nothing shifts.
Population control gives every ray its number of copies (zero to retire it, more than one to split it), a prefix sum gives each surviving copy its slot, and the copies are written in sorted order into the other half of a double-buffered pool (method.md, section 5).
Every live ray is rewritten every update anyway, because it advanced, so the compaction costs no extra writes.

A free list (retire by pushing the slot, spawn by popping it) was considered and is worse here:
- the holes stay in the pool, so threads are launched on dead slots or the pool needs compacting anyway;
- splits need new slots for their copies, which is the same allocation problem;
- it loses the sort by direction octant and Morton cell, which takes about 0.6 ms off the Church's tracer by keeping SIMD groups coherent (part 31);
- the second pool costs 16 MB at 100,000 rays, which is nothing.

What may be worth measuring: the sort's three passes run every update; sorting every second or fourth update (with a plain compaction in between) keeps most of the coherence, since rays move little in one update.

## A slower audio update

The audio update could run at 2,048 samples (42.7 ms) instead of 1,024.
What it buys and what it does not:
- Each ray then advances twice as far per update, so the walk's work per update roughly doubles; the tracing per second, which is what 100,000 rays cost, stays about the same.
- The fixed costs per update halve per second: the audio's fixed part (denoiser, kernel rebuild, fence and readback, about 2 ms per update on the Mac), the sort, and the new rays' source location.
- The response to movement lags one more update (about 21 ms more); Doppler and ITD come from delay rates, so pitch stays smooth.
- Measure the fixed and the distance-proportional shares of the walk before deciding; the estimate above assumes the walk's time is mostly proportional to the distance the rays travel.

The larger lever is the same idea without changing the period: let one update's work span more than one frame.
The 6 ms budget exists only to leave the GPU to the graphics; on async compute the walk can take most of a 21.3 ms update in the background, and 100,000 rays plus audio took about 17 ms per update even on the Mac.

## Open decisions for Clark

- Several readback batches in flight, for the station at 8,000 rays per source (0.26 ms over 6 ms today).
- Remove the walk-diffraction code (its write-up stays) and the 2D rooms (Big Room, SmollRoom).
- The beam kernel's normalisation (1.036 at 125 Hz) and the UTD lit-face term.
- Room modes on by default, after a listening check.
- A blind A/B against the measured York Minster responses.
