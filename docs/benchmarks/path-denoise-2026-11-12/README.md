# Path tracer: parallel connections, speed, the Church render and a denoiser — 12 November 2026

Follow-up to the [path tracer benchmark](../path-tracer-2026-11-12/README.md).
All numbers are measured on this Mac (Apple M5, 16 GB, Unity 6000.4.1f1 on Metal) under `tools/heavy_guard.py` (one heavy job machine-wide, 4 GiB guard, nice 19, one job worker, 50% pacing), often waiting for other projects' jobs.

## 1. Parallel connection trials

**Change.**
A queued specular-chain candidate used to retrace its exact image path and then draw trials one after another in one GPU thread until one followed the chain (the geometric count `N`).
`PathConnect` now gives each candidate 8 lanes of a 64-thread group: in round 0 lane 0 retraces the exact path and lanes 1–7 draw trials 1–7; in round r the lanes draw trials 8r … 8r + 7.
`N` is the smallest index of a successful trial, kept with an atomic minimum in group-shared memory, and each round decides whether to go on only from the previous rounds' results.
Because the trials are independent and identically distributed, the smallest successful index has exactly the distribution of the sequential count, so the estimator is unchanged; a candidate's latency falls from `N` retraces to about `N / 8`.

**Checks.**
All path checks pass unchanged in value (free field exact; rigid plane image 0.953; image lattice total 0.989, worst 20 ms window 6.1%; half-plane +0.30 dB; DD over a 2 m wall +0.49 … +1.58 dB; Eyring +4.8%; the clipped-region image 1.00 with the measured share).

Time per update (GPU busy per solve, one source and listener, all eight bands, 2 s horizon; 8 receivers per scene, one process, the measurement path with a fence per 1,024 rays):

| Scene | energy 2,048 | energy 16,384 | path 2,048 | path 16,384 | path 2,048 before | path 16,384 before |
|---|---:|---:|---:|---:|---:|---:|
| canopy (open) | 11 ms | 106 ms | 19 ms | 136 ms | 11 ms | 178 ms |
| barriers (open) | 9 ms | 59 ms | 9 ms | 81 ms | 13 ms | 110 ms |
| doorway | 56 ms | 575 ms | 104 ms | 834 ms | 140 ms | 1.30 s |
| wood shoebox | 20 ms | 329 ms | 70 ms | 596 ms | 244 ms | 1.92 s |
| stone shoebox | 50 ms | 614 ms | 109 ms | 844 ms | 283 ms | 2.33 s |
| corridor (two rooms) | 98 ms | 761 ms | 170 ms | 1.35 s | 1.10 s | – |
| furnished room | 168 ms | 1.39 s | 690 ms | 6.22 s | 3.31 s | – |

"Before" is the cost table of the first benchmark (sequential trials, all receivers).
At 2,048 rays the path tracer is now 1.0–2.2 times the energy method in the open scenes, doorway, stone shoebox and corridor, 3.5 times in the wood shoebox and 4.1 times in the furnished room; the furnished room fell from 3.3 s to 0.69 s per update, the corridor from 1.10 s to 0.17 s.
It is not realtime in rooms: 70–690 ms per update at 2,048 rays against a 23 ms budget (part 2).

**Trial cap.** With parallel trials the cap of 64 could be raised; at 1,024 the capped detections in the cluttered rooms fall from 112 to 3 (furnished room, 4,502 scored chains, 30 receivers) and from 44 to 0 (corridor, 9,785).
The level bias moved by +0.05 dB (furnished, 1.69 → 1.74 dB) and +0.02 dB (corridor), within the seed noise, while the cost rose about threefold (furnished room 2.0 s per update over its 30 receivers).
The cap stays at 64: its bias is below 0.1 dB in the worst scene of the suite, and the counter reports it.
