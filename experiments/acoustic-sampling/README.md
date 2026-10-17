# Acoustic sampling pilot

The checkpoint is `435a18c`, named `checkpoint/acoustic-sampling-2026-10-17`.
`experiment/guided-acoustics`, `experiment/listener-acoustics`, and
`experiment/bidirectional-acoustics` share this comparison harness as their starting point.
No playback, scene, runtime shader, or runtime C# changes are included.

The result does **not** justify replacing the current renderer. Joining one source-side
edge sample to one listener-side edge sample wins on the single-box fixture. It loses
on the staggered-wall fixture. Shortest-route guidance slightly helps the box but can
hurt the staggered case. Source/listener/MIS throughput efficiencies in that second
fixture are close enough that small timing differences should not pick an architecture.
See [measurements](measurements.md) for raw-cost accounting and all eight variants.

At 250 Hz and 4,096 samples, the box's expected relative complex-response error falls
from 35.03% for source-only to 12.47% for an unguided two-ended join. This is about 7.9×
less variance per sample **within this discrete pilot**, not a measured production
speedup. On staggered walls the corresponding errors are 24.42% and 25.64%; adding
guidance to the join raises error to 32.05%. Even the better box result is far from
a high-quality training reference.

## What the experiment computes

Only the signed **two-edge diffraction contribution** in static, rigid 3D geometry.
It reuses production edge extraction, `EdgeKernel`, `Visible`, and BLAS/TLAS buffers.
It does not include direct sound, one-edge paths, reflections, diffuse scattering,
transmission, Doppler, head filters, convolution, STFT, or playback. No scene is played.
This is not full bidirectional sound transport or the BEDRF algorithm.

Each finite physical edge receives Gauss-Legendre quadrature points and integration
weights. The reference enumerates every ordered pair of points, rejects blocked legs,
and sums the same signed complex contribution that the samplers estimate:

```
F(a,b) = visible(S,a,b,L)
       * edgeKernel(S,a,b) * edgeKernel(a,b,L)
       * quadratureWeight(a) * quadratureWeight(b)
       / (distance(S,a) * distance(a,b) * distance(b,L))
       * exp(-i * 2*pi*f * totalDistance / speedOfSound)
```

Immediate repeated-edge events are excluded. The model is the current renderer's
secondary-edge approximation; enumeration only makes its **discretized sum** exact.
It does not turn this approximation into a validated wave simulation. The single-box
response is stable across 16/32/64 points per edge. The staggered response still changes
1.21–1.93% from 32 to 64, so its continuum integral is not yet converged.

## Samplers

```
source:   a ~ sourceProposal; b ~ forwardProposal[a]
          q = pSource(a) * pForward(b | a)

listener: b ~ listenerProposal; a ~ reverseProposal[b]
          q = pListener(b) * pReverse(a | b)

join:     a ~ sourceProposal; b ~ listenerProposal
          q = pSource(a) * pListener(b)

MIS:      choose source, listener, or join with equal probability
          sample that strategy
          q = (qSource + qListener + qJoin) / 3

all:      trace the actual straight segments and evaluate F
          estimate += F / (q * sampleCount)
```

The join constructs a source prefix and a listener suffix and tests their connection;
it is not an average of two one-ended renders. The MIS variant accounts for all three
overlapping proposal densities. Equal mixture weights are an experiment choice, not
an optimized runtime policy. The two one-ended controls use visible inverse-square
proposal weights, **not the production Cauchy edge-position sampler**.

Guidance runs reverse Dijkstra over visible connections between real edge quadrature
points. It exponentially favors shorter remaining routes, with a 15% global exploration
mixture weighted by edge integration measure. The extra-route length scale is one
quarter of source-listener separation. These fixed research parameters were not tuned
to the results. No point is removed from support. A graph route ranks candidates;
it never substitutes for the path's geometry, pressure, delay, or visibility.

Dense probability/CDF tables keep this proof easy to audit. They are O(nodes²) per
pair and **must not become the live implementation**. Guided-table construction took
10.0/31.1 ms for box/staggered fixtures, versus 4.6/14.6 ms for baseline tables, plus
connection construction/readback and table upload. Each GPU proposal pair occupies
2.37/9.45 MB. Any runtime guide needs compact shared data and measured reuse.

## Checks and known failure

The silent EditMode test checks finite responses, positive proposal support, blocked
direct segments, source/listener reversal of every path, and GPU estimator means against
the independently enumerated target. Expected error uses the complete second moment;
observed error uses 96 estimates per variant/budget/frequency. Timings alternate variant
order, discard warmup, and retain three repeats. Budgets are 64, 512, and 4,096 samples;
frequencies are 62.5 and 250 Hz. Those are test cases, not runtime limits or band counts.

**Known production visibility defect:** if both staggered boxes have the same height,
some connecting segments lie exactly on their top/bottom faces. Visibility can say
blocked in one direction and clear in the other. With 32-point quadrature, path-response
reversal discrepancy is 0.620%; with 64 it is 1.730%. A representative segment from
`(-1.1, 0.1612385, 1.4)` to `(1.1, 1.058919, 1.4)` reproduces the problem. The ordinary
comparison uses unequal heights to separate sampling behavior from this defect.
The failing equal-height case remains reproducible with `ACOUSTIC_SAMPLING_COPLANAR=1`;
its assertion is still enabled. Baseline diagnostics are retained in `results/coplanar-*`.

Changing determinant arithmetic alone did not fix the failure and was discarded.
A robust grazing/shared-edge intersection policy needs its own regression work before
promoting reverse-path reuse. [PBRT's rounding-error discussion](https://www.pbr-book.org/4ed/Shapes/Managing_Rounding_Error)
describes why robust triangle intersections need more than an arbitrary epsilon.

## What this says about the codebase

For one live listener and many sources, listener-side path reuse remains a reasonable
candidate; for many offline listeners, source-side reuse can be useful. **Neither reuse
benefit was measured here:** this harness evaluates one pair at a time and performs no
cross-pair sharing. The existing diffuse model's known reciprocity mismatch also prevents
assuming that reversing every current transport interaction is physically equivalent.

Keep the existing renderer as the control. Preserve these small experiments separately.
Before runtime integration, resolve tangent visibility, compare the continuous production
sampler, measure moving endpoints and multi-source/listener reuse, and obtain a converged
response reference. The current data favors a two-ended option for some paths, not a
universal replacement or a shortest-path audio model.

## Reproduce without opening a player or using speakers

Use a separate Unity project with the checkpoint's Assets/Packages/ProjectSettings and
an imported Library. The script checks core runtime files, refuses the working checkout
or an open runner, temporarily installs the two experiment files, runs one low-priority
Metal batch process with two workers, and restores overwritten files on exit.

```
python3 experiments/acoustic-sampling/run.py /path/to/isolated-runner \
  --unity /Applications/Unity/Hub/Editor/6000.4.1f1/Unity.app/Contents/MacOS/Unity \
  --output /path/to/results
python3 experiments/acoustic-sampling/summarize.py /path/to/results
```

The test requires batch mode, creates an empty test scene, sets listener volume to zero,
and only dispatches compute shaders. `-nographics` is intentionally absent because this
is a GPU test. `ACOUSTIC_SAMPLING_COPLANAR=1` makes the same command reproduce the known
failure. No microphone or audio device callbacks are used.

`results/metrics.csv`, `reference.csv`, and `setup.csv` contain the numerical measurements;
`trials.csv.gz` contains individual estimates. `measurements.md` is generated by
`summarize.py results`. Local runner logs/XML are not committed because they contain
machine paths and package-import noise. `validation.json` records test status and counts.
