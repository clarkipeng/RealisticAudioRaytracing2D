# Acoustic sampling measurements

Checkpoint: `435a18c` (`checkpoint/acoustic-sampling-2026-10-17`). The original pilot is
preserved at `d2f5d47`. Guided, listener, and bidirectional experiment branches share
this harness so comparisons use the same target and code. Runtime Assets are unchanged.

[Measured results](measurements.md) include matched-error budgets, single-estimate and
batched timings, setup/transfer costs, and the production transport control. The numerical
results support further work on sampling and GPU data flow, not a live-renderer switch.
Direction alone has no universal winner. Shortest-route guidance can increase variance.

## Target and methods

The fixtures contain static rigid 3D boxes and one source/listener pair. Only the signed
two-edge diffraction contribution is compared. Geometry extraction, BLAS/TLAS, visibility,
and the edge kernel come from production. Gauss-Legendre points on each physical edge
provide an enumerable discrete target for the prototypes:

```
F(a,b) = visible(S,a,b,L) * K(S,a,b) * K(a,b,L) * weight(a) * weight(b)
       / (distance(S,a) * distance(a,b) * distance(b,L))
       * exp(-i * 2*pi*f * totalDistance / speedOfSound)

source:   a ~ pSource; b ~ pForward[a]; q = pSource(a) * pForward(b|a)
listener: b ~ pListener; a ~ pReverse[b]; q = pListener(b) * pReverse(a|b)
join:     a ~ pSource; b ~ pListener; q = pSource(a) * pListener(b)
mixture:  choose one of those three equally; q = (qSource + qListener + qJoin)/3
estimate: sum F(a,b) / (q(a,b) * sampleCount)
```

The join tests a real connection between source and listener prefixes; it is not an
average of two one-ended renders. Each method has an unguided and guided proposal.
The unguided proposal uses visible inverse-square weights. Guidance uses reverse Dijkstra
over real edge points, favoring shorter remaining routes with an exponential length scale
of one quarter of source/listener separation. A 15% global exploration mixture preserves
support. These are fixed research parameters, not runtime defaults or a shortest-path
audio model. Every sampled leg still uses the real geometric visibility test.

Two continuous controls separate proposal quality from other transport work:

- **Production:** run the unmodified `Trace3DBvh` with one frequency bin, two bounces,
  and absorbing, non-transmitting surfaces; reduce second-event arrivals. This suppresses
  surface contributions while leaving the current rigid edge kernel/opacity unchanged.
  The fixture's maximum path delay contains every candidate. Surface calculations and
  full RayPath writes remain in the timed dispatch. Independent identical source pairs
  supply 32 random trials without changing the production shader's RNG.
- **Cauchy only:** call production `SampleEdge` twice, retaining both 50% edge-event choices,
  and reduce the same two-edge contribution directly. No dense proposal tables are used.
  This isolates the estimator from surface calculations and RayPath storage; it does not
  implement the complete renderer. Both controls are checked statistically for agreement.

Direct reduction here produces one static complex coefficient. A live implementation must
still preserve changing delays, Doppler, and path continuity; this benchmark does not show
that those per-path states can be removed.

The prototype target is exact only for its discretization. It is not a wave-equation
reference. Production/Cauchy results use the 64-point reference; prototypes use 32 points.
The box converges closely across 16/32/64 points. The staggered response still changes
about 1.9% at 250 Hz between 32 and 64 and remains unconverged. No training-quality claim follows.

## Measurement and checks

Expected prototype MSE after N samples is `(sum |F|²/q - |sum F|²)/N`. This determines
budgets near 20% and 10% relative error. Production's 20% budget is calibrated from the
prior 16,384-sample observations, then tested with different seeds; a 10% production
budget is not measured. Each row combines 160 observations across five repeats after
warmup. Source/listener/join order alternates; timing tables show median and interquartile
range. The fixed-budget results and all individual estimates remain in the CSV data.

Dispatch/readback timing excludes CPU final summation and proposal construction/upload.
Those costs are reported separately. A second connection dispatch separates warm setup
from first-use initialization. Setup is a single observation, not a median. Common scene
construction/BVH upload and Unity startup are excluded. Dense O(nodes²) tables are an
audit-friendly experiment, not the proposed live implementation; no cross-pair reuse or
moving-endpoint update benefit has been measured.

Checks cover finite responses, positive proposal support, blocked direct connections,
path reciprocity, estimator means against the enumerated target, zero-diffraction
isolation, and bit-identical single/batch packing. The summary also checks production
and Cauchy-only means within six combined standard errors. This is a regression check,
not a proof of zero bias. Other production compilation warnings remain outside this scope.

The **known coplanar visibility failure remains**: equal-height staggered boxes can give
different visibility for a face-grazing segment in opposite directions. Retained diagnostics
show 0.620%/1.730% path-response reversal discrepancy at 32/64 points per edge. Ordinary
fixtures use unequal heights. `ACOUSTIC_SAMPLING_COPLANAR=1` reproduces the failing assertion;
no geometry epsilon, face-skipping rule, or production visibility change is introduced.

## Reproduce silently

Use a separate imported Unity project matching this checkpoint. The script refuses the
working checkout and an open runner, checks core runtime files, temporarily installs the
experiment files, runs one low-priority Metal batch process with two workers, and restores
the runner files. It never enters Play mode; the test creates an empty scene, mutes the
listener, and only dispatches compute shaders. No microphone/device callbacks are used.

```
python3 experiments/acoustic-sampling/run.py /path/to/isolated-runner \
  --unity /Applications/Unity/Hub/Editor/6000.4.1f1/Unity.app/Contents/MacOS/Unity \
  --output /path/to/new-results \
  --calibrate-from experiments/acoustic-sampling/results/calibration-source.csv
python3 experiments/acoustic-sampling/summarize.py /path/to/new-results
```

Omit `--calibrate-from` for a new calibration pass, then pass that output directory to
the calibrated run. The script requires empty output directories: presence of `calibration.csv`
selects the held-out production seed range. Saved calibration data uses the earlier seed
range. `results/calibration.csv` records the selected budgets; `calibration-source.csv`
contains the twenty observations that selected them.

The versioned `metrics.csv`, `reference.csv`, `setup.csv`, compressed `trials.csv.gz`, and
`validation.json` support `summarize.py results` without a Unity installation. Logs/XML
remain local because they contain machine paths and import noise. The original coplanar
diagnostics are preserved. Playback, mixed surface/edge paths, material accuracy, moving
heads, multi-source/listener sharing, and full church-scene performance remain unmeasured.
