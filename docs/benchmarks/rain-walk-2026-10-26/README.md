# Courtyard → canopy → indoor rain test

The user called the publisher rain audition better and authorized a small
spatial walk. This is a new offline direct-sound experiment, not a production
renderer integration or a fix for its previously identified indirect-path
spikes. Existing runtime propagation, 32-bounce configuration and vibration
deferral remain unchanged. The new walk still needs listening approval.

## Source and scene

A 20 × 16 m horizontal rain catchment is partitioned exactly into exposed
ground and a rectangular roof. These regions are the analytic first-hit result
for vertical rain on this particular static scene; there is no ground source
under the roof. The roof covers an open canopy and an enclosed room with one
doorway. The listener starts outside, walks beneath the canopy, pauses, then
walks through the doorway and stops inside. Total duration: 18 seconds.

There are 98 surface-clipped patches, at most 2 × 2 m. Each patch has a seeded,
absolute-time mono texture stream, built from four-second passages with two-second
hops and sine windows. Random offsets have sample resolution, avoiding a small
set of exactly shared grains across regions. The original publisher recordings
remain colored by their capture/processing; these are not dry measured material
responses. The complete recording is RMS-normalized once per profile, preserving
relative levels within it. No per-grain or listener-dependent normalization.

Ground uses PMSFX Moderate rain; roof uses its iron-plate profile. **Roof audio
is completely occluded along this below-roof route.** The canopy/inside result
is outdoor ground rain admitted by open sides/the doorway, not sound transmitted
through the roof. Structural radiation, diffraction, transmission and reflections
are not implemented in this experiment. A sealed doorway produces zero direct
sound in the room, as expected for this deliberately limited model.

## Approximation and batching

`tools/rain_area.py` casts 8 × 8 direct visibility samples per patch and response
time. For unit-power source textures it integrates `area × visibility / distance²`,
including equal-power stereo pan weights. The square root gives each channel's
pressure gain. One centroid delay is shared by the patch's texture. This is an
incoherent-power approximation within the patch, not phase-resolved propagation
of a shared waveform from every visibility sample. Refining visibility sampling
keeps patch identities and source waveforms fixed.

Response knots are fixed at 30 Hz and gains/delays interpolate continuously.
Source reads use cubic interpolation with a one-second pre-roll; this is not a
bandlimited Doppler solver or binaural HRTF. Renderer query boundaries do not
change the global knot grid, seeded grains or source clock. The absolute-time
recording generator shares its overlap mixer with the earlier rain texture
implementation; old bank-render behavior remains covered by its existing tests.

One emission gain is fixed from the initial outside expected power and used
throughout. There is no limiter, per-position gain, or indoor low-pass filter.
The audition has only 40 ms endpoint fades.

## Verification

- All 12 rain tests pass, including clipped area conservation, visibility-ray
  count invariance, stereo power conservation, roof occlusion, and an irregular
  batch split with moving delay/gain (maximum permitted residual 2e-12).
- Stationary measured output: canopy −3.27 dB and inside −12.44 dB relative to
  outside. These windows contain different rain times, so they are not pure
  transfer measurements; expected area power is also recorded separately.
- The 8 × 8 grid differs from 16 × 16 by at most 0.0172 dB over the sampled route.
  At the three stops its error versus 32 × 32 is under 0.002 dB.
- Largest adjacent expected-power change: 0.215 dB over 1/30 second. These checks
  address this fixture's geometry transitions, not production transport noise.
- Maximum measured absolute cross-correlation among 16 source-region streams:
  0.0173 over six seconds at quarter-rate. This finite check does not establish
  independence at all lags, seeds or durations.
- Output is finite, peak 0.0927 or less; delivered WAV SHA-256 verified. Both
  browser players load, with no horizontal overflow or autoplay.
- Render plus audits took 4.44 seconds on one low-priority worker. This is a
  local offline measurement, not a realtime production performance claim.

The browser includes a schematic first-person view and source minimap. Rain
streaks are illustrative, not audio-event markers; they obey the roof footprint.
Animation runs only during playback in a visible tab, capped around 30 fps.
The agent did not start playback.

## Reproduce

```sh
OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 OMP_NUM_THREADS=1 \
nice -n 19 .context/wave-planar/venv/bin/python \
docs/benchmarks/rain-walk-2026-10-26/run.py \
.context/rain-research/publisher-review/originals <new-output-directory>
```

Source filenames, original hashes, scene geometry, normalization and diagnostics
are in `report.json`. Input assets are the licensed free GDC selections documented
in the [publisher audition](../rain-library-2026-10-26/README.md). WAVs remain local
in gitignored `.context/`; do not publish the samples as an asset library.
Delivery: Conductor `.context/rain-source-review/area-walk/`.
