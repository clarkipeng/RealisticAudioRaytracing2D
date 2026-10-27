# Rain textures through the native acoustic renderer

This diagnostic feeds the previous courtyard walk's 98 seeded rain textures
through the existing `AcousticTake` → `TakeRender` → GPU transport/DSP pipeline.
It reproduces a serious loudness failure in the existing first-reflection
sampler. It does not change production source generation, Church assets or
runtime settings. The source direction received positive feedback; the native
reflection result is not suitable for promotion.

## Failure isolated

The direct baseline agrees with an independent point-source mixer within
0.008% relative waveform RMS at all three stationary stops. The 32-bounce,
1-ray/bin render raises the canopy window by about 22.1 dB versus direct sound,
and its floating-point output exceeds full scale.

`FirstReflection` in `Assets/Shaders/Transport.hlsl` samples primitive IDs and
multiplies each selected pressure contribution by `PrimitiveCount / RaysPerBin`.
The fixture has 84 triangles. At the canopy, the independent static mirror audit
finds 73 valid first-reflection paths across the 98 sources; seed 0 at 1 ray/bin
selects only two, with an 84× pressure weight each. Their combined first-mirror
RMS is 25.03 dB above enumeration of every valid first mirror. The predicted
sampled waveform matches native indirect output with 0.163% relative RMS
residual and correlation 0.9999987. This identifies the dominant contributor
to this fixture's hotspot, rather than attributing it to dry-source texture.

The same sampler selects no valid first mirrors at the outside and inside
stops for this low-budget seed, despite 48 and 17 valid paths respectively.
`mirrors.py` is a diagnostic for static boxes, not a proposed exhaustive
production renderer or a reference for all higher-order/diffuse paths.

At 2 rays/bin, seed 0 lowers the measured canopy level by **5.95 dB**. Changing
only the transport seed at that budget changes it by another **1.93 dB**, with
waveform correlation **0.044** over the focused three seconds. Source PCM and
gain are identical. The independent sampled-mirror prediction matches both
higher-budget native indirect windows within 0.21% RMS. This remains a failed
low-budget sampling result, not a convergence study or listening acceptance.

Do not repair it by deleting the inverse-probability weight or reducing rain
gain: those would change the estimator's mean or hide the propagation failure.
Improve first-mirror coverage and validate it against explicit small-fixture
connections while preserving the existing higher-order transport. Source
batching is a separate cost issue; neither needs a new rain waveform kernel.

## Measured cost and checks

All four native jobs completed with finite float output and tails covering
their configured response support. The low-budget reflected walk contains
27 samples above full scale, with peak 1.85434. All six rendered listening
previews use the same **−12 dB** gain; the raw failure remains in `report.json`.

| Job | Input / tail | Render wall time at 20% duty | Numeric working storage | Peak sampled process RSS |
| --- | --- | ---: | ---: | ---: |
| Direct walk | 19 s / 1.5 s | 430.08 s | 355.4 MiB | 723.3 MiB |
| Reflected walk, 1 ray/bin, seed 0 | 19 s / 1.5 s | 540.37 s | 1,035.4 MiB | 906.2 MiB |
| Canopy, 2 rays/bin, seed 0 | 5 s / 1.5 s | 157.77 s | 1,484.6 MiB | 886.5 MiB |
| Canopy, 2 rays/bin, seed 1 | 5 s / 1.5 s | 158.19 s | 1,484.6 MiB | 1,247.4 MiB |

Numeric storage includes preloaded offline PCM and allocated CPU/GPU buffers;
sampled process RSS is a different measure and not total system/GPU memory.
The full reflected walk reports 77.44 s in audio work versus 4.18 s in solving
transport, excluding launcher startup and throttle sleeps. The current GPU DSP
submits per-pair work at each small audio hop and reserves filter storage by
path capacity. One ordinary source per patch is useful for this diagnostic,
but these results do not support a realtime full-scene rollout.

Checks cover authenticated PCM equality, exact focused-source alignment,
unchanged source gains, settings and renderer identity, finite output, waveform
hashes, configured tail coverage and the independent direct oracle. The mirror
audit was also checked against an analytic single-box reflection distance,
finite-face rejection and an intervening occluder. Browser checks verify 18 s
walk / 3 s canopy / 12 s reference durations, paused startup, retained seek
position on version changes, working stage buttons, no console errors and no
horizontal overflow at desktop and 390 px width. No audio was played by the agent.

## What is held fixed

The same 20 × 16 m catchment, seven opaque wall/roof boxes, 18-second listener
route, four-second texture grains, source seed 23 and emission gain are used.
One second of actual renderer warm-up precedes the walk. Each region's PCM is
generated once on the absolute source clock, then reused byte-for-byte by all
jobs. The C# adapter uses the ordinary authenticated take writer. After the
full walk reproduced the failure, the remaining two-budget/seed checks were
focused on the canopy. That five-second stationary take contains exactly
original source seconds 6–11, with two seconds of warm-up. Its three-second
audition aligns to original walk time 7–10. The central second (original time
7.5–8.5) is used for level measurements; the beginning of the full-walk excerpt
still includes the listener stopping. Payload equality, source gain, input
identity within each take and output SHA-256 checks are part of the analysis.

The native trial places one independently seeded source at each patch center,
with pressure gain `emissionGain × sqrt(patchArea)`. It does not launch several
coherent copies of a region's texture or multiply pressure by inverse ray
probability to represent missing rain sources. Changing ray budget changes
transport sampling only. Patch refinement would change this source realization
and is deliberately not part of this comparison.

Replacing the previous 8 × 8 direct-area quadrature by these centers changes
predicted total direct power by −0.481 to +0.380 dB over the sampled route.
This is a geometry/power diagnostic, not a bound on waveform or spatial error.
The output is mono to isolate transport; the earlier audition was equal-power
stereo. Compare the native versions to each other at unchanged volume.

## Render settings and limits

Two jobs share the full walk: direct-only and 32 bounces at 1 ray/bin, seed 0.
Two jobs share the focused canopy take: 32 bounces at 2 rays/bin with seeds 0/1.
All use FFT 128/hop 32, response step 1472 samples, sinc radius 16, one response
per batch, a 1.25 s path horizon and 1.5 s recorded tail at 44.1 kHz. The 30 Hz
input poses use the native take reader's position/interval-velocity interpolation.
Stored observations have no explicit velocity; the stationary oracle windows
avoid motion-model differences from the previous Python prototype.
The auditions trim declared DSP latency and warm-up, retaining the 18-second
walk or three-second canopy interval. Native float WAVs preserve failing peaks.
All browser previews receive the same documented attenuation to fit playback
range, plus 40 ms endpoint fades. There is no limiter or per-version level match;
measurements always use the original unattenuated signals.

The fixture material is explicitly illustrative: flat absorption 0.35,
scattering 0.15, zero transmission. Air absorption and experimental diffraction
are disabled for this isolation test. Existing production settings are untouched.
There is no reflecting ground mesh in the earlier fixture or this derivative.
Roof impact texture is generated above the roof, but the roof blocks its direct
path to this listener. No underside source, roof transmission, structural
vibration or arbitrary indoor gain/filter is introduced.

The recordings retain their original acoustics and processing. These are
uncalibrated empirical textures, not isolated drops, a physical material model,
or rain intensity in mm/h. Two transport budgets and seeds can reveal instability;
they cannot establish physical accuracy or convergence. The production renderer's
previously identified indirect-path issues remain separate from this adapter.

## Reproduction

All sample PCM and WAVs stay in private, gitignored `.context/`. See the
[publisher audition](../rain-library-2026-10-26/README.md) for sources and license.

1. Run `prepare.py <publisher-originals-directory> <new-work-directory>` with
   the existing NumPy/SciPy environment, numerical libraries restricted to one
   thread, and `nice -n 19`.
2. Copy `RainTransportReview.cs` into the isolated runner's `Assets/Editor`.
   Run Unity in batch mode with `-disable-audio -nographics -job-worker-count 1
   -executeMethod RainTransportReview.Prepare -rainFixture <work>/fixture.json`.
   Remove this temporary helper after the experiment.
3. Run the two full-walk `*-job.json` files with `tools/capture_sequence.py --project
   <isolated-runner> --output <work>/<job-name> --render-job <work>/<job-name>-job.json
   --work-fraction .2 --max-resident-mib 3072`, one process at a time.
4. Run `focus.py <work> <new-focused-work-directory>`, prepare its take with the
   same C# adapter, then run its two jobs with the same guarded launcher.
5. Run `mirrors.py <work> --focus <focused-work>` for the first-mirror audit.
   Run `analyze.py <work> <focused-work> <new-delivery-directory>`. It verifies shared inputs,
   measures three stationary windows, compares budgets/seeds, checks direct
   output against a separate point-source mixer and builds the browser audition.

The browser retains the earlier schematic view and source minimap. Its rain
streaks are illustrative, not audio-event markers. No autoplay; changing version
pauses and retains the listening position. Animation runs only during visible
playback. No foreground editor, microphone, game networking or speaker playback
is started by the diagnostic.
