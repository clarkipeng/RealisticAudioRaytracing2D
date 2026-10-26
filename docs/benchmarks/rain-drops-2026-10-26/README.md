# Recorded drop candidates and accumulation audition

The user requested the rejected cache comparison again and an online search for
raindrop SFX that can accumulate into rain. The old comparison was reopened;
there is still no clarification about whether both halves or only the cache
half sounded bad. No previous source or propagation audition is accepted.

## Search results

Checked 2026-10-26. Product descriptions establish intended use, not a listening
pass. No paid files were purchased, downloaded or evaluated.

- [Sound of Essen — Rain Designer: Int, Ext and Single Drops](https://www.asoundeffect.com/sound-library/rain-designer-int-ext-and-single-drops/):
  the closest described match. Studio-recorded single drops in multiple sizes on
  glass, dirt, grass, wood, stone, concrete, water, cloth, plastic and metal;
  clusters and pours are separate categories. Includes Sound Particles projects
  for constructing rain. The marketplace's parsed zero-price field is unreliable;
  this is not treated as a free library.
- [PMSFX — Rain Designer](https://pmsfx.gumroad.com/l/raindesigner): a different
  product with the same short name. Listed at $49.99; 396 files including 268
  source recordings of single/multiple drips and trickles, plus designed loops
  and construction beds. Useful candidate, not proof that arbitrary layered
  single drops alone produce natural dense rain.
- Elements-Library on Freesound provides CC0 recordings of drops on
  [concrete](https://freesound.org/people/Elements-Library/sounds/683772/),
  [wood](https://freesound.org/people/Elements-Library/sounds/683781/) and
  [thin metal](https://freesound.org/people/Elements-Library/sounds/683778/).
  Original listings specify mono 48 kHz/24-bit recordings using a Rode NT1.
  We acquired only the publicly served HQ MP3 previews, not login-required WAVs.
  Concrete and metal contain separated transients suitable for a small trial.
  The wood preview has appreciable energy between peaks; it was inspected but
  not turned into a drop bank.
- [Mega-X-stream — Single drip](https://freesound.org/people/Mega-X-stream/sounds/546279/)
  is CC0 and 246 ms long, but gives one variation with unspecified impact
  surface. It was not selected over the varied surface recordings.

## Listening experiment

`run.py` extracts 11 concrete and 13 metal snippets from explicitly recorded
sample ranges in `report.json`. The selection followed waveform/envelope
inspection; **there was no agent listening evaluation**. Each snippet is 250 ms,
starting 35 ms before a selected peak. Mean removal, an 8 ms attack fade and a
20 ms release fade are the only snippet processing. Relative levels within
each bank remain intact. Snippets may contain secondary impacts, background
noise or room sound: these are not certified isolated dry single-drop responses.
The metal recording also retains the particular sheet's resonance.

A seeded Poisson stream triggers these snippets at 6, 60 and 600 events/s.
Lower densities retain subsets of the same events with unchanged timing and
waveform selection: 32, 309 and 3,046 events in each five-second section. Events
are summed directly in mono. There is no added rain ambience, synthetic noise,
propagation, pitch shifting, limiter or material simulation.

The three sections are **RMS matched at 0.015** for texture comparison, so this
is explicitly not a prediction of loudness as rainfall increases. Gains are
recorded in the report. The combined clip is 16 seconds:

| Time | Trigger rate |
| --- | --- |
| 0–5 s | 6/s |
| 5–5.5 s | silence |
| 5.5–10.5 s | 60/s |
| 10.5–11 s | silence |
| 11–16 s | 600/s |

Separate players expose six bank snippets and the full source recording with
playback gain adjustment. No audio autoplays. All output samples were checked
finite and below 0.51 full-scale peak. This demonstrates literal accumulation
of empirical snippets; whether it sounds like useful rain remains a listening
question. Runtime sources, transport, bounce count and vibration scope are
unchanged.

## Reproduce

The input directory must contain `concrete-preview.mp3`, `metal-preview.mp3`
and their decoded mono float32/44.1 kHz `concrete.wav`, `metal.wav` files.
Public preview URLs and MP3 hashes are in `report.json`; the runner verifies
those MP3 hashes. Decode with:

```sh
ffmpeg -v error -nostdin -i concrete-preview.mp3 -ar 44100 -ac 1 -c:a pcm_f32le concrete.wav
ffmpeg -v error -nostdin -i metal-preview.mp3 -ar 44100 -ac 1 -c:a pcm_f32le metal.wav
OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 OMP_NUM_THREADS=1 \
nice -n 19 .context/wave-planar/venv/bin/python \
docs/benchmarks/rain-drops-2026-10-26/run.py <input-directory> <new-output-directory>
```

Copy `listen.html` as `index.html` beside the output audio. The delivery folder
is Conductor's `.context/rain-source-review/recorded-drops/`, alongside the
existing `cache-test/`. Large audio files and downloaded previews stay local.
