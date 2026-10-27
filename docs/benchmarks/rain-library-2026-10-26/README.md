# Publisher rain audition

After rejecting the layered concrete/metal snippets, the user authorized checking
purpose-built library demos before another synthesis attempt. This is a source
audition only. No new rain algorithm, source integration or listening approval.

The [PMSFX Rain Designer product page](https://pmsfx.gumroad.com/l/raindesigner)
has an embedded SoundCloud playlist with two tracks: **Designed** and
**Construction Kit and Source**. The second previews ingredients. The complete
pack remains unpurchased; public product descriptions do not establish quality.

Four library samples are available in [Sonniss's free GDC 2020 bundle](https://sonniss.com/gameaudiogdc/).
Their filenames and supplier were checked against the
[official linked track list](https://docs.google.com/spreadsheets/d/1QqIcxJ7y0Acq3HAj9qQvFgvyeBjJ_7JvX8-GguUHIVk/edit).
The WAVs came from the Your.Org mirror's individual-file directory, avoiding
the 48.2 GB archive. `report.json` records download URLs, SHA-256 hashes,
original format and excerpt measurements. These are finished rain, background
wash, tarp/foliage texture and iron-plate texture—not isolated single drops.

The private listening page starts with the publisher's finished moderate rain.
All four excerpts select seconds 5–17, preserve stereo, keep the publisher’s original levels
with 20 ms endpoint fades, and convert to 44.1 kHz/16-bit WAV. No per-file
normalization, EQ, compression, synthesis, added bed or propagation. This tests
whether the source direction is useful, not whether one-shot accumulation works.
No agent listening assessment or speaker playback was performed.

Files stay in gitignored `.context/`. They are licensed samples, not CC0; keep
the [GDC bundle terms](https://sonniss.com/gdc-bundle-license/) with their
provenance. The local audition is for this project; do not publish its sample
files as a downloadable sound library. No purchase or account creation occurred.

Reproduce from `.context/rain-research/publisher-review/` containing
`downloads.json` and the four original WAVs in `originals/`:

```sh
OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 OMP_NUM_THREADS=1 \
nice -n 19 .context/wave-planar/venv/bin/python \
docs/benchmarks/rain-library-2026-10-26/prepare.py \
.context/rain-research/publisher-review <new-output-directory>
```

Copy `listen.html` as `index.html` beside the excerpt WAVs and `report.json`.
Delivery is Conductor's `.context/rain-source-review/library-audition/`.
Rendering uses one worker; browser playback is user-controlled with no autoplay.
