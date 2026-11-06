# Method comparison output format (`method-compare-v1`)

Written 6 November 2026.
Every acoustic method (`hybrid`, `pressure`, `energy`, …) and the wave ground truth (`gt`) write this format, and `tools/method_compare.py` scores and visualises whatever it finds under the root.
The format is the one requested for the comparison, with only additive fields; each addition is marked **(addition)** with its reason.

## Directory layout

```text
<root>/                                   default root: .context/method-compare/data
  suite.json                              scene list (addition: lets every method enumerate the suite)
  <scene-id>/
    scene.json                            the one scene description every method reads
    <method>/<budget>/<seed>/
      meta.json                           required
      energy.f32                          required
      complex.f32                         optional (methods with phase)
      direction.f32                       optional (methods with directional energy)
```

- Ground truth uses method `gt`, budget `ref`, seed `0`.
- `<method>` and `<budget>` are lowercase `[a-z0-9.-]+` strings; `<seed>` is a non-negative decimal integer.
- Budgets of one method should be named so they sort sensibly (e.g. `r00128`, `r01024`), but ordering for convergence plots comes from `meta.json` `budgetValue`, not from the name.
- A method may write any subset of scenes; the scorer reports what exists and lists what is missing.

## `suite.json` (addition)

```json
{"schema": "method-compare-suite-v1",
 "scenes": [{"id": "shoebox-0003-presets", "family": "shoebox", "variant": "presets",
             "closed": true, "gtBands": [125, 250, 500]}]}
```

`gtBands` lists the band centres (Hz) for which ground truth exists in that scene; every other band has no ground truth and is never scored.

## `scene.json`

A superset of the dataset scene schema `dlss-scene-v1` (see [dataset](../../audio-dlss/dataset.md)): metres, y up, axis-aligned boxes `walls` `[{center, half}]`, `source` `[x,y,z]`, flat receiver list `points` `[x0,y0,z0,x1,…]`, `speedOfSound` (343 m/s), `surfaceMaterials` (one material name per wall).
Receivers are always in `points` order, and every output array uses that order.

Additions used by the comparison:

| Key | Meaning |
|---|---|
| `schema` | `method-compare-scene-v1` |
| `sceneId`, `family`, `variant` | id as in the directory name; family (`shoebox`, `doorway`, `corridor`, `furnished`, `canopy`, `barriers`, `canopy-bem`, `blocks-bem`); variant `rigid` or `presets` |
| `closed` | true for rooms without openings (`shoebox`, `furnished`) |
| `shadow` | per receiver, 1 when the straight segment from the source is blocked by a box (no direct line of sight), else 0; `lineOfSight` is its complement |
| `bands` | `[125, 250, 500, 1000, 2000, 4000, 8000, 16000]` |
| `materials` | `{name: {absorption: [8], scattering: [8], transmission: [8], source: "..."}}`, per band; `rigid` is all zeros |
| `airAbsorption` | `false`: no method applies air absorption in this suite (the ground truth has none) |
| `gtBands` | as in `suite.json` |
| `origin` | where the scene came from (generator version and index, or the BEM fixture file and its hash) |

**Materials.**
`absorption` is the existing preset's energy curve (`Assets/Materials/Church <name>.asset`) evaluated at each band centre with the renderer's rule (linear in Hz between knots, clamped outside).
`scattering` and `transmission` are **zero** in this suite: the ground truth has smooth, opaque, locally reacting walls, so a method that scatters or transmits would be compared against physics the reference does not contain.
The preset's own scattering is kept beside it as `presetScattering` for information only.

## `meta.json`

```json
{
  "schema": "method-compare-v1",
  "method": "energy", "budget": "r01024", "seed": 3,
  "budgetValue": 1024, "budgetUnit": "rays per band",
  "sceneId": "shoebox-0003-presets", "sceneSha256": "<sha256 of the scene.json bytes read>",
  "code": {"name": "LateEnergy.hlsl", "sha256": "<hash of solver sources or git commit>"},
  "wallSeconds": 0.42, "gpuSeconds": 0.31,
  "units": {"energy": "free-field 1 m direct sound = 1 per band", "time": "s", "complex": "free-field exp(-ikr)/r"},
  "bandCentersHz": [125, 250, 500, 1000, 2000, 4000, 8000, 16000],
  "bandEdgesHz": [[88.39, 176.78], "..."],
  "bandsProvided": [125, 250, 500, 1000, 2000, 4000, 8000, 16000],
  "timeBinSeconds": 0.001, "timeBins": 2000,
  "validSeconds": 1.25,
  "frequenciesHz": [125, 250, 375],
  "receiverOrder": "scene.json points",
  "receivers": 72,
  "files": {"energy.f32": [72, 8, 2000], "complex.f32": [72, 3, 2], "direction.f32": [72, 8, 2000, 6]},
  "notes": "free text"
}
```

Required: `schema`, `method`, `budget`, `seed`, `sceneId`, `sceneSha256`, `code`, `wallSeconds`, `units`, `bandCentersHz`, `bandEdgesHz`, `bandsProvided`, `timeBinSeconds`, `timeBins`, `validSeconds`, `receiverOrder`, `receivers`, `files`; `frequenciesHz` when `complex.f32` exists.
`budgetValue` (a number that increases with cost) is required for convergence plots; without it a method's budgets are ordered by `wallSeconds`.
`validSeconds` **(the truncation record)** is the time up to which the histogram is complete: a method that stops at a 1.25 s delay horizon writes 1.25 and zero bins after it; the scorer never computes a time-domain metric past the smaller of the method's and ground truth's `validSeconds`.
`wallSeconds` is the cost of producing this output for all receivers of this scene, excluding one-off compilation; `gpuSeconds` is optional.

## `energy.f32` (required)

- Little-endian float32, C order, shape `[receiver, band, timeBin]` = `[R, 8, 2000]`.
- Bands are octaves centred 125 … 16000 Hz with nominal edges `fc/√2 … fc·√2`.
- Bin `k` holds the energy arriving in `[k, k+1)` ms after the source emits; 2,000 bins (2 s), zero-padded, longer responses truncated (record it in `validSeconds` and `notes`).
- **Normalisation:** a free-field direct path at 1 m puts energy 1 into its bin in every band; at distance `d` it puts `1/d²`.
  For a ray method this is the renderer's energy convention (direct sound at `d` has energy `(ReferenceDistance/d)²` with `ReferenceDistance = 1 m`): each arrival adds its band energy gain to the bin of its arrival time.
- Bands a method (or the ground truth) does not provide are **NaN** in every bin; a receiver a method failed on is NaN in every band and bin.
- Energy is the direction-summed total, including direct sound.

**Band filters and energy from a coherent response.**
A method that has a coherent response derives its band energies exactly as the ground truth does, with the helpers in `tools/method_compare.py`:

- `energy_from_impulse(h, sample_rate)`: discrete impulse responses `[R, N]` (free field at 1 m is a unit sample); bands whose nominal upper edge exceeds Nyquist are NaN, and a band whose upper skirt alone exceeds it is normalised by the part of the filter below Nyquist.
- `energy_from_spectrum(H, df)`: a dense spectrum on a uniform grid `k·df` from 0 Hz (its period `1/df` must exceed 2 s plus 0.1 s).
- `energy_from_arrivals(delays, gains)`: one receiver's discrete arrivals with signed real gains per band; arrivals within a band filter's time resolution add as pressures.
- `levels_from_complex(H, hz)`: on a sparse grid (for example the renderer's 125 Hz steps) only the band total (the `|G_b|²`-weighted mean of `|H|²`) is defined, with no time structure; the scorer uses it for the level metric when a method writes `complex.f32` but NaN energy for a band.

The definition they implement:

1. Band `b` with centre `fc` uses a zero-phase filter `G_b` whose power response is a raised cosine in log frequency between neighbouring centres: `|G_b(f)|² = sin²(π/2 · log2(2f/fc))` on `[fc/2, fc]`, `cos²(π/2 · log2(f/fc))` on `[fc, 2fc]`, zero elsewhere.
   Adjacent bands are power complementary (`Σ_b |G_b|² = 1` between 125 Hz and 16 kHz), so band energies add up to the broadband energy, and the shape mirrors the renderer's late-field synthesis bands.
2. `p_b(t) = IFFT[G_b(f) H(f)]`, the band-limited impulse response on a fine time grid.
3. `E[b, k] = ∫_{bin k} p_b(t)² dt / ∫ g_b(t)² dt`, where `g_b` is the band filter's impulse response and `∫ g_b² dt = 2 ∫_0^∞ |G_b(f)|² df ≈ 1.43 fc`; this makes a free-field `1/d` path integrate to `1/d²`.
   Energy at negative times (pre-ringing of an arrival shortly after emission) is added to bin 0.
   Consequently the band total equals the `|G_b|²`-weighted mean of `|H(f)|²` over the band.

A zero-phase band filter spreads each arrival over a few milliseconds (about ±4 ms at 125 Hz, ±1 ms at 500 Hz).
The scorer therefore convolves histograms that were not produced through a band filter (`"bandLimited": false` in `meta.json`, the default) with the same band's energy kernel `|g_b(t)|² / ∫|g_b|²` before every metric, so both sides have the same time resolution (`method_compare.smooth`); a method writes plain arrival-time histograms and should not smooth them itself.
Methods that derive energy through `energy_from_complex` set `"bandLimited": true` **(addition)**.

## `complex.f32` (optional)

- Little-endian float32, shape `[receiver, frequency, 2]` (real, imaginary), frequencies listed in `meta.json` `frequenciesHz` (ascending, Hz).
- Renderer convention: time dependence `exp(+iωt)`, unit point source, free field `exp(−ikr)/r` (no `1/4π`), source emission at t = 0.
- The scorer compares only frequencies present in both the method's and the ground truth's grid (equal to 1e-6 Hz) and inside a band with ground truth.
- The ground truth writes complex pressure on a 15.625 Hz grid inside every band it provides; the renderer's 8 kHz / 64-point grid (multiples of 125 Hz) is a subset.

## `direction.f32` (optional)

- Little-endian float32, shape `[receiver, band, timeBin, 6]`, the energy of `energy.f32` split into the directions the sound arrives **from**, as seen from the listener, in world axes: `+x, −x, +y, −y, +z, −z` (listener orientation is the identity in this suite).
- The six components sum to `energy.f32` (within float rounding).
- The ground truth does not provide it; the scorer only checks the sum and plots it.

## Writing and checking an output

```sh
python tools/method_compare.py check <root>/<scene-id>/<method>/<budget>/<seed>
```

`tools/method_compare.py` also exposes `write_output(directory, meta, energy, complex=None, direction=None)`, which fills the band and time fields, validates shapes and writes the files.
From C#, write the arrays with `BinaryWriter` in the stated order (it is little-endian on every Unity platform) and `meta.json` with the fields above.

## Method colours

The viewer and every figure use fixed colour-blind-safe colours (Okabe–Ito): `gt` black, `hybrid` blue `#0072B2`, `pressure` vermillion `#D55E00`, `energy` bluish green `#009E73`; other methods take orange, sky blue, reddish purple and yellow in name order.
