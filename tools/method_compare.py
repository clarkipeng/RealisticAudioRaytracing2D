"""Common output format, band energies and scoring for the acoustic method comparison.

Every method (`hybrid`, `pressure`, `energy`, ...) and the wave ground truth
(`gt`) write `<root>/<scene-id>/<method>/<budget>/<seed>/` with `meta.json`,
`energy.f32` [receiver, band, timeBin] and optional `complex.f32` and
`direction.f32`; the exact contract is docs/benchmarks/method-compare/FORMAT.md.
This module writes and checks that format, derives band energies from coherent
responses (impulse responses, dense spectra or delayed arrivals) with the same
band filters as the ground truth, computes the room-acoustic metrics and scores
every method found under a root against `gt`.

    python tools/method_compare.py check <output-dir>
    python tools/method_compare.py score [--root R] [--out D]
    python tools/method_compare.py view  [--root R] [--out D]     (tools/method_compare_view.py)

Pure NumPy (matplotlib only for figures); no GPU or Unity needed.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import sys

import numpy as np

SCHEMA = 'method-compare-v1'
SUITE_SCHEMA = 'method-compare-suite-v1'
SCENE_SCHEMA = 'method-compare-scene-v1'
ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ROOT = ROOT / '.context/method-compare/data'
BANDS = np.array([125., 250., 500., 1000., 2000., 4000., 8000., 16000.])
EDGES = np.stack([BANDS / math.sqrt(2), BANDS * math.sqrt(2)], 1)
BIN_SECONDS = 1e-3
TIME_BINS = 2000
DIRECTIONS = ('+x', '-x', '+y', '-y', '+z', '-z')
NAME = re.compile(r'^[a-z0-9.-]+$')
# Okabe-Ito: colour-blind safe, fixed per method across every figure and page.
METHOD_COLORS = dict(gt='#000000', hybrid='#0072B2', pressure='#D55E00', energy='#009E73', bem='#CC79A7')
EXTRA_COLORS = ['#E69F00', '#56B4E9', '#F0E442', '#999999']

# Metric thresholds (declared with the format; see README "Harness").
ONSET_DB = -12.0          # first bin within 12 dB of the strongest bin (band-filter side lobes reach -20 dB)
ETC_WINDOW = 5            # bins summed per window for the ETC distance
ETC_FLOOR_DB = -60.0      # windows more than 60 dB below the GT peak window are floored
TRUNCATION_BINS = 50      # a decay fit needs the final 50 ms to sit TRUNCATION_MARGIN_DB below its range
TRUNCATION_MARGIN_DB = 10.0
DECAY_FITS = dict(EDT=(0, -10), T20=(-5, -25), T30=(-5, -35))   # EDC fit ranges, dB


def method_color(name, others=()):
    if name in METHOD_COLORS:
        return METHOD_COLORS[name]
    rest = sorted(n for n in others if n not in METHOD_COLORS)
    return EXTRA_COLORS[rest.index(name) % len(EXTRA_COLORS)] if name in rest else EXTRA_COLORS[0]


# --------------------------------------------------------------------------- band filters

def band_power(f, fc):
    """|G_b(f)|^2: raised cosine in log2 frequency between neighbouring octave centres.

    Power complementary: summed over octave centres it is 1 between the lowest
    and highest centre. Zero outside [fc/2, 2 fc].
    """
    f = np.asarray(f, float)
    x = np.log2(np.maximum(f, 1e-30) / fc)
    out = np.where((x >= -1) & (x <= 1), np.cos(np.pi / 2 * x) ** 2, 0.0)
    return np.where(f > 0, out, 0.0)


def band_filter_energy(fc):
    """Energy of the zero-phase band filter's impulse response, 2 * int_0^inf |G|^2 df.

    With f = fc 2^x the integral is int_{-1}^{1} cos^2(pi x/2) 2^x ln2 fc dx
    (about 0.73 fc, slightly more than the nominal octave width 0.707 fc).
    """
    a = math.log(2)
    x = np.linspace(-1, 1, 200001)
    integral = np.trapezoid(np.cos(np.pi / 2 * x) ** 2 * 2.0 ** x * a * fc, x)
    return 2 * integral


_FILTER_ENERGY = {float(fc): band_filter_energy(fc) for fc in BANDS}


def filter_energy(fc):
    return _FILTER_ENERGY.get(float(fc)) or band_filter_energy(fc)


def bin_energy(p, dt, t0=0.0, bins=TIME_BINS, bin_seconds=BIN_SECONDS):
    """Integrate p(t)^2 dt into bins [k, k+1) * bin_seconds; t = t0 + n dt.

    Negative times go into bin 0. Returns (energy [..., bins], energy beyond the last bin).
    """
    p = np.asarray(p, np.float64)
    n = p.shape[-1]
    t = t0 + np.arange(n) * dt
    k = np.clip(np.floor(t / bin_seconds).astype(int), 0, None)
    inside = k < bins
    out = np.zeros(p.shape[:-1] + (bins,))
    sq = p * p * dt
    flat = sq.reshape(-1, n)
    res = out.reshape(-1, bins)
    for r in range(flat.shape[0]):
        res[r] = np.bincount(k[inside], weights=flat[r, inside], minlength=bins)[:bins]
    beyond = sq[..., ~inside].sum(-1)
    return out, beyond


def energy_from_spectrum(H, df, bands=BANDS, t0=0.0, bins=TIME_BINS, negative_seconds=.1):
    """Band energies [R, B, bins] from a spectrum on a uniform rfft grid k*df, k = 0..F-1.

    H is the continuous Fourier transform of the impulse response (free field
    exp(-i w d/c)/d), exp(+iwt) convention, so p_b(t) = IFFT[G_b H]. The time
    period 1/df must exceed the response length plus `negative_seconds`
    (samples in the last `negative_seconds` of the period are read as negative
    times: the zero-phase filters' pre-ringing of an early arrival).
    Returns (energy, beyond) with beyond = energy past the last bin per [R, B].
    """
    H = np.atleast_2d(np.asarray(H))
    F = H.shape[-1]
    n = 2 * (F - 1)
    dt = 1 / (n * df)
    f = np.arange(F) * df
    n_neg = int(round(negative_seconds / dt))
    energy = np.full((H.shape[0], len(bands), bins), np.nan)
    beyond = np.full((H.shape[0], len(bands)), np.nan)
    for b, fc in enumerate(bands):
        if fc * math.sqrt(2) > f[-1]:
            continue      # nominal band above the spectrum's Nyquist frequency: not provided
        g = np.sqrt(band_power(f, fc))
        p = np.fft.irfft(H * g, n) / dt
        p = np.concatenate([p[:, n - n_neg:], p[:, :n - n_neg]], axis=1)
        e, out = bin_energy(p, dt, t0 - n_neg * dt, bins)
        norm = filter_energy(fc)
        if 2 * fc > f[-1]:      # skirt cut at Nyquist: normalise by the part of the filter that exists
            x = np.linspace(-1, math.log2(f[-1] / fc), 20001)
            norm = 2 * np.trapezoid(np.cos(np.pi / 2 * x) ** 2 * 2.0 ** x * math.log(2) * fc, x)
        energy[:, b] = e / norm
        beyond[:, b] = out / norm
    return energy, beyond


def energy_from_impulse(h, sample_rate, bands=BANDS, t0=0.0, bins=TIME_BINS, pad_seconds=.25):
    """Band energies from discrete impulse responses h [R, N] (free field at 1 m = a unit sample).

    Sample n is at t0 + n / sample_rate after emission. Bands whose nominal
    upper edge fc*sqrt(2) exceeds the Nyquist frequency are NaN; when only the
    upper skirt lies above Nyquist, the band is normalised by the part of the
    filter below it, so a free-field path still gives 1/d^2.
    """
    h = np.atleast_2d(np.asarray(h, np.float64))
    n = h.shape[1] + int(pad_seconds * sample_rate)
    n += n % 2
    H = np.fft.rfft(h, n)            # sum h_n exp(-i w n dt): a unit sample is a unit delay
    return energy_from_spectrum(H, sample_rate / n, bands, t0, bins, negative_seconds=pad_seconds / 2)


def energy_from_arrivals(delays, gains, bands=BANDS, bins=TIME_BINS, oversample=16):
    """Coherent band energies from discrete arrivals of one receiver.

    delays [K] seconds after emission; gains [K, B] signed real pressure gains
    per band (free field at distance d: +1/d). Each band's band-limited response
    sum_k gain_kb g_b(t - tau_k) is formed on a grid of oversample*fc samples per
    second (linear-interpolation deposit, amplitude error 1.3% at 2 fc and 0.3%
    at fc for oversample 16), squared and binned. Interfering arrivals inside a
    filter's time resolution therefore add as pressures, not energies.
    """
    delays = np.asarray(delays, float)
    gains = np.asarray(gains, float).reshape(len(delays), -1)
    out = np.full((len(bands), bins), np.nan)
    horizon = bins * BIN_SECONDS + .25
    for b, fc in enumerate(bands):
        if b >= gains.shape[1] or not np.all(np.isfinite(gains[:, b])):
            continue
        rate = oversample * fc
        n = int(math.ceil(horizon * rate))
        n += n % 2
        h = np.zeros(n)
        u = (delays + .125) * rate           # 0.125 s of lead room for pre-ringing
        i0 = np.floor(u).astype(int)
        w = u - i0
        ok = (i0 >= 0) & (i0 + 1 < n)
        np.add.at(h, i0[ok], gains[ok, b] * (1 - w[ok]))
        np.add.at(h, i0[ok] + 1, gains[ok, b] * w[ok])
        e, _ = energy_from_impulse(h[None], rate, [fc], t0=-.125, bins=bins, pad_seconds=.25)
        out[b] = e[0, 0]
    return out


def levels_from_complex(H, hz, bands=BANDS, minimum=3):
    """Band totals (time-integrated energy) from complex pressure on any frequency grid.

    The |G_b|^2-weighted mean of |H|^2 over the frequencies inside the band's
    support; NaN with fewer than `minimum` frequencies there. This is the
    band total a dense grid gives exactly; a sparse grid (e.g. 125 Hz steps)
    gives only a coarse estimate and no time structure.
    """
    H = np.atleast_2d(np.asarray(H))
    hz = np.asarray(hz, float)
    out = np.full((H.shape[0], len(bands)), np.nan)
    for b, fc in enumerate(bands):
        w = band_power(hz, fc)
        if np.count_nonzero(w) >= minimum:
            out[:, b] = (np.abs(H) ** 2 * w).sum(1) / w.sum()
    return out


def energy_kernel(fc, bin_seconds=BIN_SECONDS, half_width=None):
    """Binned energy kernel |g_b(t)|^2 / int |g_b|^2 centred on bin 0 (odd length, sums to ~1)."""
    half_width = half_width or max(3, int(math.ceil(8 / fc / bin_seconds)))
    rate = max(64 * fc, 20 / bin_seconds)
    n = int(2 ** math.ceil(math.log2(4 * half_width * bin_seconds * rate + 16)))
    f = np.fft.rfftfreq(n, 1 / rate)
    g = np.fft.irfft(np.sqrt(band_power(f, fc)), n) * rate
    g = np.roll(g, n // 2)
    t = (np.arange(n) - n // 2) / rate
    k = np.round(t / bin_seconds).astype(int)
    keep = np.abs(k) <= half_width
    kernel = np.bincount(k[keep] + half_width, weights=(g[keep] ** 2) / rate, minlength=2 * half_width + 1)
    return kernel / filter_energy(fc)


def smooth(energy, bands=BANDS):
    """Convolve each band of energy [..., B, T] with that band's energy kernel."""
    out = np.array(energy, float, copy=True)
    for b, fc in enumerate(bands):
        k = energy_kernel(fc)
        h = len(k) // 2
        x = out[..., b, :]
        flat = x.reshape(-1, x.shape[-1])
        for r in range(flat.shape[0]):
            if np.all(np.isfinite(flat[r])):
                full = np.convolve(flat[r], k, mode='full')
                lead = full[:h].sum()             # spread before emission goes to bin 0, as for band-filtered data
                flat[r] = full[h:h + flat.shape[1]]
                flat[r, 0] += lead
        out[..., b, :] = flat.reshape(x.shape)
    return out


# --------------------------------------------------------------------------- format I/O

def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def sha256_file(path):
    return sha256_bytes(Path(path).read_bytes())


def write_output(directory, meta, energy, complex_pressure=None, direction=None):
    """Validate and write one output directory (atomically via a .partial sibling).

    `meta` needs method, budget, seed, sceneId, sceneSha256, code, wallSeconds,
    validSeconds; band/time/unit fields are filled in. energy [R, 8, 2000];
    complex_pressure [R, F] complex with meta['frequenciesHz']; direction [R, 8, 2000, 6].
    """
    directory = Path(directory)
    energy = np.asarray(energy, np.float32)
    R = energy.shape[0]
    if energy.shape != (R, len(BANDS), TIME_BINS):
        raise ValueError(f'energy must be [R, {len(BANDS)}, {TIME_BINS}], got {energy.shape}')
    meta = dict(meta)
    meta.update(schema=SCHEMA, bandCentersHz=BANDS.tolist(), bandEdgesHz=np.round(EDGES, 4).tolist(),
                timeBinSeconds=BIN_SECONDS, timeBins=TIME_BINS, receiverOrder='scene.json points', receivers=R)
    meta.setdefault('units', dict(energy='free-field 1 m direct sound = 1 per band', time='s',
                                  complex='free-field exp(-ikr)/r, exp(+iwt)'))
    meta.setdefault('bandsProvided', [float(fc) for b, fc in enumerate(BANDS) if np.isfinite(energy[:, b]).any()])
    meta.setdefault('bandLimited', False)
    files = {'energy.f32': list(energy.shape)}
    payload = {'energy.f32': energy}
    if complex_pressure is not None:
        c = np.asarray(complex_pressure)
        hz = meta.get('frequenciesHz')
        if hz is None or c.shape != (R, len(hz)):
            raise ValueError('complex pressure must be [R, len(frequenciesHz)]')
        arr = np.stack([c.real, c.imag], -1).astype(np.float32)
        files['complex.f32'] = list(arr.shape)
        payload['complex.f32'] = arr
    if direction is not None:
        d = np.asarray(direction, np.float32)
        if d.shape != energy.shape + (6,):
            raise ValueError('direction must be [R, 8, 2000, 6]')
        files['direction.f32'] = list(d.shape)
        payload['direction.f32'] = d
    meta['files'] = files
    problems = check_meta(meta)
    if problems:
        raise ValueError('; '.join(problems))
    partial = directory.with_name(directory.name + '.partial')
    partial.mkdir(parents=True, exist_ok=True)
    for name, arr in payload.items():
        arr.astype('<f4').tofile(partial / name)
    (partial / 'meta.json').write_text(json.dumps(meta, indent=1, allow_nan=False) + '\n')
    if directory.exists():
        for p in directory.iterdir():
            p.unlink()
        directory.rmdir()
    partial.rename(directory)
    return meta


REQUIRED = ('schema', 'method', 'budget', 'seed', 'sceneId', 'sceneSha256', 'code', 'wallSeconds', 'units',
            'bandCentersHz', 'bandEdgesHz', 'bandsProvided', 'timeBinSeconds', 'timeBins', 'validSeconds',
            'receiverOrder', 'receivers', 'files')


def check_meta(meta):
    problems = [f'missing {k}' for k in REQUIRED if k not in meta]
    if problems:
        return problems
    if meta['schema'] != SCHEMA:
        problems.append(f'schema {meta["schema"]}')
    for key in ('method', 'budget'):
        if not NAME.match(str(meta[key])):
            problems.append(f'{key} {meta[key]!r} is not [a-z0-9.-]+')
    if not isinstance(meta['seed'], int) or meta['seed'] < 0:
        problems.append('seed must be a non-negative integer')
    if meta['timeBinSeconds'] != BIN_SECONDS or meta['timeBins'] != TIME_BINS:
        problems.append('time bins must be 1 ms x 2000')
    if list(map(float, meta['bandCentersHz'])) != BANDS.tolist():
        problems.append('band centres differ')
    if 'complex.f32' in meta['files'] and 'frequenciesHz' not in meta:
        problems.append('complex.f32 needs frequenciesHz')
    if not 0 < float(meta['validSeconds']) <= TIME_BINS * BIN_SECONDS + 1e-9:
        problems.append('validSeconds must lie in (0, 2]')
    return problems


def read_output(directory):
    """meta, energy [R,8,T], complex [R,F] or None, direction or None."""
    directory = Path(directory)
    meta = json.loads((directory / 'meta.json').read_text())
    R = meta['receivers']
    energy = np.fromfile(directory / 'energy.f32', '<f4').reshape(R, len(BANDS), TIME_BINS)
    complex_pressure = direction = None
    if (directory / 'complex.f32').exists():
        c = np.fromfile(directory / 'complex.f32', '<f4').reshape(R, -1, 2)
        complex_pressure = c[..., 0].astype(np.float64) + 1j * c[..., 1]
    if (directory / 'direction.f32').exists():
        direction = np.fromfile(directory / 'direction.f32', '<f4').reshape(R, len(BANDS), TIME_BINS, 6)
    return meta, energy, complex_pressure, direction


def check_output(directory, scene=None):
    """List of problems with one output directory (empty when valid)."""
    directory = Path(directory)
    try:
        meta = json.loads((directory / 'meta.json').read_text())
    except (OSError, ValueError) as e:
        return [f'meta.json: {e}']
    problems = check_meta(meta)
    if problems:
        return problems
    R = meta['receivers']
    for name, shape in meta['files'].items():
        path = directory / name
        expect = int(np.prod(shape)) * 4
        if not path.exists():
            problems.append(f'{name} missing')
        elif path.stat().st_size != expect:
            problems.append(f'{name} has {path.stat().st_size} bytes, expected {expect}')
    if problems:
        return problems
    meta, energy, cp, direction = read_output(directory)
    if 'complex.f32' in meta['files'] and cp.shape[1] != len(meta['frequenciesHz']):
        problems.append('complex.f32 frequency count differs from frequenciesHz')
    finite = np.isfinite(energy)
    if np.any(energy[finite] < 0):
        problems.append('negative energy')
    for b, fc in enumerate(BANDS):
        provided = fc in [float(v) for v in meta['bandsProvided']]
        band = energy[:, b]
        if not provided and np.isfinite(band).any():
            problems.append(f'{fc:g} Hz not in bandsProvided but finite')
        partial = np.isfinite(band).any(1) & ~np.isfinite(band).all(1)
        if np.any(partial):
            problems.append(f'{fc:g} Hz has receivers with partly NaN histograms')
    valid_bins = int(round(meta['validSeconds'] / BIN_SECONDS))
    tail = energy[:, :, valid_bins:]
    if tail.size and np.nanmax(np.abs(np.nan_to_num(tail)), initial=0) > 0:
        problems.append('nonzero energy after validSeconds')
    if direction is not None:
        total = np.nansum(direction, -1)
        ok = np.isfinite(energy)
        scale = np.nanmax(energy) if np.isfinite(energy).any() else 1
        if np.nanmax(np.abs(total[ok] - energy[ok]), initial=0) > 1e-4 * scale:
            problems.append('direction.f32 does not sum to energy.f32')
    if scene is not None:
        scene_path = Path(scene)
        if len(json.loads(scene_path.read_text())['points']) // 3 != R:
            problems.append('receiver count differs from scene.json')
        if meta['sceneSha256'] != sha256_file(scene_path):
            problems.append('sceneSha256 differs from scene.json')
    return problems


def discover(root):
    """{scene: {method: {budget: {seed: path}}}} for every output with a meta.json."""
    root = Path(root)
    out = {}
    for meta in sorted(root.glob('*/*/*/*/meta.json')):
        seed_dir = meta.parent
        scene, method, budget, seed = seed_dir.relative_to(root).parts
        if not seed.isdigit():
            continue
        out.setdefault(scene, {}).setdefault(method, {}).setdefault(budget, {})[int(seed)] = seed_dir
    return out


def load_scene(root, scene_id):
    return json.loads((Path(root) / scene_id / 'scene.json').read_text())


# --------------------------------------------------------------------------- metrics

def schroeder(e):
    """Energy decay curve in dB (normalised to 0 at t = 0) from a histogram [..., T]."""
    rev = np.cumsum(e[..., ::-1], -1)[..., ::-1]
    with np.errstate(divide='ignore', invalid='ignore'):
        return 10 * np.log10(rev / rev[..., :1])


def decay_time(edc, start, stop, bin_seconds=BIN_SECONDS):
    """60 dB decay time (s) from a least-squares line through the EDC between start and stop dB.

    NaN when the EDC does not reach `stop` within its valid length (ISO 3382:
    EDT 0..-10, T20 -5..-25, T30 -5..-35 dB, each extrapolated to 60 dB), or
    when truncation could bend it: the last TRUNCATION_BINS of the histogram
    must hold at most `stop - TRUNCATION_MARGIN_DB` of the energy (the EDC there),
    so the missing tail cannot steepen the fitted range.
    """
    tail = len(edc) - TRUNCATION_BINS
    if tail <= 0 or not np.isfinite(edc[tail]) and edc[tail] > -np.inf or edc[tail] > stop - TRUNCATION_MARGIN_DB:
        return np.nan
    t = (np.arange(len(edc)) + .5) * bin_seconds
    if not np.isfinite(edc).any() or np.nanmin(edc) > stop:
        return np.nan
    first = np.argmax(edc <= start) if start < 0 else 0
    last = np.argmax(edc <= stop)
    sel = slice(first, last + 1)
    if last - first < 3:
        return np.nan
    slope = np.polyfit(t[sel], edc[sel], 1)[0]
    return -60.0 / slope if slope < 0 else np.nan


def onset_bin(e, threshold_db=ONSET_DB):
    """First bin within threshold_db of the strongest bin (NaN for an empty histogram)."""
    if not np.isfinite(e).all() or e.max() <= 0:
        return np.nan
    return float(np.argmax(e >= e.max() * 10 ** (threshold_db / 10)))


def receiver_metrics(e, valid_bins, t0_bin):
    """Metrics of one histogram e [T] limited to its first valid_bins bins.

    t0_bin is the onset used for clarity/definition (ISO 3382 counts from the
    direct sound; here the ground truth's onset is used for both sides, so an
    arrival-time error is reported separately and not mixed into C50/C80).
    """
    e = np.asarray(e[:valid_bins], float)
    total = e.sum()
    out = dict(level=10 * math.log10(total) if total > 0 else -np.inf, onset=onset_bin(e))
    edc = schroeder(e[int(t0_bin):]) if total > 0 and np.isfinite(t0_bin) else np.full(1, np.nan)
    for name, (start, stop) in DECAY_FITS.items():
        out[name] = decay_time(edc, start, stop)
    if np.isfinite(t0_bin) and total > 0:
        k0 = int(t0_bin)
        early50, early80 = e[k0:k0 + 50].sum(), e[k0:k0 + 80].sum()
        late = e[k0:].sum()
        out['C50'] = 10 * math.log10(early50 / max(late - early50, 1e-300)) if early50 > 0 else np.nan
        out['C80'] = 10 * math.log10(early80 / max(late - early80, 1e-300)) if early80 > 0 else np.nan
        out['D50'] = early50 / late if late > 0 else np.nan
    else:
        out.update(C50=np.nan, C80=np.nan, D50=np.nan)
    return out


def etc_distance(e_method, e_gt, valid_bins, window=ETC_WINDOW, floor_db=ETC_FLOOR_DB):
    """RMS dB difference of the windowed energy-time curves (log-spectral distance of the envelope)."""
    n = valid_bins // window * window
    if n == 0:
        return np.nan
    a = np.asarray(e_method[:n], float).reshape(-1, window).sum(1)
    b = np.asarray(e_gt[:n], float).reshape(-1, window).sum(1)
    if b.max() <= 0:
        return np.nan
    floor = b.max() * 10 ** (floor_db / 10)
    keep = (b > floor) | (a > floor)
    if not keep.any():
        return np.nan
    d = 10 * np.log10(np.maximum(a[keep], floor) / np.maximum(b[keep], floor))
    return float(np.sqrt(np.mean(d * d)))


METRICS = ('level', 'onset', 'EDT', 'T20', 'T30', 'C50', 'C80', 'D50', 'etc')
RELATIVE = ('EDT', 'T20', 'T30')        # reported as relative error (%)


def compare_output(scene, gt, out, smooth_method=None):
    """Per receiver x band metric errors of one output against the ground truth.

    gt/out are (meta, energy, complex, direction). Returns dict of arrays
    [R, B] (NaN where undefined) plus complex errors per band.
    """
    gmeta, ge, gc, _ = gt
    mmeta, me, mc, _ = out
    R = ge.shape[0]
    if me.shape[0] != R:
        raise ValueError(f'{mmeta["method"]}: {me.shape[0]} receivers, ground truth has {R}')
    gt_bands = [b for b, fc in enumerate(BANDS) if float(fc) in map(float, gmeta['bandsProvided'])]
    if smooth_method is None:
        smooth_method = not mmeta.get('bandLimited', False)
    e = smooth(me) if smooth_method else me
    valid = int(round(min(gmeta['validSeconds'], mmeta['validSeconds']) / BIN_SECONDS))
    res = {f'{k}_gt': np.full((R, len(BANDS)), np.nan) for k in METRICS[:-1]}
    res.update({f'{k}_m': np.full((R, len(BANDS)), np.nan) for k in METRICS[:-1]})
    res['etc'] = np.full((R, len(BANDS)), np.nan)
    complex_levels = None
    if mc is not None:
        complex_levels = levels_from_complex(mc, mmeta['frequenciesHz'])
    for b in gt_bands:
        for r in range(R):
            g = ge[r, b]
            t0 = onset_bin(g[:valid])
            mg = receiver_metrics(g, valid, t0)
            for k, v in mg.items():
                res[f'{k}_gt'][r, b] = v
            if np.all(np.isfinite(e[r, b])):
                mm = receiver_metrics(e[r, b], valid, t0)
                for k, v in mm.items():
                    res[f'{k}_m'][r, b] = v
                res['etc'][r, b] = etc_distance(e[r, b], g, valid)
            elif complex_levels is not None and np.isfinite(complex_levels[r, b]) and complex_levels[r, b] > 0:
                res['level_m'][r, b] = 10 * math.log10(complex_levels[r, b])
    err = {}
    for k in METRICS[:-1]:
        a, g = res[f'{k}_m'], res[f'{k}_gt']
        with np.errstate(invalid='ignore', divide='ignore'):
            err[k] = 100 * (a - g) / g if k in RELATIVE else a - g
    err['onset'] = err['onset'] * BIN_SECONDS * 1e3        # ms
    err['etc'] = res['etc']
    err['complex'] = complex_errors(gmeta, gc, mmeta, mc)
    err['raw'] = res
    return err


def complex_errors(gmeta, gc, mmeta, mc):
    """Per band: dict(all, view mask applied later) of complex relative errors per receiver.

    Returns arrays [R, B] of |m - gt|^2 and |gt|^2 summed over the shared
    frequencies in each band, so groups can be pooled before the square root.
    """
    if gc is None or mc is None:
        return None
    ghz = np.asarray(gmeta['frequenciesHz'], float)
    mhz = np.asarray(mmeta['frequenciesHz'], float)
    R = gc.shape[0]
    num = np.full((R, len(BANDS)), np.nan)
    den = np.full((R, len(BANDS)), np.nan)
    for b, (lo, hi) in enumerate(EDGES):
        pairs = [(i, int(np.argmin(np.abs(mhz - f)))) for i, f in enumerate(ghz)
                 if lo <= f < hi and np.min(np.abs(mhz - f)) < 1e-6]
        if not pairs:
            continue
        gi, mi = map(list, zip(*pairs))
        d = mc[:, mi] - gc[:, gi]
        if not np.isfinite(d).all():
            continue
        num[:, b] = (np.abs(d) ** 2).sum(1)
        den[:, b] = (np.abs(gc[:, gi]) ** 2).sum(1)
    return dict(num=num, den=den)


def seed_spread(energies, valid_bins):
    """Seed spread of one budget: level SD (dB) [R,B] and energy-weighted relative histogram SD [R,B]."""
    e = np.asarray(energies, float)[..., :valid_bins]           # [S, R, B, T]
    if e.shape[0] < 2:
        return None
    totals = e.sum(-1)
    with np.errstate(divide='ignore', invalid='ignore'):
        level_sd = np.std(10 * np.log10(totals), axis=0, ddof=1)
        mean = e.mean(0)
        sd = e.std(0, ddof=1)
        rel = sd.sum(-1) / mean.sum(-1)
    return dict(level=level_sd, histogram=rel)


# --------------------------------------------------------------------------- scoring

GROUPS = ('all', 'view', 'shadow')


def pooled(values, mask, kind):
    v = values[mask]
    v = v[np.isfinite(v)]
    if v.size == 0:
        return None
    if kind == 'rms':
        return float(np.sqrt(np.mean(v * v)))
    if kind == 'mean':
        return float(np.mean(v))
    if kind == 'median':
        return float(np.median(np.abs(v)))
    raise ValueError(kind)


def score(root, out_dir=None, methods=None):
    """Score every method under root against gt; returns the report dict and writes it."""
    root = Path(root)
    found = discover(root)
    suite_path = root / 'suite.json'
    suite = json.loads(suite_path.read_text()) if suite_path.exists() else \
        dict(scenes=[dict(id=s) for s in found])
    rows, spreads, missing = [], [], []
    for entry in suite['scenes']:
        sid = entry['id']
        runs = found.get(sid, {})
        if 'gt' not in runs or 'ref' not in runs['gt'] or 0 not in runs['gt']['ref']:
            missing.append(dict(scene=sid, method='gt'))
            continue
        scene = load_scene(root, sid)
        gt = read_output(runs['gt']['ref'][0])
        shadow = np.asarray(scene.get('shadow', np.zeros(gt[1].shape[0], int)), bool)
        info = dict(scene=sid, family=scene.get('family', entry.get('family')), variant=scene.get('variant', ''),
                    closed=bool(scene.get('closed', False)))
        for method, budgets in sorted(runs.items()):
            if methods and method not in methods:
                continue
            for budget, seeds in sorted(budgets.items()):
                energies = []                       # gt/ref/0 scores itself: the zero-error self-test row
                for seed, path in sorted(seeds.items()):
                    out = read_output(path)
                    problems = check_output(path)
                    err = compare_output(scene, gt, out)
                    energies.append(out[1])
                    meta = out[0]
                    base = dict(info, method=method, budget=budget, seed=seed,
                                budgetValue=meta.get('budgetValue'), wallSeconds=meta.get('wallSeconds'),
                                problems=problems)
                    rows.extend(summarise(base, err, shadow))
                valid = int(round(gt[0]['validSeconds'] / BIN_SECONDS))
                spread = seed_spread(energies, valid)
                if spread is not None:
                    for b, fc in enumerate(BANDS):
                        for group in GROUPS:
                            mask = group_mask(group, shadow)
                            lv = pooled(spread['level'][:, b], mask, 'rms')
                            hv = pooled(spread['histogram'][:, b], mask, 'rms')
                            if lv is None and hv is None:
                                continue
                            spreads.append(dict(info, method=method, budget=budget, band=float(fc), group=group,
                                                seeds=len(energies), levelSD=lv, histogramSD=hv,
                                                budgetValue=meta.get('budgetValue')))
    report = dict(schema='method-compare-report-v1', root=str(root), rows=rows, spreads=spreads, missing=missing,
                  metrics=dict(level='dB (method - gt), RMS over receivers', onset='ms', EDT='%', T20='%', T30='%',
                               C50='dB', C80='dB', D50='absolute', etc='dB RMS', complex='relative L2'))
    if out_dir:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / 'report.json').write_text(json.dumps(report, indent=1, allow_nan=False, default=_json) + '\n')
        (out_dir / 'summary.md').write_text(summary_markdown(report))
    return report


def _json(v):
    if isinstance(v, (np.floating, np.integer)):
        return v.item()
    if isinstance(v, Path):
        return str(v)
    raise TypeError(type(v))


def group_mask(group, shadow):
    if group == 'all':
        return np.ones(len(shadow), bool)
    return shadow if group == 'shadow' else ~shadow


def summarise(base, err, shadow):
    """One row per (band, group) with pooled metric errors."""
    rows = []
    for b, fc in enumerate(BANDS):
        if not np.isfinite(err['raw']['level_gt'][:, b]).any():
            continue
        for group in GROUPS:
            mask = group_mask(group, shadow)
            if not mask.any():
                continue
            row = dict(base, band=float(fc), group=group, receivers=int(mask.sum()))
            row['levelBias'] = pooled(err['level'][:, b], mask, 'mean')
            for k in ('level', 'onset', 'EDT', 'T20', 'T30', 'C50', 'C80', 'D50', 'etc'):
                row[k] = pooled(err[k][:, b], mask, 'rms')
            row['coverage'] = float(np.isfinite(err['level'][mask, b]).mean())
            if err['complex'] is not None:
                num, den = err['complex']['num'][mask, b], err['complex']['den'][mask, b]
                ok = np.isfinite(num) & np.isfinite(den)
                row['complex'] = float(math.sqrt(num[ok].sum() / den[ok].sum())) if ok.any() and den[ok].sum() > 0 else None
                row['complexNum'] = float(num[ok].sum()) if ok.any() else None
                row['complexDen'] = float(den[ok].sum()) if ok.any() else None
            rows.append(row)
    return rows


def aggregate(rows, keys, metric):
    """Pool RMS errors over rows grouped by keys (receiver-weighted)."""
    groups = {}
    for r in rows:
        v = r.get(metric)
        if v is None:
            continue
        k = tuple(r[key] for key in keys)
        n = r['receivers'] * r.get('coverage', 1.0)
        acc = groups.setdefault(k, [0.0, 0.0])
        if metric == 'complex':
            if r.get('complexNum') is None:
                continue
            acc[0] += r['complexNum']
            acc[1] += r['complexDen']
        elif metric == 'levelBias':
            acc[0] += v * n
            acc[1] += n
        else:
            acc[0] += v * v * n
            acc[1] += n
    out = {}
    for k, (s, n) in groups.items():
        if n <= 0:
            continue
        if metric == 'levelBias':
            out[k] = s / n
        else:
            out[k] = math.sqrt(s / n)
    return out


def top_budget(rows, method):
    """The method's largest budget (by budgetValue, else wall time)."""
    cands = {}
    for r in rows:
        if r['method'] == method:
            cands.setdefault(r['budget'], []).append(r)
    def key(b):
        rs = cands[b]
        v = [r['budgetValue'] for r in rs if r['budgetValue'] is not None]
        w = [r['wallSeconds'] for r in rs if r['wallSeconds'] is not None]
        return (max(v) if v else -1, np.mean(w) if w else 0)
    return max(cands, key=key) if cands else None


def summary_markdown(report):
    """One table per method (its largest budget), plus a convergence table per method."""
    rows = report['rows']
    lines = ['# Method comparison summary', '',
             'Errors are method minus ground truth, RMS over receivers (pooled over scenes and seeds); '
             'level in dB, decay times in %, onset in ms, ETC distance in dB RMS, complex as relative L2.', '']
    for method in sorted({r['method'] for r in rows}):
        budget = top_budget(rows, method)
        sel = [r for r in rows if r['method'] == method and r['budget'] == budget]
        cost = [r['wallSeconds'] for r in sel if r['wallSeconds'] is not None]
        lines += [f'## {method} (budget {budget}, mean wall time per scene {np.mean(cost):.3g} s)' if cost else
                  f'## {method} (budget {budget})', '']
        head = '| band | scenes | group | level dB | bias dB | onset ms | EDT % | T20 % | T30 % | C50 dB | C80 dB | D50 | ETC dB | complex |'
        lines += [head, '|---' * (head.count('|') - 1) + '|']
        for band in sorted({r['band'] for r in sel}):
            for openness in ('open', 'closed'):
                for group in GROUPS:
                    rs = [r for r in sel if r['band'] == band and r['group'] == group and r['closed'] == (openness == 'closed')]
                    if not rs:
                        continue
                    cell = lambda m, f='{:.2f}': (lambda v: f.format(v) if v is not None else '–')(
                        aggregate(rs, ('method',), m).get((method,)))
                    lines.append(f'| {band:g} | {openness} | {group} | {cell("level")} | {cell("levelBias")} | '
                                 f'{cell("onset", "{:.1f}")} | {cell("EDT", "{:.1f}")} | {cell("T20", "{:.1f}")} | '
                                 f'{cell("T30", "{:.1f}")} | {cell("C50")} | {cell("C80")} | {cell("D50", "{:.3f}")} | '
                                 f'{cell("etc")} | {cell("complex", "{:.1%}")} |')
        lines.append('')
        conv = aggregate([r for r in rows if r['method'] == method], ('budget', 'group'), 'level')
        if len({k[0] for k in conv}) > 1:
            lines += ['Level error (dB RMS, all bands with ground truth) by budget:', '',
                      '| budget | all | view | shadow |', '|---|---|---|---|']
            for b in sorted({k[0] for k in conv}):
                lines.append(f'| {b} | ' + ' | '.join(f'{conv[(b, g)]:.2f}' if (b, g) in conv else '–' for g in GROUPS) + ' |')
            lines.append('')
    if report['missing']:
        lines += ['Scenes without ground truth: ' + ', '.join(m['scene'] for m in report['missing']), '']
    return '\n'.join(lines) + '\n'


# --------------------------------------------------------------------------- CLI

def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='command', required=True)
    c = sub.add_parser('check', help='Validate output directories')
    c.add_argument('dirs', nargs='+')
    s = sub.add_parser('score', help='Score every method under a root against gt')
    s.add_argument('--root', default=str(DEFAULT_ROOT))
    s.add_argument('--out', default=str(DEFAULT_ROOT.parent / 'score'))
    s.add_argument('--methods', nargs='*')
    v = sub.add_parser('view', help='Score, then build the static HTML viewer and figures')
    v.add_argument('--root', default=str(DEFAULT_ROOT))
    v.add_argument('--out', default=str(DEFAULT_ROOT.parent / 'view'))
    v.add_argument('--figures', help='Also export key PNG figures to this directory')
    a = p.parse_args(argv)
    if a.command == 'check':
        bad = 0
        for d in a.dirs:
            d = Path(d)
            scene = d.parents[2] / 'scene.json' if len(d.parents) > 2 else None
            problems = check_output(d, scene if scene and scene.exists() else None)
            print(f'{d}: ' + ('ok' if not problems else '; '.join(problems)))
            bad += bool(problems)
        return 1 if bad else 0
    if a.command == 'score':
        report = score(a.root, a.out, a.methods)
        print((Path(a.out) / 'summary.md').read_text())
        return 0
    if a.command == 'view':
        import method_compare_view as mv
        mv.build(a.root, a.out, a.figures)
        return 0


if __name__ == '__main__':
    sys.exit(main())
