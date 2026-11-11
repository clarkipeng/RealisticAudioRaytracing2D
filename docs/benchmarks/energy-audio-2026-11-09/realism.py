"""Realism reference: measured impulse responses of real stone cathedrals against the
energy method's rendered impulse responses of the Church, and dry recordings through both.

    PY=.context/fdtd-teacher/venv/bin/python
    B=docs/benchmarks/energy-audio-2026-11-09
    nice -n 19 $PY $B/realism.py takes     # guarded Unity: static impulse takes of the Church (doors closed)
    nice -n 19 $PY $B/realism.py render    # guarded Unity: energy-method renders of those takes (one job at a time)
    $PY $B/realism.py page                 # clips, decay/spectrum figures and index.html in .context/energy-audio/realism/

Downloads (kept in .context/energy-audio/realism/downloads, not in the repository), all
from the OpenAIR library (www.openairlib.net, Audiolab, University of York), CC BY 4.0:
  York Minster (Damian T. Murphy), stereo ORTF IR, source under the tower, receiver in
  the nave centre, 23.69 m; 140,000 m^3.
  St Paul's Cathedral (2022), mono IRs: P3 source and receiver under the dome, 9.48 m;
  P1 source under the dome, receiver in the nave, 31.64 m.
  Anechoic: speech-macbeth "Speech - Core Take" (spoken word), handel-trumpet
  tr-1888d-piece3-st (natural trumpet piece), operatic-voice singing.
The Church renders are the renderer's own binaural impulse responses (StaticImpulseTake:
every pose frozen at the walk's first frame, doors closed, listener fixed, the organ
source playing one unit sample), so a dry recording convolved with them is what the
renderer plays for a static listener at that point.
"""
import json
import math
from pathlib import Path
import sys
import wave

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(Path(__file__).parent))
import listen  # noqa: E402

OUT = ROOT / '.context/energy-audio/realism'
DOWNLOADS = OUT / 'downloads'
RATE = 48000
# Listener points (doors closed, take frame 0): about 10 m from the organ in the crossing,
# and the walk's farthest point inside, at its start by the doors.
POINTS = {'near': (0.0, 2.65, 8.0), 'far': (0.0, 2.65, -30.0)}
ORGAN = np.array([0.0, 5.0, 18.0])
REAL = {
    'stpauls-9m': dict(file='stpauls-P3-mono.wav', title="St Paul's Cathedral, under the dome", distance=9.48),
    'york-24m': dict(file='york-minster-ortf.wav', title='York Minster, nave centre', distance=23.69),
    'stpauls-32m': dict(file='stpauls-P1-mono.wav', title="St Paul's Cathedral, nave", distance=31.64),
}
DRY = {'speech': ('speech-core.wav', 0, 14.0), 'trumpet': ('trumpet-piece3.wav', 0, 14.0)}
BANDS = [125, 250, 500, 1000, 2000, 4000, 8000]


def ours():
    """Rendered Church IRs: name -> (title, render directory, distance)."""
    d = {f'church-{k}': (f'Church (energy method), {name}', OUT / 'renders' / k, float(np.linalg.norm(np.array(POINTS[k]) - ORGAN)))
         for k, name in (('near', 'crossing'), ('far', 'nave by the doors'))}
    # The previous render of the same point (before the transmission, air-absorption and tail fixes of 10 November).
    d['church-near-before'] = ('Church (energy method), crossing, before (old air table and tail fit)', OUT / 'renders-before-transmission' / 'near',
                               d['church-near'][2])
    return d


def guarded(name, arguments, work_fraction=.5):
    sys.path.insert(0, str(ROOT / 'tools'))
    import method_runs as mr
    logs, resources = mr.launch(name, arguments, work_fraction=work_fraction)
    if resources['exitCode'] != 0:
        raise SystemExit(f'{name} failed; see {logs}/unity.log')
    return resources


def takes():
    outputs = [OUT / 'takes' / k for k in POINTS]
    for o in outputs:
        if o.exists():
            raise SystemExit(f'{o} exists; remove it to derive again.')
    outputs[0].parent.mkdir(parents=True, exist_ok=True)
    guarded('realism-takes', ['-quit', '-executeMethod', 'StaticImpulseTake.Derive', '-impulseTake', str(listen.TAKE),
                              '-impulseOutput', ';'.join(map(str, outputs)), '-impulseSample', '0',
                              '-impulseListener', ';'.join(','.join(map(str, POINTS[k])) for k in POINTS), '-impulseSeconds', '0.05'])


def settings(before=False):
    """The listening render's energy settings (late field continued to 16 s, accumulated over
    solves). The scene is static, so the response is updated every 16,384 samples instead of
    every 1,024 (same late partitions), and the accumulation's time constant is 4 s so it
    still averages about 20 solves (the walk's 1 s at 1,024 samples averages about 86).
    `before`: the settings of the earlier renders (2 s horizon, one fixed-seed walk)."""
    s = dict(listen.METHODS['energy'], responseStepSamples=16384, tailSamples=int(17.5 * 44100), lateAccumulationSeconds=4.0)
    if before:
        s.pop('lateTailSeconds'); s.pop('lateAccumulationSeconds')
    return s


def render(names=('near', 'far', 'near-before')):
    for name in names:
        output = OUT / 'renders' / name
        if (output / 'run.json').exists():
            print(f'{output} exists; skipped')
            continue
        take = OUT / 'takes' / name.split('-')[0]
        job = dict(take=str(take), output=str(output), shader='Assets/Shaders/Transport.compute', settings=settings(name.endswith('before')), workFraction=.5)
        path = OUT / 'renders' / f'{name}.job.json'
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(job, indent=1) + '\n')
        resources = guarded(f'realism-{name}', ['-executeMethod', 'TakeBatch.Render', '-takeJob', str(path)])
        (output / 'resources.json').write_text(json.dumps(resources, indent=1) + '\n')


# ------------------------------------------------------------------ analysis

def read(path):
    """Any PCM/float WAV as float64 [samples, channels] and its rate."""
    data = Path(path).read_bytes()
    at, fmt = 12, None
    while at < len(data):
        tag, size = data[at:at + 4], int.from_bytes(data[at + 4:at + 8], 'little')
        if tag == b'fmt ':
            code, channels, rate = int.from_bytes(data[at + 8:at + 10], 'little'), int.from_bytes(data[at + 10:at + 12], 'little'), int.from_bytes(data[at + 12:at + 16], 'little')
            bits = int.from_bytes(data[at + 22:at + 24], 'little')
            if code == 0xFFFE:
                code = int.from_bytes(data[at + 32:at + 34], 'little')
            fmt = (code, channels, rate, bits)
        if tag == b'data':
            raw = data[at + 8:at + 8 + size]
            code, channels, rate, bits = fmt
            if code == 3:
                x = np.frombuffer(raw, '<f4' if bits == 32 else '<f8').astype(float)
            elif bits == 16:
                x = np.frombuffer(raw, '<i2') / 32768.0
            elif bits == 24:
                b = np.frombuffer(raw[:len(raw) // 3 * 3], np.uint8).reshape(-1, 3).astype(np.int32)
                v = b[:, 0] | (b[:, 1] << 8) | (b[:, 2] << 16)
                x = np.where(v >= 1 << 23, v - (1 << 24), v) / float(1 << 23)
            else:
                x = np.frombuffer(raw, '<i4') / 2147483648.0
            return x.reshape(-1, channels), rate
        at += 8 + size + (size & 1)
    raise ValueError(path)


def resample(x, rate):
    from scipy.signal import resample_poly
    if rate == RATE:
        return x
    g = math.gcd(RATE, rate)
    return resample_poly(x, RATE // g, rate // g, axis=0)


def write16(path, x):
    with wave.open(str(path), 'wb') as w:
        w.setnchannels(x.shape[1]); w.setsampwidth(2); w.setframerate(RATE)
        w.writeframes((np.clip(x, -1, 1) * 32767).astype('<i2').tobytes())


def onset(h):
    """Direct-sound onset: first sample within 20 dB of the peak (channels summed in energy)."""
    e = (h ** 2).sum(1)
    return int(np.argmax(e > e.max() * 1e-2))


def octave(x, f):
    from scipy.signal import butter, sosfiltfilt
    sos = butter(3, [f / math.sqrt(2), min(f * math.sqrt(2), .45 * RATE)], 'bandpass', fs=RATE, output='sos')
    return sosfiltfilt(sos, x, axis=0)


def edc(e):
    """Schroeder curve (dB, 0 at the start) of an energy envelope, truncated where the
    1/8 s smoothed envelope reaches 5 dB above the noise of the last tenth (with that
    noise subtracted)."""
    n = len(e)
    tail = e[int(.9 * n):]
    noise = tail.mean() if len(tail) else 0
    w = max(1, RATE // 8)
    smooth = np.convolve(e, np.ones(w) / w, 'same')
    above = np.nonzero(smooth > 3.16 * noise)[0]
    end = int(above[-1]) if len(above) and noise > 0 else n
    c = np.cumsum((np.maximum(e[:end] - noise, 0))[::-1])[::-1]
    return 10 * np.log10(np.maximum(c / c[0], 1e-12))


def decay_time(curve, top, bottom):
    t = np.arange(len(curve)) / RATE
    m = (curve <= top) & (curve >= bottom)
    if m.sum() < 10:
        return float('nan')
    slope = np.polyfit(t[m], curve[m], 1)[0]
    return -60 / slope


def metrics(h):
    """Per octave band: T30, EDT, C80, D50 from the onset-aligned IR (energy summed over channels)."""
    out = {}
    for f in BANDS:
        e = (octave(h, f) ** 2).sum(1)
        c = edc(e)
        k80, k50 = int(.08 * RATE), int(.05 * RATE)
        out[f] = dict(T30=decay_time(c, -5, -35), EDT=decay_time(c, 0, -10), C80=10 * math.log10(e[:k80].sum() / max(e[k80:].sum(), 1e-30)),
                      D50=e[:k50].sum() / e.sum())
    return out


def load_irs():
    irs = {}
    for name, r in REAL.items():
        x, rate = read(DOWNLOADS / r['file'])
        irs[name] = (r['title'], r['distance'], resample(x, rate))
    for name, (title, directory, distance) in ours().items():
        wav = directory / 'listener-0.wav'
        if wav.exists():
            x, rate = read(wav)
            irs[name] = (title, distance, resample(x, rate))
    aligned = {}
    for name, (title, distance, x) in irs.items():
        start = max(0, onset(x) - int(.002 * RATE))
        aligned[name] = (title, distance, x[start:])
    return aligned


def loudness(x):
    """Integrated loudness (ITU-R BS.1770-4: K weighting, 400 ms blocks, absolute and relative gates), LUFS."""
    from scipy.signal import lfilter
    b1, a1 = [1.53512485958697, -2.69169618940638, 1.19839281085285], [1, -1.69065929318241, .73248077421585]
    b2, a2 = [1, -2, 1], [1, -1.99004745483398, .99007225036621]
    y = lfilter(b2, a2, lfilter(b1, a1, x, axis=0), axis=0)
    block, hop = int(.4 * RATE), int(.1 * RATE)
    z = np.array([(y[i:i + block] ** 2).mean(0).sum() for i in range(0, max(1, len(y) - block), hop)])
    l = -.691 + 10 * np.log10(np.maximum(z, 1e-30))
    z = z[l > -70]
    if not len(z):
        return -70.0
    rel = -.691 + 10 * math.log10(z.mean()) - 10
    z = z[(-.691 + 10 * np.log10(z)) > rel]
    return -.691 + 10 * math.log10(z.mean())


def clips(irs):
    from scipy.signal import fftconvolve
    made = {}
    (OUT / 'clips').mkdir(parents=True, exist_ok=True)
    for dry_name, (file, start, seconds) in DRY.items():
        x, rate = read(DOWNLOADS / file)
        x = resample(x[:, :1], rate)[int(start * RATE):int((start + seconds) * RATE)]
        x = x * np.minimum(1, np.arange(len(x))[::-1] / (.05 * RATE))[:, None]   # 50 ms fade at the excerpt's end
        tracks = {'dry': np.repeat(x, 2, 1)}
        for name, (_, _, h) in irs.items():
            h2 = h if h.shape[1] == 2 else np.repeat(h, 2, 1)
            h2 = h2[:int(12 * RATE)]
            tracks[name] = np.stack([fftconvolve(x[:, 0], h2[:, c]) for c in range(2)], 1)[:len(x) + int(4 * RATE)]
        for name, y in tracks.items():
            gain = 10 ** ((-23 - loudness(y)) / 20)
            y = y * gain
            peak = np.abs(y).max()
            if peak > .99:
                y *= .99 / peak
            path = OUT / 'clips' / f'{dry_name}-{name}.wav'
            write16(path, y)
            made[(dry_name, name)] = path.name
    return made


def figures(irs):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colours = {'stpauls-9m': '#0072B2', 'york-24m': '#009E73', 'stpauls-32m': '#56B4E9', 'church-near': '#D55E00', 'church-far': '#E69F00',
               'church-near-before': '#CC79A7'}
    styles = {'church-near-before': ':'}
    # Decay curves per band
    fig, axes = plt.subplots(2, 4, figsize=(17, 7.5), sharex=True, sharey=True)
    for ax, f in zip(axes.flat, BANDS):
        for name, (title, distance, h) in irs.items():
            c = edc((octave(h, f) ** 2).sum(1))
            t = np.arange(len(c)) / RATE
            ax.plot(t, c, styles.get(name, '-'), color=colours[name], lw=1.1, label=f'{title} ({distance:.1f} m)')
        ax.set_title(f'{f} Hz', fontsize=10); ax.set_ylim(-70, 2); ax.set_xlim(0, 14); ax.grid(alpha=.3)
    axes.flat[-1].axis('off')
    h, l = axes.flat[0].get_legend_handles_labels(); axes.flat[-1].legend(h, l, fontsize=8, loc='center')
    for ax in axes[:, 0]:
        ax.set_ylabel('energy decay (dB)')
    for ax in axes[1]:
        ax.set_xlabel('time after the direct sound (s)')
    fig.suptitle('Energy decay curves (Schroeder), octave bands', fontsize=11); fig.tight_layout()
    fig.savefig(OUT / 'decay.png', dpi=95); plt.close(fig)
    # Early part: energy envelope, first 300 ms
    fig, axes = plt.subplots(len(irs), 1, figsize=(12, 1.9 * len(irs)), sharex=True)
    for ax, (name, (title, distance, h)) in zip(axes, irs.items()):
        e = (octave(h, 1000) ** 2).sum(1)[:int(.3 * RATE)]
        w = int(.001 * RATE)
        env = 10 * np.log10(np.maximum(np.convolve(e, np.ones(w) / w, 'same'), 1e-30))
        ax.plot(np.arange(len(env)) / RATE * 1000, env - env.max(), color=colours[name], lw=.8)
        ax.set_ylim(-50, 3); ax.set_ylabel('dB'); ax.set_title(f'{title} ({distance:.1f} m), 1 kHz octave, 1 ms envelope', fontsize=9); ax.grid(alpha=.3)
    axes[-1].set_xlabel('time after the direct sound (ms)')
    fig.tight_layout(); fig.savefig(OUT / 'early.png', dpi=95); plt.close(fig)
    # Spectra: whole IR, and late part (after 80 ms), 1/3-octave smoothed, 0 dB at 1 kHz
    fig, axes = plt.subplots(1, 2, figsize=(15, 4.8), sharey=True)
    for ax, (label, a) in zip(axes, (('whole impulse response', 0), ('late part (after 80 ms)', .08))):
        for name, (title, distance, h) in irs.items():
            x = h[int(a * RATE):int(8 * RATE)].sum(1)
            spec = np.abs(np.fft.rfft(x)) ** 2
            f = np.fft.rfftfreq(len(x), 1 / RATE)
            centres = 31.25 * 2 ** (np.arange(0, 31) / 3)
            sm = np.array([spec[(f >= c / 2 ** (1 / 6)) & (f < c * 2 ** (1 / 6))].mean() for c in centres])
            ref = sm[np.argmin(abs(centres - 1000))]
            ax.semilogx(centres, 10 * np.log10(sm / ref), styles.get(name, '-'), color=colours[name], lw=1.1, label=f'{title} ({distance:.1f} m)')
        ax.set_title(f'Spectrum, {label}, 1/3 octave, 0 dB at 1 kHz', fontsize=10); ax.set_xlabel('Hz'); ax.grid(alpha=.3, which='both')
        ax.set_xlim(50, 18000); ax.set_ylim(-40, 15)
    axes[0].set_ylabel('dB'); axes[1].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(OUT / 'spectra.png', dpi=95); plt.close(fig)


def scene(path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import church_map as cm
    import church_room as cr
    scene_, tri, owner, _ = cr.world_triangles(Path(listen.TAKE))
    names = [o['name'] for o in scene_['objects']]
    fig, ax = plt.subplots(figsize=(6.4, 8))
    for s in cm.cut_segments(tri, owner, names, 2.65):
        ax.plot(s[:, 0], s[:, 1], color='#555555', lw=.6)
    ax.plot(ORGAN[0], ORGAN[2], '*', color='#D55E00', ms=15, label='source: organ')
    for k, p in POINTS.items():
        d = np.linalg.norm(np.array(p) - ORGAN)
        ax.plot(p[0], p[2], 'o', color='#0072B2', ms=7)
        ax.annotate(f'{k} listener, {d:.1f} m', (p[0], p[2]), textcoords='offset points', xytext=(8, 4), fontsize=9, color='#0072B2')
    ax.set_aspect('equal'); ax.set_xlabel('x (m)'); ax.set_ylabel('z (m)'); ax.grid(alpha=.25); ax.legend(fontsize=8, loc='upper left')
    ax.set_title('Church, cut at ear height; doors closed', fontsize=10)
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)


def page():
    irs = load_irs()
    figures(irs)
    scene(OUT / 'scene.png')
    made = clips(irs)
    table = {name: metrics(h) for name, (_, _, h) in irs.items()}
    (OUT / 'metrics.json').write_text(json.dumps({n: {str(f): v for f, v in m.items()} for n, m in table.items()}, indent=1) + '\n')
    rows = []
    for name, (title, distance, _) in irs.items():
        m = table[name]
        rows.append(f'<tr><td>{title}</td><td>{distance:.1f}</td>' + ''.join(f'<td>{m[f]["T30"]:.1f} / {m[f]["EDT"]:.1f} / {m[f]["C80"]:+.1f}</td>' for f in BANDS) + '</tr>')
    notes = {'dry': 'the anechoic recording itself',
             'stpauls-9m': "measured: St Paul's Cathedral, London, source and receiver under the dome, 9.5 m apart (mono microphone, same signal in both ears)",
             'york-24m': 'measured: York Minster, source under the tower, receiver in the nave centre, 23.7 m (ORTF stereo pair)',
             'stpauls-32m': "measured: St Paul's Cathedral, source under the dome, receiver in the nave, 31.6 m (mono)",
             'church-near': 'our Church, energy method, binaural, listener in the crossing 10.3 m from the organ (doors closed)',
             'church-far': 'our Church, energy method, binaural, listener in the nave by the doors, 48 m from the organ (walk start)',
             'church-near-before': 'our Church as rendered before the 10 November material and absorption fixes (same point as "near")'}
    html = ['<!doctype html><meta charset="utf-8"><title>Church realism reference</title>',
            '<style>body{font:14px/1.45 -apple-system,sans-serif;margin:1.5em;max-width:1300px;color:#222}td,th{padding:.2em .5em;border-bottom:1px solid #ddd;font-size:12px}'
            'img{max-width:100%}audio{width:260px}</style>',
            '<h1>Church against real cathedrals: same dry sounds, loudness-matched</h1>',
            '<p>Every clip is one dry (anechoic) recording convolved with one impulse response, normalised to -23 LUFS. '
            "The measured responses are York Minster (140,000 m³) and St Paul's Cathedral (about 150,000 m³); our Church is 31,000 m³. "
            'Recordings and responses: OpenAIR library, www.openairlib.net, Audiolab, University of York (York Minster: Damian T. Murphy), CC BY 4.0.</p>',
            '<img src="scene.png" style="max-width:420px;float:right;margin-left:1em">']
    for dry_name in DRY:
        html.append(f'<h2>{dry_name}</h2><table>')
        for name in ['dry'] + list(irs):
            if (dry_name, name) in made:
                html.append(f'<tr><td><audio controls preload="none" src="clips/{made[(dry_name, name)]}"></audio></td><td>{notes.get(name, name)}</td></tr>')
        html.append('</table>')
    html.append('<h2 style="clear:both">Measured differences</h2><p>Per octave band: T30 (s) / EDT (s) / C80 (dB), computed here the same way for every response.</p>')
    html.append('<table><tr><th>response</th><th>m</th>' + ''.join(f'<th>{f} Hz</th>' for f in BANDS) + '</tr>' + ''.join(rows) + '</table>')
    for img, caption in (('decay.png', 'Energy decay curves per octave band'), ('early.png', 'First 300 ms at 1 kHz: early reflections'),
                         ('spectra.png', 'Spectra (colouration and high-frequency roll-off)')):
        html.append(f'<h3>{caption}</h3><img src="{img}">')
    if (OUT / 'notes.html').exists():
        html.append((OUT / 'notes.html').read_text())
    (OUT / 'index.html').write_text('\n'.join(html))
    print(OUT / 'index.html')
    return table


if __name__ == '__main__':
    cmd = sys.argv[1] if len(sys.argv) > 1 else ''
    if cmd == 'takes':
        takes()
    elif cmd == 'render':
        render(tuple(sys.argv[2:]) or ('near', 'far', 'near-before'))
    elif cmd == 'page':
        page()
    else:
        raise SystemExit(__doc__)
