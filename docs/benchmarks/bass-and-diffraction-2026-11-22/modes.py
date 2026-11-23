"""Track A studies for README part A (room modes against the FDTD ground truth).

    PY=.context/fdtd-teacher/venv/bin/python
    G="python3 tools/heavy_guard.py run --limit-gib 3.3 --"
    $G $PY $B/modes.py numeric SCENE HMAX [--correct]   # grid bake vs the analytic box modes, both against the FDTD
    $PY $B/modes.py truncation SCENE                    # modes needed past the crossover's top
    $PY $B/modes.py score                               # path, path-modal, modal on the closed rooms -> modes.json
"""
import json
import math
from pathlib import Path
import sys
import time

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / 'tools'))
import method_compare as mc  # noqa: E402
import room_modes as rm  # noqa: E402

WORK = ROOT / '.context/bass-and-diffraction'
CLOSED = ['shoebox-0020-presets', 'shoebox-0010-presets', 'furnished-0005-presets']


def load(sid):
    root = mc.DEFAULT_ROOT
    scene = json.loads((root / sid / 'scene.json').read_text())
    gt = mc.read_output(root / sid / 'gt' / 'ref' / '0')
    return scene, gt


def metrics(gt, E, bands=(0, 1, 2)):
    """RMS level error (and bias), EDT and T30 RMS % per band, against the ground truth."""
    ge = gt[1]
    valid = int(round(gt[0]['validSeconds'] / mc.BIN_SECONDS))
    out = {}
    for b in bands:
        if not np.isfinite(ge[:, b]).all() or not np.isfinite(E[:, b]).all():
            continue
        lev, edt, t30 = [], [], []
        for r in range(len(ge)):
            t0 = mc.onset_bin(ge[r, b][:valid])
            mg, mm = mc.receiver_metrics(ge[r, b], valid, t0), mc.receiver_metrics(E[r, b], valid, t0)
            lev.append(mm['level'] - mg['level'])
            for lst, key in ((edt, 'EDT'), (t30, 'T30')):
                if np.isfinite(mm[key]) and np.isfinite(mg[key]):
                    lst.append(100 * (mm[key] - mg[key]) / mg[key])
        rms = lambda x: float(np.sqrt(np.mean(np.square(x)))) if len(x) else None
        out[float(mc.BANDS[b])] = dict(level=rms(lev), bias=float(np.mean(lev)), EDT=rms(edt), T30=rms(t30))
    return out


def show(label, m):
    for band, v in m.items():
        f = lambda x: f'{x:6.2f}' if x is not None else '     -'
        print(f'{label:24s} {band:5.0f} Hz: level {f(v["level"])} ({v["bias"]:+.2f})  EDT {f(v["EDT"])}  T30 {f(v["T30"])}', flush=True)


def modal_only(scene, k, S, a_s, a_r, top):
    adm = [scene['materials'][m]['admittance'] for m in scene['materials']]
    delta = rm.damping(k, S, adm)
    freqs = np.arange(0, top + rm.DF, rm.DF)
    H = rm.transfer(k, delta, a_s, a_r, freqs)
    H[:, 0] = 0
    bands = [b for b, fc in enumerate(mc.BANDS) if 2 * fc <= top]
    E, _ = mc.energy_from_spectrum(H, rm.DF, bands=mc.BANDS[bands])
    out = np.full((len(a_r), len(mc.BANDS), mc.TIME_BINS), np.nan)
    out[:, bands] = E
    return out, delta


def numeric(sid, hmax, correct=False):
    """Bake on the grid up to the analytic crossover top; score both against the FDTD."""
    scene, gt = load(sid)
    k, S, a_s, a_r = rm.scene_modes(scene, fmax=1100, analytic=True)
    adm = [scene['materials'][m]['admittance'] for m in scene['materials']]
    fs = rm.schroeder_frequency(k, rm.damping(k, S, adm))
    fmax = math.sqrt(2) * fs          # the crossover's top: modes past it change levels by < 0.1 dB (truncation)
    boxes, names = rm.boxes_of(scene)
    bank = rm.bake_boxes(boxes, names, hmax, fmax, out=WORK / f'bank-{sid}-{hmax:g}-to-crossover')
    kn, Sn, a_sn, a_rn = rm.scene_modes(scene, bank=bank)
    sel = k <= 2 * math.pi * fmax / rm.C
    print(f'{sid}: analytic modes below {fmax:.0f} Hz {sel.sum()}, grid {len(kn)}', flush=True)
    fa, fn = np.sort(k[sel]) * rm.C / (2 * math.pi), np.sort(kn) * rm.C / (2 * math.pi)
    m = min(len(fa), len(fn))
    print('  frequency error of the n-th mode (grid - analytic), %: ' +
          ', '.join(f'{q:.0f}th pct {np.percentile(100 * (fn[:m] - fa[:m]) / np.maximum(fa[:m], 1), q):+.2f}' for q in (5, 50, 95)), flush=True)
    top = 2 * mc.BANDS[1]
    Ea, _ = modal_only(scene, k[sel], S[sel], a_s[sel], a_r[:, sel], top)
    En, _ = modal_only(scene, kn, Sn, a_sn, a_rn, top)
    show('analytic modes', metrics(gt, Ea, (0, 1)))
    show(f'grid modes, h {hmax:g}', metrics(gt, En, (0, 1)))
    dn = rm.damping(kn, Sn, adm)
    print(f'  Schroeder frequency: analytic {fs:.0f} Hz, grid {rm.schroeder_frequency(kn, dn)} Hz (needs modes past the top)', flush=True)


def truncation(sid):
    """Crossover result with the modes cut at 1, 1.1, 1.25, 1.5x the crossover top, against all modes to 1.1 kHz."""
    scene, gt = load(sid)
    k, S, a_s, a_r = rm.scene_modes(scene, fmax=1100, analytic=True)
    adm = [scene['materials'][m]['admittance'] for m in scene['materials']]
    fs = rm.schroeder_frequency(k, rm.damping(k, S, adm))
    pe = mc.read_output(mc.DEFAULT_ROOT / sid / 'path' / 'r02048' / '0')[1].astype(float)
    ref, _, info = rm.combine(scene, k, S, a_s, a_r, pe)
    for factor in (1.0, 1.1, 1.25, 1.5):
        sel = k <= factor * math.sqrt(2) * fs * 2 * math.pi / rm.C
        out, _, _ = rm.combine(scene, k[sel], S[sel], a_s[sel], a_r[:, sel], pe, fs=fs)
        b = [mc.BANDS.tolist().index(x) for x in info['touchedBands']]
        with np.errstate(divide='ignore', invalid='ignore'):
            lev = 10 * np.log10(out[:, b].sum(-1) / ref[:, b].sum(-1))
        print(f'{sid}: modes to {factor:.2f} x {math.sqrt(2) * fs:.0f} Hz ({sel.sum()}): level change vs all modes, '
              f'RMS per band ' + ' '.join(f'{v:.3f}' for v in np.sqrt(np.mean(lev ** 2, 0))) + ' dB', flush=True)


def score(methods=('path', 'path-modal', 'modal', 'energy'), bands=(125.0, 250.0, 500.0)):
    """The suite's scorer on the closed rooms: per scene and band, and pooled, level / bias / EDT / T30 / ETC."""
    rep = mc.score(mc.DEFAULT_ROOT, None, methods=list(methods))
    rows = [r for r in rep['rows'] if r['closed'] and r['group'] == 'all']
    keys = ('level', 'levelBias', 'EDT', 'T30', 'etc')
    table = {}
    for m in methods:
        sel_m = [r for r in rows if r['method'] == m]
        if m != 'modal':
            sel_m = [r for r in sel_m if r['budget'] == ('r02048' if m != 'energy' else 'r02048')]
        for b in bands:
            for scene in sorted({r['scene'] for r in rows}) + ['closed']:
                sel = [r for r in sel_m if r['band'] == b and (scene == 'closed' or r['scene'] == scene)]
                if not sel:
                    continue
                table.setdefault(m, {}).setdefault(scene, {})[b] = {k: mc.aggregate(sel, ('method',), k).get((m,)) for k in keys}
    (HERE / 'modes.json').write_text(json.dumps(table, indent=1) + '\n')
    for m, t in table.items():
        for scene, per in t.items():
            for b, v in per.items():
                f = lambda x: f'{x:6.2f}' if x is not None else '     -'
                print(f'{m:10s} {scene:24s} {b:5.0f}: level {f(v["level"])} bias {f(v["levelBias"])} EDT {f(v["EDT"])} T30 {f(v["T30"])} ETC {f(v["etc"])}')
    return table


def oracle(budget='r02048', bands=(0, 1)):
    """The ceiling of any correction of the path method's band envelope that does not resolve the modes
    receiver by receiver: per scene and band, remove the mean level error and the mean relative EDT and
    T30 errors exactly (an oracle that knows them), and report what is left."""
    out = {}
    for sid in CLOSED:
        scene, gt = load(sid)
        ge = gt[1]
        valid = int(round(gt[0]['validSeconds'] / mc.BIN_SECONDS))
        for seed in (0, 1):
            pe = mc.smooth(mc.read_output(mc.DEFAULT_ROOT / sid / 'path' / budget / str(seed))[1].astype(float))
            for b in bands:
                lev, rel = [], {'EDT': [], 'T30': []}
                for r in range(len(ge)):
                    t0 = mc.onset_bin(ge[r, b][:valid])
                    mg, mm = mc.receiver_metrics(ge[r, b], valid, t0), mc.receiver_metrics(pe[r, b], valid, t0)
                    lev.append(mm['level'] - mg['level'])
                    for key in rel:
                        rel[key].append(100 * (mm[key] - mg[key]) / mg[key] if np.isfinite(mm[key]) and np.isfinite(mg[key]) else np.nan)
                lev = np.array(lev)
                row = out.setdefault(float(mc.BANDS[b]), {'level': [], 'EDT': [], 'T30': []})
                row['level'] += list(lev - lev.mean())
                for key, v in rel.items():
                    v = np.array(v)
                    v = v[np.isfinite(v)]
                    if len(v):
                        row[key] += list(v - v.mean())
    res = {b: {k: float(np.sqrt(np.mean(np.square(v)))) if v else None for k, v in row.items()} for b, row in out.items()}
    print('oracle-corrected path (per scene and band: mean errors removed), RMS over closed-room receivers:', res)
    return res


def candidates(update_seconds=1024 / 44100):
    """Cost of the two physical candidates up to the crossover's top, sqrt2 f_S, per room: the modal bank
    (modes, fp16 shapes) and a runtime FDTD at the ground truth's own resolution rule (10 points per
    wavelength at the top frequency, Courant 0.99/sqrt3), timed at the ground truth runs' measured median
    of GPU cell updates per second on this Mac."""
    import glob
    rate = float(np.median([json.loads(Path(f).read_text())['timing']['gcellPerSecond']
                            for f in glob.glob(str(ROOT / '.context/method-compare/fdtd/*/*.json'))
                            if 'timing' in json.loads(Path(f).read_text())])) * 1e9
    rooms = json.loads((WORK / 'suite.json').read_text())
    church = WORK / 'church-0.75.json'
    if church.exists():
        c = json.loads(church.read_text())
        rooms['church'] = dict(schroederHz=c['schroederHz'], cells=None, volume=c['volume'], modes=None)
    out = {}
    for name, r in rooms.items():
        fs = r['schroederHz']
        top = math.sqrt(2) * fs
        bank = rm.load_bank(WORK / (f'bank-{name}-0.125' if name != 'church' else 'bank-church-0.75'))
        modes = int(np.sum(bank['k'] * rm.C / (2 * math.pi) <= top))
        volume = float(bank['volume'])
        h = rm.C / (10 * top)
        cells = volume / h ** 3
        steps = rm.C * math.sqrt(3) / (0.99 * h)
        gpu = cells * steps / rate * update_seconds
        out[name] = dict(schroederHz=fs, topHz=top, volume=volume, modes=modes,
                         bankMiB=2 * modes * len(bank['volumes']) / 2 ** 20,
                         fdtdCells=cells, fdtdStepsPerSecond=steps, fdtdGpuMsPerUpdate=1e3 * gpu,
                         fdtdStateMiB=16 * cells / 2 ** 20)
        print(name, {k: round(v, 2) for k, v in out[name].items()}, flush=True)
    (HERE / 'candidates.json').write_text(json.dumps(dict(gcellPerSecond=rate / 1e9, updateSeconds=update_seconds, rooms=out), indent=1) + '\n')
    return out


if __name__ == '__main__':
    cmd = sys.argv[1]
    if cmd == 'candidates':
        candidates()
        sys.exit()
    if cmd == 'oracle':
        oracle()
        sys.exit()
    if cmd == 'score':
        score()
        sys.exit()
    if cmd == 'numeric':
        numeric(sys.argv[2], float(sys.argv[3]), '--correct' in sys.argv)
    elif cmd == 'truncation':
        truncation(sys.argv[2])
    else:
        raise SystemExit(__doc__)
