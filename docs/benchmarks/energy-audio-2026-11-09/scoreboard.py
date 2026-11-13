"""Scoreboard: every change must beat the baseline.

    .context/fdtd-teacher/venv/bin/python docs/benchmarks/energy-audio-2026-11-09/scoreboard.py

One command: rebuilds the static-IR baseline on the wave-ground-truth suite
(tools/baseline_static_ir.py), scores every method there (tools/method_compare.py score), and
writes scoreboard.md and scoreboard.json beside this file with
1. the suite: RMS errors against the wave ground truth (level dB, EDT %, T30 %, C50 dB, C80 dB,
   ETC distance dB) for baseline-static-ir, energy, hybrid and path, overall, in view, in shadow, in
   open and in closed scenes, and whether the energy and path methods beat the baseline;
2. the Church: T30 / EDT / C80 per octave band and the echo density of our renders, the static
   baseline and the measured cathedrals (from .context/energy-audio/realism/metrics.json, written
   by realism.py page). There is no ground truth there; the measured cathedrals are the reference
   for "natural", and the blind A/B page (.context/energy-audio/realism/ab.html) is the listening test.
"""
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / 'tools'))
import baseline_static_ir as base  # noqa: E402
import method_compare as mc  # noqa: E402

METHODS = ('baseline-static-ir', 'energy', 'hybrid', 'path')
METRICS = (('level', 'level dB'), ('EDT', 'EDT %'), ('T30', 'T30 %'), ('C50', 'C50 dB'), ('C80', 'C80 dB'), ('etc', 'ETC dB'))
GROUPS = (('all', lambda r: r['group'] == 'all'), ('in view', lambda r: r['group'] == 'view'), ('shadow', lambda r: r['group'] == 'shadow'),
          ('open scenes', lambda r: r['group'] == 'all' and not r['closed']), ('closed scenes', lambda r: r['group'] == 'all' and r['closed']))
BANDS = ['125', '250', '500', '1000', '2000', '4000', '8000']


def suite():
    root = mc.DEFAULT_ROOT
    for entry in json.loads((root / 'suite.json').read_text())['scenes']:
        if (root / entry['id'] / 'gt' / 'ref' / '0').exists():
            base.build(root, entry['id'])
    report = mc.score(root, root.parent / 'score')
    rows = report['rows']
    table = {}
    for method in METHODS:
        budget = mc.top_budget(rows, method)
        sel = [r for r in rows if r['method'] == method and r['budget'] == budget]
        table[method] = dict(budget=budget, groups={g: {k: mc.aggregate([r for r in sel if f(r)], ('method',), k).get((method,)) for k, _ in METRICS}
                                                    for g, f in GROUPS})
    return table


def church():
    path = ROOT / '.context/energy-audio/realism/metrics.json'
    return json.loads(path.read_text()) if path.exists() else {}


def fmt(v, spec='{:.2f}'):
    return '–' if v is None else spec.format(v)


def markdown(s, c):
    lines = ['# Scoreboard: ours against the baseline', '',
             'Regenerate with `.context/fdtd-teacher/venv/bin/python docs/benchmarks/energy-audio-2026-11-09/scoreboard.py`.', '',
             '## Wave-ground-truth suite (11 scenes)', '',
             'RMS error against the wave ground truth, lower is better, all bands with ground truth pooled.',
             '`baseline-static-ir`: the ground truth of one central in-view receiver used at every receiver, with only the direct sound moved to each receiver\'s distance (no occlusion), i.e. a game convolution reverb.',
             'A ✓ marks where the energy or path method beats the baseline.', '']
    for g, _ in GROUPS:
        lines += [f'### {g}', '', '| method | ' + ' | '.join(n for _, n in METRICS) + ' |', '|---' * (len(METRICS) + 1) + '|']
        for m in METHODS:
            row = s[m]['groups'][g]
            cells = []
            for k, _ in METRICS:
                v = row[k]; b = s['baseline-static-ir']['groups'][g][k]
                mark = ' ✓' if m in ('energy', 'path') and v is not None and b is not None and v < b else ''
                cells.append(fmt(v) + mark)
            lines.append(f'| {m} | ' + ' | '.join(cells) + ' |')
        lines.append('')
    if c:
        lines += ['## Church (no ground truth; measured cathedrals as the reference)', '',
                  'T30 (s) / EDT (s) / C80 (dB) per octave band; echo density: mean over the first 150 ms / time it reaches 0.9 (ms).', '',
                  '| response | ' + ' | '.join(f'{b} Hz' for b in BANDS) + ' | echo density |', '|---' * (len(BANDS) + 2) + '|']
        for name, m in c.items():
            e = m.get('echo') or {}
            cells = [f"{fmt(m[b]['T30'], '{:.1f}')} / {fmt(m[b]['EDT'], '{:.1f}')} / {fmt(m[b]['C80'], '{:+.1f}')}" for b in BANDS if b in m]
            lines.append(f'| {name} | ' + ' | '.join(cells) + f" | {fmt(e.get('mean150'))} / {fmt(e.get('mixingMs'), '{:.0f}')} |")
        lines.append('')
    return '\n'.join(lines) + '\n'


def main():
    s, c = suite(), church()
    (HERE / 'scoreboard.json').write_text(json.dumps(dict(suite=s, church=c), indent=1, default=float) + '\n')
    (HERE / 'scoreboard.md').write_text(markdown(s, c))
    print((HERE / 'scoreboard.md').read_text())


if __name__ == '__main__':
    main()
