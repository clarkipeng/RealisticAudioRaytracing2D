"""Scoreboard of the Path method against the static-IR baseline and the earlier methods.

    .context/fdtd-teacher/venv/bin/python docs/benchmarks/path-tracer-2026-11-12/scoreboard.py

Scores every method under the method-comparison root (tools/method_compare.py score, after
rebuilding the baseline as the 11 November scoreboard does) and writes scoreboard.md and
scoreboard.json beside this file: RMS errors against the wave ground truth at each method's
largest budget, overall, in view, in shadow, open and closed. A check mark shows where the path
method beats the static-IR baseline; a plus where it beats the energy method.
"""
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / 'docs/benchmarks/energy-audio-2026-11-09'))
import scoreboard as previous  # noqa: E402

previous.METHODS = ('baseline-static-ir', 'energy', 'hybrid', 'path')


def suite():
    """Every method at 2,048 rays (energy, hybrid, path; budgets are not comparable across methods
    in cost, but are the ones the 11 November scoreboard used), the baseline, and path at 16,384 rays
    on the scenes rendered at that budget, with every method restricted to those scenes."""
    import method_compare as mc
    root = mc.DEFAULT_ROOT
    for entry in json.loads((root / 'suite.json').read_text())['scenes']:
        if (root / entry['id'] / 'gt' / 'ref' / '0').exists():
            previous.base.build(root, entry['id'])
    rows = mc.score(root, root.parent / 'score')['rows']
    budgets = {'baseline-static-ir': None, 'energy': 'r02048', 'hybrid': 'r02048', 'path': 'r02048', 'path-16k': 'r16384'}
    high = {r['scene'] for r in rows if r['method'] == 'path' and r['budget'] == 'r16384'}
    tables = {}
    for name, scenes in (('all scenes', None), ('scenes with a 16,384-ray path render', high)):
        table = {}
        for label, budget in budgets.items():
            method = 'path' if label.startswith('path') else label
            if scenes is None and label == 'path-16k':
                continue
            sel = [r for r in rows if r['method'] == method and (budget is None or r['budget'] == budget) and (scenes is None or r['scene'] in scenes)]
            table[label] = dict(budget=budget or 'ref', groups={g: {k: mc.aggregate([r for r in sel if f(r)], ('method',), k).get((method,)) for k, _ in previous.METRICS}
                                                               for g, f in previous.GROUPS})
        tables[name] = dict(scenes=sorted(scenes) if scenes else 'all', table=table)
    return tables


def markdown(tables):
    lines = ['# Scoreboard: the path tracer against the baseline', '',
             'Regenerate with `.context/fdtd-teacher/venv/bin/python docs/benchmarks/path-tracer-2026-11-12/scoreboard.py`.', '',
             'RMS error against the wave ground truth (11 scenes), lower is better, all bands with ground truth pooled; seeds pooled.',
             '`baseline-static-ir`: the ground truth of one central in-view receiver used at every receiver, with only the direct sound moved to each receiver\'s distance (no occlusion), i.e. a game convolution reverb.',
             'A ✓ marks where a path render beats the baseline, a + where it beats the energy method.', '']
    for name, entry in tables.items():
        s = entry['table']
        lines += [f'## {name}', '']
        if entry['scenes'] != 'all':
            lines += ['Scenes: ' + ', '.join(f'`{x}`' for x in entry['scenes']) + '.', '']
        for g, _ in previous.GROUPS:
            lines += [f'### {g}', '', '| method | budget | ' + ' | '.join(n for _, n in previous.METRICS) + ' |', '|---' * (len(previous.METRICS) + 2) + '|']
            for m in s:
                row = s[m]['groups'][g]
                cells = []
                for k, _ in previous.METRICS:
                    v = row[k]; b = s['baseline-static-ir']['groups'][g][k]; e = s['energy']['groups'][g][k]
                    mark = ''
                    if m.startswith('path') and v is not None:
                        mark = (' ✓' if b is not None and v < b else '') + (' +' if e is not None and v < e else '')
                    cells.append(previous.fmt(v) + mark)
                lines.append(f"| {m} | {s[m]['budget']} | " + ' | '.join(cells) + ' |')
            lines.append('')
    return '\n'.join(lines) + '\n'


def main():
    s = suite()
    (HERE / 'scoreboard.json').write_text(json.dumps(s, indent=1, default=float) + '\n')
    (HERE / 'scoreboard.md').write_text(markdown(s))
    print((HERE / 'scoreboard.md').read_text())


if __name__ == '__main__':
    main()
