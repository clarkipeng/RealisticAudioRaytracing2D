"""Before/after table of the energy method against the wave ground truth (and the hybrid).

    PY=.context/fdtd-teacher/venv/bin/python
    $PY tools/method_compare.py score                                   # after: .context/method-compare/score
    $PY docs/benchmarks/energy-audio-2026-11-09/rescore.py              # prints Markdown, writes rescore.json

"Before" is the 6 November score (.context/method-compare/score-before-2026-11-09, copied before
the energy outputs were re-rendered); "after" the current score. Pooled exactly as
tools/method_compare.py does (receiver-weighted RMS over scenes, seeds and bands with ground truth).
"""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'tools'))
import method_compare as mc  # noqa: E402

BEFORE = ROOT / '.context/method-compare/score-before-2026-11-09/report.json'
AFTER = ROOT / '.context/method-compare/score/report.json'
METRICS = [('level', 'level dB'), ('levelBias', 'bias dB'), ('EDT', 'EDT %'), ('T20', 'T20 %'), ('T30', 'T30 %'), ('etc', 'ETC dB'), ('onset', 'onset ms')]


def pooled(rows, method, budget, closed, metric):
    sel = [r for r in rows if r['method'] == method and r['budget'] == budget and r['closed'] == closed and r['group'] == 'all']
    out = mc.aggregate(sel, ['method'], metric)
    return out.get((method,))


def table(before, after):
    lines, data = [], {}
    for closed in (True, False):
        lines.append(f"\n**{'Closed rooms' if closed else 'Open scenes'}** (2,048 rays, all bands with ground truth)\n")
        lines.append('| Metric | energy before | energy after | hybrid |')
        lines.append('|---|---:|---:|---:|')
        for metric, label in METRICS:
            values = [pooled(before, 'energy', 'r02048', closed, metric), pooled(after, 'energy', 'r02048', closed, metric),
                      pooled(after, 'hybrid', 'r02048', closed, metric)]
            data[f"{'closed' if closed else 'open'}.{metric}"] = values
            fmt = (lambda v: '–' if v is None else f'{v:+.1f}') if metric == 'levelBias' else (lambda v: '–' if v is None else f'{v:.1f}')
            lines.append(f'| {label} | ' + ' | '.join(fmt(v) for v in values) + ' |')
    lines.append('\n**All scenes, level error (dB RMS)**\n')
    lines.append('| Budget | energy before | energy after | hybrid |')
    lines.append('|---|---:|---:|---:|')
    for budget in ('r00256', 'r02048'):
        values = []
        for rows, method in ((before, 'energy'), (after, 'energy'), (after, 'hybrid')):
            sel = [r for r in rows if r['method'] == method and r['budget'] == budget and r['group'] == 'all']
            values.append(mc.aggregate(sel, ['method'], 'level').get((method,)))
        data[f'all.level.{budget}'] = values
        lines.append(f'| {int(budget[1:]):,} | ' + ' | '.join('–' if v is None else f'{v:.2f}' for v in values) + ' |')
    return '\n'.join(lines), data


def wall(rows, method, budget):
    seconds = {}
    for r in rows:
        if r['method'] == method and r['budget'] == budget and r['wallSeconds'] is not None:
            seconds[(r['scene'], r['seed'])] = r['wallSeconds'] / max(1, r['receivers'])
    return sum(seconds.values()) / max(1, len(seconds))


if __name__ == '__main__':
    before = json.loads(BEFORE.read_text())['rows']
    after = json.loads(AFTER.read_text())['rows']
    text, data = table(before, after)
    text += '\n\nMean busy time per solve at 2,048 rays (one receiver): ' + \
            f"energy before {1000 * wall(before, 'energy', 'r02048'):.0f} ms, after {1000 * wall(after, 'energy', 'r02048'):.0f} ms, " \
            f"hybrid {1000 * wall(after, 'hybrid', 'r02048'):.0f} ms."
    print(text)
    (Path(__file__).parent / 'rescore.json').write_text(json.dumps(data, indent=1) + '\n')
