"""Summarize physical-reference and mesh-representation controls; no fitting."""
import csv, hashlib, json, sys
from pathlib import Path
import numpy as np
root=Path(sys.argv[1]).resolve();base=root/'.context/finite-diffraction';out=Path(__file__).resolve().parent
ref=np.load(root/'.context/wave-canopy-2026-10-27/reference-0.25-125.npz')['pressure']
ids=[0,323,324,700,900]
def field(name,indices):
    path=base/name/'renderer.csv'
    if not path.exists():return []
    rows=list(csv.DictReader(path.open()));result=[]
    cfg=json.loads(path.with_name('scene.json').read_text())
    targetPoints=np.load(root/'.context/wave-canopy-2026-10-27/reference-0.25-125.npz')['points']
    assert np.allclose(np.array(cfg['points']).reshape(-1,3),targetPoints[indices])
    for seed in sorted({int(r['seed']) for r in rows}):
        chosen=[r for r in rows if r['diffraction']=='1' and int(r['seed'])==seed]
        if len(chosen)!=len(indices):continue
        chosen.sort(key=lambda r:int(r['point']))
        p=np.array([complex(float(r['real']),float(r['imag'])) for r in chosen]);target=ref[indices]
        shadow=np.array([r['direct_visible']=='0' for r in chosen])
        result.append(dict(seed=seed,complexRelativeL2=float(np.linalg.norm(p-target)/np.linalg.norm(target)),
          shadowComplexRelativeL2=float(np.linalg.norm((p-target)[shadow])/np.linalg.norm(target[shadow])) if np.any(shadow) else None,
          zeros=int(np.count_nonzero(p==0)),medianSolveMs=1000*float(np.median([float(r['solve_seconds']) for r in chosen])),
          rendererSha256=hashlib.sha256(path.read_bytes()).hexdigest(),
          pressure=[[float(x.real),float(x.imag)] for x in p],pointIds=indices,
          solveSeconds=[float(r['solve_seconds']) for r in chosen]))
    return result
report=dict(reference323=[float(ref[323].real),float(ref[323].imag)],reference323Amplitude=float(abs(ref[323])),
    scope='Finite 3D room and roof, rigid surfaces, 125 Hz; unchanged numerical BEM reference. No pressure, phase or output fitting.',
    convergence={str(n):field(f'convergence-{n}',[323]) for n in [65536,262144,1048576]},
    projectedProbes={str(n):field(f'probes-{n}',ids) for n in [128,2048,65536]},
    coverageProbes={str(n):field(f'coverage-probes-{n}',ids) for n in [128,65536]},
    coverageIndoor=field('coverage-indoor',[323]),unionBefore=field('union-probe',[323]),unionAfter=field('coverage-union',[323]),
    fullField=field('coverage-full',list(range(len(ref)))))
lit=[]
paths=[base/x/'renderer.csv' for x in ['lit-solid','lit-split']]
if all(p.exists() for p in paths):
    data=[list(csv.DictReader(p.open())) for p in paths]
    for seed in [0,17]:
        fields=[]
        for rows in data:
            selected=sorted([r for r in rows if r['diffraction']=='1' and int(r['seed'])==seed],key=lambda r:(int(r['point']),float(r['hz'])))
            fields.append(np.array([complex(float(r['real']),float(r['imag'])) for r in selected]))
        if all(len(p)==12 for p in fields):
            a,b=fields;lit.append(dict(seed=seed,relativeDifference=float(np.linalg.norm(b-a)/np.linalg.norm(a)),maximumAbsoluteDifference=float(np.max(abs(b-a)))))
report['litRepresentation']=lit
(out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:v for k,v in report.items() if k not in ['projectedProbes','coverageProbes','fullField']},indent=2))
