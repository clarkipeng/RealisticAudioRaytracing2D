"""Unfitted finite-3D boundary comparison. Run with repo/evidence root as argument."""
import csv, hashlib, json, sys, xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
root=Path(sys.argv[1]).resolve();out=Path(__file__).resolve().parent
base=root/'.context/solid-boundary';reference=root/'.context/wave-canopy-2026-10-27/reference-0.25-125.npz'
ref=np.load(reference);gt=ref['pressure'];points=ref['points']
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def field(directory,indices):
    c=json.loads((directory/'scene.json').read_text());r=np.genfromtxt(directory/'renderer.csv',delimiter=',',names=True)
    assert np.allclose(np.array(c['points']).reshape(-1,3),points[indices],rtol=0,atol=1e-12)
    results=[];fields=[]
    for seed in sorted(set(r['seed'])):
        a=np.sort(r[(r['diffraction']==1)&(r['seed']==seed)],order='point')
        assert np.array_equal(a['point'],np.arange(len(indices)))
        p=a['real']+1j*a['imag'];target=gt[indices];shadow=a['direct_visible']==0
        results.append(dict(seed=int(seed),samples=c['completeRouteSamples'],receivers=len(indices),
          complexRelativeL2=float(np.linalg.norm(p-target)/np.linalg.norm(target)),
          shadowComplexRelativeL2=float(np.linalg.norm((p-target)[shadow])/np.linalg.norm(target[shadow])),
          zeros=int(np.count_nonzero(p==0)),medianSolveMs=float(np.median(a['solve_seconds'])*1000),
          indoorPressure=[float(p[indices.index(323)].real),float(p[indices.index(323)].imag)],
          rendererSha256=sha(directory/'renderer.csv')))
        fields.append(p)
    return results,fields
old,before=field(root/'.context/finite-diffraction/coverage-full',list(range(len(gt))))
new,after=field(base/'ordered-full',list(range(len(gt))))
report=dict(scope='Rigid finite 3D room/roof, 125 Hz, original reference, source, receivers, 32 residual slots and 1.25 s support. No gain/phase fitting.',
 referenceSha256=sha(reference),baseline=old,baked=new,referenceIndoor=[float(gt[323].real),float(gt[323].imag)],
 probes={str(n):field(base/f'ordered-probes-{n}',[0,323,324,700,900])[0] for n in [128,65536]},
 indoor=field(base/'ordered-indoor',[323])[0])
boundary=json.loads((base/'ordered-full/boundary.json').read_text())
report['boundary']={k:v for k,v in boundary.items() if k not in ['vertices','triangles','materials']}
families=list(csv.DictReader((base/'ordered-indoor/families.csv').open()))
for entry in report['indoor']:
    parts={r['family']:complex(float(r['real']),float(r['imag'])) for r in families if int(r['seed'])==entry['seed']}
    residual=complex(*entry['indoorPressure'])-sum(parts.values())
    entry['completeFamilies']={k:[v.real,v.imag] for k,v in parts.items()}
    entry['residualPressure']=[residual.real,residual.imag]
report['tests']={name:ET.parse(base/name/'tests.xml').getroot().attrib for name in ['initial','compiled','final']}
report['tests']['exclusion']=ET.parse(base/'exclusion/tests.xml').getroot().attrib if (base/'exclusion/tests.xml').exists() else 'pending'
report['partitionRelativeDifference']=[dict(seed=0,moving=False,value=0),dict(seed=17,moving=False,value=0),dict(seed=17,moving=True,value=0)]
# Assert recorded exact-match results, not just their test pass status.
log=(base/'final/unity.log').read_text()
for c in report['partitionRelativeDifference']:
    assert f"SOLID_PARTITION: seed={c['seed']} moving={c['moving']} relative=0" in log
(out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
c=json.loads((base/'ordered-full/scene.json').read_text());indices=c['gridIndices']
def grid(v):
    a=np.full(len(c['x'])*len(c['y']),np.nan);a[indices]=v
    return a.reshape(len(c['y']),len(c['x']))
fig,axes=plt.subplots(1,3,figsize=(13.6,4.4),layout='constrained')
maximum=max(float(np.max(abs(v))) for v in [gt,before[0],after[0]])
for ax,v,title in zip(axes,[gt,before[0],after[0]],['Numerical wave reference',f'Separate pieces · {old[0]["complexRelativeL2"]:.2%} error',f'Baked exterior · {new[0]["complexRelativeL2"]:.2%} error']):
    im=ax.pcolormesh(c['x'],c['y'],grid(abs(v)),shading='nearest',cmap='magma',vmin=0,vmax=maximum)
    for w in c['walls']:
        x,y,z=w['center'];a,b,d=w['half'];roof=y-b>1.6
        ax.add_patch(Rectangle((x-a,z-d),2*a,2*d,fill=not roof,facecolor='#bbc0ca',edgecolor='#94a3b8',linestyle='--' if roof else '-',linewidth=.7))
    ax.plot(c['source'][0],c['source'][2],'c*',ms=8)
    ax.plot(points[323,0],points[323,2],'wo',ms=4,mec='black',mew=.7)
    ax.set(title=title,xlabel='x (m)',ylabel='z (m)',aspect='equal')
fig.colorbar(im,ax=axes,shrink=.75,label='Pressure magnitude (shared scale)')
fig.suptitle('Same finite room and roof · 125 Hz · 65,536 complete samples · no gain or phase fitting\nError includes phase. Dashed outline: roof; white dot: indoor probe.',fontsize=11)
fig.savefig(out/'comparison.png',dpi=160);plt.close(fig)
print(json.dumps({k:report[k] for k in ['baseline','baked','indoor']},indent=2))
