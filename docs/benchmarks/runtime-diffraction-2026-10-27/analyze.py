"""Score unchanged complex fields without amplitude or phase fitting."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np

def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def metric(a,b):return float(np.linalg.norm(a-b)/np.linalg.norm(b))
def field(rows):return rows['real']+1j*rows['imag']
def main(root,output):
 work=root/'.context/diffraction-integration';result={'metric':'relative L2 complex pressure, no fitting/normalization','planar':[],'finite':[],'archiveParity':[]}
 for family in ['barrier','slit']:
  p=work/('full-'+family);cfg=json.loads((p/'scene.json').read_text());ref=root/'.context/wave-study/pilot'/f'{family}-1'/'reference-0.25-250.npz'
  target=np.load(ref);assert np.allclose(np.array(cfg['points']).reshape(-1,3),target['points'])
  rows=np.genfromtxt(p/'renderer.csv',delimiter=',',names=True);rows=np.sort(rows[rows['diffraction']==1],order='point')
  assert np.array_equal(rows['point'],np.arange(len(target['pressure'])))
  old=np.genfromtxt(work/f'map-{family}-65536'/'baseline.csv',delimiter=',',names=True);old=np.sort(old[old['bvh']==1],order='point')
  result['planar'].append(dict(family=family,points=len(rows),samples=cfg['completeRouteSamples'],runtimeVsWave=metric(field(rows),target['pressure']),originalVsWave=metric(field(old),target['pressure']),runtimeVsOriginal=metric(field(rows),field(old)),medianSolveMs=float(np.median(rows['solve_seconds'])*1000),rendererSha256=digest(p/'renderer.csv'),referenceSha256=digest(ref)))
 ref=root/'.context/wave-canopy-2026-10-27/reference-0.25-125.npz';target=np.load(ref)
 for name in ['finite-128','finite-512','finite-2048','finite-65536','full-finite']:
  p=work/name
  if not (p/'renderer.csv').exists():continue
  cfg=json.loads((p/'scene.json').read_text());points=np.array(cfg['points']).reshape(-1,3)
  ids=np.array([np.argmin(np.linalg.norm(target['points']-pt,axis=1)) for pt in points]);assert np.allclose(points,target['points'][ids]);gt=target['pressure'][ids]
  data=np.genfromtxt(p/'renderer.csv',delimiter=',',names=True)
  for seed in cfg['seeds']:
   for diff in [0,1]:
    rows=np.sort(data[(data['seed']==seed)&(data['diffraction']==diff)],order='point');assert np.array_equal(rows['point'],np.arange(len(ids)))
    value=field(rows);item=dict(name=name,samples=cfg['completeRouteSamples'],maxBounces=cfg['maxBounces'],seed=seed,diffraction=diff,points=len(rows),complexRelativeL2=metric(value,gt),medianSolveMs=float(np.median(rows['solve_seconds'])*1000),zeros=int(np.count_nonzero(abs(value)==0)),rendererSha256=digest(p/'renderer.csv'),referenceSha256=digest(ref))
    shadow=rows['direct_visible']==0
    if np.any(shadow):item.update(shadowPoints=int(np.count_nonzero(shadow)),shadowComplexRelativeL2=metric(value[shadow],gt[shadow]))
    if 323 in ids:
     i=list(ids).index(323);item['indoor323']=dict(gtAmplitude=float(abs(gt[i])),runtimeAmplitude=float(abs(value[i])),real=float(value[i].real),imag=float(value[i].imag))
    result['finite'].append(item)
 for name in ['map-1-65536','map-barrier-65536','map-slit-65536','map-corridor-65536']:
  p=work/name;a=np.genfromtxt(p/'baseline.csv',delimiter=',',names=True);b=np.genfromtxt(p/'scalable.csv',delimiter=',',names=True)
  a=np.sort(a[a['bvh']==1],order='point');b=np.sort(b[b['bvh']==1],order='point');assert np.array_equal(a['point'],b['point'])
  result['archiveParity'].append(dict(name=name,points=len(a),relativeDifference=metric(field(a),field(b)),maxAbsoluteDifference=float(np.max(abs(field(a)-field(b)))),baselineSha256=digest(p/'baseline.csv'),currentSha256=digest(p/'scalable.csv')))
 output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('output',type=Path);a=p.parse_args();main(a.root,a.output)
