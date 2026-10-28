"""Unfitted finite-room third-order audit; argument is the extracted evidence root."""
import csv,hashlib,json,sys,xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=Path(sys.argv[1]).resolve();base=root/'.context/third-order';out=Path(__file__).resolve().parent
refpath=root/'.context/wave-canopy-2026-10-27/reference-0.25-125.npz';ref=np.load(refpath)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def measure(folder,ids):
 c=json.loads((folder/'scene.json').read_text());r=np.atleast_1d(np.genfromtxt(folder/'renderer.csv',delimiter=',',names=True))
 assert np.allclose(np.array(c['points']).reshape(-1,3),ref['points'][ids],rtol=0,atol=1e-12)
 result=[]
 for seed in sorted(set(r['seed'])):
  a=np.sort(r[(r['diffraction']==1)&(r['seed']==seed)],order='point');assert np.array_equal(a['point'],np.arange(len(ids)))
  p=a['real']+1j*a['imag'];target=ref['pressure'][ids];shadow=a['direct_visible']==0
  result.append(dict(seed=int(seed),samples=c['completeRouteSamples'],order=c.get('completeRouteOrder',2),receivers=len(ids),
   complexRelativeL2=float(np.linalg.norm(p-target)/np.linalg.norm(target)),shadowComplexRelativeL2=float(np.linalg.norm((p-target)[shadow])/np.linalg.norm(target[shadow])),
   medianSolveMs=float(np.median(a['solve_seconds'])*1000),pressure=[[z.real,z.imag] for z in p] if len(ids)<10 else None,sha256=sha(folder/'renderer.csv')))
 return result
report=dict(acceptedBakeRevision='0122256',productionRevision='0cbd066',referenceSha256=sha(refpath),referenceIndoor=[ref['pressure'][323].real,ref['pressure'][323].imag],
 scope='Same finite seven-box room/roof, rigid materials, 125 Hz numerical BEM reference. No gain/phase fitting, clipping or smoothing.',
 runs={},measurements={})
for name in ['pilot-run','compile2','controls','culled','checked','balanced','guided','guided-fresh','production-final']:
 folder=base/name
 if not (folder/'tests.xml').exists():continue
 xml=ET.parse(folder/'tests.xml').getroot();resource=json.loads((folder/'resources.json').read_text())
 checks=[dict(name=t.get('fullname'),result=t.get('result'),failure=t.findtext('failure/message')) for t in xml.iter('test-case')]
 report['runs'][name]=dict(results=xml.attrib,checks=checks,exitCode=resource['exitCode'],wallSeconds=resource['wallSeconds'],
  peakParentRssBytes=max([s['rssBytes'] for s in resource['samples']],default=0))
 provenance=json.loads((folder/'provenance.json').read_text());snapshot=folder/'source'
 if snapshot.exists():
  for path,expected in provenance.items():assert sha(snapshot/path)==expected,(name,path)
  report['runs'][name]['verifiedSourceFiles']=len(provenance)
for stage in ['checked','balanced']:
 for name,ids in [('indoor',[323]),('probes-128',[0,323,324,700,900]),('probes-65536',[0,323,324,700,900]),('full',list(range(len(ref['pressure'])))),('two-order',[0,323,324,700,900])]:
  folder=base/(stage+'-'+name)
  if (folder/'renderer.csv').exists():report['measurements'][stage+'-'+name]=measure(folder,ids)
for name in ['pilot','guided-indoor','guided-fresh-indoor']:
 folder=base/name
 if (folder/'renderer.csv').exists():report['measurements'][name]=measure(folder,[323])
for name,ids in [('indoor',[323]),('full',list(range(len(ref['pressure'])))),('probes-65536',[0,323,324,700,900])]:
 report['measurements']['accepted-'+name]=measure(root/'.context/solid-boundary'/('ordered-'+name),ids)
report['twoOrderParity']={}
for stage in ['checked','balanced']:
 rows=[]
 for a,b in zip(report['measurements']['accepted-probes-65536'],report['measurements'][stage+'-two-order']):
  x=np.array(a['pressure']);y=np.array(b['pressure']);rows.append(dict(seed=a['seed'],maxAbsolute=float(np.max(np.linalg.norm(x-y,axis=1))),relativeL2=float(np.linalg.norm(x-y)/np.linalg.norm(x))))
 report['twoOrderParity'][stage]=rows
for name in ['checked-indoor','balanced-indoor','guided-indoor','guided-fresh-indoor']:
 folder=base/name
 if (folder/'families.csv').exists():report.setdefault('families',{})[name]=list(csv.DictReader((folder/'families.csv').open()))
 if (folder/'edge-classification.json').exists():report.setdefault('edgeClassification',{})[name]=json.loads((folder/'edge-classification.json').read_text())
report['invalidValidationRun']={'guided':'Source changed but fields/controls and shader warning line numbers match the prior binary; conditional include was not recompiled. Use guided-fresh to assess the remaining-path proposal.'}
(out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
fig,axes=plt.subplots(1,2,figsize=(11.3,5.2),layout='constrained')
ax=axes[0];gt=complex(*report['referenceIndoor']);ax.plot(gt.real,gt.imag,'k*',ms=13,label='Numerical wave reference')
for key,label,color in [('accepted-indoor','Accepted bake / two interactions','#0284c7'),('checked-indoor','Third order / global budget','#a16207'),('balanced-indoor','Third order / family budgets','#e11d48'),('guided-fresh-indoor','Third order / remaining-path proposal','#7c3aed')]:
 if key not in report['measurements']:continue
 rows=report['measurements'][key];p=np.array([r['pressure'][0] for r in rows]);ax.plot(p[:,0],p[:,1],'o--',color=color,label=label)
 for r,v in zip(rows,p):ax.annotate(str(r['seed']),v,xytext=(4,4),textcoords='offset points',fontsize=8)
ax.axhline(0,color='#aaa',lw=.6);ax.axvline(0,color='#aaa',lw=.6);ax.set(xlabel='Real pressure',ylabel='Imaginary pressure',title='Indoor receiver · 1,048,576 samples / seed')
ax.legend(fontsize=7,loc='upper left',bbox_to_anchor=(0,-.17));ax.grid(alpha=.18)
ax=axes[1];keys=['accepted-full','checked-full','balanced-full'];labels=['Accepted bake','Third / global','Third / families'];x=np.arange(3)
for i,(metric,label,color) in enumerate([('complexRelativeL2','All 1,166 receivers','#0284c7'),('shadowComplexRelativeL2','Occluded receivers','#e11d48')]):
 values=[report['measurements'][k][0][metric]*100 for k in keys];bars=ax.bar(x+(i-.5)*.34,values,width=.34,color=color,label=label)
 ax.bar_label(bars,fmt='%.1f%%',fontsize=8,padding=3)
ax.set(xticks=x,xticklabels=labels,ylabel='Complex relative L2 error (%)',title='Full room · 65,536 samples · seed 0');ax.legend(fontsize=8);ax.set_ylim(0,390)
fig.suptitle('Longer-path prototype fails the room-quality gate · 125 Hz · no fitting or output clipping',fontsize=12)
fig.savefig(out/'comparison.png',dpi=160);plt.close(fig)
print(json.dumps({k:v for k,v in report['measurements'].items() if k in ['guided-fresh-indoor','balanced-full']},indent=2))
