"""Preserve numerical inputs/results and implementations; never package source PCM."""
import hashlib,json,zipfile,xml.etree.ElementTree as ET
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];out=Path(__file__).resolve().parent;work=ROOT/'.context/diffraction-integration'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
checks=[];latest={}
for name in ['straightline-controls','accepted-controls','deeper-control','final-controls','final-fields']:
 p=work/name/'tests.xml'
 if not p.exists():continue
 r=ET.parse(p).getroot();tests=[]
 for t in r.findall('.//test-case'):
  item=dict(name=t.get('fullname'),result=t.get('result'),failure=t.findtext('failure/message'));tests.append(item);latest[item['name']]=item['result']
 checks.append(dict(run=name,result=r.get('result'),total=int(r.get('total')),passed=int(r.get('passed')),failed=int(r.get('failed')),tests=tests))
validation=dict(latestDistinctChecks=len(latest),latestPassed=sum(v=='Passed' for v in latest.values()),runs=checks)
(out/'validation.json').write_text(json.dumps(validation,indent=2)+'\n')
files={}
def add(p,name):
 if p.exists():files[name]=p
for p in work.rglob('*'):
 if p.is_file() and p.suffix in ['.json','.csv','.txt','.bin','.xml','.py'] and p.name!='sample.txt':add(p,str(p.relative_to(work)))
for family in ['barrier','slit']:
 add(ROOT/f'.context/wave-study/pilot/{family}-1/reference-0.25-250.npz',f'reference/{family}.npz')
add(ROOT/'.context/wave-canopy-2026-10-27/reference-0.25-125.npz','reference/canopy.npz')
add(ROOT/'.context/gpu-scalable/ScalableRoutes.compute','original/ScalableRoutes.compute')
for folder in ['Assets/Scripts','Assets/Shaders','Assets/Resources/Acoustics','Assets/Tests/Editor','Assets/Editor']:
 for p in (ROOT/folder).rglob('*'):
  if p.is_file() and p.suffix in ['.cs','.compute','.hlsl','.asmdef']:add(p,'sources/'+str(p.relative_to(ROOT)))
for p in out.iterdir():
 if p.is_file() and p.suffix in ['.md','.json','.py']:add(p,'report/'+p.name)
with zipfile.ZipFile(out/'evidence.zip','w',compression=zipfile.ZIP_DEFLATED) as z:
 for name,p in files.items():z.write(p,name)
 z.writestr('sha256.json',json.dumps({n:sha(p) for n,p in sorted(files.items())},indent=2)+'\n')
print(json.dumps(dict(files=len(files),bytes=(out/'evidence.zip').stat().st_size,validation=validation['latestPassed']),indent=2))
