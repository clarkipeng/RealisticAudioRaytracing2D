import argparse, json, hashlib
from pathlib import Path
import numpy as np
from scipy.io import wavfile
from scipy.signal import welch

p=argparse.ArgumentParser();p.add_argument('capture',type=Path);p.add_argument('audio',type=Path);p.add_argument('output',type=Path);a=p.parse_args()
meta=json.loads((a.capture/'take/take.json').read_text());run=json.loads((a.audio/'run.json').read_text())
rate,pcm=wavfile.read(a.audio/'listener-0.wav');pcm=np.asarray(pcm,dtype=np.float64)
assert run['complete'] and np.isfinite(pcm).all()
assert all(run[x]==meta[y] for x,y in [('sceneHash','sceneHash'),('takeStateHash','stateHash'),('takeEventHash','eventHash')])
pcm=pcm[run['dspLatencySamples']:]
result={'sampleRate':rate,'duration':len(pcm)/rate,'peak':float(abs(pcm).max()),'overFullScale':int((abs(pcm)>1).sum()),'rms':float(np.sqrt(np.mean(pcm*pcm))),
 'geometryHash':hashlib.sha256((a.capture/'take/geometry.bin').read_bytes()).hexdigest(),'stateHash':meta['stateHash'],'eventHash':meta['eventHash'],'sceneHash':meta['sceneHash'],'workingNumericBytes':run['workingNumericBytes'],'renderWallSeconds':run['wallSeconds'],
 'sha256':hashlib.sha256((a.audio/'listener-0.wav').read_bytes()).hexdigest(),'windows':{}}
for name,start,end in [('outside',1,5),('closedInterior',17,21),('outsideReturn',32,35.5)]:
 x=pcm[round(start*rate):round(end*rate)];f,power=welch(x,rate,nperseg=4096)
 rms=float(np.sqrt(np.mean(x*x)));low=np.trapezoid(power[(f>=80)&(f<1000)],f[(f>=80)&(f<1000)]);high=np.trapezoid(power[(f>=1000)&(f<10000)],f[(f>=1000)&(f<10000)])
 blocks=[np.sqrt(np.mean(b*b)) for b in np.array_split(x,max(1,round((end-start)*4)))]
 result['windows'][name]={'seconds':[start,end],'rms':rms,'dBFS':20*np.log10(max(1e-20,rms)),'peak':float(abs(x).max()),'quarterSecondRmsMin':float(min(blocks)),'quarterSecondRmsMax':float(max(blocks)),
 'lowBandPower80To1000':float(low),'highBandPower1000To10000':float(high)}
w=result['windows'];result['insideVsOutsideDb']=w['closedInterior']['dBFS']-w['outside']['dBFS']
a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
