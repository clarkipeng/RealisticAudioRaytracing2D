"""Compare identical native canopy takes with an independent first-image PCM control."""
import argparse,hashlib,importlib.util,json,sys
from pathlib import Path
import numpy as np
from scipy.io import wavfile
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'tools'))
from compare_wave_field import shadow_mask
spec=importlib.util.spec_from_file_location('mirrors',ROOT/'docs/benchmarks/rain-transport-2026-10-26/mirrors.py')
mirrors=importlib.util.module_from_spec(spec);spec.loader.exec_module(mirrors)
def rms(a):return float(np.sqrt(np.mean(np.asarray(a,dtype=float)**2)))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main(focus,old,new,out):
 out.mkdir(parents=True,exist_ok=False)
 fixture=json.loads((focus/'fixture.json').read_text());before=json.loads((old/'run.json').read_text());after=json.loads((new/'run.json').read_text())
 assert before['complete'] and after['complete']
 assert before['tailCoversResponseSupport'] and after['tailCoversResponseSupport']
 for key in ['sceneHash','takeSourceHash','takeStateHash','takeEventHash','sampleRate','outputSamples','dspLatencySamples']:
  assert before[key]==after[key],f'Mismatched {key}'
 for key in ['maxBounces','raysPerBin','diffractionFactor','fftSize','spectralHop','maximumDelaySeconds','seed']:
  assert before['settings'][key]==after['settings'][key],f'Mismatched setting {key}'
 for key,value in before['settings'].items():
  if key not in ['reflectionSampling','reflectionCandidates']:
   assert after['settings'].get(key)==value,f'Mismatched setting {key}'
 rate=fixture['rate'];latency=after['dspLatencySamples']
 rb,b=wavfile.read(old/'listener-0.wav');ra,a=wavfile.read(new/'listener-0.wav')
 assert ra==rb==rate and b.shape==a.shape and a.ndim==1 and np.isfinite(a).all() and np.isfinite(b).all()
 # Three seconds after the identical two-second input warmup. No normalization.
 first,last=2*rate,5*rate;n=np.arange(first,last,dtype=float)
 walls=[dict(center=[box['center'][axis] for axis in 'xyz'],half=[box['half'][axis] for axis in 'xyz']) for box in fixture['walls']]
 listener=np.array([fixture['positions'][0]['position'][axis] for axis in 'xyz'])
 assert all(p['position']==fixture['positions'][0]['position'] for p in fixture['positions']), 'Static oracle only'
 direct=np.zeros(len(n));reflected=np.zeros(len(n));count=0
 for source in fixture['sources']:
  path=focus/source['file'];assert sha(path)==source['sha256']
  pcm=np.memmap(path,dtype='<f4',mode='r');position=np.array([source['position'][axis] for axis in 'xyz'])
  length=np.linalg.norm(listener-position)
  if not shadow_mask(dict(source=position,walls=walls),listener[None])[0]:
   direct+=mirrors.delayed(pcm,n-length/343*rate)*source['gain']/length
  for length in mirrors.paths(position,listener,walls).values():
   reflected+=mirrors.delayed(pcm,n-length/343*rate)*source['gain']*np.sqrt(.65*.85)/length;count+=1
 oracle=direct+reflected;audio={'before':b[first+latency:last+latency],'integrated':a[first+latency:last+latency],'first-images':oracle}
 for name,pcm in audio.items():wavfile.write(out/(name+'.wav'),rate,np.asarray(pcm,dtype=np.float32))
 def metrics(x):return dict(rms=rms(x),peak=float(np.max(np.abs(x))),samplesAboveUnity=int(np.count_nonzero(np.abs(x)>1)))
 def cost(run):return {k:run[k] for k in ['workingNumericBytes','preparedPcmBytes','solveSeconds','audioSeconds','preparationSeconds','wallSeconds','gpuAudio']}
 report=dict(scope='Static canopy, identical recorded rain sources; 32 surface interactions retained; diffraction disabled as in the original take. The independent PCM control includes direct and every first image, not full wave propagation or higher-order scattering.',
  inputHashes={k:after[k] for k in ['sceneHash','takeSourceHash','takeStateHash','takeEventHash']},rate=rate,sourceCount=len(fixture['sources']),validFirstMirrors=count,listenSeconds=3,
  sourceTimeWindowSeconds=[2,5],dspLatencySamples=latency,normalization='None; common original gain',
  before=dict(cost=cost(before),audio=metrics(audio['before']),fullRender=metrics(b),wavSha256=sha(old/'listener-0.wav')),
  integrated=dict(cost=cost(after),audio=metrics(audio['integrated']),fullRender=metrics(a),wavSha256=sha(new/'listener-0.wav')),
  control=dict(audio=metrics(oracle),directRms=rms(direct),mirrorRms=rms(reflected)),
  comparedToFirstImages={k:dict(relativeRmsError=rms(v-oracle)/rms(oracle),correlation=float(np.corrcoef(v,oracle)[0,1])) for k,v in audio.items() if k!='first-images'})
 (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
 template=Path(__file__).with_name('view.html').read_text()
 (out/'index.html').write_text(template.replace('{{REPORT}}',json.dumps(report).replace('<','\\u003c')))
 # Shared axes, no per-trace scaling. The short window shows delay/cancellation;
 # envelopes show the entire listen interval without presenting it as wave truth.
 import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
 plt.rcParams.update({'font.size':10,'figure.facecolor':'#0b1018','axes.facecolor':'#111a26','text.color':'#dce7f4','axes.labelcolor':'#a6b8cc','xtick.color':'#8da2bb','ytick.color':'#8da2bb','axes.edgecolor':'#344354','grid.color':'#293547'})
 fig,axes=plt.subplots(2,1,figsize=(11,6),layout='constrained')
 colors={'before':'#ed9c77','integrated':'#6accca','first-images':'#b6a1ee'}
 for key,x in audio.items():
  y=np.asarray(x,dtype=float);samples=range(rate,rate+882)
  axes[0].plot(np.arange(882)/rate*1000,y[rate:rate+882],lw=.8,color=colors[key],label=key)
  size=rate//20;chunks=y[:len(y)//size*size].reshape(-1,size);env=np.sqrt(np.mean(chunks*chunks,axis=1))
  axes[1].plot((np.arange(len(env))+.5)*size/rate,env,color=colors[key],label=key)
 axes[0].set(xlabel='Time within a 20 ms excerpt (ms)',ylabel='Pressure amplitude',title='Same source recording and gain · first-reflection integration')
 axes[1].set(xlabel='Audition time (s)',ylabel='50 ms RMS amplitude');axes[0].legend(loc='upper right')
 for ax in axes:ax.grid(alpha=.4)
 fig.savefig(out/'waveforms.png',dpi=160);plt.close(fig)
 print(json.dumps(report,indent=2))
if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('focus',type=Path);p.add_argument('old',type=Path);p.add_argument('new',type=Path);p.add_argument('output',type=Path)
 a=p.parse_args();main(a.focus,a.old,a.new,a.output)
