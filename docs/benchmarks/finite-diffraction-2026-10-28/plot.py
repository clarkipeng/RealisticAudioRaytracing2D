"""Unfitted 125 Hz complex-pressure comparison on the actual finite canopy."""
import json,sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
root=Path(sys.argv[1]);out=Path(__file__).resolve().parent
base=root/'.context/finite-diffraction/coverage-full'
config=json.loads((base/'scene.json').read_text())
ref=np.load(root/'.context/wave-canopy-2026-10-27/reference-0.25-125.npz')
rows=np.genfromtxt(base/'renderer.csv',delimiter=',',names=True)
rows=np.sort(rows[(rows['diffraction']==1)&(rows['seed']==0)],order='point')
assert np.array_equal(rows['point'],np.arange(len(ref['pressure'])))
actual=rows['real']+1j*rows['imag'];gt=ref['pressure'];ids=config['gridIndices']
def grid(values):
 v=np.full(len(config['x'])*len(config['y']),np.nan);v[ids]=values
 return v.reshape(len(config['y']),len(config['x']))
error=np.linalg.norm(actual-gt)/np.linalg.norm(gt)
fig,axes=plt.subplots(1,3,figsize=(13,4.3),layout='constrained')
maximum=max(np.max(abs(gt)),np.max(abs(actual)))
for ax,values,title,limit in zip(axes,[abs(gt),abs(actual),abs(actual-gt)],['Numerical wave reference','Finite 3D renderer','Absolute complex-pressure error'],[maximum,maximum,np.max(abs(actual-gt))]):
 im=ax.pcolormesh(config['x'],config['y'],grid(values),shading='nearest',cmap='magma',vmin=0,vmax=limit)
 for w in config['walls']:
  x,y,z=w['center'];a,b,c=w['half'];roof=y-b>1.6
  ax.add_patch(Rectangle((x-a,z-c),2*a,2*c,fill=not roof,facecolor='#bbc0ca',edgecolor='#94a3b8',linestyle='--' if roof else '-',linewidth=.7))
 ax.plot(config['source'][0],config['source'][2],'c*',ms=8)
 point=ref['points'][323];ax.plot(point[0],point[2],'wo',ms=4,mec='black',mew=.7)
 ax.set(title=title,xlabel='x (m)',ylabel='z (m)',aspect='equal');fig.colorbar(im,ax=ax,shrink=.7,label='Pressure magnitude')
fig.suptitle(f'Finite room + roof · 125 Hz · 65,536 samples · complex L2 error {100*error:.2f}%\nNo gain/phase fitting. Dashed outline: roof above the receiver plane; white dot: indoor probe.',fontsize=11)
fig.savefig(out/'comparison.png',dpi=160);plt.close(fig)
