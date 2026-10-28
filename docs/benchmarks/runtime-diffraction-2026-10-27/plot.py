"""Show complete planar fields on fixed shared scales, including complex error."""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('output',type=Path);a=p.parse_args()
fig,axes=plt.subplots(3,3,figsize=(12,10),layout='constrained')
for row,family in enumerate(['barrier','slit','finite']):
 base=a.root/'.context/diffraction-integration'/('full-'+family);cfg=json.loads((base/'scene.json').read_text())
 reference=np.load(a.root/(f'.context/wave-study/pilot/{family}-1/reference-0.25-250.npz' if family!='finite' else '.context/wave-canopy-2026-10-27/reference-0.25-125.npz'))
 gt=reference['pressure'];axes2=[0,1] if family!='finite' else [0,2];xy=reference['points'][:,axes2];data=np.genfromtxt(base/'renderer.csv',delimiter=',',names=True)
 data=np.sort(data[data['diffraction']==1],order='point');runtime=data['real']+1j*data['imag'];assert len(gt)==len(runtime)
 limit=max(np.max(abs(gt)),np.max(abs(runtime)));err=np.linalg.norm(runtime-gt)/np.linalg.norm(gt)
 for col,values in enumerate([abs(gt),abs(runtime),abs(runtime-gt)]):
  ax=axes[row,col];im=ax.scatter(xy[:,0],xy[:,1],c=values,s=25,marker='s',vmin=0,vmax=limit,cmap='magma')
  for wall in cfg['walls']:
   if family=='finite' and abs(wall['center'][1]-1.6)>wall['half'][1]:continue
   x,y=np.array(wall['center'])[axes2];hx,hy=np.array(wall['half'])[axes2];ax.add_patch(Rectangle((x-hx,y-hy),2*hx,2*hy,fc='#a5a5a5',ec='#444'))
  ax.set_aspect('equal');ax.set_xlabel('x (m)');ax.set_ylabel('y (m)' if family!='finite' else 'z (m)')
  ax.set_title([f'{family.title()}: wave GT','Production transport',f'Complex error: {100*err:.2f}% L2'][col]);fig.colorbar(im,ax=ax,shrink=.7,label='Pressure magnitude')
fig.suptitle('Production transport vs unchanged wave references · 65,536 early-route samples\nTop: extruded walls, 250 Hz. Bottom: finite 3D canopy, 125 Hz. No gain or phase fit.',fontsize=13)
fig.savefig(a.output,dpi=160)
