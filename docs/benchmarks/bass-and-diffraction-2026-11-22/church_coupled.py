"""README part A, the Church: first-order (diagonal) wall damping against the wall admittance projected on the
rigid modes in full (a non-diagonal damping matrix, solved exactly in that basis), both against the FDTD on the
same voxel church (church_modes.py fdtd), octave band at the voxel model's Schroeder frequency.

    python3 tools/heavy_guard.py run --limit-gib 3.3 -- $PY $B/church_coupled.py
"""
import sys, json, math, numpy as np
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / 'tools')); sys.path.insert(0, str(HERE))
import room_modes as rm, church_modes as cm, method_compare_suite as mcs, fdtd_wave as fw
from scipy import ndimage
h=0.75; W=cm.WORK
rep=json.loads((W/'church-0.75.json').read_text()); bank=rm.load_bank(W/'bank-church-0.75')
mats=[str(m) for m in bank['materials']]; adm=np.array([rep['admittance'][m][0] for m in mats])
edges=[bank['edges0'],bank['edges1'],bank['edges2']]; parts,vol,cells,(fc,fa,fm)=rm.operator(edges,bank['solid'])
k=bank['k']; N=len(k); psi=np.asarray(bank['psi'][fc],np.float64)   # faces x N
C=np.zeros((N,N))
for m in range(len(mats)):
    sel=fm==m
    if sel.any(): C+=adm[m]*(psi[sel].T*fa[sel])@psi[sel]
print('diag vs first-order delta check', np.allclose(0.5*343*np.diag(C), rm.damping(k,bank['surfaceVoxel'],[[a]*8 for a in adm])))
off=np.abs(C-np.diag(np.diag(C))); print('max offdiag/diag ratio median', np.median(off.max(1)/np.diag(C)))
# FDTD reference (free case), recompute quickly
solid=bank['solid']; ids=np.where(solid<0,0,solid+1).astype(np.uint8)
grid=fw.Grid(h,tuple(float(e[0]) for e in edges),tuple(int(n) for n in solid.shape),0)
clear=ndimage.distance_transform_edt(solid<0)*h; lo=np.array([e[0] for e in edges]); rng=np.random.default_rng(1)
fs=rep['schroederHz']; src=lo+(np.array(np.unravel_index(np.argmax(clear),clear.shape))+0.5)*h
pool=np.argwhere(clear>=343/fs/4); pts=lo+(pool[rng.choice(len(pool),32,replace=False)]+rng.uniform(size=(32,3)))*h
sim=fw.Simulation(grid,ids,admittance={m+1:float(adm[m]) for m in range(len(mats))},memory_limit=2*fw.GIB)
res=sim.run(src,pts,2.05,1.2*fs); H,df=mcs.fdtd_transfer_spectrum(res['pressure'],res['source'],res['dt'])
bands=np.array([fs]); Eg,_=rm.mc.energy_from_spectrum(H,df,bands)
a=rm.probe(bank,np.vstack([src,pts])); freqs=np.arange(0,2*fs+rm.DF,rm.DF)
delta=0.5*343*np.diag(C); Hd=rm.transfer(k,delta,a[0],a[1:],freqs); Hd[:,0]=0
w=2*np.pi*np.maximum(freqs,1e-6); Om2=(k*343)**2; Hc=np.zeros_like(Hd)
for i,ww in enumerate(w):
    x=np.linalg.solve(np.diag(Om2-ww**2)+1j*ww*343*C, a[0]); Hc[:,i]=4*np.pi*343**2*(a[1:]@x)
Hc[:,0]=0
for lab,Hm in (('diagonal',Hd),('coupled',Hc)):
    Em,_=rm.mc.energy_from_spectrum(Hm,rm.DF,bands)
    lev=10*np.log10(Em[:,0,:2000].sum(1)/Eg[:,0,:2000].sum(1))
    edt=[]
    for r in range(len(pts)):
        t0=rm.mc.onset_bin(Eg[r,0,:2000]); mg=rm.mc.receiver_metrics(Eg[r,0],2000,t0); mm=rm.mc.receiver_metrics(Em[r,0],2000,t0); edt.append(100*(mm['EDT']-mg['EDT'])/mg['EDT'])
    print(lab,'level rms %.2f bias %+.2f EDT rms %.1f bias %+.1f'%(np.sqrt(np.mean(lev**2)),lev.mean(),np.sqrt(np.mean(np.square(edt))),np.mean(edt)))
