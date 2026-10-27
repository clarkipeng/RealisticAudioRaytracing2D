"""Finite-3D wave reference for one rain patch and the exact seven-box canopy."""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import numpy as np

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'tools'))
from compare_wave_field import save_json, solve


def union_surface(walls, spacing):
    """Conforming exterior mesh of an exact axis-aligned box union.

    Split at every box boundary before subdivision. Shared tensor-grid vertices
    ensure adjacent faces agree; interior faces never enter the BEM boundary.
    """
    low=np.array([np.array(w['center'])-w['half'] for w in walls])
    high=np.array([np.array(w['center'])+w['half'] for w in walls])
    axes=[]
    for axis in range(3):
        edges=np.unique(np.round(np.r_[low[:,axis],high[:,axis]],12))
        axes.append(np.r_[np.concatenate([np.linspace(a,b,int(np.ceil((b-a)/spacing))+1)[:-1]
            for a,b in zip(edges[:-1],edges[1:])]),edges[-1]])
    centers=np.meshgrid(*[(a[:-1]+a[1:])/2 for a in axes],indexing='ij')
    xyz=np.stack(centers,axis=-1)
    occupied=np.any(np.all((xyz[...,None,:]>=low)&(xyz[...,None,:]<=high),axis=-1),axis=-1)
    vertices=[];triangles=[];lookup={}
    def vertex(index):
        key=tuple(index)
        if key not in lookup:
            lookup[key]=len(vertices);vertices.append([axes[d][key[d]] for d in range(3)])
        return lookup[key]
    for cell in np.argwhere(occupied):
        for axis in range(3):
            u,v=(axis+1)%3,(axis+2)%3
            for side in [-1,1]:
                neighbor=cell.copy();neighbor[axis]+=side
                if np.all(neighbor>=0) and np.all(neighbor<occupied.shape) and occupied[tuple(neighbor)]:continue
                corner=cell.copy();corner[axis]+=side>0
                a=vertex(corner);corner[u]+=1;b=vertex(corner);corner[v]+=1;c=vertex(corner);corner[u]-=1;e=vertex(corner)
                triangles.extend([(a,b,c),(a,c,e)] if side>0 else [(a,c,b),(a,e,c)])
    vertices=np.array(vertices);triangles=np.array(triangles)
    edges=Counter(tuple(sorted((int(a),int(b)))) for t in triangles for a,b in zip(t,np.roll(t,-1)))
    directed=Counter((int(a),int(b)) for t in triangles for a,b in zip(t,np.roll(t,-1)))
    assert all(n==2 for n in edges.values()),'Boundary is not a closed 2-manifold'
    assert all(directed[(a,b)]==directed[(b,a)] for a,b in directed),'Inconsistent winding'
    a,b,c=vertices[triangles].transpose(1,0,2)
    signed_volume=float(np.einsum('ij,ij->i',a,np.cross(b,c)).sum()/6)
    volumes=np.prod(np.stack(np.meshgrid(*[np.diff(a) for a in axes],indexing='ij'),axis=-1),axis=-1)
    exact_volume=float(volumes[occupied].sum())
    assert abs(signed_volume-exact_volume)<1e-8 and signed_volume>0
    area=float(np.linalg.norm(np.cross(b-a,c-a),axis=1).sum()/2)
    return vertices,triangles,dict(triangles=len(triangles),vertices=len(vertices),area=area,
        signedVolume=signed_volume,unionVolume=exact_volume,closedManifold=True)


def prepare(output):
    fixture=ROOT/'.context/rain-research/native-first/fixture.json'
    original=json.loads(fixture.read_text())
    vector=lambda p:[p['x'],p['y'],p['z']]
    walls=[dict(center=vector(w['center']),half=vector(w['half'])) for w in original['walls']]
    source=vector(original['sources'][39]['position'])
    assert source==[3.25,.001,-1.0]
    x,z=np.linspace(-7.5,6.5,43),np.linspace(-4.5,4.5,29)
    xx,zz=np.meshgrid(x,z);points=np.c_[xx.ravel(),np.full(xx.size,1.6),zz.ravel()]
    keep=np.ones(len(points),dtype=bool)
    for w in walls:keep&=np.any(abs(points-w['center'])>np.array(w['half'])+.12,axis=1)
    output.mkdir(parents=True,exist_ok=False)
    config=dict(source=source,walls=walls,points=points[keep].ravel().tolist(),gridIndices=np.flatnonzero(keep).tolist(),
        x=x.tolist(),y=z.tolist(),frequencies=[125],sampleRate=1000,fftSize=8,speedOfSound=343,
        rays=[2,128],seeds=[0,1],maxBounces=32,maximumDelaySeconds=1.25,meshSizes=[.5,.35,.25],
        referenceConvention='exp(+i*w*t)',sourcePatch=39,rainFixtureSha256=hashlib.sha256(fixture.read_bytes()).hexdigest(),
        view=dict(axes=[0,2],fixedAxis=1,fixedValue=1.6,bounds=dict(xmin=-8,xmax=7,ymin=-5,ymax=5),
            presets=dict(clear=[4,0],first=[0,0],second=[-4,0]),presetLabels=['Outside','Under canopy','Inside room']))
    save_json(output/'scene.json',config)
    (output/'validation.json').write_bytes((ROOT/'.context/wave-field/run/validation.json').read_bytes())
    for spacing in config['meshSizes']:
        _,_,stats=union_surface(walls,spacing);print(spacing,stats,flush=True)


def reference(output,spacing):
    import bempp_cl.api as bem
    config=json.loads((output/'scene.json').read_text())
    vertices,triangles,geometry=union_surface(config['walls'],spacing)
    if len(vertices)>6500:raise RuntimeError('Mesh exceeds the bounded reference size.')
    grid=bem.Grid(vertices.T,triangles.T)
    points=np.array(config['points']).reshape(-1,3)
    field,stats=solve(grid,config['source'],points,2*np.pi*125/config['speedOfSound'])
    stem=f'reference-{spacing:g}-125'
    if (output/(stem+'.npz')).exists():raise FileExistsError(stem)
    np.savez_compressed(output/(stem+'.npz'),pressure=field,points=points)
    stats.update(geometry=geometry,hz=125,meshSize=spacing,bemppVersion=bem.__version__,
        sceneSha256=hashlib.sha256((output/'scene.json').read_bytes()).hexdigest(),
        solverSha256=hashlib.sha256((ROOT/'tools/compare_wave_field.py').read_bytes()).hexdigest(),
        geometryBuilderSha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    save_json(output/(stem+'.json'),stats)
    save_json(output/'report.json',dict(dataHashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in output.glob('*.npz')}))
    print(json.dumps(stats),flush=True)


def guarded(output,spacing):
    """One owned, single-thread, low-priority CPU worker with an RSS guard."""
    command=['nice','-n','19',str(ROOT/'.context/tone-artifacts/venv/bin/python'),str(Path(__file__).resolve()),
        'solve',str(output.resolve()),'--mesh',str(spacing)]
    process=subprocess.Popen(command,env=dict(os.environ,OPENBLAS_NUM_THREADS='1',VECLIB_MAXIMUM_THREADS='1',OMP_NUM_THREADS='1',NUMBA_NUM_THREADS='1'))
    started=time.monotonic();samples=[]
    def interrupt(signum,frame):raise KeyboardInterrupt
    previous={s:signal.signal(s,interrupt) for s in [signal.SIGINT,signal.SIGTERM]}
    try:
        while process.poll() is None:
            value=subprocess.run(['ps','-p',str(process.pid),'-o','rss=,pcpu='],capture_output=True,text=True).stdout.split()
            if len(value)==2:
                samples.append(dict(seconds=time.monotonic()-started,rssBytes=int(value[0])*1024,cpuPercent=float(value[1])))
                if samples[-1]['rssBytes']>3*1024**3:raise RuntimeError('Wave worker exceeded 3 GiB RSS.')
            time.sleep(2)
    finally:
        if process.poll() is None:
            process.terminate()
            try:process.wait(timeout=10)
            except subprocess.TimeoutExpired:process.kill();process.wait()
        for s,handler in previous.items():signal.signal(s,handler)
        save_json(output/f'resources-{spacing:g}.json',dict(exitCode=process.returncode,wallSeconds=time.monotonic()-started,samples=samples))
    if process.returncode:raise RuntimeError(f'Wave worker exited {process.returncode}')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['prepare','reference','solve']);p.add_argument('output',type=Path);p.add_argument('--mesh',type=float,default=.5)
    a=p.parse_args()
    if a.action=='prepare':prepare(a.output)
    elif a.action=='reference':guarded(a.output,a.mesh)
    else:reference(a.output,a.mesh)
