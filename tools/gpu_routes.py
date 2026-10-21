"""Prepare/check the opt-in Unity CompleteRoutes GPU integration experiment.

CPU plans complete visible routes; the GPU integrates their edge heights and
reduces complex pressure. Direct/mirror/mixed fields are outside this benchmark.
No scene windows, playback, or changes to the live renderer are required.
"""
import argparse
import hashlib
import json
import struct
import time
from pathlib import Path
import numpy as np
from scipy.stats import qmc
import edge_integral as edge

def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def save(path, value): path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')

def steps(chain, source, receiver, horizon):
    points = [source[:2]]+[e[0] for e in chain]+[receiver[:2]]
    radii = np.linalg.norm(np.diff(points, axis=0), axis=1)
    terms = [edge.angles(e, points[i], points[i+2]) for i,e in enumerate(chain)]
    terms = [p[abs(np.sin(p/2))>1e-12] for p in terms]
    if radii.sum()>=horizon or any(len(p)==0 for p in terms): return [], 1.
    result, shared = [], 1.
    for i, (e, phi) in enumerate(zip(chain,terms)):
        sine, half = np.zeros(4), np.zeros(4)
        sine[:len(phi)], half[:len(phi)] = np.sin(phi), np.sin(phi/2)
        result.append(np.r_[[radii[i],radii[i+1],sum(radii[i+1:]),np.pi/e[4]],sine,half])
        if i and chain[i-1][3]==e[3] and np.any(abs(chain[i-1][0]-e[0])<1e-10): shared *= .5
    return result, shared

def prepare(output, scene, count, depth, seed, hz, indices=None, cpu=False, capture=False):
    output.mkdir(parents=True,exist_ok=True)
    if (output/'requests.bin').exists(): raise FileExistsError(output/'requests.bin')
    if count<1 or count&(count-1) or depth<1 or not np.isfinite(hz) or hz<0:
        raise ValueError('Positive power-of-two sample count, positive depth and nonnegative finite frequency required.')
    config = json.loads(scene.read_text())
    if config.get('physicalModel','point-extruded')!='point-extruded': raise ValueError('Planar extruded boxes only.')
    allpoints = np.array(config['points']).reshape(-1,3)
    ids = np.arange(len(allpoints)) if indices is None else np.asarray(indices)
    points, source, walls = allpoints[ids], np.array(config['source']), config['walls']
    if np.any(points[:,2]!=0) or source[2]!=0: raise ValueError('XY source and receivers required.')
    started = time.perf_counter()
    edges = edge.corners(walls)
    neighbors = [[j for j,b in enumerate(edges) if i!=j and edge.visible(a[0],b[0],walls)] for i,a in enumerate(edges)]
    starts = [i for i,e in enumerate(edges) if edge.visible(source,e[0],walls)]
    random = qmc.Sobol(2*depth+1,scramble=True,bits=30,seed=seed).random_base2(count.bit_length()-1)+.5/2**30
    # Identical representable open-interval coordinates for CPU and GPU parity.
    random = np.clip(random.astype(np.float32),np.nextafter(np.float32(0),np.float32(1)),
                     np.nextafter(np.float32(1),np.float32(0))).astype(float)
    geometry_seconds = time.perf_counter()-started
    horizon, k = config['maximumDelaySeconds']*config['speedOfSound'], 2*np.pi*hz/config['speedOfSound']
    metadata = dict(samples=count,depth=depth,seed=seed,hz=hz,indices=ids.tolist(),sceneSha256=sha(scene),
        solverSha256=sha(Path(edge.__file__)),driverSha256=sha(Path(__file__)),geometrySeconds=geometry_seconds,
        scope='Pure-edge integration. CPU route preparation; GPU height sampling, signed kernels, phase and reduction. No direct/mirror/mixed field.',probes=[])
    maxroutes=maxsteps=0; expected=[]; contributions=[]; offsets=[0]
    with (output/'requests.bin').open('wb') as binary:
        binary.write(bytes(40)); binary.write(random[:,:2*depth].astype('<f4').tobytes())
        for point, receiver in enumerate(points):
            started=time.perf_counter(); jobs=[]; records=[]; end=0; value=0j; cpu_seconds=0.
            for path, selected, weight in edge.route_samples(edges,neighbors,starts,receiver,walls,depth,count,random[:,-1]):
                chain=[edges[i] for i in path]
                data, shared=steps(chain,source,receiver,horizon); end+=selected
                jobs.append(struct.pack('<IIIf',len(records),len(data),end,2*weight*shared/selected))
                records.extend(data)
                if cpu or capture:
                    before=time.perf_counter()
                    value+=edge.sample_chain(chain,source,receiver,k,horizon,random[:selected])*weight
                    if capture:
                        contributions.extend(edge.sample_chain(chain,source,receiver,k,horizon,draw[None,:])*weight/selected
                                             for draw in random[:selected])
                    cpu_seconds+=time.perf_counter()-before
            assert end in (0,count)
            values=np.asarray(records,dtype='<f4').reshape(-1,12)
            if not np.isfinite(values).all(): raise ValueError('Non-finite route geometry.')
            binary.write(struct.pack('<III',len(jobs),len(records),end));binary.write(b''.join(jobs));binary.write(values.tobytes())
            maxroutes=max(maxroutes,len(jobs));maxsteps=max(maxsteps,len(records));expected.append(value);offsets.append(len(contributions))
            metadata['probes'].append(dict(point=point,routes=len(jobs),steps=len(records),
                preparationSeconds=time.perf_counter()-started-cpu_seconds,cpuIntegrationSeconds=cpu_seconds))
        binary.seek(0)
        binary.write(struct.pack('<8I2f',0x524d4331,2*depth,count,len(points),maxroutes,maxsteps,count,int(capture),k,horizon))
    metadata['requestBytes']=(output/'requests.bin').stat().st_size
    metadata['requestSha256']=sha(output/'requests.bin')
    save(output/'metadata.json',metadata);save(output/'scene.json',config)
    np.savez_compressed(output/'cpu.npz',pressure=expected,points=points,indices=ids,
        contributions=contributions,offsets=offsets,computed=cpu or capture)
    print(output.name,'prepared',len(points),'positions;',metadata['requestBytes'],'bytes',flush=True)

def check(output):
    cpu=np.load(output/'cpu.npz'); data=np.genfromtxt(output/'gpu.csv',delimiter=',',names=True,ndmin=1)
    meta=json.loads((output/'metadata.json').read_text())
    assert sha(output/'requests.bin')==meta['requestSha256']
    np.testing.assert_array_equal(data['point'],np.arange(len(cpu['points'])))
    actual=data['real']+1j*data['imag']; assert np.isfinite(actual).all()
    report=dict(probes=len(actual),samples=meta['samples'],depth=meta['depth'],hz=meta['hz'],
        medianSolveMs=float(np.median(data['solve_seconds'])*1000),
        p95SolveMs=float(np.quantile(data['solve_seconds'],.95)*1000),gpuBytes=int(data['gpu_bytes'].max()),
        cpuPreparationSeconds=meta['geometrySeconds']+sum(r['preparationSeconds'] for r in meta['probes']),
        gpuSolveSeconds=float(data['solve_seconds'].sum()),uploadSeconds=float(data['upload_seconds'].sum()),
        inputSeconds=float(data['input_seconds'].sum()),cpuIntegrationSeconds=sum(r['cpuIntegrationSeconds'] for r in meta['probes']))
    if cpu['computed']:
        expected=cpu['pressure']; norm=np.linalg.norm(expected)
        report['cpuGpuRelativeL2']=float(np.linalg.norm(actual-expected)/max(norm,1e-30))
        report['maximumAbsoluteDifference']=float(abs(actual-expected).max())
        assert np.allclose(actual,expected,rtol=.002,atol=2e-5),report
        assert norm==0 or report['cpuGpuRelativeL2']<.001,report
    if (output/'contributions.bin').exists():
        with (output/'contributions.bin').open('rb') as binary:
            for i in range(len(actual)):
                count=struct.unpack('<I',binary.read(4))[0]
                value=np.frombuffer(binary.read(count*8),'<f4').reshape(-1,2);value=value[:,0]+1j*value[:,1]
                expected=cpu['contributions'][cpu['offsets'][i]:cpu['offsets'][i+1]]
                np.testing.assert_allclose(value,expected,rtol=.002,atol=2e-5)
                np.testing.assert_allclose(value.astype(complex).sum(),actual[i],rtol=2e-6,atol=1e-7)
            assert not binary.read()
        report['perSampleParity']=True
    save(output/'checked.json',report);print(output.name,json.dumps(report),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('command',choices=['prepare','check']);parser.add_argument('output',type=Path)
    parser.add_argument('--scene',type=Path);parser.add_argument('--samples',type=int,default=65536);parser.add_argument('--depth',type=int,default=2)
    parser.add_argument('--seed',type=int,default=17);parser.add_argument('--hz',type=float,default=250);parser.add_argument('--points',type=int,nargs='+')
    parser.add_argument('--cpu',action='store_true');parser.add_argument('--capture',action='store_true');args=parser.parse_args()
    if args.command=='check':check(args.output)
    else:
        if args.scene is None:parser.error('--scene is required for prepare')
        prepare(args.output,args.scene,args.samples,args.depth,args.seed,args.hz,args.points,args.cpu,args.capture)
