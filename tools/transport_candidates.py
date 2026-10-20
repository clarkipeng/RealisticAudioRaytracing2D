"""Silent, single-process controls for transport candidates; no Unity dependency.

Axial recurrence is a 1D plane-wave model. Cylinder kernels use the existing
2D line-source / 2.5D point-source reference conventions, not finite 3D rooms.
These experiments do not add fitted gains or a second runtime audio backend.
"""
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
from scipy.special import hankel1, h1vp, jvp

import edge_integral as edge
import planar_wave as wave


def axial(frequencies, length, source, receiver, reflection, *, speed=343., horizon=None):
    """Sum four image families, each with a common round-trip multiplier.

    Unit outgoing plane pressure in each direction. Reflection coefficients are
    physical pressure ratios; no Monte Carlo weight enters the recurrence.
    horizon is a physical path length, or None for a convergent infinite sum.
    """
    frequencies = np.atleast_1d(np.asarray(frequencies, dtype=float))
    if (frequencies.ndim != 1 or not np.isfinite(frequencies).all() or np.any(frequencies < 0) or
            not np.isfinite([length, source, receiver, speed]).all() or speed <= 0 or
            not 0 < source < length or not 0 < receiver < length or
            len(reflection) != 2 or not np.isfinite(reflection).all() or
            np.any(np.abs(reflection) > 1) or
            (horizon is not None and (not np.isfinite(horizon) or horizon < 0))):
        raise ValueError('Finite interior endpoints, passive reflections and nonnegative frequencies/support required.')
    left, right = reflection
    distance = abs(receiver-source)
    paths = np.array([distance, source+receiver, 2*length-source-receiver, 2*length-distance])
    gains = np.array([1, left, right, left*right])
    k = 2*np.pi*frequencies/speed
    q = left*right*np.exp(-2j*k*length)
    if horizon is None:
        if abs(left*right) >= 1:
            raise ValueError('Lossless loops need finite support; no artificial pole damping.')
        series = 1/(1-q)
    else:
        count = np.maximum(0, np.floor((horizon-paths)/(2*length)).astype(int)+1)
        series = np.zeros((4, frequencies.size), complex)
        # expm1 avoids cancellation near resonance. q==1 has exactly count terms.
        regular = (q != 0) & (abs(q-1) > 8*np.finfo(float).eps)
        logarithm = np.log(q[regular])
        series[:,regular] = np.expm1(count[:,None]*logarithm)/np.expm1(logarithm)
        series[:,q == 0] = (count[:,None] > 0)
        series[:,(q != 0) & ~regular] = count[:,None]
    return np.sum(gains[:,None]*np.exp(-1j*paths[:,None]*k)*series, axis=0)


class CylinderKernel:
    """Object-local diagonal scattering operator for a rigid circular cylinder.

    Incoming regular harmonics -> outgoing Hankel harmonics. Geometry is in
    object-local XY coordinates; the independent cylinder_exact remains intact.
    Uses exp(-i*w*t), matching planar_wave's internal solve (conjugate for DSP).
    """
    def __init__(self, points, k, radius, terms):
        points = np.asarray(points)
        if radius <= 0 or terms < 1 or np.any(np.linalg.norm(points, axis=1) <= radius):
            raise ValueError('Positive radius/order and exterior receivers required.')
        self.k, self.radius = k, radius
        self.modes = np.arange(-terms, terms+1)
        self.transfer = -jvp(self.modes, k*radius)/h1vp(self.modes, k*radius)
        self.outgoing = hankel1(self.modes, k*np.linalg.norm(points, axis=1)[:,None]) * np.exp(
            1j*np.arctan2(points[:,1], points[:,0])[:,None]*self.modes)

    def scattered(self, source):
        radius, angle = np.linalg.norm(source), np.arctan2(source[1], source[0])
        if radius <= self.radius:
            raise ValueError('Source must be outside the cylinder.')
        incoming = .25j*hankel1(self.modes, self.k*radius)*np.exp(-1j*self.modes*angle)
        return self.outgoing @ (self.transfer*incoming)


def axial_study():
    frequencies = np.linspace(1, 400, 2048)
    parameters = dict(length=5., source=1.1, receiver=3.2, reflection=(.92,.87))
    started = time.perf_counter()
    value = axial(frequencies, **parameters)
    closed_seconds = time.perf_counter()-started
    d = abs(parameters['receiver']-parameters['source'])
    together=parameters['source']+parameters['receiver']
    paths = np.array([d, together, 10-together, 10-d]); gains = np.array([1,.92,.87,.92*.87])
    k = 2*np.pi*frequencies/343
    started = time.perf_counter(); explicit = np.zeros(len(k), complex)
    for repeat in range(512):
        explicit += np.sum((gains*(.92*.87)**repeat)[:,None]*
                           np.exp(-1j*k*(paths[:,None]+10*repeat)), axis=0)
    explicit_seconds = time.perf_counter()-started
    error = wave.relative(value, explicit)
    assert error < 1e-12
    checks = [dict(check='infinite recurrence versus explicit images', relativeError=error)]
    for horizon in [0., 4.3, 10., 55.]:
        for coefficients in [(1.,1.),(0.,.8),(.7+.2j,.6-.1j)]:
            r0, r1 = coefficients; gain = np.array([1,r0,r1,r0*r1])
            expected = np.zeros(len(k), complex)
            for repeat in range(int(horizon/10)+1):
                lengths = paths+repeat*10; keep = lengths <= horizon
                expected += np.sum((gain[keep]*(r0*r1)**repeat)[:,None]*np.exp(-1j*k*lengths[keep,None]), axis=0)
            actual = axial(frequencies, **dict(parameters, reflection=coefficients), horizon=horizon)
            assert np.max(abs(actual-expected)) < 1e-10
    # At the fundamental, a centered source does not excite the odd axial mode.
    resonances = np.arange(1, 6)*343/10
    centered = axial(resonances, 5., 2.5, 3.2, (.999,.999))
    offcenter = axial(resonances, 5., 1.1, 3.2, (.999,.999))
    assert abs(centered[0]) < .01*abs(offcenter[0])
    checks.append(dict(check='finite support, lossless/zero/complex reflection and source node', passed=True))
    return dict(checks=checks, closedSeconds=closed_seconds, explicitSeconds=explicit_seconds,
                explicitRepeats=512, scope='Axial plane waves only; not a general 2.5D/3D cavity'), dict(
                frequency=frequencies, summed=value, explicit=explicit, resonances=resonances)


def cylinder_study(fem):
    angle = np.linspace(-np.pi, np.pi, 96, endpoint=False)
    points = 2*np.c_[np.cos(angle), np.sin(angle)]
    source, k = np.array([-3., .4]), 2*np.pi*250/343
    direct = .25j*hankel1(0, k*np.linalg.norm(points-source, axis=1))
    exact = wave.cylinder_exact(points, source, k, terms=60)
    checks, arrays = [], dict(points=points, reference=exact)
    for terms in [4,8,12,16]:
        started = time.perf_counter(); operator = CylinderKernel(points, k, 1., terms)
        build = time.perf_counter()-started
        started = time.perf_counter(); value = direct+operator.scattered(source)
        apply = time.perf_counter()-started
        error = wave.relative(value, exact)
        checks.append(dict(terms=terms, modes=2*terms+1, relativeError=error,
                           buildSeconds=build, applySeconds=apply,
                           operatorBytes=operator.transfer.nbytes+operator.outgoing.nbytes))
        arrays[f'order{terms}'] = value
    assert checks[-1]['relativeError'] < 1e-10
    # Same object operator integrated along kz for our actual point-extruded convention.
    def point_field(terms, count):
        return np.conj(wave.integrate(lambda k2: .25j*hankel1(0, np.sqrt(k2)*np.linalg.norm(points-source, axis=1))+
            CylinderKernel(points,np.sqrt(k2),1.,terms).scattered(source), k, count=count))
    target = np.conj(wave.integrate(lambda k2: wave.cylinder_exact(points,source,np.sqrt(k2),terms=60),k,count=64))
    arrays['pointReference'] = target
    point_checks = []
    for terms,count in [(12,48),(24,48),(24,64)]:
        started=time.perf_counter();value=point_field(terms,count)
        point_checks.append(dict(terms=terms,quadrature=count,relativeError=wave.relative(value,target),seconds=time.perf_counter()-started))
        arrays[f'point{terms}-{count}']=value
    assert point_checks[-1]['relativeError'] < 1e-6
    fem_checks=[]
    if fem:
        for h in [.4,.3]:
            field=wave.Field(source,points,cylinder=1.,h=h,order=4)
            value=field.solve(k*k)
            fem_checks.append(dict(mesh=h,relativeError=wave.relative(value,exact),**field.stats))
        assert fem_checks[-1]['relativeError'] < .005
    return dict(modes=checks,pointExtruded=point_checks,fem=fem_checks,
                scope='Isolated rigid cylinder; no multi-object coupling, fitted residual or runtime integration'), arrays


def edge_study(scene, reference, indices, samples, seeds):
    from scipy.stats import qmc
    config=json.loads(scene.read_text());data=np.load(reference)
    metadata=json.loads(reference.with_suffix('.json').read_text())
    if (metadata['sceneSha256'] != hashlib.sha256(scene.read_bytes()).hexdigest() or metadata['hz'] != 250 or
            config.get('physicalModel') != 'point-extruded' or config.get('materials') or
            any(config.get(key) for key in ['sourceMotion','receiverMotion','wallMotion'])):
        raise ValueError('Require a matched 250 Hz stationary rigid point-extruded reference.')
    points=np.array(config['points']).reshape(-1,3)[indices];truth=data['pressure'][indices]
    rows, arrays=[],dict(points=points,reference=truth,indices=indices)
    for name,options in [('baseline',{}),('control-variate',dict(control_variate=True)),
                         ('mixed-order-3',dict(mirror_depth=3)),('roulette',dict(roulette=.2))]:
        fields=[];seconds=[];work=[]
        for seed in seeds:
            draws=qmc.Sobol(6,scramble=True,bits=30,seed=seed).random_base2(samples.bit_length()-1)+.5/2**30
            stats={};started=time.perf_counter()
            v,c,_=edge.sampled(config,points,250,samples,seed,2,random=draws,stats=stats,**options)
            seconds.append(time.perf_counter()-started);work.append(stats.get('edgeKernels',0));fields.append(v.sum(axis=1)+c.sum(axis=1))
        fields=np.array(fields)
        if not np.isfinite(fields).all():
            raise FloatingPointError('Non-finite candidate pressure; no benchmark result will be saved.')
        expected=np.broadcast_to(truth,fields.shape);difference=fields-expected;shadow=c[:,0]==0
        rows.append(dict(method=name,samplesPerRoute=samples,seeds=len(seeds),points=len(points),
            complexRelativeL2=wave.relative(fields,expected),shadowRelativeL2=wave.relative(fields[:,shadow],expected[:,shadow]),
            residualJumpP99=float(np.quantile(abs(np.diff(difference,axis=1)),.99)),
            maximumPressure=float(abs(fields).max()),medianSeconds=float(np.median(seconds)),
            meanEdgeKernelEvaluations=float(np.mean(work))))
        arrays[name]=fields;print(rows[-1],flush=True)
    return dict(rows=rows,sceneSha256=hashlib.sha256(scene.read_bytes()).hexdigest(),
                referenceSha256=hashlib.sha256(reference.read_bytes()).hexdigest(),seeds=seeds,
                scope='CPU planar route experiment, not GPU rays/bin; elapsed costs include connections'),arrays


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('method',choices=['axial','cylinder','edges']);parser.add_argument('output',type=Path)
    parser.add_argument('--fem',action='store_true');parser.add_argument('--scene',type=Path)
    parser.add_argument('--reference',type=Path);parser.add_argument('--indices',type=int,nargs='+')
    parser.add_argument('--samples',type=int,default=256);parser.add_argument('--seeds',type=int,nargs='+',default=[17,65554,131091,196628])
    args=parser.parse_args()
    if any(args.output.with_suffix(s).exists() for s in ['.npz','.json']):
        raise FileExistsError('Choose a fresh output stem; retain previous measurements.')
    if args.method=='axial':report,arrays=axial_study()
    elif args.method=='cylinder':report,arrays=cylinder_study(args.fem)
    else:
        if args.scene is None or args.reference is None or not args.indices:parser.error('edges needs --scene, --reference and --indices')
        report,arrays=edge_study(args.scene,args.reference,args.indices,args.samples,args.seeds)
    report['sources']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in
                       [Path(__file__),Path(edge.__file__),Path(wave.__file__)]}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(args.output.with_suffix('.npz'),**arrays)
    args.output.with_suffix('.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':
    main()
