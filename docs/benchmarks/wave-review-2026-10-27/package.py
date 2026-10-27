"""Package a matched numerical-wave / current-renderer comparison."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import numpy as np

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'tools'))
from compare_wave_field import shadow_mask


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(reference, measured, output, hz, fine_mesh, coarse_mesh):
    original=json.loads((reference/'scene.json').read_text())
    config=json.loads((measured/'scene.json').read_text())
    provenance=json.loads((measured/'provenance.json').read_text())
    old_report=json.loads((reference/'report.json').read_text())
    assert digest(reference/'scene.json')==provenance['originalSceneSha256']
    assert digest(measured/'scene.json')==provenance['measurementSceneSha256']
    for key in ['source','walls','points','gridIndices','x','y','speedOfSound']:
        assert original[key]==config[key],f'Physical input mismatch: {key}'
    fine_name=f'reference-{fine_mesh:g}-{hz:g}.npz';coarse_name=f'reference-{coarse_mesh:g}-{hz:g}.npz'
    for name in [fine_name,coarse_name]:
        assert digest(reference/name)==old_report['dataHashes'][name]
        meta=json.loads((reference/(name[:-4]+'.json')).read_text())
        assert meta['sceneSha256']==provenance['originalSceneSha256']
    for name,expected in provenance['rendererFiles'].items():
        assert digest(ROOT/name)==expected,f'Renderer changed during measurement: {name}'
    assert digest(ROOT/'Assets/Tests/Editor/WaveFieldMeasurement.cs')==provenance['fixtureSha256']
    tests=ET.parse(measured/'tests.xml').getroot()
    assert tests.get('result')=='Passed' and int(tests.get('passed'))==1
    resources=json.loads((measured/'resources.json').read_text())
    assert resources['exitCode']==0
    points=np.array(config['points']).reshape(-1,3)
    fine=np.load(reference/fine_name)
    coarse=np.load(reference/coarse_name)
    fine_meta=json.loads((reference/fine_name.replace('.npz','.json')).read_text())
    assert np.array_equal(points,fine['points']) and np.array_equal(points,coarse['points'])
    target=fine['pressure'];previous=coarse['pressure']
    assert np.isfinite(target).all()
    shadow=shadow_mask(config,points)
    # Unity serializes these inputs to float32. At exact grazing a rounded
    # segment can fall on the other side; keep all points in field scoring.
    native_geometry=dict(source=np.float32(config['source']).tolist(),walls=[dict(
        center=np.float32(w['center']).tolist(),half=np.float32(w['half']).tolist()) for w in config['walls']])
    native_shadow=shadow_mask(native_geometry,points.astype(np.float32).astype(float))
    visibility_changes=np.flatnonzero(shadow!=native_shadow).tolist()
    norm=np.linalg.norm(target)
    rows=np.genfromtxt(measured/'renderer.csv',delimiter=',',names=True)
    assert len(rows)==len(points)*len(config['rays'])*len(config['seeds'])*2
    cases=[]; fields={}
    for rays in config['rays']:
        for seed in config['seeds']:
            for diffraction in [0,1]:
                select=rows[(rows['rays']==rays)&(rows['seed']==seed)&(rows['diffraction']==diffraction)&(rows['hz']==hz)]
                select=np.sort(select,order='point')
                assert np.array_equal(select['point'],np.arange(len(points)))
                assert np.array_equal(select['direct_visible'].astype(bool),~native_shadow)
                actual=select['real']+1j*select['imag']
                assert np.isfinite(actual).all()
                key=f'r{rays}-s{seed}-d{diffraction}'
                fields[key]=dict(real=actual.real.tolist(),imag=actual.imag.tolist())
                cases.append(dict(key=key,rays=rays,seed=seed,diffraction=bool(diffraction),
                    complexRelativeL2=float(np.linalg.norm(actual-target)/norm),
                    amplitudeRelativeL2=float(np.linalg.norm(abs(actual)-abs(target))/norm),
                    shadowComplexRelativeL2=float(np.linalg.norm((actual-target)[shadow])/np.linalg.norm(target[shadow])),
                    zeroPoints=int(np.sum(abs(actual)<1e-12)),maxPressure=float(abs(actual).max()),
                    medianSolveMs=float(np.median(select['solve_seconds'])*1000)))
    validation=json.loads((reference/'validation.json').read_text())
    sphere=[v for v in validation if v['hz']==hz and v['refinement']==4][0]
    canopy='sourcePatch' in config
    report=dict(scope=('Stationary 125 Hz field from rain patch 39 through the seven-box canopy with rigid surfaces. One source, no reflecting floor, no roof vibration, no broadband rain mix.' if canopy else 'Stationary 250 Hz field around two finite rigid 3D blocks; not the rain/canopy scene or broadband audio.'),
        hz=hz,referenceMesh=fine_meta,
        reference='Independent Helmholtz boundary-element solution; exp(+iwt), exp(-ikr)/r; unit amplitude at 1 m.',
        renderer='Current native GPU transport; 32 bounces, 1.25 s support; no gain fitting or output normalization.',
        referenceGlobalRefinementChange=float(np.linalg.norm(target-previous)/norm),
        referenceShadowRefinementChange=float(np.linalg.norm((target-previous)[shadow])/np.linalg.norm(target[shadow])),
        referenceSphereControlError=sphere['relativeL2'],
        points=len(points),shadowPoints=int(shadow.sum()),cases=cases,
        float32VisibilityChanges=[dict(point=i,position=points[i].tolist(),referenceGeometryBlocked=bool(shadow[i]),nativeGeometryBlocked=bool(native_shadow[i])) for i in visibility_changes],
        rendererCsvSha256=digest(measured/'renderer.csv'),
        referenceSha256=digest(reference/fine_name),
        referenceValidationSha256=digest(reference/'validation.json'),
        provenance=provenance,resourceWallSeconds=resources['wallSeconds'],
        peakSampledRssBytes=max(s['rssBytes'] for s in resources['samples']),
        timingScope='Warm transport queries including readback; excludes synthesis, startup and pacing. Four additional FFT bins are computed but not evaluated here.',
        limits='Mesh changes and analytic controls are numerical evidence, not certified error bounds or measured-world truth. Wave reference is steady-state; native support/depth are finite.')
    output.mkdir(parents=True,exist_ok=False)
    (output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    data=dict(config=config,points=points.tolist(),shadow=shadow.tolist(),nativeShadow=native_shadow.tolist(),reference=dict(real=target.real.tolist(),imag=target.imag.tolist()),
        fields=fields,report=report,referenceRms=float(np.sqrt(np.mean(abs(target)**2))))
    (output/'index.html').write_text(Path(__file__).with_name('view.html').read_text().replace('{{DATA}}',json.dumps(data)))
    if not canopy:
        ledger=ROOT/'docs/benchmarks/gpu-route-reuse-2026-10-21.json'
        prototype=json.loads(ledger.read_text())
        assert digest(ledger.parent/prototype['archive']['file'])==prototype['archive']['sha256']
        names={'barrier':'Thick wall','staggered':'Staggered walls','slit':'Slit','corridor':'Corridor','1':'One-box layout','9':'Nine boxes','25':'Twenty-five boxes'}
        by_scene={v['scene']:v for v in prototype['results']['maps'] if v['mode']==2 and v['samples']==1048576}
        rows=''.join(f'<tr><td>{label}</td><td>{by_scene[key]["errorPercent"]:.2f}%</td></tr>' for key,label in names.items())
        (output/'prototype.html').write_text(Path(__file__).with_name('prototype.html').read_text().replace('{{ROWS}}',rows))
        (output/'prototype-report.json').write_bytes(ledger.read_bytes())
        for size in ['small','large']:(output/f'prototype-{size}.png').write_bytes((ledger.parent/f'gpu-reuse-{size}-2026-10-21.png').read_bytes())
    # Export a conventional scientific figure, with identical level scales.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    plt.rcParams.update({'figure.facecolor':'#10141b','axes.facecolor':'#181e28','text.color':'#e8edf5',
        'axes.labelcolor':'#adb9cb','xtick.color':'#adb9cb','ytick.color':'#adb9cb','axes.edgecolor':'#343e4d', 'font.size':10})
    def grid(values):
        image=np.full(len(config['x'])*len(config['y']),np.nan)
        image[config['gridIndices']]=values
        return image.reshape(len(config['y']),len(config['x']))
    key='r2-s0-d0';actual=np.array(fields[key]['real'])+1j*np.array(fields[key]['imag'])
    view=config.get('view',dict(axes=[0,1],fixedAxis=2,fixedValue=.2,bounds=dict(xmin=-3.3,xmax=3.5,ymin=-2.7,ymax=2.7)))
    u,v=view['axes'];fixed=view['fixedAxis'];bounds=view['bounds']
    dx=(config['x'][1]-config['x'][0])/2;dy=(config['y'][1]-config['y'][0])/2
    extent=[config['x'][0]-dx,config['x'][-1]+dx,config['y'][0]-dy,config['y'][-1]+dy]
    fig,axes=plt.subplots(1,3,figsize=(14,4.7),layout='constrained')
    for ax,values,title in zip(axes,[target,actual,actual-target],['Numerical wave reference','Current renderer · diffraction off','Complex residual']):
        if ax is axes[2]:
            image=ax.imshow(grid(abs(values)/data['referenceRms']),origin='lower',extent=extent,cmap='magma',vmin=0,vmax=3,interpolation='nearest')
            fig.colorbar(image,ax=ax,label='|native − wave| / RMS(wave)',shrink=.8)
        else:
            image=ax.imshow(grid(20*np.log10(np.maximum(abs(values),1e-12))),origin='lower',extent=extent,cmap='magma',vmin=-60,vmax=10,interpolation='nearest')
            fig.colorbar(image,ax=ax,label='Pressure level (dB re source at 1 m)',shrink=.8)
        for wall in config['walls']:
            c,h=np.array(wall['center']),np.array(wall['half'])
            inside=abs(c[fixed]-view['fixedValue'])<=h[fixed]
            ax.add_patch(Rectangle([c[u]-h[u],c[v]-h[v]],2*h[u],2*h[v],facecolor='#83909e' if inside else 'none',edgecolor='#e8edf5',linestyle='-' if inside else '--'))
        ax.plot(config['source'][u],config['source'][v],'*',color='#1accff',markersize=10)
        ax.set(xlim=(bounds['xmin'],bounds['xmax']),ylim=(bounds['ymin'],bounds['ymax']),xlabel='xyz'[u]+' (m)',ylabel='xyz'[v]+' (m)',title=title,aspect='equal')
    fig.suptitle(f'{hz} Hz · finite 3D geometry, slice {"xyz"[fixed]} = {view["fixedValue"]} m · 2 rays/bin, seed 0, 32 bounces',fontsize=13)
    fig.savefig(output/'comparison.png',dpi=160);plt.close(fig)
    print(json.dumps({k:report[k] for k in ['referenceGlobalRefinementChange','referenceShadowRefinementChange','referenceSphereControlError','cases']},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['reference','measured','output']:p.add_argument(name,type=Path)
    p.add_argument('--hz',type=int,default=250);p.add_argument('--fine-mesh',type=float,default=.1);p.add_argument('--coarse-mesh',type=float,default=.14)
    a=p.parse_args();main(a.reference,a.measured,a.output,a.hz,a.fine_mesh,a.coarse_mesh)
