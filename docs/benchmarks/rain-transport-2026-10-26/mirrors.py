"""Independent first-mirror audit for this static box fixture only.

Enumerates valid mirror paths, then reproduces the native primitive selection
and inverse-probability weight. Not a replacement production transport solver.
"""
import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
from scipy.io import wavfile

spec = importlib.util.spec_from_file_location('audit', Path(__file__).with_name('analyze.py'))
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)
from compare_wave_field import shadow_mask


def hash_ray(x):
    x &= 0xffffffff
    x = ((x ^ (x >> 16))*0x7feb352d) & 0xffffffff
    x = ((x ^ (x >> 15))*0x846ca68b) & 0xffffffff
    return x ^ (x >> 16)


def paths(source, listener, walls):
    indices = np.array([0,2,3,0,3,1,4,5,7,4,7,6,0,1,5,0,5,4,2,6,7,2,7,3,0,4,6,0,6,2,1,3,7,1,7,5]).reshape(-1,3)
    result = {}
    for owner, box in enumerate(walls):
        center, half = np.array(box['center']), np.array(box['half'])
        corners = center+np.array([[1 if v & (1 << axis) else -1 for axis in range(3)] for v in range(8)])*half
        other = walls[:owner]+walls[owner+1:]
        accepted = []
        for face, tri in enumerate(corners[indices]):
            a, b, c = tri
            normal = np.cross(b-a, c-a); normal /= np.linalg.norm(normal)
            ds, dl = np.dot(source-a, normal), np.dot(listener-a, normal)
            if ds <= 0 or dl <= 0:
                continue
            image = source-2*ds*normal
            hit = image+(listener-image)*ds/(ds+dl)
            basis = np.column_stack((b-a, c-a))
            u, v = np.linalg.lstsq(basis, hit-a, rcond=None)[0]
            if u < -1e-9 or v < -1e-9 or u+v > 1+1e-9:
                continue
            if any(np.linalg.norm(hit-h) < 1e-4 for h in accepted):
                continue
            if any(shadow_mask(dict(source=p, walls=other), hit[None])[0] for p in [source, listener]):
                continue
            accepted.append(hit)
            result[12*owner+face] = np.linalg.norm(listener-image)
    return result


def delayed(pcm, samples):
    at = np.floor(samples).astype(int); frac = samples-at
    value = np.zeros(len(at)); norm = np.zeros(len(at))
    for k in range(-15,17):
        d = k-frac; w = np.sinc(d)*np.sinc(d/16)
        value += pcm[at+k]*w; norm += w
    return value/norm


def main(root, focus=None):
    inputs = json.loads((root/'inputs.json').read_text()); rate = inputs['sampleRate']
    walls = audit.walk.scene()['walls']; count = len(walls)*12
    report = dict(scope='Static first specular bounce only; exact fixture enumeration versus native primitive sampling; excludes other reflected/scattered paths.', primitiveCount=count, stops={})
    for label, t in [('outside',1), ('canopy',8), ('inside',16)]:
        listener = audit.walk.listener(t)
        samples = np.arange(round((t+.5)*rate), round((t+1.5)*rate))
        exact = np.zeros(rate); sampled = {name: np.zeros(rate) for name in ['r1-s0', 'r2-s0', 'r2-s1']}
        path_count, selected_count = 0, {k:0 for k in sampled}
        for i, source in enumerate(inputs['sources']):
            position = np.array([source['position'][axis] for axis in 'xyz'])
            mirrors = paths(position, listener, walls); path_count += len(mirrors)
            pcm = np.memmap(root/source['file'], dtype='<f4', mode='r')
            for primitive, length in mirrors.items():
                value = delayed(pcm, samples-length/343*rate)*source['gain']*np.sqrt(.65*.85)/length
                exact += value
                for name, rays, seed in [('r1-s0',1,0), ('r2-s0',2,0), ('r2-s1',2,1)]:
                    first = hash_ray(i*0x85ebca6b+len(inputs['sources'])+seed)
                    hits = sum(((first+r) & 0xffffffff) % count == primitive for r in range(rays))
                    selected_count[name] += hits
                    sampled[name] += value*hits*count/rays
        row = dict(validMirrorPaths=path_count, selectedPaths=selected_count, exactMirrorRms=audit.rms(exact), sampled={})
        for name, value in sampled.items():
            case = dict(mirrorRms=audit.rms(value), levelAboveExactDb=float(audit.level(value)-audit.level(exact)) if audit.rms(value)>0 else None)
            native = root/name/'listener-0.wav'
            native_report = root/name/'run.json'
            offset = 0
            if not native.exists() and focus is not None and label=='canopy':
                native = focus/name/'listener-0.wav'
                native_report = focus/name/'run.json'
                offset = 6*rate
            if native.exists() and json.loads(native_report.read_text())['complete']:
                _, full = wavfile.read(native); _, direct = wavfile.read(root/'direct/listener-0.wav')
                a, b = samples[0]+144, samples[-1]+145
                residual = full[a-offset:b-offset].astype(float)-direct[a:b].astype(float)
                case['nativeIndirectRms'] = audit.rms(residual)
                case['nativeIndirectMinusPredictedRelativeRms'] = audit.rms(residual-value)/max(audit.rms(residual),1e-30)
                case['nativeIndirectCorrelation'] = float(np.corrcoef(residual,value)[0,1]) if audit.rms(value)>0 else None
            row['sampled'][name] = case
        report['stops'][label] = row
    (root/'mirrors.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('root', type=Path)
    parser.add_argument('--focus', type=Path)
    args = parser.parse_args(); main(args.root, args.focus)
