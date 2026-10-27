"""Freeze the preferred texture field for a native-renderer diagnostic.

Writes private PCM and a numeric fixture consumed by RainTransportReview.cs.
Patch centers approximate the area field; no production weather asset is edited.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT/'tools'))
import rain_area
import rain_texture

spec = importlib.util.spec_from_file_location('walk', ROOT/'docs/benchmarks/rain-walk-2026-10-26/run.py')
walk = importlib.util.module_from_spec(spec)
spec.loader.exec_module(walk)


def vector(v):
    return dict(zip(('x', 'y', 'z'), map(float, v)))


def main(source, output):
    output.mkdir(parents=True, exist_ok=False)
    prior = json.loads((ROOT/'docs/benchmarks/rain-walk-2026-10-26/report.json').read_text())
    config, rate = walk.scene(), walk.RATE
    patches = rain_area.patches(config['surfaces'], prior['patchWidth'])
    profiles = {}
    for name, info in prior['inputs'].items():
        path = source/info['file']
        assert hashlib.sha256(path.read_bytes()).hexdigest() == info['sha256']
        raw = subprocess.check_output(['ffmpeg', '-v', 'error', '-nostdin', '-threads', '1',
            '-filter_threads', '1', '-i', str(path), '-map', '0:a:0', '-ar', str(rate),
            '-ac', '1', '-f', 'f32le', 'pipe:1'])
        pcm = np.frombuffer(raw, dtype='<f4').astype(float)
        profiles[name] = pcm/np.sqrt(np.mean(pcm*pcm))
    # One second of actual renderer warm-up precedes the original 18-second walk.
    seconds = walk.SECONDS+1
    sources = []
    for i, patch in enumerate(patches):
        pcm = rain_texture.recording(profiles[patch.profile], 4*rate, walk.SEED, i, 0, seconds*rate)
        file = f'source-{i:03}.f32'
        (output/file).write_bytes(pcm.astype('<f4').tobytes())
        sources.append(dict(position=vector(patch.center), area=patch.area, profile=patch.profile,
            gain=float(prior['emissionGain']*np.sqrt(patch.area)), file=file,
            sha256=hashlib.sha256((output/file).read_bytes()).hexdigest()))
    positions = [dict(sample=round(t*rate), position=vector(walk.listener(max(0, t-1))))
                 for t in np.linspace(0, seconds, seconds*30+1)]
    fixture = dict(rate=rate, seconds=seconds, sourceSeed=walk.SEED, sources=sources,
        positions=positions, walls=[dict(center=vector(b['center']), half=vector(b['half'])) for b in config['walls']])
    (output/'fixture.json').write_text(json.dumps(fixture, indent=2)+'\n')
    # Isolate area -> centroid visibility/power error independently of transport.
    errors, audits = [], {}
    for t in np.arange(0, walk.SECONDS+.01, .1):
        g, _ = rain_area.response(patches, rain_area.sites(patches, 8), walk.listener(t), config['walls'])
        p, _ = rain_area.response(patches, rain_area.sites(patches, 1), walk.listener(t), config['walls'])
        errors.append(10*np.log10(np.sum(p*p)/np.sum(g*g)))
    for label, t in [('outside', 1), ('canopy', 8), ('inside', 16)]:
        g, _ = rain_area.response(patches, rain_area.sites(patches, 1), walk.listener(t), config['walls'])
        audits[label] = dict(centroidPower=float(np.sum(g*g)))
    report = dict(sourceSeed=walk.SEED, sourceCount=len(patches), sampleRate=rate,
        emissionGain=prior['emissionGain'], warmupSeconds=1, walkSeconds=walk.SECONDS,
        sources=sources, inputs=prior['inputs'], audits=audits,
        centroidVsAreaPowerErrorDb=dict(min=float(min(errors)), max=float(max(errors))),
        limits='Point approximation; mono; empirical uncalibrated textures; illustrative flat material; no ground reflecting geometry, diffraction, transmission or structural radiation.')
    (output/'inputs.json').write_text(json.dumps(report, indent=2)+'\n')
    settings = dict(transport=1, synthesis=1, audioBackend=0, fftSize=128, spectralHop=32,
        responseStepSamples=1472, maximumDelaySeconds=1.25, maximumResponsesPerBatch=1,
        useBvh=True, maxBounces=32, raysPerBin=1, diffractionFactor=0, airAbsorption=False,
        tailSamples=66150, delayFilter=1, sincRadius=16)
    for name, changes in [('direct', dict(maxBounces=0)), ('r1-s0', {})]:
        job = dict(take='take', output=name, shader='Assets/Shaders/Transport.compute', settings=settings | changes)
        (output/(name+'-job.json')).write_text(json.dumps(job, indent=2)+'\n')
    print(json.dumps({k: report[k] for k in ('sourceCount', 'centroidVsAreaPowerErrorDb', 'audits')}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    main(args.source, args.output)
