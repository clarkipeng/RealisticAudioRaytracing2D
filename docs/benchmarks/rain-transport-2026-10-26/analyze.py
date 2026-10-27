"""Audit and package native rain renders without level matching or limiting."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
from scipy.io import wavfile

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT/'tools'))
import rain_area

spec = importlib.util.spec_from_file_location('walk', ROOT/'docs/benchmarks/rain-walk-2026-10-26/run.py')
walk = importlib.util.module_from_spec(spec)
spec.loader.exec_module(walk)


def rms(x):
    return float(np.sqrt(np.mean(np.asarray(x, dtype=float)**2)))


def level(x):
    return 20*np.log10(max(rms(x), 1e-30))


def direct_oracle(root, inputs, direct):
    rate = walk.RATE
    # A stationary-window, independently mixed point-source oracle. It checks
    # PCM/gain/visibility/delay wiring, not moving Doppler or indirect physics.
    patches = rain_area.patches(walk.scene()['surfaces'], 2)
    oracle = {}
    for label, t in [('outside', 1), ('canopy', 8), ('inside', 16)]:
        g, delay = rain_area.response(patches, rain_area.sites(patches, 1), walk.listener(t), walk.scene()['walls'])
        gain = np.sqrt(np.sum(g*g, axis=1))*inputs['emissionGain']
        n = np.arange(round((t+.5)*rate), round((t+1.5)*rate))
        expected = np.zeros(len(n))
        for i in np.flatnonzero(gain):
            pcm = np.memmap(root/inputs['sources'][i]['file'], dtype='<f4', mode='r')
            # Same 32-tap windowed sinc formula as the native delay reader.
            read = n-delay[i]*rate
            at = np.floor(read).astype(int)
            frac = read-at
            value = np.zeros(len(n)); norm = np.zeros(len(n))
            for k in range(-15, 17):
                d = k-frac
                weight = np.sinc(d)*np.sinc(d/16)
                value += pcm[at+k]*weight
                norm += weight
            expected += gain[i]*value/norm
        actual = direct[round((t-.5)*rate):round((t+.5)*rate)]
        oracle[label] = dict(expectedRms=rms(expected), nativeRms=rms(actual),
            relativeRmsError=rms(actual-expected)/rms(expected), correlation=float(np.corrcoef(actual, expected)[0, 1]))
    return oracle


def main(root, focus, output):
    output.mkdir(parents=True, exist_ok=False)
    inputs = json.loads((root/'inputs.json').read_text())
    take = json.loads((root/'take/take.json').read_text())
    # Verify that the numeric adapter preserved every float, and that the
    # focused take contains exactly the original source interval at equal gain.
    for i, source in enumerate(inputs['sources']):
        clip = take['scene']['clips'][i]
        assert clip['sha256'] == source['sha256']
        assert hashlib.sha256((root/'take'/clip['file']).read_bytes()).hexdigest() == source['sha256']
    focused = json.loads((focus/'fixture.json').read_text())
    focused_take = json.loads((focus/'take/take.json').read_text())
    for i, source in enumerate(inputs['sources']):
        clip = focused_take['scene']['clips'][i]
        payload = (focus/'take'/clip['file']).read_bytes()
        assert payload == (root/source['file']).read_bytes()[6*walk.RATE*4:11*walk.RATE*4]
        assert clip['sha256'] == hashlib.sha256(payload).hexdigest()
        assert focused['sources'][i]['gain'] == source['gain']
    report = dict(status='Awaiting listening; diagnostic, not production integration', inputs=inputs, renders={}, comparisons={})
    audio = {}
    hashes = {}
    runtime = None
    for name, location, warmup, duration in [('direct',root,1,18), ('r1-s0',root,1,18),
                                            ('r2-s0',focus,2,3), ('r2-s1',focus,2,3)]:
        run = json.loads((location/name/'run.json').read_text())
        resources = json.loads((location/(name+'.resources.json')).read_text())
        assert run['complete'] and resources['exitCode'] == 0
        identity = {k: run[k] for k in ['sceneHash', 'takeStateHash', 'takeEventHash', 'takeSourceHash', 'audioShaderDependencyHash', 'provenance']}
        group = 'walk' if location == root else 'canopy'
        if group not in hashes:
            hashes[group] = identity
        assert hashes[group] == identity, 'Source, scene, or DSP changed within a matched take'
        current_runtime = (run['audioShaderDependencyHash'], run['provenance'])
        if runtime is None:
            runtime = current_runtime
        assert current_runtime == runtime, 'Renderer implementation changed'
        target = json.loads((location/(name+'-job.json')).read_text())['settings']
        assert all(run['settings'][k] == v for k, v in target.items()), 'Rendered settings differ from the requested job'
        rate, pcm = wavfile.read(location/name/'listener-0.wav')
        assert rate == walk.RATE and pcm.ndim == 1 and np.isfinite(pcm).all()
        assert run['tailCoversResponseSupport'], 'Recorded tail does not cover configured response support'
        assert hashlib.sha256((location/name/'listener-0.wav').read_bytes()).hexdigest() == run['audioHashes'][0]
        begin = warmup*rate+run['dspLatencySamples']
        x = pcm[begin:begin+duration*rate].astype(float)
        assert len(x) == duration*rate
        audio[name] = x
        windows = x.reshape(-1, rate//4)
        quarters = np.sqrt(np.mean(windows*windows, axis=1))
        times = [('outside',1), ('canopy',8), ('inside',16)] if group == 'walk' else [('canopy',1)]
        stops = {label: dict(rms=rms(x[round((t-.5)*rate):round((t+.5)*rate)]),
                            dBFS=float(level(x[round((t-.5)*rate):round((t+.5)*rate)])))
                 for label, t in times}
        report['renders'][name] = dict(settings=run['settings'], stops=stops,
            auditionSeconds=duration, warmupSeconds=warmup, originalWalkStartSeconds=0 if group=='walk' else 7,
            peak=float(np.max(np.abs(x))), fullRenderPeak=float(np.max(np.abs(pcm))),
            fullRenderSamplesOverFullScale=int(np.sum(np.abs(pcm)>1)),
            peakTimeSeconds=float(np.argmax(np.abs(x))/rate), quarterSecondRms=quarters.tolist(),
            maxAdjacentQuarterLevelChangeDb=float(np.max(np.abs(np.diff(20*np.log10(np.maximum(quarters, 1e-30)))))),
            workingNumericBytes=run['workingNumericBytes'], preparedPcmBytes=run['preparedPcmBytes'],
            peakSampledRssBytes=max(s['rssBytes'] for s in resources['samples']),
            renderWallSeconds=run['wallSeconds'], launcherWallSeconds=resources['wallSeconds'],
            solveSeconds=run['solveSeconds'], audioSeconds=run['audioSeconds'], gpuAudio=run['gpuAudio'],
            tailCoversResponseSupport=run['tailCoversResponseSupport'],
            nativeSha256=run['audioHashes'][0], provenance=run['provenance'])
    # Native floating-point outputs retain the failing peaks. Apply one shared
    # playback attenuation to every version, preserving their level differences.
    peak = max(v['fullRenderPeak'] for v in report['renders'].values())
    attenuation = max(0., 6*np.ceil(20*np.log10(peak/.5)/6))
    playback_gain = 10**(-attenuation/20)
    report['previewAttenuationDb'] = float(attenuation)
    report['previewGain'] = float(playback_gain)
    report['status'] = 'Fails full-scale stability; not promoted to production' if peak>1 else report['status']
    previews = audio | {name+'-canopy': audio[name][7*rate:10*rate] for name in ['direct','r1-s0']}
    report['auditionHashes'] = {}
    for name, x in previews.items():
        audition = x.copy()*playback_gain
        fade = np.sin(np.linspace(0, np.pi/2, round(.04*rate)))**2
        audition[:len(fade)] *= fade
        audition[-len(fade):] *= fade[::-1]
        assert np.max(np.abs(audition)) <= .5
        wavfile.write(output/(name+'.wav'), rate, audition.astype(np.float32))
        report['auditionHashes'][name] = hashlib.sha256((output/(name+'.wav')).read_bytes()).hexdigest()
    for label, a, b in [('rayBudget', 'r1-s0', 'r2-s0'), ('transportSeed', 'r2-s0', 'r2-s1')]:
        x, y = previews[a+'-canopy'] if a=='r1-s0' else audio[a], audio[b]
        report['comparisons'][label] = dict(first=a, second=b,
            scope='Canopy only; original walk time 7..10. Level comparison uses original time 7.5..8.5.',
            relativeWaveformRms=float(rms(y-x)/rms(x)), correlation=float(np.corrcoef(x, y)[0, 1]),
            stopLevelDifferenceDb={s: report['renders'][b]['stops'][s]['dBFS']-report['renders'][a]['stops'][s]['dBFS']
                                   for s in ['canopy']})
    oracle = direct_oracle(root, inputs, audio['direct'])
    assert all(v['relativeRmsError'] < .001 for v in oracle.values()), 'Direct source/visibility/delay wiring failed'
    report['directOracle'] = oracle
    report['sharedInputHashes'] = hashes
    report['focusedSourceAlignment'] = json.loads((focus/'alignment.json').read_text())
    (output/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    config = walk.scene()
    patches = rain_area.patches(config['surfaces'], 2)
    display = dict(scene=config, seconds=walk.SECONDS,
        positions=[dict(time=float(t), position=walk.listener(t).tolist()) for t in np.arange(0, walk.SECONDS+.01, .1)],
        emitters=[dict(position=p.center.tolist(), area=p.area, profile=p.profile) for p in patches])
    template = Path(__file__).with_name('listen.html').read_text()
    (output/'index.html').write_text(template.replace('{{DATA}}', json.dumps(display)).replace('{{ATTENUATION}}', f'{attenuation:g}'))
    print(json.dumps(dict(renders={k: {s: v[s] for s in ['peak', 'stops', 'renderWallSeconds', 'workingNumericBytes']} for k,v in report['renders'].items()},
        comparisons=report['comparisons'], directOracle=oracle), indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('focus', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    main(args.root, args.focus, args.output)
