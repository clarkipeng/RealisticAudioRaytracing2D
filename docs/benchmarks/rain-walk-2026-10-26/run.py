"""Short offline courtyard -> canopy -> doorway rain-texture experiment."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
from scipy.io import wavfile

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'tools'))
import rain_area as area
import rain_texture

RATE, SECONDS, SEED = 44100, 18, 23


def scene():
    def surface(name, x, z, height, profile):
        return dict(name=name, x=x, z=z, height=height, profile=profile)
    def box(center, half):
        return dict(center=center, half=half)
    return dict(domainXZ=[[-10, 10], [-8, 8]], roofXZ=[[-6.25, 2.25], [-3.25, 3.25]],
        surfaces=[surface('roof', [-6.25, 2.25], [-3.25, 3.25], 3.251, 'iron'),
                  surface('west ground', [-10, -6.25], [-8, 8], .001, 'rain'),
                  surface('east ground', [2.25, 10], [-8, 8], .001, 'rain'),
                  surface('south ground', [-6.25, 2.25], [-8, -3.25], .001, 'rain'),
                  surface('north ground', [-6.25, 2.25], [3.25, 8], .001, 'rain')],
        walls=[box([-2, 3.125, 0], [4.25, .125, 3.25]),
               box([-6.125, 1.5, 0], [.125, 1.5, 3.25]),
               box([-4.125, 1.5, -3.125], [2.125, 1.5, .125]),
               box([-4.125, 1.5, 3.125], [2.125, 1.5, .125]),
               box([-2.125, 1.25, -2.075], [.125, 1.25, 1.175]),
               box([-2.125, 1.25, 2.075], [.125, 1.25, 1.175]),
               box([-2.125, 2.75, 0], [.125, .25, 3.25])])


def listener(t):
    def smooth(t, a, b):
        u = np.clip((t-a)/(b-a), 0, 1)
        return u*u*(3-2*u)
    return np.array([4.8*(1-smooth(t, 3, 7))-4.5*smooth(t, 10, 14), 1.6, 0.])


def main(source, output):
    started = time.perf_counter()
    output.mkdir(parents=True, exist_ok=False)
    original = json.loads((ROOT/'docs/benchmarks/rain-library-2026-10-26/report.json').read_text())
    profiles, inputs = {}, {}
    for name in ['rain', 'iron']:
        info = original['files'][name]
        path = source/info['file']
        assert hashlib.sha256(path.read_bytes()).hexdigest() == info['sha256']
        raw = subprocess.check_output(['ffmpeg', '-v', 'error', '-nostdin', '-threads', '1',
            '-filter_threads', '1', '-i', str(path), '-map', '0:a:0', '-ar', str(RATE),
            '-ac', '1', '-f', 'f32le', 'pipe:1'])
        pcm = np.frombuffer(raw, dtype='<f4').astype(float)
        rms = float(np.sqrt(np.mean(pcm*pcm)))
        profiles[name] = pcm/rms
        inputs[name] = dict(file=info['file'], sha256=info['sha256'], monoRms=rms,
                            source=info['url'], normalization='One RMS value for the complete recording; no per-grain gain.')
    config = scene()
    patches = area.patches(config['surfaces'], 2)
    assert abs(sum(p.area for p in patches)-320) < 1e-12
    points = area.sites(patches, 8)
    knots = np.linspace(0, SECONDS, SECONDS*30+1)
    gains, delays = [], []
    for t in knots:
        g, d = area.response(patches, points, listener(t), config['walls'])
        gains.append(g); delays.append(d)
    gains, delays = np.array(gains), np.array(delays)
    geometry_seconds = time.perf_counter()-started
    power = np.sum(gains*gains, axis=(1, 2))
    # A single emission gain, fixed using the outside expected area power.
    # It never follows listener position, active voice count or quadrature count.
    emission_gain = .012/np.sqrt(power[0]/2)
    audio = np.zeros((SECONDS*RATE, 2))
    roof = np.zeros_like(audio)
    for index, patch in enumerate(patches):
        layer = area.mix_patch(profiles[patch.profile], SEED, index, RATE, knots,
                               gains[:, index], delays[:, index], 0, len(audio))
        audio += layer
        if patch.profile == 'iron':
            roof += layer
        if index % 20 == 0:
            print('Mixed patch', index, 'of', len(patches), flush=True)
    audio *= emission_gain; roof *= emission_gain
    unnormalized_audio = audio.copy()
    # Only the beginning/end of the audition are faded.
    fade = np.sin(np.linspace(0, np.pi/2, round(.04*RATE)))**2
    audio[:len(fade)] *= fade[:, None]; audio[-len(fade):] *= fade[::-1, None]
    assert np.isfinite(audio).all() and np.max(np.abs(audio)) < .95
    wavfile.write(output/'walk.wav', RATE, audio.astype(np.float32))

    audits = {}
    for name, t in [('outside', 1.), ('canopy', 8.), ('inside', 16.)]:
        levels = {}
        for side in [4, 8, 16, 32]:
            g, _ = area.response(patches, area.sites(patches, side), listener(t), config['walls'])
            levels[str(side)] = float(np.sum(g*g))
        g_closed, _ = area.response(patches, points, listener(t), config['walls']+[
            dict(center=[-2.125, 1.25, 0], half=[.125, 1.25, .9])])
        window = unnormalized_audio[round((t-.5)*RATE):round((t+.5)*RATE)]
        roof_window = roof[round((t-.5)*RATE):round((t+.5)*RATE)]
        audits[name] = dict(expectedPowerByGrid=levels,
            grid8ErrorDb=float(10*np.log10(levels['8']/levels['32'])),
            sealedDoorPower=float(np.sum(g_closed*g_closed)),
            audioRms=float(np.sqrt(np.mean(window*window))),
            roofRms=float(np.sqrt(np.mean(roof_window*roof_window))))
    assert audits['inside']['sealedDoorPower'] == 0
    assert audits['canopy']['roofRms'] == 0 and audits['inside']['roofRms'] == 0

    # Whole-route geometry check against finer quadrature, including boundaries.
    fine_power = []
    for t in knots[::5]:
        g, _ = area.response(patches, area.sites(patches, 16), listener(t), config['walls'])
        fine_power.append(np.sum(g*g))
    fine_power = np.array(fine_power)
    route_error = 10*np.log10(power[::5]/fine_power)
    # Recording-offset seeds are only approximately independent. Measure them.
    probes = np.array([rain_texture.recording(profiles['rain'], 4*RATE, SEED, i,
                                             RATE, 7*RATE)[::4] for i in range(16)])
    correlation = np.corrcoef(probes)
    correlation[np.diag_indices_from(correlation)] = 0
    windows = unnormalized_audio.reshape(-1, round(.25*RATE), 2)
    rms_quarters = np.sqrt(np.mean(windows*windows, axis=(1, 2)))
    report = dict(version='rain-area-walk-v1', status='awaiting listening',
        sourceFeedback='User called the publisher audition better; spatialized output is not yet accepted.',
        sampleRate=RATE, seconds=SECONDS, seed=SEED, patchWidth=2,
        patchCount=len(patches), quadratureSitesPerPatch=64, responseRate=30,
        sourceGrainSeconds=4, sourceEpochSeconds=1, emissionGain=float(emission_gain),
        model='Statistical area-power direct response; one seeded mono texture and centroid delay per patch; equal-power stereo pan.',
        limits='No reflections, diffraction, transmission, structural roof radiation or HRTF; recording acoustics retained; cubic delay interpolation; source units uncalibrated.',
        scene=config, inputs=inputs, audits=audits,
        maxRouteGrid8Vs16ErrorDb=float(np.max(np.abs(route_error))),
        maxAdjacentExpectedPowerStepDb=float(np.max(np.abs(np.diff(10*np.log10(power))))),
        recordingRegionMaxAbsCorrelation=float(np.max(np.abs(correlation))),
        quarterSecondRms=rms_quarters.tolist(), peak=float(np.max(np.abs(audio))),
        geometrySeconds=geometry_seconds, totalSeconds=time.perf_counter()-started,
        outputSha256=hashlib.sha256((output/'walk.wav').read_bytes()).hexdigest())
    (output/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    display = dict(scene=config, seconds=SECONDS,
        positions=[dict(time=float(t), position=listener(t).tolist(),
                        relativePowerDb=float(10*np.log10(p/power[0]))) for t, p in zip(knots[::3], power[::3])],
        emitters=[dict(position=p.center.tolist(), area=p.area, profile=p.profile) for p in patches],
        audits=audits)
    (output/'index.html').write_text(Path(__file__).with_name('listen.html').read_text().replace('{{DATA}}', json.dumps(display)))
    print(json.dumps({k: report[k] for k in ['patchCount', 'maxRouteGrid8Vs16ErrorDb',
        'maxAdjacentExpectedPowerStepDb', 'recordingRegionMaxAbsCorrelation', 'peak', 'totalSeconds']}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    main(args.source, args.output)
