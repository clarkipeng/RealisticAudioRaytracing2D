"""Reuse exact PCM from the walk for a shorter stationary canopy diagnostic."""
import argparse
import hashlib
import json
from pathlib import Path


def main(original, output):
    output.mkdir(parents=True, exist_ok=False)
    fixture = json.loads((original/'fixture.json').read_text())
    rate = fixture['rate']
    # Original walk time 7..10 is source time 8..11. Retain two seconds of
    # source history before that interval, beyond the configured 1.25 s horizon.
    fixture['seconds'] = 5
    fixture['positions'] = [dict(sample=t*rate, position=dict(x=0, y=1.6, z=0)) for t in [0,5]]
    for source in fixture['sources']:
        pcm = (original/source['file']).read_bytes()[6*rate*4:11*rate*4]
        assert len(pcm) == 5*rate*4
        (output/source['file']).write_bytes(pcm)
        source['sha256'] = hashlib.sha256(pcm).hexdigest()
    (output/'fixture.json').write_text(json.dumps(fixture, indent=2)+'\n')
    for seed in [0,1]:
        name = f'r2-s{seed}'
        job = json.loads((original/'r1-s0-job.json').read_text())
        job['output'] = name
        job['settings'].update(raysPerBin=2, seed=seed)
        (output/(name+'-job.json')).write_text(json.dumps(job, indent=2)+'\n')
    (output/'alignment.json').write_text(json.dumps(dict(originalSourceOffsetSeconds=6,
        warmupSeconds=2, auditionSeconds=3, originalWalkIntervalSeconds=[7,10],
        pose='Stationary canopy; compare central second to avoid the original walk stopping at interval start.'), indent=2)+'\n')


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('original',type=Path);p.add_argument('output',type=Path)
    a=p.parse_args();main(a.original,a.output)
