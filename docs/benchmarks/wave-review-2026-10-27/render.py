"""Run the existing wave-field measurement in a guarded isolated Unity worker."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time

ROOT=Path(__file__).resolve().parents[3]

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(project, output, reference, hz):
    project, output = project.resolve(), output.resolve()
    if output.exists() or project==ROOT or (project/'Temp/UnityLockfile').exists():
        raise RuntimeError('Use an unused measurement directory and a closed runner.')
    files={str(p.relative_to(ROOT)):digest(p) for directory in ['Assets/Scripts','Assets/Shaders','Assets/Resources/Acoustics']
        for p in (ROOT/directory).rglob('*') if p.is_file() and p.suffix!='.meta'}
    for name,expected in files.items():
        if not (project/name).is_file() or digest(project/name)!=expected:
            raise RuntimeError(f'Isolated runner does not match current source: {name}')
    fixture=Path('Assets/Tests/Editor/WaveFieldMeasurement.cs')
    if not (project/fixture).is_file():
        raise RuntimeError('Use an existing isolated runner containing the measurement tests.')
    (project/fixture).write_bytes((ROOT/fixture).read_bytes())
    scene=json.loads((reference/'scene.json').read_text())
    scene.update(frequencies=[hz],rays=[2,128],seeds=[0,1],maxBounces=32,maximumDelaySeconds=1.25)
    output.mkdir(parents=True)
    (output/'scene.json').write_text(json.dumps(scene,indent=2)+'\n')
    (output/'provenance.json').write_text(json.dumps(dict(rendererFiles=files,fixtureSha256=digest(ROOT/fixture),
        originalSceneSha256=digest(reference/'scene.json'),measurementSceneSha256=digest(output/'scene.json')),indent=2)+'\n')
    command = ['taskpolicy','-b','nice','-n','19',
        '/Applications/Unity/Hub/Editor/6000.4.1f1/Unity.app/Contents/MacOS/Unity',
        '-batchmode','-force-metal','-disable-audio','-job-worker-count','1',
        '-projectPath',str(project),'-runTests','-testPlatform','EditMode',
        '-testFilter','TakeChecks.MeasureWaveField','-testResults',str(output/'tests.xml'),
        '-logFile',str(output/'renderer.log')]
    process = subprocess.Popen(command, env=dict(os.environ, ACOUSTIC_WAVE_OUTPUT=str(output),
        ACOUSTIC_WAVE_WORK_FRACTION='.2'))
    started, samples = time.monotonic(), []
    def interrupt(signum, frame):
        raise KeyboardInterrupt
    previous = {s:signal.signal(s, interrupt) for s in [signal.SIGINT,signal.SIGTERM]}
    print(f'Wave comparison worker {process.pid}; nice 19, 20% completed-work budget.', flush=True)
    try:
        while process.poll() is None:
            data = subprocess.run(['ps','-p',str(process.pid),'-o','rss=,pcpu=,nice='], capture_output=True,text=True).stdout.split()
            if len(data)==3:
                samples.append(dict(seconds=time.monotonic()-started,rssBytes=int(data[0])*1024,cpuPercent=float(data[1]),nice=int(data[2])))
                if samples[-1]['rssBytes']>3072*1024**2:
                    raise RuntimeError('Measurement exceeded its 3 GiB process-RSS guard.')
            time.sleep(2)
    finally:
        if process.poll() is None:
            process.terminate()
            try:process.wait(timeout=15)
            except subprocess.TimeoutExpired:process.kill();process.wait()
        for s, handler in previous.items():signal.signal(s, handler)
        (output/'resources.json').write_text(json.dumps(dict(exitCode=process.returncode,
            wallSeconds=time.monotonic()-started,workFraction=.2,samples=samples),indent=2)+'\n')
    raise SystemExit(process.returncode)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('project',type=Path);parser.add_argument('output',type=Path)
    parser.add_argument('--reference',type=Path,default=ROOT/'.context/wave-field/run')
    parser.add_argument('--hz',type=int,default=250)
    args=parser.parse_args();main(args.project,args.output,args.reference,args.hz)
