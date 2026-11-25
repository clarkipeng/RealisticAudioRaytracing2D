"""The live showcase: the path method live in the Church and the train station (README.md, "Live showcase";
docs/benchmarks/path-denoise-2026-11-12, parts 26 and 38).

    PY=.context/fdtd-teacher/venv/bin/python
    nice -n 19 $PY tools/path_live_church.py build [--development]
                                                      # guarded Unity build -> Builds/LiveShowcase/LiveShowcase.app
                                                      # (a development player -> Builds/LiveShowcaseDevelopment/: per-stage GPU times)
    nice -n 19 $PY tools/path_live_church.py smoke --windowed [--scenes Church TrainStation] [--shots] [--development]
                                                      # guarded scripted run of the build -> .context/path-live/smoke-*/
    nice -n 19 $PY tools/path_live_church.py capture --scenes Church TrainStation [-- player arguments]
                                                      # the smoke walks at a fixed timestep, every frame and the output
                                                      # -> .context/path-live/capture-*/<scene>/ (frames/, output.wav, capture.json)
    $PY tools/path_live_church.py video CAPTURE_DIR OUT.mp4   # the captured walks as one H.264/AAC film, and OUT-thumb.jpg
    nice -n 19 $PY tools/path_live_church.py stills   # guarded still images of every scene -> .context/polish/stills
    nice -n 19 $PY tools/path_live_church.py author   # guarded ShowcaseAuthoring of the showcase scenes

The build comes from the test runner's copy of the project (tools/dlss_jobs.py RUNNER, synced from this
working tree), so the editor's Library is untouched. The smoke run launches the build with -pathLiveSmoke:
each scene's scripted walk (PathLive, PathLiveScene), one scene after another, loaded as the scene picker loads
them. Per scene the output is recorded to output.wav, frame times, GPU times and audio underruns to smoke.json,
and with --shots a screenshot is saved at each phase of the walk. --windowed launches it hidden and unfocused;
without it, it runs windowless (-batchmode) and draws nothing. Arguments after `smoke --` go to the player
(e.g. -pathBudget 6).
"""
import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
import dlss_jobs as dj  # noqa: E402
import heavy_guard as hg  # noqa: E402
import method_runs as mr  # noqa: E402

WORK = ROOT / '.context/path-live'
# A build is a folder: on the Mac it holds LiveShowcase.app, on the PC LiveShowcase.exe and its data, on the Windows
# drive (a player loading from \\wsl$ would read its data slowly).
BUILDS = hg.WORKTREE_WORK / 'Builds' if hg.WSL else ROOT / 'Builds'
PLAYER = 'LiveShowcase.exe' if hg.WSL else 'LiveShowcase.app'


def player_folder(development):
    return BUILDS / ('LiveShowcaseDevelopment' if development else 'LiveShowcase')


def player_binary(development):
    app = player_folder(development) / PLAYER
    return app if hg.WSL else next((app / 'Contents/MacOS').iterdir())


def launch(name, arguments):
    """A guarded batch editor run in the runner, its scenes and content synced first."""
    mr.LOGS = WORK / 'logs'; mr.LOGS.mkdir(parents=True, exist_ok=True)
    return mr.launch(name, arguments)


def stills(scenes):
    """Still images of the scenes (SceneStills) for a visual review: .context/polish/stills."""
    out = ROOT / '.context/polish/stills'
    shutil.rmtree(out, ignore_errors=True)
    logs, resources = launch('stills', ['-quit', '-executeMethod', 'SceneStills.Capture', '-stillsOutput', str(out),
                                        *(['-stillsScenes', ','.join(scenes)] if scenes else [])])
    print(f'{len(list(out.glob("*.png")))} stills in {out}' if resources['exitCode'] == 0 else f'stills failed: see {logs}/unity.log')


def author():
    """ShowcaseAuthoring in the runner, then its scenes (and the metas of new scripts) copied back here."""
    logs, resources = launch('author', ['-quit', '-executeMethod', 'ShowcaseAuthoring.Apply'])
    if resources['exitCode'] != 0:
        raise SystemExit(f'authoring failed: see {logs}/unity.log')
    for scene in ['Assets/Scenes/Church.unity', 'Assets/Scenes/TrainStation.unity']:
        shutil.copy2(dj.RUNNER / scene, ROOT / scene)
    for folder in ['Assets/Scripts', 'Assets/Editor']:
        for meta in (dj.RUNNER / folder).rglob('*.meta'):
            target = ROOT / meta.relative_to(dj.RUNNER)
            if not target.exists() and target.with_suffix('').exists():
                shutil.copy2(meta, target)
    print('authored the showcase scenes')


def build(development=False):
    folder = player_folder(development)
    staging = BUILDS / 'staging'
    shutil.rmtree(staging, ignore_errors=True)
    staged = staging / PLAYER
    logs, resources = launch('build', ['-quit', '-executeMethod', 'PathLiveBuild.Build', '-pathLiveOutput', str(staged),
                                       *(['-pathLiveDevelopment'] if development else [])])
    if resources['exitCode'] != 0 or not staged.exists():
        raise SystemExit(f'build failed: see {logs}/unity.log')
    shutil.rmtree(folder, ignore_errors=True)
    staging.rename(folder)
    print(f'built {folder / PLAYER}')


def front_app():
    """The frontmost application's name (lsappinfo: read only, needs no permission)."""
    import subprocess
    asn = subprocess.run(['lsappinfo', 'front'], capture_output=True, text=True).stdout.strip()
    info = subprocess.run(['lsappinfo', 'info', '-only', 'name', asn], capture_output=True, text=True).stdout
    return info.split('=')[-1].strip().strip('"')


def smoke(extra=(), windowed=False, scenes=(), shots=False, development=False, capture=False):
    app = player_folder(development) / PLAYER
    out = WORK / time.strftime(('capture' if capture else 'smoke') + '-%Y%m%d-%H%M%S')
    out.mkdir(parents=True)
    binary = player_binary(development)
    # -batchmode: no window, so the run never takes the screen's focus; the player still draws the scene every frame
    # (smoke.json reports camera renders per frame) and still plays its audio callbacks (callbacks, underruns).
    size = ['1920', '1080'] if capture else ['1280', '720']
    # -force-d3d12: a Windows player in -batchmode otherwise starts on Direct3D 11, which a D3D12-only build has no
    # compute kernels for.
    command = [str(binary), '-batchmode', *(['-force-d3d12'] if hg.WSL else []), '-pathLiveSmoke', str(out), '-screen-width', size[0], '-screen-height', size[1],
               *(['-screen-fullscreen', '0', '-pathLiveCapture', '-pathLiveCaptureFps', '60'] if capture else []),
               '-logFile', str(out / 'player.log'), '-liveTrace', str(out / 'pipeline.csv'),
               *(['-pathLiveScenes', ','.join(scenes)] if scenes else []), *(['-pathLiveShots'] if shots else []), *extra]
    if windowed:
        # A window, so that the scene is drawn and the profiler's GPU timings can be read: launched hidden and not
        # activated (open -g -j). The front application is watched throughout; if the player ever became frontmost it
        # is stopped at once (it must never take the screen's focus). Its footprint is not tracked (open's child).
        import subprocess, threading
        command = ['open', '-g', '-j', '-n', '-W', '-a', str(app), '--args'] + [c for c in command[1:] if c != '-batchmode']
        before, stolen, done = front_app(), [], threading.Event()
        def watch():
            while not done.wait(.2):
                name = front_app()
                if name != before and (app.stem in name or 'AudioRayTrace' in name):
                    stolen.append(name); subprocess.run(['pkill', '-f', str(app)]); return
        threading.Thread(target=watch, daemon=True).start()
    resources = dj.Worker(command, dict(os.environ), lambda m: print(m, flush=True)).run()
    if windowed:
        done.set()
        resources['focusTaken'] = bool(stolen)
        print('the player took the focus and was stopped' if stolen else f'focus stayed with {before}')
    (out / 'resources.json').write_text(json.dumps(resources, indent=1) + '\n')
    print(f'exit {resources["exitCode"]}, {resources["wallSeconds"]:.0f} s, peak footprint {resources["peakTreeFootprintBytes"] / dj.GIB:.2f} GiB')
    reports = sorted(out.glob('*/smoke.json'))
    for report in reports:
        r = json.loads(report.read_text())
        print(f'{r["scene"]}: {r["seconds"]:.0f} s, underruns {r["underrunSamples"]} samples, frames {r["frameMsMedian"]:.2f} / '
              f'{r["frameMsP99"]:.1f} / {r["frameMsMax"]:.1f} ms (median / p99 / max), frame GPU {r["frameGpuMsMedian"]:.1f} ms median, '
              f'{r["cameraRendersPerFrame"]:.2f} camera renders per frame; {report}')
    stalls(out / 'pipeline.csv.stalls')
    if len(reports) < max(1, len(scenes)):
        print(f'{max(1, len(scenes)) - len(reports)} scene(s) without a report; see {out}/player.log')


def video(capture_dir, out, seconds=None):
    """The captured walks (capture_dir/<scene>/frames, output.wav, capture.json), one after another, as one film:
    1920x1080 H.264 High, yuv420p, BT.709, CRF 18, AAC 192 kbit/s; and a 1280x720 thumbnail beside it. Each scene's sound
    is its own binaural output, aligned to its pictures by capture.json (frame f shows take sample first + f rate / fps,
    heard at output sample that + lead). One gain for the whole film brings the loudest peak to -1 dBFS."""
    import subprocess
    capture_dir, out = Path(capture_dir).resolve(), Path(out).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    order = [s for s in ['Church', 'TrainStation'] if (capture_dir / s / 'capture.json').exists()]
    if not order:
        raise SystemExit(f'no captured scene in {capture_dir}')
    # One gain for every scene (their relative levels kept): the loudest peak to -1 dBFS.
    def peak(wav):
        text = subprocess.run(['ffmpeg', '-hide_banner', '-i', str(wav), '-af', 'volumedetect', '-f', 'null', '-'],
                              capture_output=True, text=True).stderr
        return float(text.split('max_volume:')[1].split('dB')[0])
    gain = -1 - max(peak(capture_dir / s / 'output.wav') for s in order)
    parts, fps = [], None
    for scene in order:
        meta = json.loads((capture_dir / scene / 'capture.json').read_text())
        fps, rate = meta['fps'], meta['sampleRate']
        offset = (meta['firstTakeSample'] + meta['leadSamples']) / rate
        frames = meta['frames']
        part = capture_dir / f'{scene}.mp4'
        color = ['-colorspace', 'bt709', '-color_primaries', 'bt709', '-color_trc', 'bt709', '-color_range', 'tv',
                 '-x264-params', 'colorprim=bt709:transfer=bt709:colormatrix=bt709']   # (in the stream, so it survives the concatenation)
        subprocess.run(['ffmpeg', '-v', 'error', '-y', '-framerate', str(fps), '-i', str(capture_dir / scene / 'frames/frame-%06d.jpg'),
                        '-ss', f'{offset:.6f}', '-i', str(capture_dir / scene / 'output.wav'),
                        '-t', f'{frames / fps:.6f}', '-map', '0:v', '-map', '1:a',
                        '-vf', 'scale=1920:1080:flags=lanczos:out_color_matrix=bt709,format=yuv420p,fade=t=in:st=0:d=0.5,'
                               f'fade=t=out:st={frames / fps - .5:.3f}:d=0.5',
                        '-af', f'volume={gain:.2f}dB,afade=t=in:st=0:d=0.5,afade=t=out:st={frames / fps - .5:.3f}:d=0.5',
                        '-c:v', 'libx264', '-profile:v', 'high', '-crf', '18', '-preset', 'slow', *color,
                        '-c:a', 'aac', '-b:a', '192k', '-ar', '48000', '-ac', '2', str(part)], check=True)
        parts.append(part)
    listing = capture_dir / 'parts.txt'
    listing.write_text(''.join(f"file '{p}'\n" for p in parts))
    subprocess.run(['ffmpeg', '-v', 'error', '-y', '-f', 'concat', '-safe', '0', '-i', str(listing), '-c', 'copy',
                    '-movflags', '+faststart', str(out)], check=True)
    thumb = out.with_name(out.stem + '-thumb.jpg')
    subprocess.run(['ffmpeg', '-v', 'error', '-y', '-ss', '12', '-i', str(out), '-frames:v', '1', '-vf', 'scale=1280:720:flags=lanczos',
                    '-q:v', '2', str(thumb)], check=True)
    print(f'wrote {out} and {thumb}')


def stalls(path):
    """The main-thread stalls of the run (-liveTrace FILE.stalls, README part 41), each with its kind: computing (the
    main thread's CPU time is most of it), paging (blocked, with page faults that needed I/O), or blocked (waiting or
    preempted). Scene loads and the end of each take are stalls too; they are not in a walk."""
    import csv
    if not path.exists():
        return
    for row in csv.DictReader(path.open()):
        ms, cpu = float(row['ms']), float(row.get('mainThreadCpuMs') or 'nan')
        kind = ('computing' if cpu > .5 * ms else 'paging' if int(row.get('majorFaults') or 0) > 0 else 'blocked') if cpu == cpu else '?'
        print(f"stall {row['seconds']} s, frame {row['frame']}: {row['system']} {ms:.0f} ms, {kind} (main thread CPU {cpu:.0f} ms, "
              f"{row.get('majorFaults')} page-ins, {row.get('involuntarySwitches')} preemptions)")


if __name__ == '__main__':
    hg.gpu_gate()
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('command', choices=['build', 'smoke', 'capture', 'video', 'stills', 'author'])
    p.add_argument('player', nargs='*', help='smoke: extra player arguments (after --); stills: scene names')
    p.add_argument('--windowed', action='store_true', help='smoke: with a window, launched hidden and unfocused (open -g -j)')
    p.add_argument('--scenes', nargs='+', default=[], help='smoke: the scenes to walk, in turn (default: the first)')
    p.add_argument('--shots', action='store_true', help='smoke: a screenshot at each phase of each walk')
    p.add_argument('--development', action='store_true', help='build / smoke: the development player (per-stage GPU times)')
    a = p.parse_args()
    {'build': lambda: build(a.development), 'smoke': lambda: smoke(a.player, a.windowed, a.scenes, a.shots, a.development),
     'capture': lambda: smoke(a.player, True, a.scenes, False, a.development, capture=True),
     'video': lambda: video(*a.player[:2]), 'stills': lambda: stills(a.player),
     'author': author}[a.command]()
