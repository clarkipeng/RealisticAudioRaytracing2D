#!/usr/bin/env python3
"""Run the finite-edge sampling pilot in an existing, isolated Unity project."""
import argparse
import os
from pathlib import Path
import shutil
import subprocess
import csv
import math
import statistics

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("runner", type=Path, help="Separate Unity project with matching runtime code and imported Library")
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--unity", type=Path, required=True)
parser.add_argument("--calibrate-from", type=Path, help="Previous results used to select a production budget near 20 percent error")
args = parser.parse_args()
here = Path(__file__).resolve().parent
repo = Path(subprocess.check_output(["git", "-C", str(here), "rev-parse", "--show-toplevel"], text=True).strip())
runner, output = args.runner.resolve(), args.output.resolve()
if runner == repo or not (runner / "ProjectSettings/ProjectVersion.txt").is_file():
    parser.error("Use a separate Unity project; this script never runs the working checkout.")
if (runner / "Temp/UnityLockfile").exists():
    parser.error("The runner is already open in Unity.")
for name in ("Assets/Shaders/Transport.hlsl", "Assets/Scripts/Simulation/RayTransport.cs",
             "Assets/Scripts/Simulation/TransportSolver.cs"):
    if (repo / name).read_bytes() != (runner / name).read_bytes():
        parser.error(f"Runner differs from this experiment's runtime: {name}")
if output.exists() and any(output.iterdir()):
    parser.error("Use an empty output directory so calibration and measurements cannot be mixed with an older run.")
output.mkdir(parents=True, exist_ok=True)
if args.calibrate_from:
    source = args.calibrate_from / "metrics.csv" if args.calibrate_from.is_dir() else args.calibrate_from
    rows = list(csv.DictReader(source.open()))
    with (output / "calibration.csv").open("w") as file:
        writer = csv.writer(file, lineterminator="\n"); writer.writerow(["scene", "hz", "samples", "target"])
        for scene, hz in sorted({(r["scene"], r["hz"]) for r in rows if r["method"] == "production"}):
            group = [r for r in rows if r["scene"] == scene and r["hz"] == hz and r["method"] == "production"]
            largest = max(int(r["samples"]) for r in group)
            group = [r for r in group if int(r["samples"]) == largest]
            mse = statistics.mean(float(r["empirical_rmse"]) ** 2 /
                (float(r["reference_real"]) ** 2 + float(r["reference_imag"]) ** 2) for r in group)
            samples = max(64, math.ceil(largest * mse / .2 ** 2 / 64) * 64)
            writer.writerow([scene, hz, samples, .2])
files = {here / "SamplingExperiment.cs": runner / "Assets/Tests/Editor/SamplingExperiment.cs",
         here / "SamplingExperiment.compute": runner / "Assets/Shaders/SamplingExperiment.compute",
         repo / "Assets/Shaders/Geometry.hlsl": runner / "Assets/Shaders/Geometry.hlsl"}
backups = {p: p.read_bytes() if p.exists() else None for dst in files.values()
           for p in (dst, Path(str(dst) + ".meta"))}
try:
    for src, dst in files.items():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
    env = dict(os.environ, ACOUSTIC_SAMPLING_OUTPUT=str(output))
    command = ["nice", "-n", "15", str(args.unity.resolve()), "-batchmode", "-force-metal",
               "-job-worker-count", "2", "-projectPath", str(runner), "-runTests",
               "-testPlatform", "EditMode", "-testFilter", "TakeChecks.CompareEdgeSampling",
               "-testResults", str(output / "tests.xml"), "-logFile", str(output / "run.log")]
    completed = subprocess.run(command, env=env, check=False)
finally:
    for path, data in backups.items():
        if data is None:
            path.unlink(missing_ok=True)
        else:
            path.write_bytes(data)
raise SystemExit(completed.returncode)
