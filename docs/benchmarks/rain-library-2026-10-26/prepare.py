"""Prepare short, consistently gained excerpts of four free GDC library samples."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import wave

import numpy as np


FILES = {
    "rain": "PM_RD_RAIN_8_MODERATE_DRIPPY.wav",
    "background": "PM_RD_DESIGNED_CONSTANT_BACKING_NOISE_5.wav",
    "tarp": "PM_RD_DESIGNED_DRIPPY_ON_TARP_FOLIAGE_2.wav",
    "iron": "PM_RD_DESIGNED_DRIPPY_ON_IRON_PLATE_3.wav",
}


def prepare(source, dest):
    downloads = {d["file"]: d for d in json.loads((source / "downloads.json").read_text())}
    dest.mkdir(parents=True, exist_ok=False)
    report = {
        "status": "awaiting listening", "author": "PMSFX / Phil Michalski",
        "collection": "Rain Designer selections in Sonniss GDC 2020",
        "license": "https://sonniss.com/gdc-bundle-license/",
        "archive": "https://sonniss.com/gameaudiogdc/",
        "official_tracklist": "https://docs.google.com/spreadsheets/d/1QqIcxJ7y0Acq3HAj9qQvFgvyeBjJ_7JvX8-GguUHIVk/edit",
        "processing": {"start_s": 5, "duration_s": 12, "gain": 1.0,
                       "edge_fades_s": .02, "output": "44.1 kHz stereo PCM16"},
        "purpose": "source-quality audition only; no drop extraction, synthesis or propagation",
        "files": {},
    }
    for key, filename in FILES.items():
        original = source / "originals" / filename
        metadata = downloads[filename]
        assert hashlib.sha256(original.read_bytes()).hexdigest() == metadata["sha256"]
        probe = json.loads(subprocess.check_output([
            "ffprobe", "-v", "error", "-select_streams", "a", "-show_entries",
            "stream=codec_name,sample_rate,channels,bits_per_sample:format=duration",
            "-of", "json", str(original)]))
        assert len(probe["streams"]) == 1 and probe["streams"][0]["channels"] == 2
        assert float(probe["format"]["duration"]) >= 17
        output = dest / f"{key}.wav"
        subprocess.run([
            "ffmpeg", "-v", "error", "-nostdin", "-threads", "1", "-filter_threads", "1",
            "-ss", "5", "-i", str(original), "-map", "0:a:0", "-t", "12", "-ar", "44100",
            "-af", "afade=t=in:d=0.02,afade=t=out:st=11.98:d=0.02",
            "-c:a", "pcm_s16le", str(output)], check=True)
        with wave.open(str(output)) as wav:
            assert wav.getframerate() == 44100 and wav.getnchannels() == 2
            assert wav.getsampwidth() == 2 and wav.getnframes() == 12 * 44100
            x = np.frombuffer(wav.readframes(wav.getnframes()), dtype="<i2").astype(float) / 32768
        assert np.isfinite(x).all() and np.max(np.abs(x)) < 1.0
        report["files"][key] = {
            **metadata, "original_format": probe, "excerpt": output.name,
            "excerpt_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
            "excerpt_peak": float(np.max(np.abs(x))),
            "excerpt_rms": float(np.sqrt(np.mean(x * x))),
        }
    (dest / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: {"peak": v["excerpt_peak"], "rms": v["excerpt_rms"]}
                      for k, v in report["files"].items()}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("dest", type=Path)
    args = parser.parse_args()
    prepare(args.source, args.dest)
