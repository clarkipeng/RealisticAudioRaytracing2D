"""Recheck saved float WAVs: python verify.py (NumPy/SciPy)."""
import json
from pathlib import Path

import numpy as np
from scipy.io import wavfile


def verify(root):
    evidence = json.loads((root / "church-probe.json").read_text())
    assert evidence["settingsPreserved"]
    assert all(rms > .001 for rms in evidence["clipExcerptRms"])
    checks = {}
    for name, report in zip(("outdoor", "indoor"), evidence["renders"]):
        rate, audio = wavfile.read(root / (name + ".wav"))
        assert rate == report["sampleRate"]
        assert audio.shape == (report["outputSamples"],)
        assert np.isfinite(audio).all()
        peak = float(np.max(np.abs(audio)))
        assert 0 < peak < 1, (name, peak)
        assert np.max(np.abs(audio[-rate:])) == 0, name
        assert report["complete"] and report["settings"]["maxBounces"] == 0
        assert report["gpuAudio"]["pathCapacity"] == evidence["directSlots"]
        checks[name] = dict(sampleRate=rate, samples=len(audio), peak=peak,
                            rms=float(np.sqrt(np.mean(audio.astype(float) ** 2))),
                            finalSecondPeak=0, finite=True, unclipped=True)
    return checks


if __name__ == "__main__":
    print(json.dumps(verify(Path(__file__).resolve().parent), indent=2))
