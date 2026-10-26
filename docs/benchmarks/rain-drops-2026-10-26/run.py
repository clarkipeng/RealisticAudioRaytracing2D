"""Small listening experiment: sum recorded impact snippets at three event rates.

Input: the documented Freesound HQ previews decoded to mono float32/44.1 kHz.
This is empirical sample playback, not a calibrated rain or material model.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.io import wavfile


SR = 44100
SECONDS = 5
RATES = (6, 60, 600)
SOURCES = {
    "concrete": {
        "id": 683772,
        "sha256": "328e564fc3f4dc44f802d8c4eca309960bb428919b39bc7cee49b66d368d94ef",
        "peaks_s": [.0718, .425, 2.0234, 2.9872, 3.6607, 4.1246,
                    4.6275, 5.2052, 7.9619, 9.0035, 9.8925],
    },
    "metal": {
        "id": 683778,
        "sha256": "9395e21f22477946a63da7393315ed2611ea407870f083571502f78464a7981e",
        "peaks_s": [.443, 1.6822, 2.4574, 3.4541, 4.379, 5.4955,
                    6.0672, 6.5042, 8.794, 11.4909, 11.9369, 12.9127, 16.3189],
    },
}


def rms(x):
    return float(np.sqrt(np.mean(x * x)))


def fade(x, attack, release):
    x[:attack] *= np.sin(np.linspace(0, np.pi / 2, attack)) ** 2
    x[-release:] *= np.cos(np.linspace(0, np.pi / 2, release)) ** 2
    return x


def build(input_dir, output_dir):
    output_dir.mkdir(parents=True, exist_ok=False)
    report = {"seed": 42, "sample_rate": SR, "status": "awaiting listening",
              "method": "mono overlap-add of recorded impact snippets; no added ambience or room response",
              "rates_are": "sample triggers per second, not physical rainfall units",
              "sources": {}, "audio": {}}
    for name, source in SOURCES.items():
        mp3 = input_dir / f"{name}-preview.mp3"
        assert hashlib.sha256(mp3.read_bytes()).hexdigest() == source["sha256"]
        rate, data = wavfile.read(input_dir / f"{name}.wav")
        assert rate == SR and data.ndim == 1 and data.dtype == np.float32
        assert np.isfinite(data).all()
        data = data.astype(np.float64)
        bank, cuts = [], []
        for peak in source["peaks_s"]:
            start = round((peak - .035) * SR)
            end = start + round(.25 * SR)
            assert 0 <= start < end <= len(data)
            clip = data[start:end].copy()
            clip -= clip.mean()
            fade(clip, round(.008 * SR), round(.020 * SR))
            bank.append(clip)
            cuts.append([start, end])
        bank = np.asarray(bank)
        np.save(output_dir / f"{name}-bank.npy", bank.astype(np.float32))

        # One marked Poisson stream. Thinning gives nested lower-rate subsets;
        # every retained event keeps its time and waveform at all densities.
        rng = np.random.default_rng(42)
        count = rng.poisson(max(RATES) * SECONDS)
        times = np.sort(rng.uniform(0, SECONDS, count))
        marks = rng.uniform(0, 1, count)
        voices = rng.integers(0, len(bank), count)
        offsets = np.rint(times * SR).astype(int)
        signals, counts = {}, {}
        for density in RATES:
            signal = np.zeros(SECONDS * SR + bank.shape[1])
            selected = np.flatnonzero(marks < density / max(RATES))
            for i in selected:
                offset = offsets[i]
                signal[offset:offset + bank.shape[1]] += bank[voices[i]]
            signals[f"{density}-per-second"] = fade(
                signal[:SECONDS * SR], round(.020 * SR), round(.020 * SR))
            counts[str(density)] = len(selected)

        # Equal RMS for a texture audition, not a loudness/intensity prediction.
        # Use one target for all densities; reduce all together if needed.
        target = min(.015, min(.5 * rms(x) / np.max(np.abs(x))
                               for x in signals.values()))
        gains = {}
        for key, signal in signals.items():
            gains[key] = target / rms(signal)
            signals[key] = signal * gains[key]
        gap = np.zeros(round(.5 * SR))
        signals["buildup"] = np.concatenate([
            signals["6-per-second"], gap, signals["60-per-second"], gap,
            signals["600-per-second"]])

        # Six excerpts in sequence, with their recorded relative levels kept.
        solo = np.zeros(5 * SR)
        for i, clip in enumerate(bank[:6]):
            offset = round((.2 + .75 * i) * SR)
            solo[offset:offset + len(clip)] += clip
        solo_gain = min(.25 / np.max(np.abs(solo)), .015 / rms(solo))
        signals["snippets"] = solo * solo_gain
        reference_gain = min(.4 / np.max(np.abs(data)), .015 / rms(data))
        signals["recording"] = data * reference_gain
        for key, signal in signals.items():
            assert np.isfinite(signal).all() and np.max(np.abs(signal)) < .51
            dest = output_dir / f"{name}-{key}.wav"
            wavfile.write(dest, SR, signal.astype(np.float32))
            report["audio"][dest.name] = {
                "sha256": hashlib.sha256(dest.read_bytes()).hexdigest(),
                "seconds": len(signal) / SR, "rms": rms(signal),
                "peak": float(np.max(np.abs(signal)))}
        report["sources"][name] = {
            **source, "author": "Elements-Library", "license": "CC0",
            "page": f'https://freesound.org/people/Elements-Library/sounds/{source["id"]}/',
            "download": f'https://cdn.freesound.org/previews/683/{source["id"]}_14876768-hq.mp3',
            "input_is": "lossy public HQ MP3 preview, not original WAV",
            "decoded_pcm_sha256": hashlib.sha256(data.astype("<f4").tobytes()).hexdigest(),
            "cut_sample_ranges": cuts, "bank_bytes_float32": bank.size * 4,
            "event_counts": counts, "density_audition_rms": target,
            "density_audition_gains": gains, "solo_gain": solo_gain,
            "reference_gain": reference_gain}
    (output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({name: {"snippets": len(s["peaks_s"]), "counts": s["event_counts"],
                             "rms": s["density_audition_rms"]}
                      for name, s in report["sources"].items()}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    build(args.input_dir, args.output_dir)
