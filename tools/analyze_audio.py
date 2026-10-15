#!/usr/bin/env python3
"""Silent WAV inspection: shared-scale plots and sample-aligned measurements.

Requires numpy, scipy>=1.12 and matplotlib. No playback, GUI or Unity dependency.
"""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile
from scipy.signal import ShortTimeFFT, windows


def read_wave(path):
    rate, pcm = wavfile.read(path)
    dtype = pcm.dtype
    pcm = pcm.astype(np.float64)
    if dtype.kind == "u":
        pcm = (pcm - 128) / 128
    elif dtype.kind == "i":
        pcm /= float(2 ** (8 * dtype.itemsize - 1))
    if pcm.ndim == 1:
        pcm = pcm[:, None]
    if not len(pcm) or not np.isfinite(pcm).all():
        raise ValueError(f"Empty or non-finite audio: {path}")
    return rate, pcm


def spectrum(pcm, rate, fft, hop, first, end):
    transform = ShortTimeFFT(windows.hann(fft, sym=False), hop, rate,
                             fft_mode="onesided2X", scale_to="magnitude")
    # Compute only visible columns, but read their full windows from the file.
    # Zooming therefore never introduces artificial padding at selection edges.
    p0, p1 = first // hop, min(transform.p_max(len(pcm)), (end - 1) // hop + 2)
    z = transform.stft(pcm, p0=p0, p1=p1)
    return transform.f, transform.t(len(pcm), p0=p0, p1=p1), 20 * np.log10(np.maximum(abs(z), 1e-15))


def waveform(ax, pcm, rate, first, color):
    if len(pcm) <= 2400:
        ax.plot((first + np.arange(len(pcm))) / rate, pcm, color=color, lw=.7,
                marker="." if len(pcm) < 150 else None, markersize=3)
    else:
        # Min/max buckets preserve spikes that ordinary decimation would hide.
        edges = np.linspace(0, len(pcm), 2401, dtype=int)
        ax.vlines((first + (edges[:-1] + edges[1:] - 1) / 2) / rate,
                  np.minimum.reduceat(pcm, edges[:-1]), np.maximum.reduceat(pcm, edges[:-1]),
                  color=color, linewidth=.5)


def measurements(pcm, rate, first):
    result = []
    for channel in pcm.T:
        delta = np.diff(channel)
        jump = int(np.argmax(abs(delta))) + 1 if len(delta) else 0
        edges = np.diff(np.r_[False, channel == 0, False].astype(np.int8))
        starts, ends = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
        longest = int(np.argmax(ends - starts)) if len(starts) else None
        result.append(dict(peak=float(abs(channel).max()), rms=float(np.sqrt(np.mean(channel ** 2))),
                           dc=float(channel.mean()), atOrAboveFullScale=int(np.sum(abs(channel) >= 1)),
                           maxSampleStep=float(abs(delta[jump - 1])) if len(delta) else 0,
                           maxSampleStepAt=(first + jump) / rate,
                           longestExactSilence=None if longest is None else dict(
                               start=(first + int(starts[longest])) / rate,
                               frames=int(ends[longest] - starts[longest]))))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("waves", type=Path, nargs="+")
    parser.add_argument("--labels", nargs="+")
    parser.add_argument("--trim", type=int, nargs="+", help="Leading samples to remove from each WAV; explicit clock alignment, never guessed.")
    parser.add_argument("--start", type=float, default=0)
    parser.add_argument("--end", type=float)
    parser.add_argument("--fft", type=int, default=4096)
    parser.add_argument("--hop", type=int, default=256)
    parser.add_argument("--min-hz", type=float, default=0)
    parser.add_argument("--max-hz", type=float)
    parser.add_argument("--floor-db", type=float, default=-110)
    parser.add_argument("--ceiling-db", type=float, default=0)
    parser.add_argument("--grid-samples", type=int, help="Overlay response/hop boundaries at this sample interval when zoomed in.")
    parser.add_argument("--output", type=Path, required=True, help="New directory for plots and analysis.json.")
    args = parser.parse_args()
    trim = args.trim if args.trim is not None else [0] * len(args.waves)
    labels = args.labels if args.labels is not None else [f"{i + 1}: {p.name}" for i, p in enumerate(args.waves)]
    if len(trim) != len(args.waves) or len(labels) != len(args.waves) or any(n < 0 for n in trim):
        parser.error("Supply one nonnegative trim and one label per WAV.")
    if args.fft < 4 or args.fft % 2 or not 0 < args.hop <= args.fft:
        parser.error("Use an even FFT >= 4 and a hop in 1..FFT.")
    numbers = [args.start, args.min_hz, args.floor_db, args.ceiling_db]
    numbers += [x for x in (args.end, args.max_hz) if x is not None]
    if not np.isfinite(numbers).all() or args.start < 0 or args.floor_db >= args.ceiling_db:
        parser.error("Use finite ranges, a nonnegative start, and floor < ceiling.")
    if args.grid_samples is not None and args.grid_samples <= 0:
        parser.error("Grid interval must be positive.")
    loaded = [read_wave(p) for p in args.waves]
    rate, channels = loaded[0][0], loaded[0][1].shape[1]
    if any(fs != rate or x.shape[1] != channels for fs, x in loaded):
        parser.error("Sample rates and channel counts must match; no implicit resampling or downmixing.")
    audio = [x[n:] for (_, x), n in zip(loaded, trim)]
    frames = min(map(len, audio))
    first = round(args.start * rate)
    end = frames if args.end is None else min(frames, round(args.end * rate))
    high = rate / 2 if args.max_hz is None else args.max_hz
    if not 0 <= first < end or frames < args.fft // 2:
        parser.error("Select a nonempty common range; each trimmed WAV must contain at least FFT/2 samples.")
    if not 0 <= args.min_hz < high <= rate / 2:
        parser.error("Frequency range must be within 0..Nyquist.")
    args.output.mkdir(parents=True, exist_ok=False)
    report = dict(sampleRate=rate, channels=channels, firstSample=first, endSampleExclusive=end,
                  fft=args.fft, hop=args.hop, binHz=rate / args.fft, windowSeconds=args.fft / rate,
                  spectrum="Periodic Hann; one-sided peak amplitude dBFS; centers on aligned sample clock; zero-padding only outside file.",
                  interpretation="Sample steps, exact zeros and spectral sidebands are observations, not automatic artifact classifications.",
                  gridSamples=args.grid_samples, inputs=[], differences=[])
    for path, label, n, (_, full), x in zip(args.waves, labels, trim, loaded, audio):
        with path.open("rb") as stream:
            sha = hashlib.file_digest(stream, "sha256").hexdigest()
        report["inputs"].append(dict(path=str(path.resolve()), sha256=sha, label=label, originalFrames=len(full),
                                     trimmedLeadingSamples=n, unexaminedTrailingSamples=len(x) - end,
                                     measurements=measurements(x[first:end], rate, first)))
    for i in range(1, len(audio)):
        a, b = audio[0][first:end], audio[i][first:end]
        error = a - b
        report["differences"].append(dict(reference=labels[0], compared=labels[i], frames=len(error),
                                         maximumAbsoluteError=float(abs(error).max()),
                                         rmsError=float(np.sqrt(np.mean(error ** 2))),
                                         differingSamples=int(np.count_nonzero(error))))
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for channel in range(channels):
        rows = 2 * len(audio) + (len(audio) > 1)
        fig, axes = plt.subplots(rows, 1, figsize=(12, rows * 2.1), sharex=True, layout="constrained")
        peak = max(float(abs(x[first:end, channel]).max()) for x in audio) or 1
        for i, (x, label) in enumerate(zip(audio, labels)):
            wave, spec = axes[i * 2:i * 2 + 2]
            waveform(wave, x[first:end, channel], rate, first, colors[i % len(colors)])
            wave.set(title=label, ylabel="Amplitude", ylim=(-peak * 1.05, peak * 1.05))
            f, t, db = spectrum(x[:, channel], rate, args.fft, args.hop, first, end)
            band = (f >= args.min_hz) & (f <= high)
            if not band.any():
                raise ValueError("No FFT bins in the selected frequency range; increase FFT or widen range.")
            mesh = spec.pcolormesh(t, f[band], db[band], shading="nearest", cmap="magma",
                                   vmin=args.floor_db, vmax=args.ceiling_db, rasterized=True)
            spec.set(ylabel="Frequency (Hz)", ylim=(args.min_hz, high))
            fig.colorbar(mesh, ax=spec, label="dBFS")
        if len(audio) > 1:
            for i in range(1, len(audio)):
                waveform(axes[-1], audio[i][first:end, channel] - audio[0][first:end, channel], rate, first, colors[i % len(colors)])
            axes[-1].set(title="Difference from first input (own scale; no gain matching)", ylabel="Amplitude")
        for ax in axes:
            ax.set_xlim(first / rate, end / rate)
            if args.grid_samples and (end - first) / args.grid_samples <= 50:
                for n in range((first // args.grid_samples + 1) * args.grid_samples, end, args.grid_samples):
                    ax.axvline(n / rate, color="gray", alpha=.3, lw=.6)
        axes[-1].set_xlabel("Aligned time (seconds); sample 0 after explicit leading trim")
        fig.suptitle(f"Channel {channel + 1}/{channels} · {rate} Hz · shared waveform/color scales\n"
                     f"Hann {args.fft} samples ({1000 * args.fft / rate:.1f} ms), hop {args.hop}, bins {rate / args.fft:.3f} Hz")
        fig.savefig(args.output / f"channel-{channel + 1}.png", dpi=160)
        plt.close(fig)
    (args.output / "analysis.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(args.output / "analysis.json")


if __name__ == "__main__":
    main()
