# Dense rain source experiment — 26 October 2026

The first source experiment is implemented and exported for listening. A bounded convolution mixer reproduces the explicit dense-drop mixture to floating-point precision. It is faster at high event density and slower at low density. This verifies the source-generation optimization, not realistic rain, spatial grouping or Church acoustics. Listening acceptance is pending.

## Implemented model

[`tools/rain_source.py`](../../../tools/rain_source.py) authors two response banks: a noisy hard/wet-surface impact family and a water family with optional damped bubble-like chirps. Each contains 32 responses spanning four illustrative size classes and eight variants, with 80 ms maximum support. Shapes have smooth support and zero integral. Event gain is not normalized to cancel intensity. Surface responses, size probabilities, coupling, decay and bubble occurrence are authored examples, not measured material properties. The water frequency uses the Minnaert relation; that does not calibrate airborne amplitude. No recording was fitted into these kernels and no structural resonator is enabled.

Each region uses a marked Poisson process reconstructed from `(seed, region, absolute one-second block)`, independent of audio batch boundaries. Intensity 1 means 4,000 impacts/s in the test region, not a measured rainfall rate. Uniformly distributed integer sample onsets permit simultaneous drops. A waveform index and amplitude select the immutable response; there is no per-drop GameObject, PCM copy or filter history. The prototype supports constant intensity per run; time-varying weather control is not implemented.

The explicit reference adds every complete waveform at its event sample. The compact path accumulates events into one impulse stream per response, multiplies their FFTs by the corresponding response FFTs, sums spectra, and performs one inverse transform per block. Overlap-add preserves the complete tail across blocks. Linearity makes this an exact aggregation for the chosen finite response bank at one common transfer location; it does not collapse distinct spatial paths exactly.

The reference and optimized paths use exactly the same events and kernels. There is no Gaussian replacement, event thinning or boosted missing-drop compensation in the candidate. A separately labeled thinning control keeps 1/64 of events and boosts their amplitude by eight, demonstrating the texture damage of this different approximation.

## Verification and measurements

Three independent unit checks pass: explicit versus compact mixing with simultaneous impulses and boundary/tail cases; identical events under irregular interval queries; and full authored-bank agreement across different block sizes. Run:

```sh
python -m unittest discover -s tools -p test_rain_source.py -v
```

The experiment covers two surfaces × three intensities × three seeds, each eight seconds at 44.1 kHz. Across all 18 cases, maximum sample error between explicit and compact mixing is **1.04 × 10⁻¹⁶** in float64. Source intensities change expected event count and power; they do not change a transport budget. Native outputs retain the full 80 ms response tail after the eight-second source interval.

At 16,000 expected impacts/s, the median explicit/compact mixing speed ratios are **about 3× for both surfaces**. At 4,000/s the compact method is slightly slower, and at 1,000/s explicit mixing is substantially cheaper. Timings include renderer construction, exclude event generation and analysis (generation is separately reported), and are single runs per seed. These are offline source-only CPU measurements with one FFT worker and nice 19; they are not live Unity, GPU transport or end-to-end performance claims.

Each float64 response bank uses 903,424 bytes. The 2,048-sample compact source mixer accounts for approximately 4.76 MiB of persistent and transient numerical arrays, excluding the shared bank, output, event list and FFT scratch. It does not allocate audio/filter history per drop. This does not establish how much of Church's 1.36 GiB transport/DSP storage an integrated spatial-field implementation would save.

For the first hard-surface seed, the dense candidate's excess kurtosis is about 0.49; the thinned control is 36.50. The 10 ms P90−P10 level spread is about 3.8 dB versus 43.8 dB. The real recording measures about 0.35 and 4.6 dB respectively. The comparison improves these texture statistics substantially, but they are not sufficient acceptance criteria. The candidate has less low-frequency body and weaker slow envelope modulation than the recording. No EQ or modulation was fitted to hide those differences, because the recording contains unknown surface and propagation effects.

Different region seeds produce waveform correlation of approximately −0.0035 (hard) and −0.0080 (water) in this sample. This is a descriptive check, not proof of complete independence or absence of perceptual repetition. The immutable bank has only 32 variants.

![Source measurements and mixing cost](source-review.png)

## Listening files and reference provenance

Files are exported under the Conductor workspace's `.context/rain-source-review/`:

| File | Contents |
|---|---|
| `reference-then-hard-prototype.wav` | Six seconds of recorded rain, 0.5 s silence, then six seconds of the dry hard-surface prototype. |
| `hard-rain-prototype.wav` | Eight seconds of the hard-surface candidate, plus its tail. |
| `water-rain-prototype.wav` | Eight seconds of the water candidate, plus its tail. |
| `real-rain-reference.wav` | Eight seconds of the existing real-rain excerpt. |
| `dense-then-thinned-control.wav` | Same response family: dense mixture followed by fewer/louder impacts. This is not the earlier Church render. |

All synthetic auditions use one gain of 6.016105854351045, selected with headroom for the thinning control. The recording separately uses gain 0.6768708009372549 to RMS-match the hard candidate over seconds 1–7. All cuts have 20 ms edge fades. There is no limiter or per-segment synthetic normalization. Every delivered WAV is finite and below full scale; the largest peak is 0.7781 in the deliberately bad thinning control. Native hard/water float WAVs are preserved in the experiment directory.

The reference is the existing `Assets/External/Church/Audio/RainCourtyard.wav`, derived from Joseph SARDIN's **Rain and Thunder #1 (0124)**, [BigSoundBank](https://bigsoundbank.com/rain-and-storm-s0124.html), CC0. The page identifies outdoor heavy rain; surface and microphone position are unspecified. The repository excerpt excludes the later thunder, is mono, and retains the original environmental acoustics. It is a general texture reference, not isolated dry impacts, a specific water/hard-surface reference, or calibrated ground truth. Existing asset preparation is documented in the [Church attribution](../../../Assets/External/Church/README.md).

## Reproduction and remaining work

With NumPy, SciPy and Matplotlib available:

```sh
python docs/benchmarks/rain-source-2026-10-26/run.py <new-output-directory>
```

The command refuses an existing output directory, exports banks and native/audition WAVs, verifies numerical agreement, records measurements/hashes, and draws the chart. It never plays audio. The retained [report](report.json) contains all seeds, native metrics, timings, source-bank/event hashes, provenance and audition gains. Three tests and the 18-case experiment were run after the final kernel change.

Listen to both candidate families before promoting their timbre. A small spatial/shelter comparison remains next: per-drop positions versus fixed patches, with directions, delays and occlusion accounted for. Source generation is now independent of ray count, but there is no ray renderer in this experiment. Existing Church files, 32-bounce settings, material assignments and rain clip are unchanged. The known indirect-transport spikes and missing roof radiation remain unresolved. No Unity process, foreground window, audio-device playback or persistent worker was started.
