# Scoreboard: ours against the baseline

Regenerate with `.context/fdtd-teacher/venv/bin/python docs/benchmarks/energy-audio-2026-11-09/scoreboard.py`.

## Wave-ground-truth suite (11 scenes)

RMS error against the wave ground truth, lower is better, all bands with ground truth pooled.
`baseline-static-ir`: the ground truth of one central in-view receiver used at every receiver, with only the direct sound moved to each receiver's distance (no occlusion), i.e. a game convolution reverb.
A ✓ marks where the energy method beats the baseline.

### all

| method | level dB | EDT % | T30 % | C50 dB | C80 dB | ETC dB |
|---|---|---|---|---|---|---|
| baseline-static-ir | 8.85 | 420.40 | 38.02 | 5.90 | 5.87 | 10.05 |
| energy | 3.04 ✓ | 269.21 ✓ | 78.69 | 1410.39 | 1924.76 | 15.06 |
| hybrid | 5.68 | 383.90 | 68.48 | 18.38 | 22.20 | 13.64 |

### in view

| method | level dB | EDT % | T30 % | C50 dB | C80 dB | ETC dB |
|---|---|---|---|---|---|---|
| baseline-static-ir | 3.69 | 113.05 | 41.21 | 5.38 | 4.90 | 7.40 |
| energy | 1.97 ✓ | 307.39 | 70.82 | 1171.82 | 1754.15 | 13.29 |
| hybrid | 3.46 | 488.23 | 71.43 | 15.25 | 20.73 | 12.29 |

### shadow

| method | level dB | EDT % | T30 % | C50 dB | C80 dB | ETC dB |
|---|---|---|---|---|---|---|
| baseline-static-ir | 13.36 | 656.75 | 34.70 | 6.63 | 7.12 | 13.14 |
| energy | 4.18 ✓ | 195.71 ✓ | 88.92 | 1716.16 | 2163.04 | 17.46 |
| hybrid | 7.99 | 77.85 | 64.42 | 22.40 | 24.30 | 15.51 |

### open scenes

| method | level dB | EDT % | T30 % | C50 dB | C80 dB | ETC dB |
|---|---|---|---|---|---|---|
| baseline-static-ir | 9.77 | 466.16 | 40.70 | 6.48 | 6.47 | 10.89 |
| energy | 3.23 ✓ | 298.42 ✓ | 83.89 | 1564.10 | 2134.53 | 16.50 |
| hybrid | 6.02 | 425.48 | 72.67 | 20.25 | 24.52 | 14.87 |

### closed scenes

| method | level dB | EDT % | T30 % | C50 dB | C80 dB | ETC dB |
|---|---|---|---|---|---|---|
| baseline-static-ir | 1.73 | 15.44 | 6.13 | 1.95 | 1.60 | 4.89 |
| energy | 1.99 | 18.40 | 23.94 | 1.86 ✓ | 1.70 | 5.47 |
| hybrid | 3.89 | 31.05 | 27.55 | 4.97 | 4.48 | 5.76 |

## Church (no ground truth; measured cathedrals as the reference)

T30 (s) / EDT (s) / C80 (dB) per octave band; echo density: mean over the first 150 ms / time it reaches 0.9 (ms).

| response | 125 Hz | 250 Hz | 500 Hz | 1000 Hz | 2000 Hz | 4000 Hz | 8000 Hz | echo density |
|---|---|---|---|---|---|---|---|---|
| stpauls-9m | 9.0 / 6.9 / -4.7 | 10.4 / 8.5 / -7.2 | 10.7 / 6.9 / -1.8 | 9.1 / 5.5 / +0.7 | 7.0 / 3.5 / +6.1 | 3.9 / 1.7 / +9.1 | 2.0 / 0.1 / +13.0 | 0.79 / 41 |
| york-24m | 7.8 / 7.4 / +0.7 | 8.2 / 8.1 / +0.1 | 8.2 / 8.1 / -3.6 | 7.6 / 7.0 / -3.6 | 5.8 / 5.0 / +0.1 | 3.5 / 3.0 / +2.9 | 1.4 / 1.3 / +7.2 | 0.91 / 35 |
| stpauls-32m | 9.5 / 8.7 / -3.7 | 10.8 / 9.6 / -7.2 | 10.7 / 9.9 / -6.4 | 9.1 / 8.8 / -5.9 | 6.9 / 6.0 / +1.6 | 4.1 / 3.5 / +3.6 | 2.3 / 1.9 / +6.1 | 0.89 / 35 |
| church-near | 6.8 / 5.1 / -0.1 | 6.5 / 6.4 / -3.3 | 6.2 / 5.2 / -2.2 | 5.7 / 4.8 / -1.7 | 5.0 / 4.2 / -0.1 | 3.7 / 2.8 / +2.2 | 2.3 / 1.1 / +4.8 | 0.78 / 41 |
| church-near-hybrid | 5.3 / 5.7 / -6.6 | 6.3 / 5.6 / -6.9 | 3.6 / 7.4 / -4.4 | 4.6 / 9.0 / -4.4 | 3.7 / 13.1 / -0.2 | 4.2 / 9.3 / +3.3 | 2.7 / 0.9 / +5.2 | 0.94 / 13 |
| church-far | 5.0 / 7.9 / -1.9 | 4.9 / 8.1 / -6.0 | 5.0 / 7.5 / -9.1 | 5.1 / 6.9 / -6.8 | 4.9 / 6.1 / -5.9 | 4.0 / 4.4 / -4.3 | 3.1 / 3.2 / -1.0 | 0.85 / 50 |
| church-far-hybrid | 4.0 / 7.7 / -14.5 | 4.5 / 3.7 / -13.1 | 5.9 / 9.3 / -12.4 | 3.2 / 8.4 / -14.2 | 3.9 / 6.2 / -11.7 | 3.6 / 4.7 / -9.9 | 3.6 / 1.6 / -8.7 | 0.91 / 13 |
| church-near-before | 27.1 / 19.8 / -1.0 | 24.9 / 22.7 / -6.6 | 20.2 / 8.4 / -4.3 | 15.7 / 7.5 / -3.1 | 10.1 / 5.3 / +0.3 | 5.8 / 2.1 / +1.8 | 2.6 / 1.0 / +4.8 | 0.85 / 36 |
| baseline-near | 9.0 / 6.9 / -5.3 | 10.4 / 8.4 / -4.5 | 10.7 / 6.9 / -1.6 | 9.1 / 5.5 / +0.7 | 7.0 / 3.7 / +5.7 | 4.0 / 2.2 / +7.1 | 2.0 / 0.2 / +13.6 | 0.78 / 41 |
| baseline-far | 9.5 / 8.9 / -4.7 | 10.8 / 9.5 / -7.8 | 10.7 / 10.0 / -8.5 | 9.1 / 8.8 / -6.5 | 6.9 / 6.3 / -3.3 | 4.2 / 3.7 / -1.5 | 2.3 / 2.1 / +4.2 | 0.89 / 35 |

