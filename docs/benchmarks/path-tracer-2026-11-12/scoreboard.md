# Scoreboard: the path tracer against the baseline

Regenerate with `.context/fdtd-teacher/venv/bin/python docs/benchmarks/path-tracer-2026-11-12/scoreboard.py`.

RMS error against the wave ground truth (11 scenes), lower is better, all bands with ground truth pooled; seeds pooled.
`baseline-static-ir`: the ground truth of one central in-view receiver used at every receiver, with only the direct sound moved to each receiver's distance (no occlusion), i.e. a game convolution reverb.
A ✓ marks where a path render beats the baseline, a + where it beats the energy method.

## all scenes

### all

| method | budget | level dB | EDT % | T30 % | C50 dB | C80 dB | ETC dB |
|---|---|---|---|---|---|---|---|
| baseline-static-ir | ref | 8.85 | 420.40 | 38.02 | 5.90 | 5.87 | 10.05 |
| energy | r02048 | 3.04 | 269.21 | 78.69 | 1410.39 | 1924.76 | 15.06 |
| hybrid | r02048 | 5.68 | 383.90 | 68.48 | 18.38 | 22.20 | 13.64 |
| path | r02048 | 4.05 ✓ | 1011.51 | 88.16 | 1530.19 | 2026.18 | 16.44 |

### in view

| method | budget | level dB | EDT % | T30 % | C50 dB | C80 dB | ETC dB |
|---|---|---|---|---|---|---|---|
| baseline-static-ir | ref | 3.69 | 113.05 | 41.21 | 5.38 | 4.90 | 7.40 |
| energy | r02048 | 1.97 | 307.39 | 70.82 | 1171.82 | 1754.15 | 13.29 |
| hybrid | r02048 | 3.46 | 488.23 | 71.43 | 15.25 | 20.73 | 12.29 |
| path | r02048 | 2.18 ✓ | 1261.15 | 62.64 + | 1237.11 | 1826.31 | 14.59 |

### shadow

| method | budget | level dB | EDT % | T30 % | C50 dB | C80 dB | ETC dB |
|---|---|---|---|---|---|---|---|
| baseline-static-ir | ref | 13.36 | 656.75 | 34.70 | 6.63 | 7.12 | 13.14 |
| energy | r02048 | 4.18 | 195.71 | 88.92 | 1716.16 | 2163.04 | 17.46 |
| hybrid | r02048 | 7.99 | 77.85 | 64.42 | 22.40 | 24.30 | 15.51 |
| path | r02048 | 5.88 ✓ | 375.57 ✓ | 115.25 | 1898.01 | 2302.72 | 18.94 |

### open scenes

| method | budget | level dB | EDT % | T30 % | C50 dB | C80 dB | ETC dB |
|---|---|---|---|---|---|---|---|
| baseline-static-ir | ref | 9.77 | 466.16 | 40.70 | 6.48 | 6.47 | 10.89 |
| energy | r02048 | 3.23 | 298.42 | 83.89 | 1564.10 | 2134.53 | 16.50 |
| hybrid | r02048 | 6.02 | 425.48 | 72.67 | 20.25 | 24.52 | 14.87 |
| path | r02048 | 4.39 ✓ | 1121.78 | 94.08 | 1697.10 | 2247.19 | 17.36 |

### closed scenes

| method | budget | level dB | EDT % | T30 % | C50 dB | C80 dB | ETC dB |
|---|---|---|---|---|---|---|---|
| baseline-static-ir | ref | 1.73 | 15.44 | 6.13 | 1.95 | 1.60 | 4.89 |
| energy | r02048 | 1.99 | 18.40 | 23.94 | 1.86 | 1.70 | 5.47 |
| hybrid | r02048 | 3.89 | 31.05 | 27.55 | 4.97 | 4.48 | 5.76 |
| path | r02048 | 1.95 + | 25.18 | 24.76 | 2.00 | 1.90 | 11.57 |

## scenes with a 16,384-ray path render

Scenes: `barriers-0020-presets`, `barriers-0020-rigid`, `blocks-bem`, `canopy-0009-presets`, `canopy-0009-rigid`, `canopy-bem`, `doorway-0008-presets`, `shoebox-0010-presets`, `shoebox-0020-presets`.

### all

| method | budget | level dB | EDT % | T30 % | C50 dB | C80 dB | ETC dB |
|---|---|---|---|---|---|---|---|
| baseline-static-ir | ref | 8.87 | 445.98 | 40.38 | 6.16 | 6.16 | 10.01 |
| energy | r02048 | 3.13 | 284.79 | 79.77 | 1496.31 | 2042.02 | 15.42 |
| hybrid | r02048 | 5.74 | 407.04 | 71.49 | 19.32 | 23.43 | 14.03 |
| path | r02048 | 4.19 ✓ | 1072.15 | 69.10 + | 1619.69 | 2143.42 | 16.04 |
| path-16k | r16384 | 3.26 ✓ | 325.27 ✓ | 84.07 | 1415.75 + | 1915.85 + | 14.70 + |

### in view

| method | budget | level dB | EDT % | T30 % | C50 dB | C80 dB | ETC dB |
|---|---|---|---|---|---|---|---|
| baseline-static-ir | ref | 3.79 | 118.40 | 43.39 | 5.59 | 5.10 | 7.57 |
| energy | r02048 | 1.98 | 322.06 | 73.47 | 1227.93 | 1838.14 | 13.72 |
| hybrid | r02048 | 3.18 | 511.39 | 74.26 | 15.80 | 21.62 | 12.64 |
| path | r02048 | 2.20 ✓ | 1321.51 | 65.04 + | 1296.34 | 1913.76 | 14.46 |
| path-16k | r16384 | 1.98 ✓ | 406.04 | 86.08 | 1107.00 + | 1656.23 + | 12.98 + |

### shadow

| method | budget | level dB | EDT % | T30 % | C50 dB | C80 dB | ETC dB |
|---|---|---|---|---|---|---|---|
| baseline-static-ir | ref | 13.60 | 710.67 | 37.35 | 7.01 | 7.59 | 13.09 |
| energy | r02048 | 4.41 | 208.97 | 88.76 | 1857.14 | 2340.74 | 17.89 |
| hybrid | r02048 | 8.42 | 83.15 | 67.60 | 24.04 | 26.15 | 16.07 |
| path | r02048 | 6.22 ✓ | 399.33 ✓ | 74.80 + | 2046.28 | 2477.84 | 18.36 |
| path-16k | r16384 | 4.66 ✓ | 89.59 ✓ + | 81.55 + | 1814.78 + | 2281.51 + | 17.17 + |

### open scenes

| method | budget | level dB | EDT % | T30 % | C50 dB | C80 dB | ETC dB |
|---|---|---|---|---|---|---|---|
| baseline-static-ir | ref | 9.65 | 486.79 | 42.40 | 6.67 | 6.68 | 10.72 |
| energy | r02048 | 3.31 | 310.77 | 83.78 | 1633.38 | 2229.08 | 16.71 |
| hybrid | r02048 | 6.21 | 444.26 | 75.08 | 21.07 | 25.56 | 15.19 |
| path | r02048 | 4.50 ✓ | 1170.41 | 72.54 + | 1768.20 | 2339.95 | 17.13 |
| path-16k | r16384 | 3.47 ✓ | 354.91 ✓ | 88.30 | 1545.44 + | 2091.35 + | 15.85 + |

### closed scenes

| method | budget | level dB | EDT % | T30 % | C50 dB | C80 dB | ETC dB |
|---|---|---|---|---|---|---|---|
| baseline-static-ir | ref | 1.76 | 15.82 | 5.84 | 1.95 | 1.60 | 4.83 |
| energy | r02048 | 1.93 | 19.00 | 10.27 | 1.93 | 1.80 | 4.65 |
| hybrid | r02048 | 1.96 | 17.89 | 10.40 | 1.91 | 1.77 | 4.43 |
| path | r02048 | 1.85 + | 24.32 | 11.82 | 2.02 | 1.90 | 8.37 |
| path-16k | r16384 | 1.82 + | 23.81 | 11.01 | 1.99 | 1.88 | 5.75 |

