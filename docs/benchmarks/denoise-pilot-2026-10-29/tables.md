### Label noise (early-field label, K = 8)

| Family | Scenes | Accepted | Label SE (RMS) | Label SE (worst) | Valid entries | Input SD / label SE at 1/256 / 1/64 / 1/16 / 1/4 | Total-label SE |
|---|---:|---:|---:|---:|---:|---:|---:|
| shoebox | 12 | 12 | 1.2% | 2.4% | 100% | 273 / 159 / 65 / 6 | 80% |
| doorway | 12 | 12 | 2.9% | 4.9% | 99% | 101 / 75 / 35 / 13 | 97% |
| corridor | 12 | 12 | 3.5% | 6.0% | 99% | 135 / 89 / 38 / 19 | 90% |
| canopy | 12 | 11 | 4.0% | 8.9% | 97% | 65 / 33 / 17 / 6 | 76% |
| barriers | 12 | 9 | 5.2% | 9.1% | 92% | 49 / 30 / 16 / 6 | 98% |
| furnished | 12 | 7 | 18.9% | 35.5% | 64% | 160 / 83 / 35 / 19 | 99% |
| all | 72 | 63 | 8.4% | 35.5% | 92% | 124 / 73 / 32 / 15 | 90% |

### Input error against the label (accepted scenes, early field)

| Family | 1/256 | 1/64 | 1/16 | 1/4 | 125–500 Hz at 1/64 | 2–4 kHz at 1/64 |
|---|---:|---:|---:|---:|---:|---:|
| shoebox | 336% | 195% | 79% | 8% | 196% | 193% |
| doorway | 292% | 214% | 100% | 37% | 225% | 211% |
| corridor | 471% | 307% | 137% | 67% | 342% | 299% |
| canopy | 200% | 104% | 52% | 19% | 112% | 103% |
| barriers | 196% | 121% | 61% | 25% | 130% | 119% |
| furnished | 841% | 432% | 183% | 100% | 467% | 421% |
| all | 414% | 242% | 107% | 49% | 261% | 237% |

### Outliers (early field, all accepted scenes)

| Budget | Median p99 of \|error\| / field RMS | Largest | Entries > 1× RMS | > 3× RMS | Family with the largest error in >1× entries |
|---|---:|---:|---:|---:|---|
| 1/256 | 8.53 | 148 | 41.5% | 16.06% | RR 36%, DD 24%, RD 19% |
| 1/64 | 4.81 | 33 | 41.9% | 11.28% | RR 60%, DD 17%, RD 12% |
| 1/16 | 2.24 | 14 | 23.3% | 2.29% | RR 77%, DD 11%, RD 6% |
| 1/4 | 0.43 | 8 | 6.7% | 0.14% | RR 87%, RD 5%, DD 4% |

### Baselines on the test split (early field; canopy family and wide doorways)

| Budget | Solve ms | Raw, 1 seed | Raw at the next budget (4× rays) | Mean of 4 seeds (4× solves) | Mean of 2 seeds | Grid-neighbour mean | Family clamp |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1/256 | 1.8 | 226% | 144% | 109% | 159% | 141% | 137% |
| 1/64 | 2.4 | 144% | 72% | 71% | 102% | 112% | 89% |
| 1/16 | 3.6 | 72% | 27% | 35% | 51% | 93% | 66% |
| 1/4 | 6.0 | 27% | n/a | 15% | 20% | 87% | 41% |

### Rigid tracer label against FDTD

| Family | Settled scenes 125 / 250 Hz | 125 Hz early | 125 Hz incl. walk | 125 Hz LoS / shadow | 250 Hz early | 250 Hz incl. walk | 250 Hz LoS / shadow | Label SE 125 / 250 Hz |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| doorway | 0 / 1 of 6 | n/a | n/a | n/a / n/a | 17% | 17% | 10% / 102% | n/a / 0.9% |
| corridor | 0 / 0 of 6 | n/a | n/a | n/a / n/a | n/a | n/a | n/a / n/a | n/a / n/a |
| canopy | 4 / 4 of 6 | 23% | 691% | 22% / 41% | 18% | 729% | 17% / 47% | 2.9% / 3.2% |
| barriers | 6 / 6 of 6 | 21% | 2484% | 16% / 51% | 15% | 2691% | 11% / 46% | 2.8% / 3.0% |
