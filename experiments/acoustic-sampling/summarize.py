#!/usr/bin/env python3
"""Report matched-error transport costs and validate the continuous control."""
import csv
import gzip
import json
import math
from pathlib import Path
import statistics
import sys
from collections import defaultdict
import xml.etree.ElementTree as ET

root = Path(sys.argv[1])
if (root / "tests.xml").exists():
    assert ET.parse(root / "tests.xml").find(".//test-case").get("result") == "Passed"
else:
    assert json.loads((root / "validation.json").read_text())["result"] == "Passed"
rows = list(csv.DictReader((root / "metrics.csv").open()))
reference = list(csv.DictReader((root / "reference.csv").open()))
names = {"production": "Production trace", "cauchy": "Cauchy proposal only", "source": "Source",
         "listener": "Listener", "join": "Two-ended join", "bidirectional_mis": "Source/listener/join mixture"}

def median(group, field):
    return statistics.median(float(r[field]) for r in group)

def timing(group, field):
    values = [float(r[field]) for r in group]
    q1, _, q3 = statistics.quantiles(values, n=4, method="inclusive")
    return f"{statistics.median(values):.3f} [{q1:.3f}–{q3:.3f}]"

print("# Matched-error acoustic sampling measurements\n")
print((root / "device.txt").read_text().strip().replace("\n", "; ") + ". Silent, low-priority Metal EditMode run.\n")
print("**The prototypes are promising after setup, but none is ready to replace the live renderer.** "
      "These measurements cover one static source/listener pair and its two-edge diffraction response. "
      "They exclude full-scene audio synthesis, moving listeners, and sharing across pairs.\n")
print("Prototype sample budgets come from enumerating the complete quadrature target's variance. "
      "The production budget near 20% was calibrated from a previous run and then evaluated with fresh seeds. "
      "The Cauchy-only control uses the same calibrated budget. Error is complex-response RMSE, "
      "not perceived audio quality or error against a wave solver.\n")
print("Five repeats follow a discarded warmup. Each batch contains 32 estimates; observed error combines "
      "160 estimates. A separate single-estimate dispatch is measured each repeat. Times include dispatch "
      "and blocking readback, with **median [25th–75th percentile]** shown. CPU final summation, proposal "
      "preparation, and SetData call time are outside these timings. This is wall time, not GPU timestamps; "
      "the wide single-estimate ranges should discourage precise speedup claims.\n")
for hz, target in (("250", "0.2"), ("250", "0.1"), ("62.5", "0.2")):
    print(f"## {hz} Hz, approximately {float(target):.0%} response error\n")
    print("| Scene | Method | Guided | Samples | Observed error | One estimate ms | Batch of 32 ms |")
    print("|---|---|---|---:|---:|---:|---:|")
    for scene in ("box", "staggered"):
        for guided in ("0", "1"):
            for method, label in names.items():
                group = [r for r in rows if r["scene"] == scene and r["hz"] == hz and r["method"] == method
                         and r["guided"] == guided and r["target_relative_rmse"] == target]
                if not group:
                    continue
                r = group[0]; magnitude = math.hypot(float(r["reference_real"]), float(r["reference_imag"]))
                observed = math.sqrt(statistics.mean(float(v["empirical_rmse"]) ** 2 for v in group)) / magnitude
                print(f"| {scene} | {label} | {'yes' if guided == '1' else 'no'} | {int(r['samples']):,} | "
                      f"{observed:.2%} | {timing(group, 'single_ms')} | {timing(group, 'batch_ms')} |")
    print()
print("The 10% production budget was **not measured**. That table only compares prototypes against their "
      "discrete target. All 64/512/4,096/16,384-sample controls are retained in the raw CSV.\n")
print("## Setup and transfer costs\n")
print("| Scene | Warm connections/readback ms | Baseline tables ms | Guided tables ms | GPU bytes per proposal pair | First connection dispatch ms |")
print("|---|---:|---:|---:|---:|---:|")
for r in csv.DictReader((root / "setup.csv").open()):
    print(f"| {r['scene']} | {float(r['connections_ms']):.3f} | {float(r['baseline_tables_ms']):.3f} | "
          f"{float(r['guided_tables_ms']):.3f} | {int(r['proposal_pair_bytes']):,} | {float(r['first_connections_ms']):.3f} |")
print("\nSetup is measured once per fixture; it is not a median. The first connection dispatch is recorded "
      "separately because it can include shader initialization. Common scene upload/BVH creation and Unity "
      "startup are excluded. Dense O(nodes²) tables are frequency-independent and reused throughout the run; "
      "their construction must be paid or amortized. They are not suitable as the live architecture.\n")
for scene in ("box", "staggered"):
    group = [r for r in rows if r["scene"] == scene and r["nodes"] != "0"]
    print(f"- {scene}: median proposal-pair SetData call {median(group, 'upload_ms'):.3f} ms; "
          "queued upload work may also affect the dispatch wait.")
print("\nThe production and Cauchy controls need no dense guide. Their `build_ms` field records common "
      "test-buffer/material setup, not guide creation; Cauchy-only inherits that allocation for comparison.\n")
print("## What was isolated\n")
print("1. Source, listener, join, and mixture use the same evaluator and quadrature target. Changing among "
      "them isolates proposal direction/combination; guidance changes only the proposal tables.\n"
      "2. Cauchy-only calls the production `SampleEdge` twice with the same two 50% edge-event choices, "
      "then reduces contributions directly. No dense guide is used.\n"
      "3. Production calls the unmodified `Trace3DBvh`, then reduces its second-event arrivals. Reflection "
      "and transmission are suppressed through material settings to isolate the two-edge contribution. "
      "The signed rigid-edge kernel and edge opacity remain unchanged. Surface calculations and full "
      "path writes still occur, so the gap to Cauchy-only measures more than sampling. Necessary surface "
      "transport cannot simply be removed from the complete renderer. Direct reduction here yields a "
      "static complex coefficient; changing delays, Doppler, and path continuity still need to be "
      "preserved in a live implementation.\n")
print("## Validation and limits\n")
groups = defaultdict(list)
trial_stream = (root / "trials.csv").open() if (root / "trials.csv").exists() else gzip.open(root / "trials.csv.gz", "rt")
for r in csv.DictReader(trial_stream):
    if r["method"] in ("production", "cauchy"):
        groups[(r["scene"], r["hz"], r["samples"], r["target_relative_rmse"], r["method"])].append(
            complex(float(r["real"]), float(r["imag"])))
trial_stream.close()
discrepancies = []
for key, p in groups.items():
    if key[-1] != "production":
        continue
    c = groups[(*key[:-1], "cauchy")]
    mean = lambda values: sum(values) / len(values)
    variance = lambda values: sum(abs(v - mean(values)) ** 2 for v in values) / (len(values) - 1)
    se = math.sqrt(variance(p) / len(p) + variance(c) / len(c))
    discrepancies.append(abs(mean(p) - mean(c)) / se)
assert max(discrepancies) < 6, "Continuous control disagrees with production beyond six combined standard errors."
print(f"The GPU test passed finite-output, positive-support, reciprocity, estimator-mean, zero-diffraction "
      f"isolation, and exact single/batch packing checks. Across {len(discrepancies)} continuous-control "
      f"comparisons, mean differences stayed within {max(discrepancies):.2f} combined standard errors "
      "(six-standard-error regression threshold). This is statistical consistency, not proof of zero bias.\n")
for scene in ("box", "staggered"):
    group = [r for r in reference if r["scene"] == scene and r["hz"] == "250"]
    values = {r["quadrature_order"]: complex(float(r["real"]), float(r["imag"])) for r in group}
    print(f"- {scene}, 250 Hz: changing 32 to 64 quadrature points per edge changes the reference by "
          f"{abs(values['32'] - values['64']) / abs(values['64']):.5%}.")
print("\nPrototypes use the enumerated 32-point target; continuous controls use the 64-point reference. "
      "The staggered continuum reference remains unconverged. Coplanar/grazing visibility still has a "
      "known reversal defect, reproduced by `ACOUSTIC_SAMPLING_COPLANAR=1`; these ordinary fixtures use "
      "unequal heights. No whole-church, moving-head, multi-source reuse, or training-reference claim follows.")
