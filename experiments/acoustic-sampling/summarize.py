#!/usr/bin/env python3
"""Summarize the raw CSVs without plotting dependencies or playback."""
import csv
import math
from pathlib import Path
import statistics
import sys

root = Path(sys.argv[1])
rows = list(csv.DictReader((root / "metrics.csv").open()))
reference = list(csv.DictReader((root / "reference.csv").open()))
print("# Finite-edge sampling measurements\n")
print((root / "device.txt").read_text().strip().replace("\n", "; ") + ". Silent EditMode compute, one low-priority process.\n")
print("All rows compare prototype samplers for a discretized two-edge contribution. "
      "The source-only control is **not the production tracer's Cauchy proposal**, and these are not whole-renderer gains.\n")
print("Expected error is calculated from **every candidate path**, not just observed trial error. "
      "Timings are medians of three batches after warmup. Each batch evaluates 32 independent estimates "
      "with 4,096 samples each. They include dispatch and blocking readback; they exclude table construction "
      "and the CPU SetData call, reported separately below. Queued upload work can still affect the dispatch wait. "
      "These are batch costs, not single-listener frame times or hardware GPU timestamps.\n")
for hz in ("62.5", "250"):
    print(f"## {hz} Hz\n")
    print("| Scene | Sampler | Guided | Expected relative RMSE | Observed relative RMSE | Batch ms | Upload ms | Relative MSE × time |")
    print("|---|---|---|---:|---:|---:|---:|---:|")
    for scene in ("box", "staggered"):
        base = [r for r in rows if r["scene"] == scene and r["hz"] == hz and r["samples"] == "4096"]
        control = [r for r in base if r["method"] == "source" and r["guided"] == "0"]
        control_cost = float(control[0]["expected_rmse"]) ** 2 * statistics.median(float(r["batch_ms"]) for r in control)
        for guided in ("0", "1"):
            for method in ("source", "listener", "join", "bidirectional_mis"):
                data = [r for r in base if r["method"] == method and r["guided"] == guided]
                r = data[0]
                magnitude = math.hypot(float(r["reference_real"]), float(r["reference_imag"]))
                expected = float(r["expected_rmse"])
                observed = math.sqrt(statistics.mean(float(v["empirical_rmse"]) ** 2 for v in data))
                elapsed = statistics.median(float(v["batch_ms"]) for v in data)
                upload = statistics.median(float(v["upload_ms"]) for v in data)
                cost = expected ** 2 * elapsed / control_cost
                print(f"| {scene} | {method} | {'yes' if guided == '1' else 'no'} | {expected / magnitude:.2%} | "
                      f"{observed / magnitude:.2%} | {elapsed:.3f} | {upload:.3f} | {cost:.2f} |")
    print("\nLast column is `expected MSE × measured batch time`, normalized to source-only for that scene "
          "and frequency; lower is better. It is a throughput efficiency indicator, **not a measured speedup** "
          "at a separately rerun equal-time budget. Small differences are not meaningful given timing variation.\n")
print("## Setup cost\n")
print("| Scene | Nodes | Connections/readback ms | Baseline tables ms | Guided tables ms | GPU bytes per proposal pair |")
print("|---|---:|---:|---:|---:|---:|")
for r in csv.DictReader((root / "setup.csv").open()):
    print(f"| {r['scene']} | {r['nodes']} | {float(r['connections_ms']):.3f} | "
          f"{float(r['baseline_tables_ms']):.3f} | {float(r['guided_tables_ms']):.3f} | {int(r['proposal_pair_bytes']):,} |")
print("\nThese are single setup measurements, not medians; the first connection dispatch can include shader "
      "initialization. Tables use O(nodes²) storage and rebuild work. This is a transparent research "
      "implementation, unsuitable as a live architecture. All geometry and frequency-independent setup "
      "is reused across seeds, sample budgets, methods, and frequencies in the measurement.\n")
print("## Target discretization and reciprocity\n")
print("| Scene | Hz | Change from 32 to 64 points per edge | Largest path reversal discrepancy |")
print("|---|---:|---:|---:|")
for scene in ("box", "staggered"):
    for hz in ("62.5", "250"):
        group = [r for r in reference if r["scene"] == scene and r["hz"] == hz]
        values = {r["quadrature_order"]: complex(float(r["real"]), float(r["imag"])) for r in group}
        change = abs(values["32"] - values["64"]) / abs(values["64"])
        reciprocity = max(float(r["reciprocity_relative_l2"]) for r in group)
        print(f"| {scene} | {hz} | {change:.5%} | {reciprocity:.3g} |")
print("\nReversal discrepancy is relative L2 over individual path contributions. It is **not** validation "
      "against a wave solver. The staggered target is not converged: sampling rankings concern the "
      "enumerated 32-point discretization. The equal-height coplanar variant fails reciprocity and is "
      "retained separately as a known production-visibility defect.\n")
print("## Calculation\n\nFor complex path contribution `F(a,b)` and proposal probability `q(a,b)`, "
      "the enumerated response is `H = sum F`. Expected squared error after N independent samples is "
      "`(sum |F|²/q - |H|²)/N`. Observed error uses 96 estimates per row (32 seeds × 3 repeats). "
      "The table probabilities use the actual float CDF interval widths.")
