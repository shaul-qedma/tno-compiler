"""Analyze TFI compression benchmark data from CSV.

Usage: uv run python scripts/analyze_tfi_data.py
"""

import pandas as pd
import numpy as np

df = pd.read_csv("data/tfi_benchmark.csv")
print(f"Loaded {len(df)} rows\n")

# Summary: how does diamond_bound depend on compression ratio?
print("=" * 70)
print("Diamond bound vs ansatz fraction (averaged over all parameters)")
print("=" * 70)
summary = df.groupby("ansatz_fraction").agg(
    diamond_mean=("diamond_bound", "mean"),
    diamond_median=("diamond_bound", "median"),
    diamond_min=("diamond_bound", "min"),
    diamond_max=("diamond_bound", "max"),
    best_single_mean=("best_single_frob", "mean"),
    count=("diamond_bound", "count"),
).round(4)
print(summary.to_string())

# How does dt affect compressibility at fixed fraction?
print("\n" + "=" * 70)
print("Diamond bound vs dt (ansatz_fraction=0.5)")
print("=" * 70)
filt = df[df["ansatz_fraction"] == 0.5]
if len(filt) > 0:
    by_dt = filt.groupby("dt").agg(
        diamond_mean=("diamond_bound", "mean"),
        diamond_median=("diamond_bound", "median"),
        count=("diamond_bound", "count"),
    ).round(4)
    print(by_dt.to_string())

# How does total_time affect compressibility?
print("\n" + "=" * 70)
print("Diamond bound vs total_time (ansatz_fraction=0.5)")
print("=" * 70)
if len(filt) > 0:
    by_time = filt.groupby("total_time").agg(
        diamond_mean=("diamond_bound", "mean"),
        diamond_median=("diamond_bound", "median"),
        count=("diamond_bound", "count"),
    ).round(4)
    print(by_time.to_string())

# Best cases: where does compression work?
print("\n" + "=" * 70)
print("Best compression results (ansatz_fraction < 1, lowest diamond_bound)")
print("=" * 70)
compressed = df[df["ansatz_fraction"] < 1.0].nsmallest(10, "diamond_bound")
cols = ["n_qubits", "J", "g", "h", "dt", "steps", "target_depth",
        "ansatz_depth", "diamond_bound", "best_single_frob"]
print(compressed[cols].to_string(index=False))

# Width comparison
print("\n" + "=" * 70)
print("Diamond bound by width (ansatz_fraction=0.5)")
print("=" * 70)
if len(filt) > 0:
    by_n = filt.groupby("n_qubits").agg(
        diamond_mean=("diamond_bound", "mean"),
        diamond_median=("diamond_bound", "median"),
        count=("diamond_bound", "count"),
    ).round(4)
    print(by_n.to_string())
