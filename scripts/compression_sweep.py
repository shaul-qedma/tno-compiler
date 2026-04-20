"""Grid sweep: compression accuracy vs (n, g, Trotter steps, compression ratio).

Gapped/critical/paramagnetic TFI (J=-1 fixed, h=0 fixed, dt=0.1 fixed),
compiled to brickwall ansatzes at various depth ratios. Accuracy via
`sampled_max_trace_distance`.

Output: docs/data/compression_grid.csv — one row per unique compile config.

Usage: uv run python scripts/compression_sweep.py
"""

import csv
import time
from pathlib import Path

import numpy as np

from tno_compiler.pipeline import compile_ensemble
from tno_compiler.tfi import tfi_trotter_circuit
from tno_compiler.verify import sampled_max_trace_distance


# --- Grid ---
# Skip ratio=1 (no compression — same-depth case is well-covered by
# tests and the MPO-convert phase at large depth is memory-expensive).
N_VALUES = [4, 6, 8, 10]
G_VALUES = [0.3, 1.0, 1.5]             # gapped / critical / paramagnetic
STEPS_VALUES = [2, 4, 6, 8]
COMPRESSION_RATIOS = [2, 4, 8]

# --- Fixed params ---
J, H, DT = -1.0, 0.0, 0.1
N_CIRCUITS = 3
MAX_ITER = 100
N_SAMPLES = 10
COMPILE_SEED = 42
VERIFY_SEED = 0
MAX_ANSATZ_DEPTH = 16  # cap to avoid MPO-convert blowups

out_path = Path(__file__).resolve().parent.parent / "docs" / "data" / "compression_grid.csv"
out_path.parent.mkdir(parents=True, exist_ok=True)

fieldnames = [
    "n", "g", "steps", "dt",
    "target_depth", "ansatz_depth", "compression_ratio",
    "max_td", "mean_td",
    "compile_err_min", "compile_err_max",
    "compile_err_spread",  # max - min of individual_frobs
    "seeds_agree",         # bool: did all 3 seeds converge to same basin
    "elapsed_s",
]

# Build unique configs (dedupe on (n, g, steps, ansatz_depth) to avoid
# redundant ratio-equivalent runs)
configs = []
seen = set()
for n in N_VALUES:
    for g in G_VALUES:
        for steps in STEPS_VALUES:
            target = tfi_trotter_circuit(n, J, g, H, DT, steps)
            tgt_d = target.depth()
            for ratio in COMPRESSION_RATIOS:
                ansatz_d = max(1, tgt_d // ratio)
                if ansatz_d > tgt_d or ansatz_d > MAX_ANSATZ_DEPTH:
                    continue
                key = (n, g, steps, ansatz_d)
                if key in seen:
                    continue
                seen.add(key)
                configs.append((n, g, steps, tgt_d, ansatz_d, ratio, target))

print(f"{len(configs)} configs to run", flush=True)

# Streaming CSV writer
with open(out_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    f.flush()

    for idx, (n, g, steps, tgt_d, ansatz_d, ratio, target) in enumerate(configs):
        t0 = time.perf_counter()
        try:
            result = compile_ensemble(
                target, ansatz_d,
                n_circuits=N_CIRCUITS, max_iter=MAX_ITER, seed=COMPILE_SEED)
            out = sampled_max_trace_distance(
                target, result['circuits'], result['weights'],
                n_samples=N_SAMPLES, seed=VERIFY_SEED)
            indiv = list(result['individual_frobs'])
            err_min, err_max = min(indiv), max(indiv)
            err_spread = err_max - err_min
            seeds_agree = err_spread < max(1e-6, err_min * 1e-4)
            max_td = out['max_td']
            mean_td = out['mean_td']
        except Exception as e:
            print(f"  ERROR at n={n} g={g} steps={steps} ansatz_d={ansatz_d}: {e}",
                  flush=True)
            continue

        elapsed = time.perf_counter() - t0
        row = {
            "n": n, "g": g, "steps": steps, "dt": DT,
            "target_depth": tgt_d, "ansatz_depth": ansatz_d,
            "compression_ratio": ratio,
            "max_td": max_td, "mean_td": mean_td,
            "compile_err_min": err_min, "compile_err_max": err_max,
            "compile_err_spread": err_spread,
            "seeds_agree": int(seeds_agree),
            "elapsed_s": elapsed,
        }
        writer.writerow(row)
        f.flush()
        print(f"  [{idx+1}/{len(configs)}] n={n} g={g} steps={steps} "
              f"tgt_d={tgt_d} ansatz_d={ansatz_d} -> max_td={max_td:.3e} "
              f"({elapsed:.1f}s)",
              flush=True)

print(f"\nDone. Wrote {out_path}")
