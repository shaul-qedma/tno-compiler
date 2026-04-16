"""Target <5% diamond bound for deeper TFI circuits.

Sweep: target depth (steps) × compression ratio, with more iterations
and the best LR. Report which configs achieve diamond_bound < 0.05.

Usage: uv run python scripts/deep_compression.py
Output: data/deep_compression.csv
"""

import csv
import os
import numpy as np
from tno_compiler.tfi import tfi_trotter_circuit
from tno_compiler.pipeline import compile_ensemble

J, g, h = 1.0, 0.75, 0.6
N_CIRCUITS = 5
MAX_ITER = 300
LR = 5e-2

os.makedirs("data", exist_ok=True)
OUTPUT = "data/deep_compression.csv"

configs = []
for n in [4, 6]:
    for dt in [0.1, 0.2, 0.5]:
        for steps in [1, 2, 4, 8]:
            for frac in [0.25, 0.5, 0.75, 1.0]:
                configs.append((n, dt, steps, frac))

print(f"Total configs: {len(configs)}")

with open(OUTPUT, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["n_qubits", "dt", "steps", "total_time", "target_depth",
                     "ansatz_depth", "frac", "diamond_bound", "delta_ens", "R",
                     "best_single_frob", "active_circuits"])

    for idx, (n, dt, steps, frac) in enumerate(configs):
        target = tfi_trotter_circuit(n, J, g, h, dt, steps)
        td = target.depth()
        ad = max(1, int(round(td * frac)))
        total_time = dt * steps

        print(f"[{idx+1}/{len(configs)}] n={n} dt={dt} steps={steps} "
              f"t={total_time} depth={td}→{ad}", end="", flush=True)

        try:
            result = compile_ensemble(
                target, ad, n_circuits=N_CIRCUITS,
                max_iter=MAX_ITER, lr=LR, seed=42)
            db = result['diamond_bound']
            best = min(result['individual_frobs'])
            active = sum(1 for w in result['weights'] if w > 1e-3)
            marker = " *** <5%" if db < 0.05 else (" * <50%" if db < 0.5 else "")
            print(f" → diamond={db:.4f}, best_frob={best:.4f}{marker}", flush=True)

            writer.writerow([n, dt, steps, total_time, td, ad, frac,
                             db, result['delta_ens'], result['R'],
                             best, active])
            f.flush()
        except Exception as e:
            print(f" FAILED: {e}", flush=True)

print(f"\nDone. Results in {OUTPUT}")
