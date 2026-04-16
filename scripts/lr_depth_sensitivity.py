"""LR sensitivity across depth and width for TFI circuits.

Fixed: J=1, g=0.75, h=0.6, dt=0.5. Vary: n_qubits × steps × lr.
Compile at same depth as target.

Usage: uv run python scripts/lr_depth_sensitivity.py
Output: data/lr_depth_sensitivity.csv
"""

import csv
import os
from tno_compiler.tfi import tfi_trotter_circuit
from tno_compiler.compiler import compile_circuit

J, g, h, dt = 1.0, 0.75, 0.6, 0.5
WIDTHS = [4, 6]
STEPS = [1, 2, 4]
LRS = [5e-2, 2e-2, 1e-2, 5e-3]
MAX_ITER = 300

os.makedirs("data", exist_ok=True)
OUTPUT = "data/lr_depth_sensitivity.csv"

with open(OUTPUT, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["n_qubits", "steps", "target_depth", "lr",
                     "init_cost", "final_cost", "best_cost", "converged_iter"])

    for n in WIDTHS:
        for steps in STEPS:
            target = tfi_trotter_circuit(n, J, g, h, dt, steps)
            td = target.depth()
            print(f"\nn={n}, steps={steps}, target_depth={td}")
            print(f"  {'lr':>8} | {'init':>10} | {'final':>10} | {'best':>10} | {'conv':>5}")
            print("  " + "-" * 50)

            for lr in LRS:
                _, info = compile_circuit(target, td, max_iter=MAX_ITER, lr=lr)
                hist = info['cost_history']
                best = min(hist)
                conv = next((i for i, c in enumerate(hist) if c < 0.01), -1)
                writer.writerow([n, steps, td, lr, hist[0], hist[-1], best, conv])
                print(f"  {lr:8.0e} | {hist[0]:10.4e} | {hist[-1]:10.4e} | "
                      f"{best:10.4e} | {conv:5d}")

print(f"\nResults in {OUTPUT}")
