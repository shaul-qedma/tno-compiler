"""Investigate learning rate sensitivity across circuit sizes for TFI.

Fixed TFI parameters (J=1, g=0.75, h=0.6, dt=0.5, 1 step).
Vary: n_qubits × lr, compile at same depth, report final cost.

Usage: uv run python scripts/lr_sensitivity.py
Output: data/lr_sensitivity.csv
"""

import csv
import os
import numpy as np
from tno_compiler.tfi import tfi_trotter_circuit
from tno_compiler.compiler import compile_circuit

J, g, h, dt, steps = 1.0, 0.75, 0.6, 0.5, 1
WIDTHS = [4, 6, 8]
LRS = [5e-2, 2e-2, 1e-2, 5e-3, 2e-3, 1e-3]
MAX_ITER = 300

os.makedirs("data", exist_ok=True)
OUTPUT = "data/lr_sensitivity.csv"

with open(OUTPUT, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["n_qubits", "target_depth", "lr",
                     "init_cost", "final_cost", "best_cost", "converged_iter"])

    for n in WIDTHS:
        target = tfi_trotter_circuit(n, J, g, h, dt, steps)
        td = target.depth()
        print(f"\nn={n}, target_depth={td}")
        print(f"  {'lr':>8} | {'init':>10} | {'final':>10} | {'best':>10} | {'conv_iter':>9}")
        print("  " + "-" * 55)

        for lr in LRS:
            _, info = compile_circuit(target, td, max_iter=MAX_ITER, lr=lr)
            hist = info['cost_history']
            init_cost = hist[0]
            final_cost = hist[-1]
            best_cost = min(hist)
            # Find first iteration where cost < 0.01
            conv_iter = next((i for i, c in enumerate(hist) if c < 0.01), -1)

            writer.writerow([n, td, lr, init_cost, final_cost, best_cost, conv_iter])
            print(f"  {lr:8.0e} | {init_cost:10.4e} | {final_cost:10.4e} | "
                  f"{best_cost:10.4e} | {conv_iter:9d}")

print(f"\nResults in {OUTPUT}")
