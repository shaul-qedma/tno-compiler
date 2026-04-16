"""LR sensitivity for deeper TFI circuits (same-depth compilation).

Usage: uv run python scripts/lr_deep.py
"""

import csv
import os
from tno_compiler.tfi import tfi_trotter_circuit
from tno_compiler.compiler import compile_circuit

J, g, h = 1.0, 0.75, 0.6
n = 4
MAX_ITER = 500
LRS = [1e-1, 5e-2, 2e-2, 1e-2]

os.makedirs("data", exist_ok=True)

print(f"{'steps':>5} {'depth':>5} {'lr':>8} | {'init':>10} {'final':>10} {'best':>10} {'conv':>5}")
print("-" * 65)

for dt, steps in [(0.1, 1), (0.1, 2), (0.1, 4), (0.1, 8), (0.5, 1), (0.5, 2), (0.5, 4)]:
    target = tfi_trotter_circuit(n, J, g, h, dt, steps)
    td = target.depth()
    for lr in LRS:
        _, info = compile_circuit(target, td, max_iter=MAX_ITER, lr=lr)
        hist = info['cost_history']
        best = min(hist)
        conv = next((i for i, c in enumerate(hist) if c < 0.01), -1)
        print(f"{steps:5d} {td:5d} {lr:8.0e} | {hist[0]:10.4e} {hist[-1]:10.4e} "
              f"{best:10.4e} {conv:5d}", flush=True)
    print()
