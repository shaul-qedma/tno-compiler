"""Deep TFI compression at n=6, targeting <5% superoperator norm error.

Fixed: n=6, J=1, g=0.75, h=0.6, dt=0.1.
Sweep: steps={1,2,4,8,16} × ansatz_fraction={0.25,0.5,0.75,1.0}.
LR=2e-2, 500 iter, 5 circuits, scale=0.01.

Reports both diamond bound and actual superoperator norm.

Usage: uv run python scripts/deep_n6.py
"""

import csv
import os
import time
import numpy as np
from qiskit.quantum_info import Operator
from tno_compiler.tfi import tfi_trotter_circuit
from tno_compiler.pipeline import compile_ensemble

n = 6
J, g, h, dt = 1.0, 0.75, 0.6, 0.1
N_CIRCUITS = 5
MAX_ITER = 500
LR = 2e-2

os.makedirs("data", exist_ok=True)
OUTPUT = "data/deep_n6.csv"

with open(OUTPUT, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["steps", "total_time", "target_depth", "ansatz_depth",
                     "frac", "diamond_bound", "superop_norm",
                     "best_single_frob", "active", "time_s"])

    for steps in [1, 2, 4, 8, 16]:
        target = tfi_trotter_circuit(n, J, g, h, dt, steps)
        td = target.depth()
        V = Operator(target).data
        S_tgt = np.kron(V.conj(), V)
        d = 2 ** n

        for frac in [0.25, 0.5, 0.75, 1.0]:
            ad = max(1, int(round(td * frac)))
            t0 = time.perf_counter()

            result = compile_ensemble(
                target, ad, n_circuits=N_CIRCUITS,
                max_iter=MAX_ITER, lr=LR, seed=42)

            # Compute actual superoperator norm
            S_ens = np.zeros((d**2, d**2), dtype=complex)
            for qc, p in zip(result['circuits'], result['weights']):
                if p < 1e-15:
                    continue
                U = Operator(qc).data
                S_ens += p * np.kron(U.conj(), U)
            superop = np.linalg.norm(S_ens - S_tgt, ord=2)

            elapsed = time.perf_counter() - t0
            best = min(result['individual_frobs'])
            active = sum(1 for w in result['weights'] if w > 1e-3)
            db = result['diamond_bound']

            marker = ""
            if superop < 0.05:
                marker = " *** <5%"
            elif superop < 0.5:
                marker = " * <50%"

            print(f"steps={steps:2d} depth={td:2d}→{ad:2d} | "
                  f"superop={superop:.4e} diamond={db:.4e} "
                  f"best_frob={best:.4e} active={active}/{N_CIRCUITS} "
                  f"({elapsed:.0f}s){marker}", flush=True)

            writer.writerow([steps, dt*steps, td, ad, frac,
                             db, superop, best, active, f"{elapsed:.1f}"])
            f.flush()

print(f"\nDone. Results in {OUTPUT}")
