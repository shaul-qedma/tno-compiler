"""Generate compression benchmark data for TFI Trotter circuits.

Sweeps over TFI parameters and ansatz depths, writes results to CSV.

Usage: uv run python scripts/generate_tfi_data.py
"""

import csv
import itertools
import sys
import numpy as np
from tno_compiler.tfi import tfi_trotter_circuit
from tno_compiler.pipeline import compile_ensemble

# Parameter grid
WIDTHS = [4, 6]
STEPS_LIST = [1, 2, 4]
J_VALUES = [0.5, 1.0]
G_VALUES = [0.5, 1.0]
H_VALUES = [0.0, 0.5]
DT_VALUES = [0.1, 0.5]
ANSATZ_FRACTIONS = [0.25, 0.5, 0.75, 1.0]  # fraction of target depth

N_CIRCUITS = 3
MAX_ITER = 100
LR = 5e-3
SEED = 42

OUTPUT = "data/tfi_benchmark.csv"

FIELDS = [
    "n_qubits", "J", "g", "h", "dt", "steps", "total_time",
    "target_depth", "ansatz_depth", "ansatz_fraction",
    "diamond_bound", "delta_ens", "R", "best_single_frob",
    "active_circuits", "n_circuits", "qp_value",
]


def target_depth_of(qc):
    """Count the depth (layers of 2-qubit gates) of a QuantumCircuit."""
    return qc.depth()


def main():
    import os
    os.makedirs("data", exist_ok=True)

    configs = list(itertools.product(
        WIDTHS, J_VALUES, G_VALUES, H_VALUES, DT_VALUES, STEPS_LIST))
    total = len(configs) * len(ANSATZ_FRACTIONS)
    print(f"Total configurations: {total}")

    with open(OUTPUT, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()

        for idx, (n, J, g, h, dt, steps) in enumerate(configs):
            target = tfi_trotter_circuit(n, J, g, h, dt, steps)
            td = target_depth_of(target)
            if td == 0:
                continue

            for frac in ANSATZ_FRACTIONS:
                ad = max(1, int(round(td * frac)))
                print(f"[{idx * len(ANSATZ_FRACTIONS) + ANSATZ_FRACTIONS.index(frac) + 1}"
                      f"/{total}] n={n} J={J} g={g} h={h} dt={dt} "
                      f"steps={steps} depth={td}→{ad}", flush=True)

                try:
                    result = compile_ensemble(
                        target, ad, n_circuits=N_CIRCUITS,
                        max_iter=MAX_ITER, lr=LR, seed=SEED)

                    best_single = min(result['individual_frobs'])
                    active = sum(1 for w in result['weights'] if w > 1e-3)

                    writer.writerow({
                        "n_qubits": n, "J": J, "g": g, "h": h,
                        "dt": dt, "steps": steps,
                        "total_time": dt * steps,
                        "target_depth": td, "ansatz_depth": ad,
                        "ansatz_fraction": frac,
                        "diamond_bound": result['diamond_bound'],
                        "delta_ens": result['delta_ens'],
                        "R": result['R'],
                        "best_single_frob": best_single,
                        "active_circuits": active,
                        "n_circuits": N_CIRCUITS,
                        "qp_value": result['qp_value'],
                    })
                    f.flush()
                except Exception as e:
                    print(f"  FAILED: {e}", flush=True)

    print(f"\nDone. Results in {OUTPUT}")


if __name__ == "__main__":
    main()
