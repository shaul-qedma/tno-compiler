"""Compression scaling: cost vs n for 2×, 4×, 6×, 8× compression.

For each (n, k), generates multiple random TFI circuits with different
coupling parameters. Constraint: compressed circuit has >= ceil(n/3) steps.

Usage: uv run python scripts/compression_scaling.py
Output: data/compression_scaling.csv
"""

import csv
import math
import os
import time
import numpy as np
from tno_compiler.tfi import tfi_trotter_circuit
from tno_compiler.compiler import compile_circuit

WIDTHS = [10, 20, 30, 40, 50]
COMPRESSIONS = [2, 4, 6, 8]
N_SAMPLES = 3  # random TFI instances per (n, k)
MAX_ITER = 10
MAX_TARGET_STEPS = 32  # cap target depth for feasibility
DT = 0.1

os.makedirs("data", exist_ok=True)
OUTPUT = "data/compression_scaling.csv"

rng = np.random.RandomState(42)

with open(OUTPUT, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["n", "k", "sample", "J", "g", "h",
                     "target_steps", "target_depth",
                     "ansatz_steps", "ansatz_depth",
                     "cost", "frob", "time_s"])

    for n in WIDTHS:
        min_ansatz_steps = math.ceil(n / 3)
        for k in COMPRESSIONS:
            target_steps = min(k * min_ansatz_steps, MAX_TARGET_STEPS)
            ansatz_steps = max(1, target_steps // k)

            for s in range(N_SAMPLES):
                J = rng.uniform(0.5, 2.0)
                g = rng.uniform(0.3, 1.5)
                h = rng.uniform(0.0, 1.0)

                target = tfi_trotter_circuit(n, J, g, h, DT, target_steps)
                td = target.depth()
                ad = max(1, td // k)

                t0 = time.perf_counter()
                compiled, info = compile_circuit(target, ad, method="polar",
                                                 max_iter=MAX_ITER)
                elapsed = time.perf_counter() - t0

                cost = info['compile_error']
                frob = np.sqrt(max(2**n * cost, 0))

                print(f"n={n:3d} k={k} s={s} J={J:.2f} g={g:.2f} h={h:.2f} "
                      f"steps={target_steps:3d}→{ansatz_steps:3d} "
                      f"cost={cost:.2e} {elapsed:.1f}s", flush=True)

                writer.writerow([n, k, s, f"{J:.3f}", f"{g:.3f}", f"{h:.3f}",
                                 target_steps, td, ansatz_steps, ad,
                                 cost, frob, f"{elapsed:.1f}"])
                f.flush()

print(f"\nDone. Results in {OUTPUT}")
