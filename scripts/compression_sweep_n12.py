"""Fill in n=12 configs missing from compression_grid.csv + compression_grid2.csv.

Appends to compression_grid2.csv. Skips configs already present.
"""

import csv
import time
from pathlib import Path

import numpy as np

from tno_compiler.pipeline import compile_ensemble
from tno_compiler.tfi import tfi_trotter_circuit
from tno_compiler.verify import sampled_max_trace_distance


N_VALUES = [12]
G_VALUES = [0.3, 1.0, 1.5]
STEPS_VALUES = [4, 6, 8, 12, 16]
ANSATZ_DEPTH_VALUES = [2, 3, 4, 5, 6, 8, 10, 12]

J, H, DT = -1.0, 0.0, 0.1
N_CIRCUITS = 3
MAX_ITER = 100
N_SAMPLES = 10
COMPILE_SEED = 42
VERIFY_SEED = 0
MAX_ANSATZ_DEPTH = 16

base = Path(__file__).resolve().parent.parent
out_path = base / "docs" / "data" / "compression_grid2.csv"
prior_paths = [
    base / "docs" / "data" / "compression_grid.csv",
    base / "docs" / "data" / "compression_grid2.csv",
]

fieldnames = [
    "n", "g", "steps", "dt",
    "target_depth", "ansatz_depth",
    "max_td", "mean_td",
    "compile_err_min", "compile_err_max", "compile_err_spread",
    "seeds_agree", "elapsed_s",
]

seen = set()
for p in prior_paths:
    if p.exists():
        with open(p) as f:
            for r in csv.DictReader(f):
                seen.add((int(r['n']), float(r['g']), int(r['steps']),
                          int(r['ansatz_depth'])))
print(f"Prior configs to skip: {len(seen)}", flush=True)

configs = []
for n in N_VALUES:
    for g in G_VALUES:
        for steps in STEPS_VALUES:
            target = tfi_trotter_circuit(n, J, g, H, DT, steps)
            tgt_d = target.depth()
            for ad in ANSATZ_DEPTH_VALUES:
                if ad > tgt_d or ad > MAX_ANSATZ_DEPTH:
                    continue
                key = (n, g, steps, ad)
                if key in seen:
                    continue
                seen.add(key)
                configs.append((n, g, steps, tgt_d, ad, target))

print(f"New configs: {len(configs)}", flush=True)

with open(out_path, "a", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    f.flush()
    for idx, (n, g, steps, tgt_d, ansatz_d, target) in enumerate(configs):
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
            print(f"  ERROR n={n} g={g} steps={steps} ansatz_d={ansatz_d}: {e}",
                  flush=True)
            continue
        elapsed = time.perf_counter() - t0
        writer.writerow({
            "n": n, "g": g, "steps": steps, "dt": DT,
            "target_depth": tgt_d, "ansatz_depth": ansatz_d,
            "max_td": max_td, "mean_td": mean_td,
            "compile_err_min": err_min, "compile_err_max": err_max,
            "compile_err_spread": err_spread,
            "seeds_agree": int(seeds_agree),
            "elapsed_s": elapsed,
        })
        f.flush()
        print(f"  [{idx+1}/{len(configs)}] n={n} g={g} steps={steps} "
              f"tgt_d={tgt_d} ansatz_d={ansatz_d} -> max_td={max_td:.3e} "
              f"({elapsed:.1f}s)", flush=True)

print(f"\nDone. Appended to {out_path}")
