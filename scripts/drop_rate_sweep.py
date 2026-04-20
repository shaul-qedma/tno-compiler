"""Drop-rate sweep: characterize how dropout affects compile outcomes
across representative configs.

For each (config, drop_rate, init_seed) we run a single compile_circuit
and record final cost + timing. Goals:
1. Identify optimal drop_rate ranges per regime (easy / over-parameterized /
   under-parameterized / critical).
2. Measure whether dropout reduces init-seed variance.

Output: docs/data/drop_rate_sweep.csv.
"""

import csv
import time
from pathlib import Path

import numpy as np

from tno_compiler.brickwall import random_brickwall
from tno_compiler.compiler import compile_circuit
from tno_compiler.pipeline import _qc_to_gate_tensors
from tno_compiler.tfi import tfi_trotter_circuit


# Representative configs (n, g, steps, ansatz_d, label, comment)
CONFIGS = [
    # label, n, g, steps, ansatz_d, regime_note
    ("easy",             6, 0.3,  2, 2,  "well within expressivity"),
    ("moderate",         8, 0.3,  8, 4,  "rule-predicted depth"),
    ("overparam_g1",    10, 1.0,  8, 10, "non-monotonic baseline (d=10 was 15x worse than d=8 without dropout)"),
    ("overparam_g15",    4, 1.5, 16, 8,  "non-monotonic n=4 case"),
    ("critical",         8, 1.0, 12, 6,  "critical coupling, mid compression"),
    ("underparam",       8, 0.5,  8, 2,  "expressivity-limited"),
]

DROP_RATES = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.7]
INIT_SEEDS = [1, 2, 3, 4, 5]   # 5 init variations per (config, drop_rate)
MAX_ITER = 100
J, H, DT = -1.0, 0.0, 0.1

out = Path(__file__).resolve().parent.parent / "docs" / "data" / "drop_rate_sweep.csv"
out.parent.mkdir(parents=True, exist_ok=True)

fieldnames = [
    "config_label", "n", "g", "steps", "ansatz_d",
    "drop_rate", "init_seed", "seed",
    "target_depth", "final_cost", "elapsed_s",
    "last10_std",   # std of last 10 cost-history values (basin stability)
]

total = len(CONFIGS) * len(DROP_RATES) * len(INIT_SEEDS)
print(f"{total} runs total", flush=True)

with open(out, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    f.flush()

    idx = 0
    for label, n, g, steps, ansatz_d, _note in CONFIGS:
        target = tfi_trotter_circuit(n, J=J, g=g, h=H, dt=DT, steps=steps)
        tgt_d = target.depth()
        for init_seed in INIT_SEEDS:
            init_qc = random_brickwall(n, ansatz_d, seed=init_seed)
            init_tensors = _qc_to_gate_tensors(init_qc)
            for drop_rate in DROP_RATES:
                idx += 1
                # Derive one seed from (init_seed, drop_rate-index) to decorrelate
                master = init_seed * 1000 + int(drop_rate * 1000)
                t0 = time.perf_counter()
                try:
                    _, info = compile_circuit(
                        target, ansatz_d,
                        max_iter=MAX_ITER, tol=1e-10, method="polar",
                        init_gates=init_tensors,
                        drop_rate=drop_rate, seed=master,
                    )
                    final = info['cost_history'][-1]
                    last10_std = float(np.std(info['cost_history'][-10:]))
                except Exception as e:
                    print(f"  ERROR {label} drop={drop_rate} init={init_seed}: {e}",
                          flush=True)
                    continue
                elapsed = time.perf_counter() - t0
                writer.writerow({
                    "config_label": label,
                    "n": n, "g": g, "steps": steps, "ansatz_d": ansatz_d,
                    "drop_rate": drop_rate, "init_seed": init_seed, "seed": master,
                    "target_depth": tgt_d, "final_cost": float(final),
                    "elapsed_s": elapsed, "last10_std": last10_std,
                })
                f.flush()
                print(f"  [{idx}/{total}] {label:15s} drop={drop_rate:.2f} "
                      f"init={init_seed}: final_cost={final:.3e} "
                      f"last10_std={last10_std:.1e} ({elapsed:.1f}s)",
                      flush=True)

print(f"\nDone. Wrote {out}")
