"""Benchmark: how well does the ensemble pipeline compress circuits?

For a target at depth D, compile ensembles at depth D' ≤ D and report
the diamond bound and comparison vs single circuit.
"""

import numpy as np
from tno_compiler.brickwall import random_brickwall
from tno_compiler.pipeline import compile_ensemble

n = 4
seed = 42

for target_depth in [2, 4]:
    target = random_brickwall(n, target_depth, seed=seed)
    print(f"\n{'='*60}")
    print(f"Target: n={n}, depth={target_depth}")
    print(f"{'='*60}")
    print(f"{'depth':>5} | {'diamond':>10} | {'δ_ens':>10} | {'R':>10} | "
          f"{'best_single':>12} | {'active':>6}")
    print("-" * 65)

    for ansatz_depth in range(1, target_depth + 1):
        result = compile_ensemble(
            target, ansatz_depth,
            n_circuits=5, max_iter=200, lr=5e-3, seed=seed)

        best_single = min(result['individual_frobs'])
        active = sum(1 for w in result['weights'] if w > 1e-3)

        print(f"{ansatz_depth:5d} | {result['diamond_bound']:10.4f} | "
              f"{result['delta_ens']:10.4f} | {result['R']:10.4f} | "
              f"{best_single:12.4f} | {active:6d}/{len(result['circuits'])}")
