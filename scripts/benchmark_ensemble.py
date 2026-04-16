"""Benchmark: how well does the ensemble pipeline compress circuits?

For a target at depth D, compile ensembles at depth D' < D and report
the diamond bound, ensemble Frobenius error, and comparison vs single circuit.
"""

import numpy as np
from tno_compiler.brickwall import random_haar_gates
from tno_compiler.pipeline import compile_ensemble

n = 6
seed = 42

for target_depth in [4, 6, 8]:
    tg = random_haar_gates(n, target_depth, seed=seed)
    print(f"\n{'='*60}")
    print(f"Target: n={n}, depth={target_depth}")
    print(f"{'='*60}")

    for compile_depth in [1, 2, 3, 4]:
        if compile_depth > target_depth:
            continue
        # Note: compile_ensemble compiles at compile_depth, not target_depth
        # The target gates define V, the ansatz depth is compile_depth
        result = compile_ensemble(
            tg, n, compile_depth,
            n_circuits=5, max_iter=200, lr=5e-3, seed=seed)

        best_single = min(result['individual_frobs'])
        active = sum(1 for w in result['weights'] if w > 1e-3)

        print(f"  depth={compile_depth}: "
              f"diamond={result['diamond_bound']:.4f}, "
              f"δ_ens={result['delta_ens']:.4f}, "
              f"R={result['R']:.4f}, "
              f"best_single_frob={best_single:.4f}, "
              f"active_circuits={active}/{len(result['circuits'])}")
