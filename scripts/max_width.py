"""Find the max width where to_dense() + superoperator norm is feasible.

The superoperator matrix is d² × d² = 4^n × 4^n.
The SVD for operator norm is O((4^n)³).

Usage: uv run python scripts/max_width.py
"""

import time
import numpy as np
from qiskit.quantum_info import Operator
from tno_compiler.tfi import tfi_trotter_circuit
from tno_compiler.pipeline import compile_ensemble

J, g, h, dt, steps = 1.0, 0.75, 0.6, 0.1, 1

for n in [4, 6, 8, 10, 12]:
    d = 2 ** n
    print(f"n={n}: d={d}, superop matrix={d**2}×{d**2} = {d**2}² entries, "
          f"memory ~{d**4 * 16 / 1e9:.1f} GB")

    if d ** 4 * 16 > 4e9:  # >4GB
        print("  SKIP (too large)")
        continue

    target = tfi_trotter_circuit(n, J, g, h, dt, steps)

    t0 = time.perf_counter()
    result = compile_ensemble(target, target.depth(), n_circuits=2,
                              max_iter=50, lr=2e-2, seed=42)
    t_compile = time.perf_counter() - t0

    t0 = time.perf_counter()
    V = Operator(target).data
    S_tgt = np.kron(V.conj(), V)

    S_ens = np.zeros((d**2, d**2), dtype=complex)
    for qc, p in zip(result['circuits'], result['weights']):
        if p < 1e-15:
            continue
        U = Operator(qc).data
        S_ens += p * np.kron(U.conj(), U)

    superop_norm = np.linalg.norm(S_ens - S_tgt, ord=2)
    t_norm = time.perf_counter() - t0

    print(f"  compile: {t_compile:.1f}s, norm: {t_norm:.1f}s, "
          f"superop_norm={superop_norm:.2e}, diamond_bound={result['diamond_bound']:.2e}")
