"""Test whether the overlap error is due to bond truncation."""

import numpy as np
from tno_compiler.brickwall import random_haar_gates, gates_to_unitary, target_mpo
from tno_compiler.gradient import compute_cost_and_grad

n, d = 4, 2
tg = random_haar_gates(n, d, seed=0)
cg = random_haar_gates(n, d, seed=5000)
target_arrays = target_mpo(tg, n, d)

V = gates_to_unitary(tg, n, d)
U = gates_to_unitary(cg, n, d)
exact_cost = 2.0 - 2.0 * np.trace(V.conj().T @ U).real / (2 ** n)

for mb in [16, 32, 64, 128, 256, 512]:
    cost, _ = compute_cost_and_grad(target_arrays, cg, n, d, max_bond=mb)
    print(f"max_bond={mb:4d}: cost={cost:.10f}, error={abs(cost-exact_cost):.2e}")
