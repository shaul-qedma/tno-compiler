"""Test compute_cost_and_grad for the simplest case: 1 layer."""

import numpy as np
from tno_compiler.brickwall import random_haar_gates, gates_to_unitary, target_mpo
from tno_compiler.gradient import compute_cost_and_grad

for n in [2, 4, 6]:
    d = 1
    tg = random_haar_gates(n, d, seed=0)
    cg = random_haar_gates(n, d, seed=100)

    ta = target_mpo(tg, n, d)
    cost, _ = compute_cost_and_grad(ta, cg, n, d)

    V = gates_to_unitary(tg, n, d)
    U = gates_to_unitary(cg, n, d)
    exact_cost = 2 - 2 * np.trace(V.conj().T @ U).real / (2**n)
    print(f"n={n}, d={d}: cost={cost:.10f}, exact={exact_cost:.10f}, err={abs(cost-exact_cost):.2e}")
