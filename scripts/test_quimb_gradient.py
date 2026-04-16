"""Test whether the gradient computation works with quimb MPOs."""

import numpy as np
from tno_compiler.brickwall import random_haar_gates, target_mpo, circuit_to_mpo
from tno_compiler.brickwall import mpo_to_arrays
from tno_compiler.gradient import compute_cost_and_grad

n, d = 4, 2
tg = random_haar_gates(n, d, seed=0)
cg = random_haar_gates(n, d, seed=5000)

# Build target MPO via quimb
tmpo = target_mpo(tg, n, d)
print(f"Target type: {type(tmpo)}")
print(f"Target bonds: {tmpo.bond_sizes()}")

# The gradient code expects a list of numpy arrays (bl, k, b, br).
# quimb_mpo_to_arrays was written for this but may need updating.
# Let's see if it still exists and works.
try:
    arrays = mpo_to_arrays(tmpo)
    print(f"Converted to arrays: {[a.shape for a in arrays]}")

    cost, grad = compute_cost_and_grad(arrays, cg, n, d)
    print(f"Cost: {cost}")
    print(f"Grad shape: {grad.shape}")
except Exception as e:
    print(f"Failed: {e}")

# Also verify against exact
V = np.array(circuit_to_mpo(tg, n, d).to_dense())
U = np.array(circuit_to_mpo(cg, n, d).to_dense())
exact_cost = 2.0 - 2.0 * np.trace(V.conj().T @ U).real / (2**n)
print(f"Exact cost: {exact_cost}")
print(f"Match: {abs(cost - exact_cost) < 1e-4}")
