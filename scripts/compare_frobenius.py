"""Compare our DP-based Frobenius compressor vs quimb's rsum2 cutoff_mode.

For various tol values, compress the same circuit both ways and compare:
- Actual Frobenius error (dense, testing only)
- Bond dimensions achieved
"""

import numpy as np
import quimb.tensor as qtn
from tno_compiler.brickwall import random_haar_gates, circuit_to_tn, circuit_to_mpo

n, d, seed = 8, 4, 42
gates = random_haar_gates(n, d, seed=seed)

# Build exact MPO for reference
exact, _ = circuit_to_mpo(gates, n, d, tol=0.0)
V_exact = np.array(exact.to_dense())

print(f"Circuit: n={n}, d={d}, seed={seed}")
print(f"Exact MPO bonds: {exact.bond_sizes()}")
print(f"{'tol':>10} | {'method':>10} | {'bonds':>20} | {'F-error':>10} | {'max_bond':>8}")
print("-" * 75)

for tol in [1e-1, 1e-2, 1e-4, 1e-6]:
    # Our DP method
    mpo_ours, err_ours = circuit_to_mpo(gates, n, d, tol=tol, norm="frobenius")
    V_ours = np.array(mpo_ours.to_dense())
    actual_ours = np.linalg.norm(V_exact - V_ours, ord='fro')
    bonds_ours = mpo_ours.bond_sizes()

    # quimb rsum2 with same tolerance (cutoff = tol^2 since rsum2 bounds sum of squares)
    tn = circuit_to_tn(gates, n, d)
    mpo_quimb = qtn.tensor_network_1d_compress(
        tn, cutoff=tol**2, cutoff_mode="rsum2", method="dm")
    mpo_quimb.view_as_(qtn.MatrixProductOperator, cyclic=False, L=n)
    V_quimb = np.array(mpo_quimb.to_dense())
    actual_quimb = np.linalg.norm(V_exact - V_quimb, ord='fro')
    bonds_quimb = mpo_quimb.bond_sizes()

    print(f"{tol:10.0e} | {'ours':>10} | {str(bonds_ours):>20} | {actual_ours:10.2e} | {max(bonds_ours):8}")
    print(f"{'':>10} | {'quimb':>10} | {str(bonds_quimb):>20} | {actual_quimb:10.2e} | {max(bonds_quimb):8}")
