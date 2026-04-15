"""Check if the trace factors correctly for a product-state target."""

import numpy as np
from tno_compiler.brickwall import random_haar_gates, gates_to_unitary, target_mpo
from tno_compiler.mpo_ops import (
    quimb_mpo_to_arrays, merge_gate_with_mpo_pair, split_merged_tensor,
    trace_mpo,
)

n, d = 4, 1  # 1 odd layer on 4 qubits
tg = random_haar_gates(n, d, seed=0)
cg = random_haar_gates(n, d, seed=100)

# The target V has gates on (0,1) and (2,3) -- product across the (1,2) cut.
# So Tr(V†U) = Tr_01(V0†·U0) × Tr_23(V1†·U1) where V0,U0 act on (0,1)
# and V1,U1 act on (2,3).

V = gates_to_unitary(tg, n, d)
U = gates_to_unitary(cg, n, d)

# V = V0 ⊗ V1, U = U0 ⊗ U1
V0 = tg[0].reshape(4, 4)
V1 = tg[1].reshape(4, 4)
U0 = cg[0].reshape(4, 4)
U1 = cg[1].reshape(4, 4)

tr01 = np.trace(V0.conj().T @ U0)
tr23 = np.trace(V1.conj().T @ U1)
product = tr01 * tr23
exact = np.trace(V.conj().T @ U)

print(f"Tr_01(V0†U0) = {tr01}")
print(f"Tr_23(V1†U1) = {tr23}")
print(f"Product = {product}")
print(f"Exact Tr(V†U) = {exact}")
print(f"Match: {np.allclose(product, exact, atol=1e-10)}")

# Now check: what does the merged MPO trace give for each pair?
T = quimb_mpo_to_arrays(target_mpo(tg, n, d))

# Merge gate0 with T[0],T[1]
m1 = merge_gate_with_mpo_pair(cg[0], T[0], T[1], gate_is_left=True)
T0, T1 = split_merged_tensor(m1, max_bond=128)
ov01 = trace_mpo([T0, T1])
print(f"\nMPO trace pair(0,1): {ov01}")
print(f"Should be Tr(V0†U0) = {tr01}")
print(f"Match: {np.allclose(ov01, tr01, atol=1e-8)}")

# Merge gate1 with T[2],T[3]
m2 = merge_gate_with_mpo_pair(cg[1], T[2], T[3], gate_is_left=True)
T2, T3 = split_merged_tensor(m2, max_bond=128)
ov23 = trace_mpo([T2, T3])
print(f"MPO trace pair(2,3): {ov23}")
print(f"Should be Tr(V1†U1) = {tr23}")
print(f"Match: {np.allclose(ov23, tr23, atol=1e-8)}")

print(f"\nProduct of MPO traces: {ov01 * ov23}")
print(f"Full 4-site trace: {trace_mpo([T0, T1, T2, T3])}")
