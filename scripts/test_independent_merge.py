"""Test: merge gates independently (no R-carry) on a 4-qubit target."""

import numpy as np
from tno_compiler.brickwall import random_haar_gates, gates_to_unitary, target_mpo
from tno_compiler.mpo_ops import merge_gate_with_mpo_pair, split_merged_tensor, trace_mpo

tg = random_haar_gates(4, 1, seed=0)
cg = random_haar_gates(4, 1, seed=100)

T = target_mpo(tg, 4, 1)
print(f"Target shapes: {[t.shape for t in T]}")

# Merge gate0 with (T[0], T[1]), gate1 with (T[2], T[3]) -- NO carry
m0 = merge_gate_with_mpo_pair(cg[0], T[0], T[1], gate_is_left=True)
T0, T1 = split_merged_tensor(m0, max_bond=128)

m1 = merge_gate_with_mpo_pair(cg[1], T[2], T[3], gate_is_left=True)
T2, T3 = split_merged_tensor(m1, max_bond=128)

print(f"Shapes: T0={T0.shape}, T1={T1.shape}, T2={T2.shape}, T3={T3.shape}")
print(f"Bond T1-right={T1.shape[-1]}, T2-left={T2.shape[0]}")

ov = trace_mpo([T0, T1, T2, T3])
exact = np.trace(gates_to_unitary(tg, 4, 1).conj().T @ gates_to_unitary(cg, 4, 1))
print(f"Overlap: {ov}")
print(f"Exact: {exact}")
print(f"Match: {np.allclose(ov, exact, atol=1e-6)}")

# The bond between T1 and T2 is T1.shape[-1]=16 and T2.shape[0]=16
# so they can communicate. But the result is still wrong.
# Let me check: contract the full 4-site network directly without splitting
# merged0: (1, ..., 16), merged1: (16, ..., 1)
# For the closed TN: contract all physical pairs and sum over bonds
# Actually the merged tensors have 6 indices. Let me contract everything directly.

# Full contraction: sum over all physical and bond indices
# Merge T[0],T[1] with G0 -> 6-index tensor
# Merge T[2],T[3] with G1 -> 6-index tensor
# Contract over the bond between site 1 and site 2
# Then trace all physical pairs

# m0: (bl=1, k1_out, b1_out, k2_out, b2_out, bond_12)
# m1: (bond_12, k3_out, b3_out, k4_out, b4_out, br=1)
# Contract bond_12, then trace: k1=b1, k2=b2, k3=b3, k4=b4
contracted = np.einsum('iabcdf,fghijk->iabcdghij', m0, m1)
# Now trace all physical pairs
tr = np.einsum('iaabbccdi->', contracted)
print(f"\nDirect contraction trace: {tr}")
print(f"Match: {np.allclose(tr, exact, atol=1e-6)}")
