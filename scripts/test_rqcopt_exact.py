"""Test rqcopt functions directly from the reference code."""

import sys
sys.path.insert(0, '_reference_rqcopt')

import jax.numpy as jnp
from jax import config
config.update("jax_enable_x64", True)
import numpy as np
from qiskit.quantum_info import random_unitary

from rqcopt_mpo.tn_helpers import (
    get_mpo_from_matrix, get_matrix_from_mpo, fully_contract_mpo,
    merge_two_mpos_and_gate, split_tensor_into_half_canonical_mpo_pair,
    left_to_right_QR_sweep,
)

# 4 qubits, 1 odd layer
n = 4

# Target V and circuit gate
V0_mat = random_unitary(4, seed=0).data
V1_mat = random_unitary(4, seed=1).data
G0_mat = random_unitary(4, seed=100).data
G1_mat = random_unitary(4, seed=101).data

V = np.kron(V0_mat, V1_mat)
U = np.kron(G0_mat, G1_mat)
exact = np.trace(V.conj().T @ U)
print(f"Exact Tr(V†U) = {exact}")

# Build V† as MPO using rqcopt's own function
Vd = V.conj().T
Vd_mpo = get_mpo_from_matrix(jnp.asarray(Vd))
print(f"V† MPO shapes: {[t.shape for t in Vd_mpo]}")

# Verify roundtrip
Vd_recon = np.array(get_matrix_from_mpo(Vd_mpo))
print(f"V† roundtrip: {np.allclose(Vd_recon, Vd, atol=1e-10)}")

# Merge gate0 with Vd[0], Vd[1]
G0_tn = jnp.asarray(G0_mat.reshape(2, 2, 2, 2))
merged = merge_two_mpos_and_gate(G0_tn, Vd_mpo[0], Vd_mpo[1], gate_is_left=True)
print(f"Merged shape: {merged.shape}")
T0, T1 = split_tensor_into_half_canonical_mpo_pair(merged, canonical_mode='left')
print(f"T0={T0.shape}, T1={T1.shape}")

# Merge gate1 with Vd[2], Vd[3]
G1_tn = jnp.asarray(G1_mat.reshape(2, 2, 2, 2))
merged1 = merge_two_mpos_and_gate(G1_tn, Vd_mpo[2], Vd_mpo[3], gate_is_left=True)
T2, T3 = split_tensor_into_half_canonical_mpo_pair(merged1, canonical_mode='left')

# Full trace
result_mpo = [T0, T1, T2, T3]
ov = np.array(fully_contract_mpo(list(result_mpo)))
print(f"rqcopt trace: {ov}")
print(f"Match: {np.allclose(ov, exact, atol=1e-6)}")
