"""Compare our split_merged_tensor with rqcopt's split_tensor_into_half_canonical_mpo_pair."""

import sys
sys.path.insert(0, '_reference_rqcopt')

import numpy as np
import jax.numpy as jnp
from jax import config
config.update("jax_enable_x64", True)
from qiskit.quantum_info import random_unitary

from rqcopt_mpo.tn_helpers import (
    get_mpo_from_matrix, merge_two_mpos_and_gate,
    split_tensor_into_half_canonical_mpo_pair, fully_contract_mpo,
)
from tno_compiler.mpo_ops import (
    matrix_to_mpo, merge_gate_with_mpo_pair, split_merged_tensor, trace_mpo,
)

# Same setup as the working rqcopt test
V0 = random_unitary(4, seed=0).data
V1 = random_unitary(4, seed=1).data
G0 = random_unitary(4, seed=100).data
G1 = random_unitary(4, seed=101).data

V = np.kron(V0, V1)
Vd = V.conj().T

mpo = matrix_to_mpo(Vd)  # Same as rqcopt's
G0_tn = G0.reshape(2, 2, 2, 2)

# Merge (should be identical)
merged_rqcopt = np.array(merge_two_mpos_and_gate(
    jnp.asarray(G0_tn), jnp.asarray(mpo[0]), jnp.asarray(mpo[1]), gate_is_left=True))
merged_ours = merge_gate_with_mpo_pair(G0_tn, mpo[0], mpo[1], gate_is_left=True)
print(f"Merged match: {np.allclose(merged_rqcopt, merged_ours, atol=1e-10)}")
print(f"Merged shapes: rqcopt={merged_rqcopt.shape}, ours={merged_ours.shape}")

# Split
T0_rq, T1_rq = split_tensor_into_half_canonical_mpo_pair(
    jnp.asarray(merged_ours), canonical_mode='left')
T0_rq, T1_rq = np.array(T0_rq), np.array(T1_rq)

T0_ours, T1_ours = split_merged_tensor(merged_ours, canonical='left', max_bond=128)

print(f"\nSplit shapes: rqcopt=({T0_rq.shape}, {T1_rq.shape}), ours=({T0_ours.shape}, {T1_ours.shape})")
print(f"T0 match: {np.allclose(T0_rq, T0_ours, atol=1e-10)}")
print(f"T1 match: {np.allclose(T1_rq, T1_ours, atol=1e-10)}")

# Check traces
G1_tn = G1.reshape(2, 2, 2, 2)
m1 = merge_gate_with_mpo_pair(G1_tn, mpo[2], mpo[3], gate_is_left=True)
T2_ours, T3_ours = split_merged_tensor(m1, canonical='left', max_bond=128)

ov_ours = trace_mpo([T0_ours, T1_ours, T2_ours, T3_ours])
print(f"\nOurs trace: {ov_ours}")

T2_rq, T3_rq = split_tensor_into_half_canonical_mpo_pair(
    jnp.asarray(m1), canonical_mode='left')
T2_rq, T3_rq = np.array(T2_rq), np.array(T3_rq)

ov_rq = float(fully_contract_mpo([jnp.asarray(T0_rq), jnp.asarray(T1_rq),
                                    jnp.asarray(T2_rq), jnp.asarray(T3_rq)]))
print(f"rqcopt trace: {ov_rq}")

exact = np.trace(Vd @ np.kron(G0, G1))
print(f"Exact: {exact}")
