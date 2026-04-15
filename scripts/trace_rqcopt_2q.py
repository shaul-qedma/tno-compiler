"""Trace rqcopt's exact contraction for a 2-qubit, 1-layer case.

Goal: determine what scalar compute_full_gradient actually computes.
"""

import sys
sys.path.insert(0, '_reference_rqcopt')

import numpy as np
import jax.numpy as jnp
from jax import config
config.update("jax_enable_x64", True)

from rqcopt_mpo.tn_helpers import get_mpo_from_matrix, get_id_mpo
from rqcopt_mpo.tn_brickwall_methods import compute_full_gradient

# 2 qubits, 1 odd layer = 1 gate
n = 2
V = np.array([[1, 0, 0, 0],
              [0, 0, 1, 0],
              [0, 1, 0, 0],
              [0, 0, 0, 1]], dtype=complex)  # SWAP gate
G = np.eye(4, dtype=complex)  # Identity circuit

V_mpo = get_mpo_from_matrix(jnp.asarray(V))
G_tn = jnp.asarray(G.reshape(2, 2, 2, 2))

# 1 layer, odd
gates_per_layer = [jnp.expand_dims(G_tn, 0)]  # shape (1, 2, 2, 2, 2)
layer_is_odd = [True]

grad, ov = compute_full_gradient(V_mpo, gates_per_layer, layer_is_odd, 128, compute_overlap=True)
print(f"V=SWAP, G=I: overlap={ov}")
print(f"Tr(SWAP) = {np.trace(V)}, Tr(SWAP @ I) = {np.trace(V @ np.eye(4))}")
print(f"Tr(SWAP^dag @ I) = {np.trace(V.conj().T @ np.eye(4))}")

# Now try V=I, G=SWAP
V2 = np.eye(4, dtype=complex)
G2 = V.copy()

V2_mpo = get_mpo_from_matrix(jnp.asarray(V2))
G2_tn = jnp.asarray(G2.reshape(2, 2, 2, 2))
gates2 = [jnp.expand_dims(G2_tn, 0)]

_, ov2 = compute_full_gradient(V2_mpo, gates2, layer_is_odd, 128, compute_overlap=True)
print(f"\nV=I, G=SWAP: overlap={ov2}")
print(f"Tr(I @ SWAP) = {np.trace(np.eye(4) @ V)}")

# General random case
from qiskit.quantum_info import random_unitary
V3 = random_unitary(4, seed=0).data
G3 = random_unitary(4, seed=100).data

V3_mpo = get_mpo_from_matrix(jnp.asarray(V3))
G3_tn = jnp.asarray(G3.reshape(2, 2, 2, 2))
gates3 = [jnp.expand_dims(G3_tn, 0)]

_, ov3 = compute_full_gradient(V3_mpo, gates3, layer_is_odd, 128, compute_overlap=True)
print(f"\nRandom V, G: overlap={ov3}")

# Check all possibilities
for label, val in [
    ("Tr(V†G)", np.trace(V3.conj().T @ G3)),
    ("Tr(V G†)", np.trace(V3 @ G3.conj().T)),
    ("Tr(V G)", np.trace(V3 @ G3)),
    ("Tr(V^T G)", np.trace(V3.T @ G3)),
    ("Tr(V^T G†)", np.trace(V3.T @ G3.conj().T)),
    ("sum V*G", np.sum(V3.conj() * G3)),
    ("sum V*G†", np.sum(V3.conj() * G3.conj().T)),
    ("sum V G*", np.sum(V3 * G3.conj())),
]:
    if np.allclose(ov3, val, atol=1e-8):
        print(f"  MATCHES: {label} = {val}")
