"""Determine what rqcopt's compute_full_gradient actually computes."""

import sys
sys.path.insert(0, '_reference_rqcopt')

import numpy as np
import jax.numpy as jnp
from jax import config
config.update("jax_enable_x64", True)
from qiskit.quantum_info import random_unitary

from rqcopt_mpo.tn_helpers import get_mpo_from_matrix
from rqcopt_mpo.tn_brickwall_methods import compute_full_gradient

from tno_compiler.brickwall import random_haar_gates, gates_to_unitary, partition_gates, layer_structure
from tno_compiler.mpo_ops import matrix_to_mpo

n, d = 4, 2

target_gates = random_haar_gates(n, d, seed=0)
circuit_gates = random_haar_gates(n, d, seed=5000)

V = gates_to_unitary(target_gates, n, d)
U = gates_to_unitary(circuit_gates, n, d)

structure = layer_structure(n, d)
gate_layers = partition_gates(circuit_gates, n, d)
is_odd = [s[0] for s in structure]

# Test with V as MPO (not V†)
V_mpo = [jnp.asarray(a) for a in matrix_to_mpo(V)]
gl_jax = [jnp.asarray(gl) for gl in gate_layers]
_, ov_V = compute_full_gradient(V_mpo, gl_jax, is_odd, 128, compute_overlap=True)

# Test with V† as MPO
Vd_mpo = [jnp.asarray(a) for a in matrix_to_mpo(V.conj().T)]
_, ov_Vd = compute_full_gradient(Vd_mpo, gl_jax, is_odd, 128, compute_overlap=True)

print("Various overlaps:")
print(f"Tr(V†U) = {np.trace(V.conj().T @ U)}")
print(f"Tr(V U†) = {np.trace(V @ U.conj().T)}")
print(f"Tr(V U) = {np.trace(V @ U)}")
print(f"Tr(V^T U) = {np.trace(V.T @ U)}")
print()
print(f"rqcopt with V as MPO:  {float(ov_V.real)} + {float(ov_V.imag)}j")
print(f"rqcopt with V† as MPO: {float(ov_Vd.real)} + {float(ov_Vd.imag)}j")

# Check which one matches
for label, val in [
    ("Tr(V†U)", np.trace(V.conj().T @ U)),
    ("Tr(V U†)", np.trace(V @ U.conj().T)),
    ("Tr(V U)", np.trace(V @ U)),
    ("Tr(V^T U)", np.trace(V.T @ U)),
]:
    if np.allclose(ov_V, val, atol=1e-4):
        print(f"  V-MPO matches {label}")
    if np.allclose(ov_Vd, val, atol=1e-4):
        print(f"  V†-MPO matches {label}")
