"""Build the target unitary directly from gate tensors (no Qiskit).

The MPO convention has site 0 = most significant qubit.
Build V = gate_on_(0,1) ⊗ gate_on_(2,3) directly using np.kron.
This avoids any Qiskit qubit ordering issues.
"""

import sys
sys.path.insert(0, '_reference_rqcopt')

import numpy as np
import jax.numpy as jnp
from jax import config
config.update("jax_enable_x64", True)

from rqcopt_mpo.tn_brickwall_methods import compute_full_gradient
from tno_compiler.brickwall import random_haar_gates, partition_gates, layer_structure
from tno_compiler.mpo_ops import matrix_to_mpo
from qiskit.quantum_info import random_unitary

n, d = 4, 1  # 1 odd layer: gates on (0,1) and (2,3)

# Generate gates as 4x4 matrices
G0 = random_unitary(4, seed=0).data  # gate on qubits (0,1)
G1 = random_unitary(4, seed=1).data  # gate on qubits (2,3)

# Build V directly: V = G0 ⊗ G1 (big-endian, site 0 = MSB)
V = np.kron(G0, G1)
print(f"V shape: {V.shape}")
print(f"Tr(V) = {np.trace(V)}")
print(f"Tr(G0)*Tr(G1) = {np.trace(G0)*np.trace(G1)}")
print(f"Match: {np.allclose(np.trace(V), np.trace(G0)*np.trace(G1), atol=1e-10)}")

# Circuit: same structure, different gates
H0 = random_unitary(4, seed=100).data
H1 = random_unitary(4, seed=101).data
U = np.kron(H0, H1)

exact = np.trace(V @ U)  # Tr(V U) since rqcopt computes this
print(f"\nExact Tr(VU) = {exact}")

# MPO from V
V_mpo = [jnp.asarray(a) for a in matrix_to_mpo(V)]

# Circuit gates as (2,2,2,2) tensors
cg = [H0.reshape(2,2,2,2), H1.reshape(2,2,2,2)]
structure = layer_structure(n, d)
is_odd = [s[0] for s in structure]
gl = partition_gates(cg, n, d)
gl_jax = [jnp.asarray(g) for g in gl]

_, ov = compute_full_gradient(V_mpo, gl_jax, is_odd, 128, compute_overlap=True)
print(f"rqcopt overlap: {ov}")
print(f"Match Tr(VU): {np.allclose(ov, exact, atol=1e-6)}")

# Now also check Tr(V†U):
exact_vdu = np.trace(V.conj().T @ U)
print(f"\nExact Tr(V†U) = {exact_vdu}")

Vd_mpo = [jnp.asarray(a) for a in matrix_to_mpo(V.conj().T)]
_, ov2 = compute_full_gradient(Vd_mpo, gl_jax, is_odd, 128, compute_overlap=True)
print(f"rqcopt with V† MPO: {ov2}")
print(f"Match Tr(V†U): {np.allclose(ov2, exact_vdu, atol=1e-6)}")
