"""Verify that reversing qubit order fixes the overlap."""

import sys
sys.path.insert(0, '_reference_rqcopt')

import numpy as np
import jax.numpy as jnp
from jax import config
config.update("jax_enable_x64", True)

from rqcopt_mpo.tn_brickwall_methods import compute_full_gradient
from tno_compiler.brickwall import random_haar_gates, gates_to_unitary, partition_gates, layer_structure
from tno_compiler.mpo_ops import matrix_to_mpo

n, d = 4, 1
tg = random_haar_gates(n, d, seed=0)
cg = random_haar_gates(n, d, seed=100)

V = gates_to_unitary(tg, n, d)
U = gates_to_unitary(cg, n, d)
exact_VdU = np.trace(V.conj().T @ U)

# Permutation matrix to reverse qubit order
# For n qubits, this swaps qubit i with qubit (n-1-i)
d_hilbert = 2**n
perm = np.zeros((d_hilbert, d_hilbert))
for i in range(d_hilbert):
    # Reverse the bit string
    bits = format(i, f'0{n}b')
    j = int(bits[::-1], 2)
    perm[j, i] = 1

V_rev = perm @ V @ perm.T  # Reverse qubit order
U_rev = perm @ U @ perm.T

# Tr(V†U) is invariant under the same unitary on both
print(f"Tr(V†U) original: {exact_VdU}")
print(f"Tr(V_rev†U_rev):  {np.trace(V_rev.conj().T @ U_rev)}")
print(f"Match: {np.allclose(exact_VdU, np.trace(V_rev.conj().T @ U_rev), atol=1e-10)}")

# Now build MPO from V_rev (reversed qubit order)
V_rev_mpo = [jnp.asarray(a) for a in matrix_to_mpo(V_rev)]

structure = layer_structure(n, d)
is_odd = [s[0] for s in structure]

# Circuit gates also need reversed qubit pairs
# Original: gate 0 on (0,1), gate 1 on (2,3)
# Reversed: gate 0 should be on MPO sites (2,3), gate 1 on (0,1)
# So we swap the gate order within each layer
gate_layers_rev = []
for layer_gates, (odd, pairs) in zip(partition_gates(cg, n, d), structure):
    gate_layers_rev.append(list(reversed(layer_gates)))
gl_rev_jax = [jnp.asarray(gl) for gl in gate_layers_rev]

_, ov = compute_full_gradient(V_rev_mpo, gl_rev_jax, is_odd, 128, compute_overlap=True)
cost = 2 - 2 * float(ov.real) / (2**n)
exact_cost = 2 - 2 * exact_VdU.real / (2**n)
print(f"\nReversed: overlap={ov}")
print(f"cost={cost:.10f}, exact={exact_cost:.10f}, err={abs(cost-exact_cost):.2e}")

# Alternative simpler fix: just reverse the Qiskit unitary matrix
# by permuting rows and columns
V_perm = perm @ V @ perm.T
V_perm_mpo = [jnp.asarray(a) for a in matrix_to_mpo(V_perm)]

# And use the original gate order (since gates are now aligned with MPO)
gl_jax = [jnp.asarray(gl) for gl in partition_gates(cg, n, d)]
_, ov2 = compute_full_gradient(V_perm_mpo, gl_jax, is_odd, 128, compute_overlap=True)
cost2 = 2 - 2 * float(ov2.real) / (2**n)
print(f"\nPermuted MPO, original gates: overlap={ov2}")
print(f"cost={cost2:.10f}, exact={exact_cost:.10f}, err={abs(cost2-exact_cost):.2e}")
