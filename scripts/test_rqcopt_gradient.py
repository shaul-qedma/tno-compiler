"""Run rqcopt's compute_full_gradient directly and compare with ours."""

import sys
sys.path.insert(0, '_reference_rqcopt')

import numpy as np
import jax.numpy as jnp
from jax import config
config.update("jax_enable_x64", True)
from qiskit.quantum_info import random_unitary

from rqcopt_mpo.tn_helpers import get_mpo_from_matrix
from rqcopt_mpo.tn_brickwall_methods import compute_full_gradient

from tno_compiler.brickwall import random_haar_gates, gates_to_unitary, target_mpo, partition_gates, layer_structure
from tno_compiler.gradient import compute_cost_and_grad

n, d = 4, 2
seed = 0

target_gates = random_haar_gates(n, d, seed=seed)
circuit_gates = random_haar_gates(n, d, seed=seed + 5000)

# Exact
V = gates_to_unitary(target_gates, n, d)
U = gates_to_unitary(circuit_gates, n, d)
exact = np.trace(V.conj().T @ U)
exact_cost = 2.0 - 2.0 * exact.real / (2 ** n)
print(f"Exact Tr(V†U) = {exact}")
print(f"Exact cost = {exact_cost}")

# Our code
target_arrays = target_mpo(target_gates, n, d)
our_cost, our_grad = compute_cost_and_grad(target_arrays, circuit_gates, n, d)
print(f"\nOur cost = {our_cost}")

# rqcopt code -- need to set up gates_per_layer and layer_is_odd
structure = layer_structure(n, d)
gate_layers_ours = partition_gates(circuit_gates, n, d)
is_odd_ours = [s[0] for s in structure]

# Convert to jax
U_mpo_jax = [jnp.asarray(a) for a in target_arrays]
gates_per_layer_jax = [jnp.asarray(gl) for gl in gate_layers_ours]

rqcopt_grad, rqcopt_overlap = compute_full_gradient(
    U_mpo_jax, gates_per_layer_jax, is_odd_ours, 128, compute_overlap=True)
rqcopt_cost = 2.0 - 2.0 * float(rqcopt_overlap.real) / (2 ** n)
print(f"rqcopt cost = {rqcopt_cost}")
print(f"rqcopt overlap = {rqcopt_overlap}")
print(f"\nOur cost matches rqcopt: {np.allclose(our_cost, rqcopt_cost, atol=1e-6)}")
print(f"rqcopt cost matches exact: {np.allclose(rqcopt_cost, exact_cost, atol=1e-6)}")
