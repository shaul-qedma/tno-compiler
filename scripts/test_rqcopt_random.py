"""Test rqcopt compute_full_gradient with random Haar gates (not Trotter)."""

import sys
sys.path.insert(0, '_reference_rqcopt')

import numpy as np
import jax.numpy as jnp
from jax import config
config.update("jax_enable_x64", True)

from rqcopt_mpo.tn_helpers import get_mpo_from_matrix, left_to_right_QR_sweep
from rqcopt_mpo.tn_brickwall_methods import compute_full_gradient

from tno_compiler.brickwall import random_haar_gates, gates_to_unitary, partition_gates, layer_structure
from tno_compiler.mpo_ops import matrix_to_mpo

n, d = 4, 2

tg = random_haar_gates(n, d, seed=0)
cg = random_haar_gates(n, d, seed=5000)

V = gates_to_unitary(tg, n, d)
U = gates_to_unitary(cg, n, d)

structure = layer_structure(n, d)
gate_layers = partition_gates(cg, n, d)
is_odd = [s[0] for s in structure]
gl_jax = [jnp.asarray(gl) for gl in gate_layers]

# Test 1: V as MPO (NOT left-canonicalized)
V_mpo = [jnp.asarray(a) for a in matrix_to_mpo(V)]
_, ov1 = compute_full_gradient(V_mpo, gl_jax, is_odd, 128, compute_overlap=True)
cost1 = 2 - 2 * float(ov1.real) / (2**n)
print(f"V MPO (no canon):  cost={cost1:.10f}, overlap={ov1}")

# Test 2: V as MPO, left-canonicalized
V_lc = left_to_right_QR_sweep(V_mpo, normalize=False)
_, ov2 = compute_full_gradient(V_lc, gl_jax, is_odd, 128, compute_overlap=True)
cost2 = 2 - 2 * float(ov2.real) / (2**n)
print(f"V MPO (left canon): cost={cost2:.10f}, overlap={ov2}")

# Test 3: V† as MPO, left-canonicalized
Vd_mpo = [jnp.asarray(a) for a in matrix_to_mpo(V.conj().T)]
Vd_lc = left_to_right_QR_sweep(Vd_mpo, normalize=False)
_, ov3 = compute_full_gradient(Vd_lc, gl_jax, is_odd, 128, compute_overlap=True)
cost3 = 2 - 2 * float(ov3.real) / (2**n)
print(f"V† MPO (left canon): cost={cost3:.10f}, overlap={ov3}")

# Exact costs
exact_VdU = np.trace(V.conj().T @ U)
exact_VUd = np.trace(V @ U.conj().T)
print(f"\nExact Tr(V†U) cost = {2 - 2*exact_VdU.real / 2**n:.10f}")
print(f"Exact Tr(V U†) cost = {2 - 2*exact_VUd.real / 2**n:.10f}")
print(f"(These should be equal since Re(Tr(V†U)) = Re(Tr(VU†)*)= Re(Tr(VU†)))")
