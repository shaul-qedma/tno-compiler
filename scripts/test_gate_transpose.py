"""Test all possible gate tensor transpositions for the 4-qubit overlap."""

import sys
sys.path.insert(0, '_reference_rqcopt')

import numpy as np
import jax.numpy as jnp
from jax import config
config.update("jax_enable_x64", True)

from rqcopt_mpo.tn_brickwall_methods import compute_full_gradient
from tno_compiler.brickwall import random_haar_gates, gates_to_unitary, partition_gates, layer_structure
from tno_compiler.mpo_ops import matrix_to_mpo
from itertools import permutations

n, d = 4, 1
tg = random_haar_gates(n, d, seed=0)
cg = random_haar_gates(n, d, seed=100)

V = gates_to_unitary(tg, n, d)
U = gates_to_unitary(cg, n, d)

structure = layer_structure(n, d)
is_odd = [s[0] for s in structure]

# Try V as MPO with all possible gate transpositions
V_mpo = [jnp.asarray(a) for a in matrix_to_mpo(V)]
exact_TrVU = np.trace(V @ U)  # We know rqcopt computes Tr(V·G) for 2-qubit

for perm in permutations(range(4)):
    cg_perm = [g.transpose(perm) for g in cg]
    gl = partition_gates(cg_perm, n, d)
    gl_jax = [jnp.asarray(g) for g in gl]
    _, ov = compute_full_gradient(V_mpo, gl_jax, is_odd, 128, compute_overlap=True)
    if np.allclose(ov, exact_TrVU, atol=1e-6):
        print(f"MATCH with perm {perm}: overlap={ov}")

# Also try: transpose the TARGET gates when building the MPO
for perm in permutations(range(4)):
    tg_perm = [g.transpose(perm) for g in tg]
    V_perm = gates_to_unitary(tg_perm, n, d)
    V_perm_mpo = [jnp.asarray(a) for a in matrix_to_mpo(V_perm)]
    gl_jax = [jnp.asarray(g) for g in partition_gates(cg, n, d)]
    _, ov = compute_full_gradient(V_perm_mpo, gl_jax, is_odd, 128, compute_overlap=True)
    exact = np.trace(V_perm @ U)
    if np.allclose(ov, exact, atol=1e-6):
        print(f"TARGET perm {perm}: overlap={ov}, exact Tr(V_perm U)={exact}")
