"""Call rqcopt's compute_partial_derivatives_in_layer directly."""

import sys
sys.path.insert(0, '_reference_rqcopt')

import numpy as np
import jax.numpy as jnp
from jax import config
config.update("jax_enable_x64", True)

from rqcopt_mpo.tn_brickwall_methods import compute_partial_derivatives_in_layer
from rqcopt_mpo.tn_helpers import get_mpo_from_matrix

from tno_compiler.brickwall import random_haar_gates, gates_to_unitary
from tno_compiler.mpo_ops import matrix_to_mpo, identity_mpo

n, d = 4, 1
tg = random_haar_gates(n, d, seed=0)
cg = random_haar_gates(n, d, seed=100)

V = gates_to_unitary(tg, n, d)
U = gates_to_unitary(cg, n, d)
exact = np.trace(V.conj().T @ U)
print(f"Exact Tr(V†U) = {exact}")

# Test 1: V† as upper, I as lower, U as gates
upper = [jnp.asarray(a) for a in matrix_to_mpo(V.conj().T)]
lower = [jnp.asarray(a) for a in identity_mpo(n)]
gates_jax = jnp.asarray(cg)

grads1 = compute_partial_derivatives_in_layer(gates_jax, True, upper, lower)
ov1 = jnp.einsum('abcd,abcd->', grads1[0].conj(), gates_jax[0])
print(f"V† upper, I lower, U gates: overlap={ov1}")

# Test 2: V as upper, I as lower, U as gates
upper2 = [jnp.asarray(a) for a in matrix_to_mpo(V)]
grads2 = compute_partial_derivatives_in_layer(gates_jax, True, upper2, lower)
ov2 = jnp.einsum('abcd,abcd->', grads2[0].conj(), gates_jax[0])
print(f"V upper, I lower, U gates: overlap={ov2}")
print(f"  Tr(V U) = {np.trace(V @ U)}")
print(f"  Matches Tr(V U)? {np.allclose(ov2, np.trace(V @ U), atol=1e-8)}")
