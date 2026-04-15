"""Test rqcopt on its home turf: Ising model Trotter gates."""

import sys
sys.path.insert(0, '_reference_rqcopt')

import numpy as np
import jax.numpy as jnp
from jax import config
config.update("jax_enable_x64", True)

from rqcopt_mpo.spin_systems import construct_ising_hamiltonian, get_brickwall_trotter_gates_spin_chain
from rqcopt_mpo.tn_helpers import get_mpo_from_matrix, get_matrix_from_mpo
from rqcopt_mpo.brickwall_circuit import get_gates_per_layer
from rqcopt_mpo.tn_brickwall_methods import compute_full_gradient, fully_contract_swap_network_mpo

n_sites = 4
t = 0.5
n_reps = 1
degree = 2

_, J, g, h = construct_ising_hamiltonian(n_sites, J=1., g=0.75, h=0.6, get_matrix=False)

# Target: exp(-iHt) as MPO
from scipy.linalg import expm
H_full, _, _, _ = construct_ising_hamiltonian(n_sites, J=1., g=0.75, h=0.6, get_matrix=True)
V = expm(-1j * t * np.array(H_full).reshape(2**n_sites, 2**n_sites))
V_mpo = get_mpo_from_matrix(jnp.asarray(V))

# Circuit: Trotter gates with NEGATED Hamiltonian (= V†)
Vlist = get_brickwall_trotter_gates_spin_chain(t, n_sites, n_reps, degree, 'ising-1d',
                                                use_TN=True, J=-J, g=-g, h=-h)
gates_per_layer, layer_is_odd = get_gates_per_layer(Vlist, n_sites, degree, n_reps, hamiltonian='ising-1d')

# Compute gradient
grad, overlap = compute_full_gradient(V_mpo, gates_per_layer, layer_is_odd, 128, compute_overlap=True)

print(f"rqcopt overlap = {overlap}")
print(f"rqcopt cost = {2 - 2*overlap.real / (2**n_sites)}")

# Now check: what is the gate tensor convention?
# Look at one Trotter gate
g0 = np.array(Vlist[0])
print(f"\nTrotter gate shape: {g0.shape}")
print(f"Is unitary (reshape 4x4)? {np.allclose(g0.reshape(4,4) @ g0.reshape(4,4).conj().T, np.eye(4), atol=1e-10)}")

# Check gate leg ordering: is it (k0, k1, b0, b1) or (k0, b0, k1, b1)?
g_mat1 = g0.reshape(4, 4)  # (k0*k1, b0*b1) if first two are row, last two are col
g_mat2 = g0.transpose(0, 2, 1, 3).reshape(4, 4)  # (k0*b0, k1*b1) alternative
print(f"reshape(4,4) unitary: {np.allclose(g_mat1 @ g_mat1.conj().T, np.eye(4), atol=1e-10)}")
print(f"transpose+reshape unitary: {np.allclose(g_mat2 @ g_mat2.conj().T, np.eye(4), atol=1e-10)}")
