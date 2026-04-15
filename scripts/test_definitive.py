"""Definitive test: call rqcopt compute_full_gradient for n=4, d=1
with both V and V† as MPO, and check against ALL possible overlaps."""

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

structure = layer_structure(n, d)
gate_layers = partition_gates(cg, n, d)
is_odd = [s[0] for s in structure]
gl_jax = [jnp.asarray(gl) for gl in gate_layers]

for label, mat in [("V", V), ("V†", V.conj().T)]:
    mpo = [jnp.asarray(a) for a in matrix_to_mpo(mat)]
    _, ov = compute_full_gradient(mpo, gl_jax, is_odd, 128, compute_overlap=True)
    print(f"\nMPO={label}: overlap = {ov}")

    # Check against every possible contraction
    candidates = {
        "Tr(V†U)":  np.trace(V.conj().T @ U),
        "Tr(VU†)":  np.trace(V @ U.conj().T),
        "Tr(VU)":   np.trace(V @ U),
        "Tr(V†U†)": np.trace(V.conj().T @ U.conj().T),
        "Tr(V^TU)": np.trace(V.T @ U),
        "sum(V*U)":  np.sum(V.conj() * U),  # = Tr(V†U)
        "sum(VU*)":  np.sum(V * U.conj()),   # = Tr(VU†)* = conj(Tr(V†U))
    }
    for name, val in candidates.items():
        if np.allclose(ov, val, atol=1e-6):
            print(f"  *** MATCHES {name} = {val}")

# Also test: what if we use the ADJOINT of the circuit gates?
# rqcopt Trotter uses -H gates = V†
cg_adj = [g.conj().transpose(2, 3, 0, 1) for g in cg]
gl_adj = partition_gates(cg_adj, n, d)
gl_adj_jax = [jnp.asarray(gl) for gl in gl_adj]

for label, mat in [("V", V), ("V†", V.conj().T)]:
    mpo = [jnp.asarray(a) for a in matrix_to_mpo(mat)]
    _, ov = compute_full_gradient(mpo, gl_adj_jax, is_odd, 128, compute_overlap=True)
    print(f"\nMPO={label}, gates=U†: overlap = {ov}")

    U_adj = gates_to_unitary(cg_adj, n, d)
    candidates = {
        "Tr(V†U†)": np.trace(V.conj().T @ U_adj),
        "Tr(VU††)=Tr(VU)": np.trace(V @ U),
        "Tr(V U†)": np.trace(V @ U.conj().T),
        "Tr(V†U)":  np.trace(V.conj().T @ U),
    }
    for name, val in candidates.items():
        if np.allclose(ov, val, atol=1e-6):
            print(f"  *** MATCHES {name} = {val}")
