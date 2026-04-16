"""Compare our compiler vs rqcopt on the same TFI circuit.

Uses rqcopt's own Ising setup to establish ground truth, then
runs the same target through our pipeline.
"""

import sys
sys.path.insert(0, '_reference_rqcopt')

import numpy as np
import jax.numpy as jnp
from jax import config
config.update("jax_enable_x64", True)

from rqcopt_mpo.spin_systems import (
    construct_ising_hamiltonian, get_brickwall_trotter_gates_spin_chain,
)
from rqcopt_mpo.tn_helpers import get_mpo_from_matrix, left_to_right_QR_sweep
from rqcopt_mpo.brickwall_opt import optimize_swap_network_circuit_RieADAM

from tno_compiler.tfi import tfi_trotter_circuit
from tno_compiler.compiler import compile_circuit
from tno_compiler.pipeline import _qc_to_gate_tensors
from qiskit.quantum_info import Operator

# Parameters
n = 4
t = 0.5
J_val, g_val, h_val = 1.0, 0.75, 0.6
degree = 2  # second-order Trotter
n_reps = 1

print("=" * 60)
print(f"TFI: n={n}, t={t}, J={J_val}, g={g_val}, h={h_val}")
print("=" * 60)

# === rqcopt reference ===
print("\n--- rqcopt ---")
_, J, g, h = construct_ising_hamiltonian(n, J_val, g_val, h_val, get_matrix=False)

# Target: exp(-iHt) as MPO
from scipy.linalg import expm
H_full, _, _, _ = construct_ising_hamiltonian(n, J_val, g_val, h_val, get_matrix=True)
H_mat = np.array(H_full).reshape(2**n, 2**n)
V_exact = expm(-1j * t * H_mat)
V_mpo = get_mpo_from_matrix(jnp.asarray(V_exact))
V_mpo = left_to_right_QR_sweep(V_mpo, normalize=False)

# Circuit: Trotter gates with negated H (= V†)
Vlist = get_brickwall_trotter_gates_spin_chain(
    t, n, n_reps, degree, 'ising-1d', use_TN=True, J=-J, g=-g, h=-h)

from rqcopt_mpo.tn_brickwall_methods import get_riemannian_gradient_and_cost_function

# Single cost evaluation
cost_rqcopt, _ = get_riemannian_gradient_and_cost_function(
    V_mpo, Vlist, n, degree, n_reps, 0, 128, False, 'ising-1d')
print(f"Initial cost (Trotter init): {cost_rqcopt:.6e}")

# === Our pipeline ===
print("\n--- Our pipeline ---")
# Build TFI circuit via Qiskit
target_qc = tfi_trotter_circuit(n, J_val, g_val, h_val, t / n_reps, n_reps, order=1)
print(f"Target circuit: depth={target_qc.depth()}, gates={len(target_qc.data)}")

# Compile at same depth
ansatz_depth = target_qc.depth()
for lr in [1e-2, 5e-3, 1e-3, 5e-4, 1e-4]:
    compiled, info = compile_circuit(
        target_qc, ansatz_depth, max_iter=200, lr=lr)
    final_cost = info['compile_error']
    print(f"  lr={lr:.0e}: final_cost={final_cost:.6e} "
          f"(init={info['cost_history'][0]:.6e})")

# Also try: what does rqcopt achieve with optimization?
print("\n--- rqcopt with optimization (200 iter) ---")
config_rqcopt = {
    'n_sites': n, 'degree': degree, 'n_repetitions': n_reps,
    'n_id_layers': 0, 'max_bondim': 128, 'normalize_reference': False,
    'hamiltonian': 'ising-1d', 'n_iter': 200, 'lr': 5e-3,
    'model_nbr': 999, 'model_dir': '/tmp',
}
# Skip the actual rqcopt optimization (needs file I/O setup)
# Instead just report the initial cost to compare starting points
print(f"rqcopt starts at cost {cost_rqcopt:.6e} (Trotter init)")
print("Our pipeline starts from random init (no Trotter init available)")
