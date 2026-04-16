"""Test that circuit_to_mpo respects the target tolerance end-to-end.

The key test: apply the exact circuit and the compressed MPO to random
basis states, verify the output difference is within the operator norm bound.
"""

import numpy as np
from hypothesis import given, settings, strategies as st
from qiskit.quantum_info import Statevector

from tno_compiler.brickwall import (
    random_haar_gates, circuit_to_mpo, layer_structure,
)

n_qubits_st = st.sampled_from([4, 6, 8])
n_layers_st = st.integers(1, 4)
tol_st = st.sampled_from([1e-2, 1e-4, 1e-6, 1e-8])
seed_st = st.integers(0, 9999)


def _simulate_circuit(gates, n, d, input_state):
    """Apply brickwall circuit to a statevector (big-endian)."""
    psi = input_state.copy()
    idx = 0
    for _, pairs in layer_structure(n, d):
        for q1, q2 in pairs:
            gate = np.asarray(gates[idx]).reshape(4, 4)
            # Apply gate to qubits (q1, q2) in the 2^n vector
            psi = psi.reshape([2] * n)
            psi = np.tensordot(gate.reshape(2, 2, 2, 2), psi,
                               axes=([2, 3], [q1, q2]))
            # tensordot puts the gate output axes first; move them back
            psi = np.moveaxis(psi, [0, 1], [q1, q2])
            idx += 1
    return psi.ravel()


def _simulate_mpo(mpo, n, input_state):
    """Apply quimb MPO to a statevector (big-endian, via dense matrix)."""
    mat = np.array(mpo.to_dense())
    return mat @ input_state


@given(n=n_qubits_st, d=n_layers_st, tol=tol_st, seed=seed_st)
@settings(max_examples=20, deadline=60000)
def test_error_bound_respected(n, d, tol, seed):
    """The reported error bound should be ≤ tol."""
    gates = random_haar_gates(n, d, seed=seed)
    _, error = circuit_to_mpo(gates, n, d, tol=tol)
    assert error <= tol, f"error {error:.2e} > tol {tol:.2e}"


@given(n=n_qubits_st, d=n_layers_st, tol=tol_st, seed=seed_st)
@settings(max_examples=20, deadline=60000)
def test_operator_norm_via_basis_states(n, d, tol, seed):
    """For random basis |k⟩: ||(V - V_mpo)|k⟩|| ≤ error_bound.

    Directly tests the operator norm guarantee by sampling basis states.
    """
    gates = random_haar_gates(n, d, seed=seed)
    mpo, error = circuit_to_mpo(gates, n, d, tol=tol)

    rng = np.random.RandomState(seed)
    for _ in range(5):
        k = rng.randint(0, 2 ** n)
        basis = np.zeros(2 ** n, dtype=complex)
        basis[k] = 1.0

        exact_out = _simulate_circuit(gates, n, d, basis)
        mpo_out = _simulate_mpo(mpo, n, basis)

        diff = np.linalg.norm(exact_out - mpo_out)
        assert diff <= error + 1e-12, (
            f"||(V-V_mpo)|{k}⟩|| = {diff:.2e} > error = {error:.2e}")
