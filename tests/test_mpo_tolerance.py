"""Test that circuit_to_mpo respects the target operator norm tolerance."""

import numpy as np
from hypothesis import given, settings, strategies as st

from conftest import seed_st
from tno_compiler.brickwall import random_haar_gates, circuit_to_mpo

n_qubits_st = st.sampled_from([4, 6, 8])
n_layers_st = st.integers(1, 4)
tol_st = st.sampled_from([1e-2, 1e-4, 1e-6, 1e-8])


@given(n=n_qubits_st, d=n_layers_st, tol=tol_st, seed=seed_st)
@settings(max_examples=20, deadline=60000)
def test_error_bound_respected(n, d, tol, seed):
    """The reported error bound should be ≤ tol."""
    gates = random_haar_gates(n, d, seed=seed)
    _, error = circuit_to_mpo(gates, n, d, tol=tol)
    assert error <= tol


@given(n=n_qubits_st, d=n_layers_st, tol=tol_st, seed=seed_st)
@settings(max_examples=20, deadline=60000)
def test_operator_norm_within_bound(n, d, tol, seed):
    """The actual operator norm ||V - V_mpo||_op should be ≤ error_bound."""
    gates = random_haar_gates(n, d, seed=seed)
    mpo_exact, _ = circuit_to_mpo(gates, n, d, tol=0.0)
    mpo_compressed, error = circuit_to_mpo(gates, n, d, tol=tol)
    diff = np.array(mpo_exact.to_dense()) - np.array(mpo_compressed.to_dense())
    assert np.linalg.norm(diff, ord=2) <= error + 1e-12
