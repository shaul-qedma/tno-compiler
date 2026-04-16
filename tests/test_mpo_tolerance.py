"""Test that circuit_to_mpo respects the target tolerance end-to-end."""

import numpy as np
from hypothesis import given, settings, strategies as st

from tno_compiler.brickwall import random_haar_gates, circuit_to_mpo, circuit_to_tn

n_qubits_st = st.sampled_from([4, 6, 8])
n_layers_st = st.integers(1, 4)
tol_st = st.sampled_from([1e-2, 1e-4, 1e-6, 1e-8])
seed_st = st.integers(0, 9999)


@given(n=n_qubits_st, d=n_layers_st, tol=tol_st, seed=seed_st)
@settings(max_examples=20, deadline=60000)
def test_error_bound_respected(n, d, tol, seed):
    """The reported error bound should be ≤ tol."""
    gates = random_haar_gates(n, d, seed=seed)
    _, error = circuit_to_mpo(gates, n, d, tol=tol)
    assert error <= tol, f"error {error:.2e} > tol {tol:.2e}"


@given(n=n_qubits_st, d=n_layers_st, tol=tol_st, seed=seed_st)
@settings(max_examples=20, deadline=60000)
def test_actual_distance_below_bound(n, d, tol, seed):
    """The actual Frobenius distance should be consistent with the error bound.

    The reported error bounds the operator norm. The normalized Frobenius
    distance (from quimb) should also be small when the error bound is small.
    """
    gates = random_haar_gates(n, d, seed=seed)
    tn_exact = circuit_to_tn(gates, n, d)
    mpo, error = circuit_to_mpo(gates, n, d, tol=tol)
    dist = mpo.distance_normalized(tn_exact)
    # Normalized distance should be at most O(error) for small error
    assert dist < max(tol * 10, 1e-10), (
        f"distance {dist:.2e} too large for tol={tol:.2e}, error={error:.2e}")
