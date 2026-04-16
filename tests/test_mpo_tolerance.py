"""Test that circuit_to_mpo respects tolerance in both norms."""

import numpy as np
from hypothesis import given, settings, strategies as st

from conftest import seed_st
from tno_compiler.brickwall import random_brickwall, circuit_to_mpo

n_qubits_st = st.sampled_from([4, 6, 8])
n_layers_st = st.integers(1, 4)
tol_st = st.sampled_from([1e-2, 1e-4, 1e-6, 1e-8])


@given(n=n_qubits_st, d=n_layers_st, tol=tol_st, seed=seed_st)
@settings(max_examples=20, deadline=60000)
def test_operator_norm_bound_respected(n, d, tol, seed):
    qc = random_brickwall(n, d, seed=seed)
    _, error = circuit_to_mpo(qc, tol=tol, norm="operator")
    assert error <= tol


@given(n=n_qubits_st, d=n_layers_st, tol=tol_st, seed=seed_st)
@settings(max_examples=20, deadline=60000)
def test_operator_norm_actual(n, d, tol, seed):
    qc = random_brickwall(n, d, seed=seed)
    exact, _ = circuit_to_mpo(qc, tol=0.0)
    compressed, error = circuit_to_mpo(qc, tol=tol, norm="operator")
    diff = np.array(exact.to_dense()) - np.array(compressed.to_dense())
    assert np.linalg.norm(diff, ord=2) <= error + 1e-12


@given(n=n_qubits_st, d=n_layers_st, tol=tol_st, seed=seed_st)
@settings(max_examples=20, deadline=60000)
def test_frobenius_norm_bound_respected(n, d, tol, seed):
    qc = random_brickwall(n, d, seed=seed)
    _, error = circuit_to_mpo(qc, tol=tol, norm="frobenius")
    assert error <= tol


@given(n=n_qubits_st, d=n_layers_st, tol=tol_st, seed=seed_st)
@settings(max_examples=20, deadline=60000)
def test_frobenius_norm_actual(n, d, tol, seed):
    qc = random_brickwall(n, d, seed=seed)
    exact, _ = circuit_to_mpo(qc, tol=0.0)
    compressed, error = circuit_to_mpo(qc, tol=tol, norm="frobenius")
    diff = np.array(exact.to_dense()) - np.array(compressed.to_dense())
    assert np.linalg.norm(diff, ord='fro') <= error + 1e-12
