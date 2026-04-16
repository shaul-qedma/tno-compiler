"""Tests for the Kalloor ensemble QP."""

import numpy as np
from hypothesis import given, settings, strategies as st

from conftest import n_qubits_st, seed_st
from tno_compiler.brickwall import random_haar_gates, circuit_to_mpo
from tno_compiler.compiler import compile_circuit
from tno_compiler.ensemble import ensemble_qp

n_layers_st = st.integers(1, 2)


def _dense_gram_and_overlaps(compiled_circuits, target_gates, n, d):
    """Gram matrix and target overlaps from dense matrices (testing only)."""
    V = np.array(circuit_to_mpo(target_gates, n, d, tol=0.0)[0].to_dense())
    Us = [np.array(circuit_to_mpo(c, n, d, tol=0.0)[0].to_dense()) for c in compiled_circuits]
    M = len(Us)
    gram = np.zeros((M, M))
    overlaps = np.zeros(M)
    for i in range(M):
        overlaps[i] = np.trace(Us[i].conj().T @ V).real
        for j in range(M):
            gram[i, j] = np.trace(Us[i].conj().T @ Us[j]).real
    return gram, overlaps


def test_qp_weights_are_valid():
    """Weights should be non-negative and sum to 1."""
    M = 5
    rng = np.random.default_rng(0)
    gram = rng.standard_normal((M, M))
    gram = gram @ gram.T
    overlaps = rng.standard_normal(M)
    weights, _ = ensemble_qp(gram, overlaps)
    assert np.all(weights >= -1e-10)
    assert abs(np.sum(weights) - 1) < 1e-8


def test_qp_single_circuit():
    """With one circuit, the weight should be 1."""
    weights, _ = ensemble_qp(np.array([[4.0]]), np.array([3.5]))
    assert abs(weights[0] - 1.0) < 1e-8


@given(n=n_qubits_st, d=n_layers_st, seed=seed_st)
@settings(max_examples=5, deadline=300000)
def test_ensemble_improves_over_single(n, d, seed):
    """The ensemble QP value should be ≤ the best single-circuit QP value."""
    tg = random_haar_gates(n, d, seed=seed)
    circuits = []
    for s in range(3):
        gates, _ = compile_circuit(
            tg, n, d, max_iter=30, lr=5e-3,
            init_gates=random_haar_gates(n, d, seed=seed + 1000 * s))
        circuits.append(gates)

    gram, overlaps = _dense_gram_and_overlaps(circuits, tg, n, d)
    _, qp_val = ensemble_qp(gram, overlaps)
    best_single = min(0.5 * gram[i, i] - overlaps[i] for i in range(len(circuits)))
    assert qp_val <= best_single + 1e-8
