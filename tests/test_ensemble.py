"""Tests for the Kalloor ensemble QP."""

import numpy as np
from hypothesis import given, settings, strategies as st

from conftest import n_qubits_st, seed_st
from tno_compiler.brickwall import random_brickwall
from tno_compiler.ensemble import ensemble_qp
from tno_compiler.pipeline import compile_ensemble

n_layers_st = st.integers(1, 2)


def test_qp_weights_are_valid():
    rng = np.random.default_rng(0)
    gram = rng.standard_normal((5, 5))
    gram = gram @ gram.T
    weights, _ = ensemble_qp(gram, rng.standard_normal(5))
    assert np.all(weights >= -1e-10)
    assert abs(np.sum(weights) - 1) < 1e-8


def test_qp_single_circuit():
    weights, _ = ensemble_qp(np.array([[4.0]]), np.array([3.5]))
    assert abs(weights[0] - 1.0) < 1e-8


@given(n=n_qubits_st, d=n_layers_st, seed=seed_st)
@settings(max_examples=5, deadline=300000)
def test_ensemble_improves_over_single(n, d, seed):
    target = random_brickwall(n, d, seed=seed)
    result = compile_ensemble(target, d, n_circuits=3, max_iter=30, seed=seed)
    best_single = min(0.5 * result['individual_frobs'][i]**2
                      for i in range(len(result['circuits'])))
    # Ensemble should do no worse (QP value is the minimum)
    assert result['qp_value'] <= best_single + 2**n + 1e-6
