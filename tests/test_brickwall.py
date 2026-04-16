"""Tests for brickwall MPO construction via quimb."""

import numpy as np
from hypothesis import given, settings

from conftest import n_qubits_st, n_layers_st, seed_st
from tno_compiler.brickwall import (
    random_haar_gates, circuit_to_mpo, circuit_to_tn, target_mpo,
    total_gates, layer_structure,
)


def test_gate_counts():
    assert total_gates(6, 4) == 10
    assert total_gates(8, 3) == 4 + 3 + 4


def test_layer_structure():
    s = layer_structure(6, 3)
    assert s[0] == (True, [(0, 1), (2, 3), (4, 5)])
    assert s[1] == (False, [(1, 2), (3, 4)])
    assert s[2] == (True, [(0, 1), (2, 3), (4, 5)])


@given(n=n_qubits_st, d=n_layers_st, seed=seed_st)
@settings(max_examples=10, deadline=30000)
def test_mpo_faithful(n, d, seed):
    """circuit_to_mpo should be close to the exact TN."""
    gates = random_haar_gates(n, d, seed=seed)
    tn = circuit_to_tn(gates, n, d)
    mpo, _ = circuit_to_mpo(gates, n, d)
    assert mpo.distance_normalized(tn) < 1e-6


@given(n=n_qubits_st, d=n_layers_st, seed=seed_st)
@settings(max_examples=10, deadline=30000)
def test_target_is_adjoint(n, d, seed):
    """target_mpo should be the adjoint of circuit_to_mpo."""
    gates = random_haar_gates(n, d, seed=seed)
    V = np.array(circuit_to_mpo(gates, n, d)[0].to_dense())
    Vd = np.array(target_mpo(gates, n, d)[0].to_dense())
    assert np.allclose(Vd, V.conj().T, atol=1e-8)
