"""Tests for brickwall circuit construction and MPO conversion."""

import numpy as np
from hypothesis import given, settings
from qiskit.quantum_info import Operator

from conftest import n_qubits_st, n_layers_st, seed_st
from tno_compiler.brickwall import (
    random_brickwall, circuit_to_mpo, circuit_to_quimb_tn,
)


def test_random_brickwall_is_unitary():
    qc = random_brickwall(6, 3, seed=42)
    assert qc.num_qubits == 6
    U = Operator(qc).data
    assert np.allclose(U @ U.conj().T, np.eye(64), atol=1e-10)


@given(n=n_qubits_st, d=n_layers_st, seed=seed_st)
@settings(max_examples=10, deadline=30000)
def test_mpo_faithful(n, d, seed):
    """circuit_to_mpo should be close to the exact TN."""
    qc = random_brickwall(n, d, seed=seed)
    tn = circuit_to_quimb_tn(qc)
    mpo, _ = circuit_to_mpo(qc)
    assert mpo.distance_normalized(tn) < 1e-6


