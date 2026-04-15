"""Tests for brickwall circuit representation and MPO conversion."""

import numpy as np
import pytest
from hypothesis import given, settings, strategies as st
from qiskit.quantum_info import process_fidelity, Operator

from tno_compiler.brickwall import (
    random_haar_gates, gates_to_unitary, gates_to_mpo, gates_to_qiskit,
    total_gates, layer_structure,
)


def test_gate_counts():
    # n=6, odd: (0,1)(2,3)(4,5)=3, even: (1,2)(3,4)=2
    # 4 layers odd-even-odd-even: 3+2+3+2=10
    assert total_gates(6, 4) == 10
    assert total_gates(8, 3) == 4 + 3 + 4


def test_layer_structure():
    s = layer_structure(6, 3)
    assert s[0] == (True, [(0, 1), (2, 3), (4, 5)])
    assert s[1] == (False, [(1, 2), (3, 4)])
    assert s[2] == (True, [(0, 1), (2, 3), (4, 5)])


@given(seed=st.integers(0, 9999))
@settings(max_examples=5, deadline=60000)
def test_mpo_matches_exact_unitary(seed):
    """MPO from gates_to_mpo matches the exact Qiskit unitary."""
    n, d = 6, 4
    gates = random_haar_gates(n, d, seed=seed)
    U_exact = gates_to_unitary(gates, n, d)
    mpo = gates_to_mpo(gates, n, d)
    U_mpo = np.array(mpo.to_dense())
    assert np.allclose(U_mpo, U_exact, atol=1e-10)


@given(seed=st.integers(0, 9999))
@settings(max_examples=5, deadline=60000)
def test_mpo_trace_matches_exact(seed):
    """Tr(MPO) matches Tr(exact unitary)."""
    n, d = 6, 3
    gates = random_haar_gates(n, d, seed=seed)
    U_exact = gates_to_unitary(gates, n, d)
    mpo = gates_to_mpo(gates, n, d)
    assert np.allclose(mpo.trace(), np.trace(U_exact), atol=1e-8)


@given(seed=st.integers(0, 9999))
@settings(max_examples=3, deadline=60000)
def test_mpo_overlap_is_trace_inner_product(seed):
    """mpo1.overlap(mpo2) = conj(Tr(mpo1^dag mpo2)) in quimb convention."""
    n, d = 6, 2
    g1 = random_haar_gates(n, d, seed=seed)
    g2 = random_haar_gates(n, d, seed=seed + 10000)
    U1 = gates_to_unitary(g1, n, d)
    U2 = gates_to_unitary(g2, n, d)
    mpo1 = gates_to_mpo(g1, n, d)
    mpo2 = gates_to_mpo(g2, n, d)

    ov_quimb = mpo1.overlap(mpo2)
    ov_exact = np.trace(U1.conj().T @ U2)
    # quimb returns conj of the standard inner product
    assert np.allclose(ov_quimb, np.conj(ov_exact), atol=1e-8)
