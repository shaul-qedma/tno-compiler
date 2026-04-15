"""Tests for brickwall circuit representation and target MPO construction."""

import numpy as np
import pytest
from hypothesis import given, settings, strategies as st

from tno_compiler.brickwall import (
    random_haar_gates, gates_to_unitary, target_mpo,
    total_gates, layer_structure,
)
from tno_compiler.mpo_ops import matrix_to_mpo, trace_mpo


def test_gate_counts():
    assert total_gates(6, 4) == 10  # 3+2+3+2
    assert total_gates(8, 3) == 4 + 3 + 4


def test_layer_structure():
    s = layer_structure(6, 3)
    assert s[0] == (True, [(0, 1), (2, 3), (4, 5)])
    assert s[1] == (False, [(1, 2), (3, 4)])
    assert s[2] == (True, [(0, 1), (2, 3), (4, 5)])


@given(seed=st.integers(0, 9999))
@settings(max_examples=5, deadline=60000)
def test_mpo_roundtrip(seed):
    """matrix_to_mpo -> reconstruct should match the original."""
    n = 4
    U = gates_to_unitary(random_haar_gates(n, 2, seed=seed), n, 2)
    mpo = matrix_to_mpo(U)
    # Reconstruct
    A = mpo[0]
    for B in mpo[1:]:
        C = np.einsum('iabj,jcdk->iacbdk', A, B)
        s = C.shape
        A = C.reshape(s[0], s[1]*s[2], s[3]*s[4], s[-1])
    U_recon = np.einsum('iabj->ab', A)
    assert np.allclose(U_recon, U, atol=1e-10)


@given(seed=st.integers(0, 9999))
@settings(max_examples=5, deadline=60000)
def test_mpo_trace_matches_exact(seed):
    """Tr(MPO) should match Tr(matrix)."""
    n, d = 4, 2
    gates = random_haar_gates(n, d, seed=seed)
    U = gates_to_unitary(gates, n, d)
    mpo = matrix_to_mpo(U)
    assert np.allclose(trace_mpo(mpo), np.trace(U), atol=1e-8)


@given(seed=st.integers(0, 9999))
@settings(max_examples=3, deadline=60000)
def test_gates_to_unitary_is_unitary(seed):
    """The constructed unitary should actually be unitary."""
    n, d = 6, 3
    gates = random_haar_gates(n, d, seed=seed)
    U = gates_to_unitary(gates, n, d)
    assert np.allclose(U @ U.conj().T, np.eye(2**n), atol=1e-10)
