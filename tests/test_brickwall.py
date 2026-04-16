"""Tests for brickwall circuit representation and MPO decomposition."""

import numpy as np
from hypothesis import given, settings, strategies as st

from tno_compiler.brickwall import random_haar_gates, gates_to_unitary, total_gates, layer_structure
from tno_compiler.mpo_ops import matrix_to_mpo

n_qubits_st = st.sampled_from([4, 6])
n_layers_st = st.integers(1, 3)
seed_st = st.integers(0, 9999)


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
def test_mpo_roundtrip(n, d, seed):
    """matrix_to_mpo followed by full contraction recovers the matrix."""
    U = gates_to_unitary(random_haar_gates(n, d, seed=seed), n, d)
    mpo = matrix_to_mpo(U)
    A = mpo[0]
    for B in mpo[1:]:
        C = np.einsum('iabj,jcdk->iacbdk', A, B)
        s = C.shape
        A = C.reshape(s[0], s[1] * s[2], s[3] * s[4], s[-1])
    assert np.allclose(np.einsum('iabj->ab', A), U, atol=1e-10)
