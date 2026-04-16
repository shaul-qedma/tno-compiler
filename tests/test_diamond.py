"""Verify the Mixing Lemma at the channel level using dense superoperators."""

import numpy as np
from qiskit.quantum_info import Operator, SuperOp

from tno_compiler.brickwall import random_brickwall
from tno_compiler.pipeline import compile_ensemble


def _ensemble_superop(circuits, weights):
    """Dense superoperator of Σ p_i U_i · U_i†."""
    dim = 2 ** circuits[0].num_qubits
    S = np.zeros((dim ** 2, dim ** 2), dtype=complex)
    for qc, p in zip(circuits, weights):
        if p < 1e-15:
            continue
        U = Operator(qc).data
        S += p * np.kron(U.conj(), U)
    return S


def test_channel_distance_within_bound():
    """||E - V||_superop should be ≤ diamond_bound."""
    target = random_brickwall(4, 1, seed=42)
    result = compile_ensemble(target, 1, n_circuits=3, max_iter=50, seed=42)

    S_ens = _ensemble_superop(result['circuits'], result['weights'])
    V = Operator(target).data
    S_tgt = np.kron(V.conj(), V)

    diff_norm = np.linalg.norm(S_ens - S_tgt, ord=2)
    assert diff_norm <= result['diamond_bound'] + 1e-6


def test_perfect_channel():
    """Target as sole ensemble member → zero channel distance."""
    target = random_brickwall(4, 1, seed=42)
    V = Operator(target).data
    S = np.kron(V.conj(), V)
    assert np.linalg.norm(S - S, ord=2) < 1e-10
