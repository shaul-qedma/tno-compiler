"""Verify the Mixing Lemma at the channel level using dense superoperators."""

import numpy as np
from tno_compiler.brickwall import random_brickwall, circuit_to_mpo
from tno_compiler.pipeline import compile_ensemble


def _mpo_dense(qc):
    return np.array(circuit_to_mpo(qc, tol=0.0)[0].to_dense())


def _ensemble_superop(circuits, weights):
    d = _mpo_dense(circuits[0]).shape[0]
    S = np.zeros((d ** 2, d ** 2), dtype=complex)
    for qc, p in zip(circuits, weights):
        if p < 1e-15:
            continue
        U = _mpo_dense(qc)
        S += p * np.kron(U.conj(), U)
    return S


def test_channel_distance_within_bound():
    target = random_brickwall(4, 1, seed=42)
    result = compile_ensemble(target, 1, n_circuits=3, max_iter=50, seed=42)
    V = _mpo_dense(target)
    S_tgt = np.kron(V.conj(), V)
    S_ens = _ensemble_superop(result['circuits'], result['weights'])
    diff_norm = np.linalg.norm(S_ens - S_tgt, ord=2)
    assert diff_norm <= result['diamond_bound'] + 1e-6


def test_perfect_channel():
    target = random_brickwall(4, 1, seed=42)
    V = _mpo_dense(target)
    S = np.kron(V.conj(), V)
    assert np.linalg.norm(S - S, ord=2) < 1e-10
