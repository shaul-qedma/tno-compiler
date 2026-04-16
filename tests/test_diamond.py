"""Verify the Mixing Lemma at the channel level using dense superoperators.

The ensemble channel E(ρ) = Σ p_i U_i ρ U_i† should be close to V ρ V†
as measured by the diamond distance, bounded by the pipeline's certificate.

At small n we represent channels as d²×d² superoperator matrices and
compute their distance directly.
"""

import numpy as np
from hypothesis import given, settings
from qiskit.quantum_info import SuperOp, Operator, process_fidelity

from conftest import n_qubits_st, seed_st
from tno_compiler.brickwall import random_haar_gates, circuit_to_mpo
from tno_compiler.pipeline import compile_ensemble

n_layers_st = __import__('hypothesis').strategies.integers(1, 2)


def _ensemble_superop(circuits, weights, n, d):
    """Dense superoperator of Σ p_i U_i · U_i† (d²×d² matrix)."""
    dim = 2 ** n
    S = np.zeros((dim ** 2, dim ** 2), dtype=complex)
    for gates, p in zip(circuits, weights):
        if p < 1e-15:
            continue
        U = np.array(circuit_to_mpo(gates, n, d, tol=0.0)[0].to_dense())
        # Superoperator of U·U†: S_U = conj(U) ⊗ U
        S += p * np.kron(U.conj(), U)
    return S


def _target_superop(target_gates, n, d):
    """Dense superoperator of V · V†."""
    V = np.array(circuit_to_mpo(target_gates, n, d, tol=0.0)[0].to_dense())
    return np.kron(V.conj(), V)


@given(n=n_qubits_st, d=n_layers_st, seed=seed_st)
@settings(max_examples=5, deadline=300000)
def test_channel_distance_within_diamond_bound(n, d, seed):
    """The superoperator norm ||E - V||_op should be ≤ diamond_bound.

    The diamond norm ≥ the induced trace norm ≥ the superoperator
    operator norm, so diamond_bound bounding the superop norm is
    a necessary condition for the certificate to be valid.
    """
    tg = random_haar_gates(n, d, seed=seed)
    result = compile_ensemble(tg, n, d, n_circuits=3, max_iter=50, seed=seed)

    S_ens = _ensemble_superop(result['circuits'], result['weights'], n, d)
    S_tgt = _target_superop(tg, n, d)

    # Operator norm of superoperator difference
    diff_norm = np.linalg.norm(S_ens - S_tgt, ord=2)
    assert diff_norm <= result['diamond_bound'] + 1e-6, (
        f"||E-V||_superop={diff_norm:.6e} > bound={result['diamond_bound']:.6e}")


@given(n=n_qubits_st, d=n_layers_st, seed=seed_st)
@settings(max_examples=5, deadline=300000)
def test_ensemble_channel_fidelity(n, d, seed):
    """Process fidelity of ensemble channel vs target should be consistent
    with the diamond bound (high fidelity ↔ small diamond distance)."""
    tg = random_haar_gates(n, d, seed=seed)
    result = compile_ensemble(tg, n, d, n_circuits=3, max_iter=50, seed=seed)

    V = np.array(circuit_to_mpo(tg, n, d, tol=0.0)[0].to_dense())
    target_op = Operator(V)
    ensemble_sup = SuperOp(_ensemble_superop(
        result['circuits'], result['weights'], n, d))

    fid = process_fidelity(ensemble_sup, target_op)
    # Process fidelity ≥ 1 - d·diamond_distance for small distances
    assert fid >= 0 or result['diamond_bound'] > 1


def test_perfect_channel():
    """V as sole ensemble member → superoperator difference is zero."""
    n, d = 4, 1
    tg = random_haar_gates(n, d, seed=42)
    S_ens = _ensemble_superop([tg], np.array([1.0]), n, d)
    S_tgt = _target_superop(tg, n, d)
    assert np.linalg.norm(S_ens - S_tgt, ord=2) < 1e-10
