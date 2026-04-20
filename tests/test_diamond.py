"""Channel-level verification of the ensemble pipeline.

Two complementary checks:
- Dense superoperator norm (rigorous but only feasible up to n≈4).
- Max trace distance over random pure inputs (Kalloor et al. 2025,
  results §Demonstration): a lower bound on (½)·‖E - V‖_◇, usable at
  any n where the dense unitary fits in memory.
"""

import numpy as np
from tno_compiler.brickwall import random_brickwall, circuit_to_mpo
from tno_compiler.pipeline import compile_ensemble
from tno_compiler.verify import sampled_max_trace_distance


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


# --- Sampled max trace distance (Kalloor-style channel verification) ---

def test_sampled_tracedist_zero_for_identity_ensemble():
    target = random_brickwall(4, 1, seed=42)
    out = sampled_max_trace_distance(target, [target], np.array([1.0]),
                                     n_samples=5, seed=0)
    assert out['max_td'] < 1e-10


def test_sampled_tracedist_in_unit_interval():
    target = random_brickwall(4, 2, seed=42)
    unrelated = random_brickwall(4, 2, seed=99)
    out = sampled_max_trace_distance(target, [unrelated], np.array([1.0]),
                                     n_samples=5, seed=0)
    assert 0.0 <= out['max_td'] <= 1.0 + 1e-10


def test_sampled_tracedist_trained_beats_random():
    target = random_brickwall(4, 2, seed=42)
    compiled = compile_ensemble(target, 2, n_circuits=3, max_iter=100, seed=42)
    trained_td = sampled_max_trace_distance(
        target, compiled['circuits'], compiled['weights'],
        n_samples=10, seed=7)['max_td']

    random_circs = [random_brickwall(4, 2, seed=900 + i) for i in range(3)]
    random_weights = np.ones(3) / 3
    random_td = sampled_max_trace_distance(
        target, random_circs, random_weights,
        n_samples=10, seed=7)['max_td']

    assert trained_td < random_td, (
        f"trained={trained_td:.3e} not better than random={random_td:.3e}")


def test_sampled_tracedist_pipeline_same_depth():
    """Same-depth target at n=4: ensemble should reproduce the target well."""
    target = random_brickwall(4, 2, seed=42)
    result = compile_ensemble(target, 2, n_circuits=3, max_iter=200, seed=42)
    out = sampled_max_trace_distance(
        target, result['circuits'], result['weights'],
        n_samples=10, seed=0)
    assert out['max_td'] < 0.1, (
        f"max trace distance {out['max_td']:.3e} exceeds 0.1")
