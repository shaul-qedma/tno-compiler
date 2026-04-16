"""End-to-end compilation pipeline: target tolerance + depth budget.

Compiles an ensemble of circuits at a given depth, solves the Kalloor QP,
and certifies the diamond distance. Optionally binary searches over depth.
"""

import numpy as np
from .brickwall import random_haar_gates, circuit_to_mpo
from .compiler import compile_circuit
from .ensemble import ensemble_qp


def compile_ensemble(target_gates, n_qubits, n_layers, n_circuits=5,
                     tol=1e-2, compress_fraction=0.0, max_bond=None,
                     max_iter=200, lr=5e-3, first_odd=True, seed=0):
    """Compile an ensemble of circuits and solve the Kalloor QP.

    Returns dict with:
        weights: optimal ensemble probabilities.
        circuits: list of compiled gate lists.
        delta_ens: ensemble Frobenius error ||Σ p_i U_i - V_mpo||_F.
        R: max individual error max_i ||U_i - V_mpo||_F.
        compress_error: Frobenius MPO compression error.
        diamond_bound: 2*delta_ens_total + R_total² (certified upper bound).
    """
    # Compile M circuits from different initializations
    circuits = []
    compile_errors = []
    for i in range(n_circuits):
        init = random_haar_gates(n_qubits, n_layers, first_odd, seed=seed + 1000 * i)
        gates, info = compile_circuit(
            target_gates, n_qubits, n_layers,
            tol=tol, compress_fraction=compress_fraction,
            max_bond=max_bond, max_iter=max_iter, lr=lr,
            first_odd=first_odd, init_gates=init)
        circuits.append(gates)
        compile_errors.append(info['compile_error'])

    compress_error = info['compress_error']

    # Compute Gram matrix and target overlaps (dense, small n)
    V = np.array(circuit_to_mpo(target_gates, n_qubits, n_layers, tol=0.0)[0].to_dense())
    Us = [np.array(circuit_to_mpo(c, n_qubits, n_layers, tol=0.0)[0].to_dense())
          for c in circuits]

    M = len(Us)
    gram = np.zeros((M, M))
    overlaps = np.zeros(M)
    for i in range(M):
        overlaps[i] = np.trace(Us[i].conj().T @ V).real
        for j in range(M):
            gram[i, j] = np.trace(Us[i].conj().T @ Us[j]).real

    # Solve QP
    weights, qp_val = ensemble_qp(gram, overlaps)

    # Certification (SPEC.md Lemma D)
    # delta_ens = ||Σ p_i U_i - V||_F (from QP + compression)
    d = 2 ** n_qubits
    # QP gives: Σ p_i p_j Re Tr(U_i†U_j) - 2 Σ p_i Re Tr(U_i†V) = qp_val * 2
    # ||Σ p_i U_i - V||_F² = above + Tr(V†V) = 2*qp_val + d
    ensemble_frob_sq = 2 * qp_val + d
    ensemble_frob = np.sqrt(max(ensemble_frob_sq, 0))

    # Individual Frobenius errors: ||U_i - V||_F = sqrt(2d - 2 Re Tr(U_i†V))
    individual_frobs = [np.sqrt(max(2 * d - 2 * overlaps[i], 0)) for i in range(M)]
    R = max(individual_frobs[i] for i in range(M) if weights[i] > 1e-10)

    # Diamond bound with compression error
    delta_ens = ensemble_frob + compress_error
    R_total = R + compress_error
    diamond_bound = 2 * delta_ens + R_total ** 2

    return {
        'weights': weights,
        'circuits': circuits,
        'delta_ens': delta_ens,
        'R': R_total,
        'compress_error': compress_error,
        'diamond_bound': diamond_bound,
        'individual_frobs': individual_frobs,
        'qp_value': qp_val,
    }


def find_min_depth(target_gates, n_qubits, tol, max_depth=20, **kwargs):
    """Binary search for the minimum depth achieving diamond_bound ≤ tol."""
    lo, hi = 1, max_depth

    best = None
    while lo <= hi:
        mid = (lo + hi) // 2
        result = compile_ensemble(
            target_gates, n_qubits, mid, tol=tol, **kwargs)
        if result['diamond_bound'] <= tol:
            best = (mid, result)
            hi = mid - 1
        else:
            lo = mid + 1

    if best is None:
        # Try max_depth as fallback
        result = compile_ensemble(
            target_gates, n_qubits, max_depth, tol=tol, **kwargs)
        return max_depth, result

    return best
