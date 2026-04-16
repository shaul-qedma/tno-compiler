"""Tests for gradient computation against exact overlaps and finite differences."""

import numpy as np
from hypothesis import given, settings

from conftest import n_qubits_st, n_layers_st, seed_st
from tno_compiler.brickwall import random_haar_gates, target_mpo, circuit_to_mpo
from tno_compiler.mpo_ops import mpo_to_arrays
from tno_compiler.gradient import compute_cost_and_grad


def _exact_overlap(tg, cg, n, d):
    """Tr(V†U) via dense matrices (testing only)."""
    V = np.array(circuit_to_mpo(tg, n, d, tol=0.0)[0].to_dense())
    U = np.array(circuit_to_mpo(cg, n, d, tol=0.0)[0].to_dense())
    return np.trace(V.conj().T @ U)


@given(n=n_qubits_st, d=n_layers_st, seed=seed_st)
@settings(max_examples=10, deadline=60000)
def test_overlap_matches_exact(n, d, seed):
    """The MPO-computed Frobenius cost should match the exact cost."""
    tg = random_haar_gates(n, d, seed=seed)
    cg = random_haar_gates(n, d, seed=seed + 5000)
    tmpo, _ = target_mpo(tg, n, d, tol=0.0)
    cost, _ = compute_cost_and_grad(mpo_to_arrays(tmpo), cg, n, d)
    exact_cost = 2.0 - 2.0 * _exact_overlap(tg, cg, n, d).real / (2 ** n)
    assert abs(cost - exact_cost) < 1e-4, f"cost={cost}, exact={exact_cost}"


@given(n=n_qubits_st, d=n_layers_st, seed=seed_st)
@settings(max_examples=5, deadline=60000)
def test_gradient_finite_difference(n, d, seed):
    """Analytic gradient should match finite differences."""
    eps = 1e-5
    tg = random_haar_gates(n, d, seed=seed)
    cg = random_haar_gates(n, d, seed=seed + 5000)
    tmpo, _ = target_mpo(tg, n, d, tol=0.0)
    _, grad = compute_cost_and_grad(mpo_to_arrays(tmpo), cg, n, d)

    rng = np.random.RandomState(seed)
    g_idx = rng.randint(0, len(cg))
    direction = rng.randn(2, 2, 2, 2) + 1j * rng.randn(2, 2, 2, 2)
    gates_p, gates_m = list(cg), list(cg)
    gates_p[g_idx] = gates_p[g_idx] + eps * direction
    gates_m[g_idx] = gates_m[g_idx] - eps * direction

    fd = (_exact_overlap(tg, gates_p, n, d) -
          _exact_overlap(tg, gates_m, n, d)) / (2 * eps)
    analytic = np.einsum('abcd,abcd->', grad[g_idx].conj(), direction)
    rel_err = abs(fd - analytic) / max(abs(analytic), 1e-10)
    assert rel_err < 0.01, f"gate={g_idx}, fd={fd:.6e}, analytic={analytic:.6e}"
