"""Tests for gradient computation, verified against finite differences."""

import numpy as np
from hypothesis import given, settings, strategies as st

from tno_compiler.brickwall import random_haar_gates, target_mpo, gates_to_unitary
from tno_compiler.gradient import compute_cost_and_grad


def _exact_overlap(target_gates, circuit_gates, n, d):
    V = gates_to_unitary(target_gates, n, d)
    U = gates_to_unitary(circuit_gates, n, d)
    return np.trace(V.conj().T @ U)


@given(seed=st.integers(0, 9999))
@settings(max_examples=5, deadline=120000)
def test_overlap_matches_exact(seed):
    """MPO cost should match exact Frobenius cost."""
    n, d = 4, 2
    tg = random_haar_gates(n, d, seed=seed)
    cg = random_haar_gates(n, d, seed=seed + 5000)

    cost, _ = compute_cost_and_grad(target_mpo(tg, n, d), cg, n, d)
    exact_cost = 2.0 - 2.0 * _exact_overlap(tg, cg, n, d).real / (2 ** n)

    assert abs(cost - exact_cost) < 1e-6, f"cost={cost}, exact={exact_cost}"


@given(seed=st.integers(0, 9999))
@settings(max_examples=3, deadline=120000)
def test_gradient_finite_difference(seed):
    """Analytic gradient should match finite differences."""
    n, d = 4, 2
    eps = 1e-5

    tg = random_haar_gates(n, d, seed=seed)
    cg = random_haar_gates(n, d, seed=seed + 5000)
    ta = target_mpo(tg, n, d)

    _, grad = compute_cost_and_grad(ta, cg, n, d)

    rng = np.random.RandomState(seed)
    g_idx = rng.randint(0, len(cg))
    direction = rng.randn(2, 2, 2, 2) + 1j * rng.randn(2, 2, 2, 2)

    gates_p, gates_m = list(cg), list(cg)
    gates_p[g_idx] = gates_p[g_idx] + eps * direction
    gates_m[g_idx] = gates_m[g_idx] - eps * direction

    fd = (_exact_overlap(tg, gates_p, n, d) - _exact_overlap(tg, gates_m, n, d)) / (2 * eps)
    analytic = np.einsum('abcd,abcd->', grad[g_idx].conj(), direction)

    rel_err = abs(fd - analytic) / max(abs(analytic), 1e-10)
    assert rel_err < 1e-3, f"gate={g_idx}, fd={fd:.6e}, analytic={analytic:.6e}"
