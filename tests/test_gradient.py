"""Tests for gradient computation, verified against finite differences and Qiskit."""

import numpy as np
import pytest
from hypothesis import given, settings, strategies as st
from qiskit.quantum_info import Operator

from tno_compiler.brickwall import (
    random_haar_gates, target_mpo, gates_to_unitary,
    partition_gates, layer_structure,
)
from tno_compiler.mpo_ops import quimb_mpo_to_arrays
from tno_compiler.gradient import compute_cost_and_grad


def _exact_overlap(target_gates, circuit_gates, n, d):
    V = gates_to_unitary(target_gates, n, d)
    U = gates_to_unitary(circuit_gates, n, d)
    return np.trace(V.conj().T @ U)


@given(seed=st.integers(0, 9999))
@settings(max_examples=5, deadline=120000)
def test_overlap_matches_exact(seed):
    """MPO-computed overlap should match exact Tr(V†U)."""
    n, d = 4, 2
    target_gates = random_haar_gates(n, d, seed=seed)
    circuit_gates = random_haar_gates(n, d, seed=seed + 5000)

    tmpo = target_mpo(target_gates, n, d)
    target_arrays = quimb_mpo_to_arrays(tmpo)

    cost, _ = compute_cost_and_grad(target_arrays, circuit_gates, n, d)

    exact_ov = _exact_overlap(target_gates, circuit_gates, n, d)
    exact_cost = 2.0 - 2.0 * exact_ov.real / (2 ** n)

    assert abs(cost - exact_cost) < 1e-6, (
        f"cost={cost}, exact_cost={exact_cost}")


@given(seed=st.integers(0, 9999))
@settings(max_examples=3, deadline=120000)
def test_gradient_finite_difference(seed):
    """Analytic gradient should match finite differences."""
    n, d = 4, 2
    eps = 1e-5

    target_gates = random_haar_gates(n, d, seed=seed)
    circuit_gates = random_haar_gates(n, d, seed=seed + 5000)

    tmpo = target_mpo(target_gates, n, d)
    target_arrays = quimb_mpo_to_arrays(tmpo)

    _, grad = compute_cost_and_grad(target_arrays, circuit_gates, n, d)

    rng = np.random.RandomState(seed)
    g_idx = rng.randint(0, len(circuit_gates))
    direction = rng.randn(2, 2, 2, 2) + 1j * rng.randn(2, 2, 2, 2)

    # Perturb one gate
    gates_p = list(circuit_gates)
    gates_m = list(circuit_gates)
    gates_p[g_idx] = gates_p[g_idx] + eps * direction
    gates_m[g_idx] = gates_m[g_idx] - eps * direction

    ov_p = _exact_overlap(target_gates, gates_p, n, d)
    ov_m = _exact_overlap(target_gates, gates_m, n, d)
    fd = (ov_p - ov_m) / (2 * eps)

    # Analytic: d/deps Tr(V†U) = Tr(env† direction)
    analytic = np.einsum('abcd,abcd->', grad[g_idx].conj(), direction)

    rel_err = abs(fd - analytic) / max(abs(analytic), 1e-10)
    assert rel_err < 1e-3, (
        f"gate={g_idx}, fd={fd:.6e}, analytic={analytic:.6e}, "
        f"rel_err={rel_err:.6e}")
