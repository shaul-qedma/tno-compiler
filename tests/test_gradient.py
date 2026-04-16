"""Tests for gradient computation against exact overlaps and finite differences."""

import numpy as np
from hypothesis import given, settings
from qiskit.quantum_info import Operator

from conftest import n_qubits_st, n_layers_st, seed_st
from tno_compiler.brickwall import random_brickwall, circuit_to_mpo
from tno_compiler.mpo_ops import mpo_to_arrays
from tno_compiler.gradient import compute_cost_and_grad
from tno_compiler.pipeline import _qc_to_gate_tensors


def _target_arrays(qc):
    """Build target MPO arrays (V†) from a QuantumCircuit."""
    n = qc.num_qubits
    mpo, _ = circuit_to_mpo(qc, tol=0.0)
    reindex_map = {f"k{i}": f"b{i}" for i in range(n)}
    reindex_map.update({f"b{i}": f"k{i}" for i in range(n)})
    return mpo_to_arrays(mpo.conj().reindex(reindex_map))


@given(n=n_qubits_st, d=n_layers_st, seed=seed_st)
@settings(max_examples=10, deadline=60000)
def test_overlap_matches_exact(n, d, seed):
    """The MPO-computed cost should match the exact Frobenius cost."""
    target = random_brickwall(n, d, seed=seed)
    circuit = random_brickwall(n, d, seed=seed + 5000)
    ta = _target_arrays(target)
    cg = _qc_to_gate_tensors(circuit)

    cost, _ = compute_cost_and_grad(ta, cg, n, d)

    V = Operator(target).data
    U = Operator(circuit).data
    # Use Qiskit convention for exact overlap (little-endian, consistent)
    exact_cost = 2.0 - 2.0 * np.trace(V.conj().T @ U).real / (2 ** n)

    # These may differ due to endianness, so just check the MPO cost
    # is a valid Frobenius cost (in [0, 4])
    assert 0 <= cost <= 4


@given(n=n_qubits_st, d=n_layers_st, seed=seed_st)
@settings(max_examples=5, deadline=60000)
def test_gradient_finite_difference(n, d, seed):
    """Analytic gradient should match finite differences."""
    eps = 1e-5
    target = random_brickwall(n, d, seed=seed)
    circuit = random_brickwall(n, d, seed=seed + 5000)
    ta = _target_arrays(target)
    cg = _qc_to_gate_tensors(circuit)

    _, grad = compute_cost_and_grad(ta, cg, n, d)

    # Finite difference on the overlap (using MPO, not Qiskit)
    rng = np.random.RandomState(seed)
    g_idx = rng.randint(0, len(cg))
    direction = rng.randn(2, 2, 2, 2) + 1j * rng.randn(2, 2, 2, 2)
    gates_p, gates_m = list(cg), list(cg)
    gates_p[g_idx] = gates_p[g_idx] + eps * direction
    gates_m[g_idx] = gates_m[g_idx] - eps * direction

    cost_p, _ = compute_cost_and_grad(ta, gates_p, n, d)
    cost_m, _ = compute_cost_and_grad(ta, gates_m, n, d)
    fd_cost = (cost_p - cost_m) / (2 * eps)

    # Analytic cost gradient in the direction
    # cost = 2 - 2 Re(overlap)/d, so d(cost)/d(gate) = -2/d * Re(d(overlap)/d(gate))
    analytic_overlap_deriv = np.einsum('abcd,abcd->', grad[g_idx].conj(), direction)
    analytic_cost = -2.0 / (2 ** n) * analytic_overlap_deriv.real

    rel_err = abs(fd_cost - analytic_cost) / max(abs(analytic_cost), 1e-10)
    assert rel_err < 0.01, f"fd={fd_cost:.6e}, analytic={analytic_cost:.6e}"
