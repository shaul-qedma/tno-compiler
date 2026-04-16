"""End-to-end compiler tests."""

import numpy as np
from hypothesis import given, settings, strategies as st

from conftest import n_qubits_st, seed_st
from tno_compiler.compiler import compile_circuit
from tno_compiler.brickwall import random_haar_gates, target_mpo, circuit_to_mpo

n_layers_st = st.integers(1, 2)  # tighter for compiler (slower)


def _fidelity(tg, compiled_gates, n, d):
    """|Tr(V†U)|²/d² via dense matrices (testing only)."""
    V = np.array(circuit_to_mpo(tg, n, d)[0].to_dense())
    U = np.array(circuit_to_mpo(compiled_gates, n, d)[0].to_dense())
    return abs(np.trace(V.conj().T @ U)) ** 2 / (2 ** n) ** 2


@given(n=n_qubits_st, d=n_layers_st, seed=seed_st)
@settings(max_examples=6, deadline=60000)
def test_self_compilation(n, d, seed):
    """Initializing with the exact answer should preserve fidelity ~1."""
    tg = random_haar_gates(n, d, seed=seed)
    gates, _ = compile_circuit(target_mpo(tg, n, d)[0], n, d,
                               max_iter=10, lr=1e-3, init_gates=tg)
    assert _fidelity(tg, gates, n, d) > 0.999


@given(n=n_qubits_st, d=n_layers_st, seed=seed_st)
@settings(max_examples=6, deadline=120000)
def test_fidelity_consistent_with_cost(n, d, seed):
    """Process fidelity should be consistent with the optimizer's cost."""
    tg = random_haar_gates(n, d, seed=seed)
    gates, history = compile_circuit(target_mpo(tg, n, d)[0], n, d,
                                     max_iter=100, lr=5e-3)
    fid = _fidelity(tg, gates, n, d)
    lower_bound = max(0, 1 - history[-1] / 2) ** 2
    assert fid >= lower_bound - 1e-4
