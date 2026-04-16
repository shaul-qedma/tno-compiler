"""End-to-end compiler tests."""

import numpy as np
from hypothesis import given, settings, strategies as st

from conftest import n_qubits_st, seed_st
from tno_compiler.brickwall import random_brickwall, circuit_to_mpo
from tno_compiler.compiler import compile_circuit
from tno_compiler.pipeline import _qc_to_gate_tensors

n_layers_st = st.integers(1, 2)


def _mpo_dense(qc):
    return np.array(circuit_to_mpo(qc, tol=0.0)[0].to_dense())


@given(n=n_qubits_st, d=n_layers_st, seed=seed_st)
@settings(max_examples=6, deadline=60000)
def test_self_compilation(n, d, seed):
    """Initializing with the exact answer should preserve fidelity ~1."""
    target = random_brickwall(n, d, seed=seed)
    init = _qc_to_gate_tensors(target)
    compiled, _ = compile_circuit(target, d, max_iter=10, lr=1e-3, init_gates=init)
    V = _mpo_dense(target)
    U = _mpo_dense(compiled)
    fid = abs(np.trace(V.conj().T @ U)) ** 2 / (2**n) ** 2
    assert fid > 0.999


@given(n=n_qubits_st, d=n_layers_st, seed=seed_st)
@settings(max_examples=6, deadline=120000)
def test_compilation_reduces_cost(n, d, seed):
    """Compilation should reduce the Frobenius cost."""
    target = random_brickwall(n, d, seed=seed)
    _, info = compile_circuit(target, d, max_iter=100, lr=2e-2)
    assert info['compile_error'] < info['cost_history'][0]
