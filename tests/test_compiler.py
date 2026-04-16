"""End-to-end compiler tests with QuantumCircuit interface."""

import numpy as np
from hypothesis import given, settings, strategies as st
from qiskit.quantum_info import Operator, process_fidelity

from conftest import n_qubits_st, seed_st
from tno_compiler.brickwall import random_brickwall
from tno_compiler.compiler import compile_circuit
from tno_compiler.pipeline import _qc_to_gate_tensors

n_layers_st = st.integers(1, 2)


@given(n=n_qubits_st, d=n_layers_st, seed=seed_st)
@settings(max_examples=6, deadline=60000)
def test_self_compilation(n, d, seed):
    """Initializing with the exact answer should preserve fidelity ~1."""
    target = random_brickwall(n, d, seed=seed)
    init = _qc_to_gate_tensors(target)
    compiled, _ = compile_circuit(target, d, max_iter=10, lr=1e-3, init_gates=init)
    fid = process_fidelity(Operator(compiled), Operator(target))
    assert fid > 0.999


@given(n=n_qubits_st, d=n_layers_st, seed=seed_st)
@settings(max_examples=6, deadline=120000)
def test_compilation_reduces_cost(n, d, seed):
    """Compilation should reduce the Frobenius cost."""
    target = random_brickwall(n, d, seed=seed)
    _, info = compile_circuit(target, d, max_iter=100, lr=5e-3)
    assert info['compile_error'] < info['cost_history'][0]
