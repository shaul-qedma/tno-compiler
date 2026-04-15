"""End-to-end compiler tests."""

import numpy as np
import pytest
from hypothesis import given, settings, strategies as st

from tno_compiler.compiler import compile_circuit
from tno_compiler.brickwall import random_haar_gates, gates_to_unitary, target_mpo


def test_self_compilation():
    """Compiling with the answer as init should stay near zero cost."""
    n, d = 4, 2
    tg = random_haar_gates(n, d, seed=42)
    ta = target_mpo(tg, n, d)

    gates, history = compile_circuit(ta, n, d, max_iter=10, lr=1e-3, init_gates=tg)
    assert history[-1] < 1e-4, f"Self-compile cost {history[-1]} too high"


def test_compilation_improves():
    """Compilation from identity init should reduce cost."""
    n, d = 4, 2
    tg = random_haar_gates(n, d, seed=42)
    ta = target_mpo(tg, n, d)

    gates, history = compile_circuit(ta, n, d, max_iter=50, lr=5e-3)
    assert history[-1] < history[0], "Optimization did not improve"


def test_compiled_overlap_matches_exact():
    """The MPO cost should match the exact matrix cost."""
    n, d = 4, 2
    tg = random_haar_gates(n, d, seed=42)
    ta = target_mpo(tg, n, d)

    gates, history = compile_circuit(ta, n, d, max_iter=30, lr=5e-3)

    V = gates_to_unitary(tg, n, d)
    U = gates_to_unitary(gates, n, d)
    exact_cost = 2 - 2 * np.trace(V.conj().T @ U).real / (2 ** n)

    assert abs(history[-1] - exact_cost) < 0.01, (
        f"MPO cost {history[-1]} != exact {exact_cost}")
