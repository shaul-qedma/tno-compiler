"""End-to-end compiler tests.

Verifies that the compiled circuit reproduces the target by checking
process_fidelity = |Tr(V†U)|²/d² ≈ 1, computed from the big-endian
matrices (same convention as the optimizer).
"""

import numpy as np
from hypothesis import given, settings, strategies as st

from tno_compiler.compiler import compile_circuit
from tno_compiler.brickwall import random_haar_gates, target_mpo, gates_to_unitary

n_qubits_st = st.sampled_from([4, 6])
n_layers_st = st.integers(1, 2)
seed_st = st.integers(0, 9999)


def _fidelity(target_gates, compiled_gates, n, d):
    """|Tr(V†U)|²/d² from big-endian matrices."""
    V = gates_to_unitary(target_gates, n, d)
    U = gates_to_unitary(compiled_gates, n, d)
    d2 = (2 ** n) ** 2
    return abs(np.trace(V.conj().T @ U)) ** 2 / d2


@given(n=n_qubits_st, d=n_layers_st, seed=seed_st)
@settings(max_examples=6, deadline=60000)
def test_self_compilation(n, d, seed):
    """Initializing with the exact answer should preserve fidelity ~1."""
    tg = random_haar_gates(n, d, seed=seed)
    gates, _ = compile_circuit(target_mpo(tg, n, d), n, d,
                               max_iter=10, lr=1e-3, init_gates=tg)
    assert _fidelity(tg, gates, n, d) > 0.999


@given(n=n_qubits_st, d=n_layers_st, seed=seed_st)
@settings(max_examples=6, deadline=120000)
def test_fidelity_consistent_with_cost(n, d, seed):
    """Process fidelity should be consistent with the optimizer's cost.

    cost = 2 - 2·Re(Tr(V†U))/d  =>  Re(overlap)/d = 1 - cost/2
    fidelity = |overlap|²/d² >= (Re(overlap)/d)² = (1 - cost/2)²
    """
    tg = random_haar_gates(n, d, seed=seed)
    gates, history = compile_circuit(target_mpo(tg, n, d), n, d,
                                     max_iter=100, lr=5e-3)
    fid = _fidelity(tg, gates, n, d)
    lower_bound = max(0, 1 - history[-1] / 2) ** 2
    assert fid >= lower_bound - 1e-4, (
        f"fidelity {fid:.6f} < lower bound {lower_bound:.6f} "
        f"from cost {history[-1]:.6f}")
