"""compile_state with a non-|0⟩ initial MPS."""

import numpy as np
import quimb.tensor as qtn
from qiskit.circuit import QuantumCircuit

from tno_compiler.brickwall import random_brickwall
from tno_compiler.compile_state import (
    circuit_to_state_mps_arrays,
    compile_state,
    state_mps_to_target_arrays,
    state_mps_to_target_arrays_general,
)


def _zero_mps_arrays(n: int) -> list[np.ndarray]:
    """|0…0⟩ as tno-convention 'lpr' MPS arrays — 3D with bond=1 everywhere."""
    out = []
    for i in range(n):
        a = np.zeros((1, 2, 1), dtype=complex)
        a[0, 0, 0] = 1.0
        out.append(a)
    return out


def test_general_embedding_with_zero_initial_matches_fastpath():
    """state_mps_to_target_arrays_general with |0⟩ MPS as initial must
    equal the fast-path state_mps_to_target_arrays (both encode
    |target⟩⟨0|, the rank-1 MPO)."""
    n = 4
    qc = random_brickwall(n, 2, seed=0)
    target_mps, _ = circuit_to_state_mps_arrays(qc, max_bond=2 ** n, cutoff=0.0)

    fast = state_mps_to_target_arrays(target_mps)
    general = state_mps_to_target_arrays_general(target_mps, _zero_mps_arrays(n))
    for i, (a, b) in enumerate(zip(fast, general)):
        assert a.shape == b.shape, f"site {i}: {a.shape} vs {b.shape}"
        assert np.allclose(a, b, atol=1e-12), f"site {i} differs"


def test_compile_state_with_zero_initial_matches_default():
    """compile_state with explicit |0⟩-MPS initial should give the same
    optimized state-fid as the default (initial=None)."""
    n = 4
    qc = random_brickwall(n, 3, seed=1, first_odd=True)
    target_mps, _ = circuit_to_state_mps_arrays(qc, max_bond=2 ** n, cutoff=0.0)

    _, info_default = compile_state(
        qc, ansatz_depth=3, target_state_mps=target_mps,
        max_iter=20, first_odd=True, seed=42,
    )
    _, info_explicit = compile_state(
        qc, ansatz_depth=3, target_state_mps=target_mps,
        initial_state_mps=_zero_mps_arrays(n),
        max_iter=20, first_odd=True, seed=42,
    )
    assert abs(info_default["state_infidelity"]
               - info_explicit["state_infidelity"]) < 1e-8


def test_compile_state_with_nontrivial_initial_recovers_identity():
    """If target = U·|φ⟩ where |φ⟩ = U_init·|0⟩, compile_state with
    initial=|φ⟩, ansatz_depth=depth(U) and warm-init=U's gates should
    recover state-fid 1 (the V=U solution)."""
    from tno_compiler.pipeline import _qc_to_gate_tensors
    n = 4
    # Build initial state via a small random brickwall.
    qc_init = random_brickwall(n, 1, seed=7, first_odd=True)
    initial_mps, _ = circuit_to_state_mps_arrays(qc_init, max_bond=2 ** n, cutoff=0.0)

    # Build target = U_target · |φ⟩ as MPS.
    qc_target_op = random_brickwall(n, 2, seed=8, first_odd=True)
    qc_full = qc_init.compose(qc_target_op)
    target_mps, _ = circuit_to_state_mps_arrays(qc_full, max_bond=2 ** n, cutoff=0.0)

    # Compile with initial=|φ⟩, ansatz=2, init_gates=U_target's gates.
    init = _qc_to_gate_tensors(qc_target_op)
    _, info = compile_state(
        qc_full, ansatz_depth=2, target_state_mps=target_mps,
        initial_state_mps=initial_mps,
        max_iter=30, first_odd=True, seed=0,
        init_gates=init,
    )
    # V=U_target satisfies V|φ⟩ = U_target·|φ⟩ = U_target·U_init|0⟩ = full|0⟩ = target.
    assert info["state_infidelity"] < 1e-6, info["state_infidelity"]
