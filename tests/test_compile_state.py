"""Sanity + small end-to-end tests for `compile_state`."""

import numpy as np
import quimb.tensor as qtn
from hypothesis import given, settings
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import Statevector

from conftest import n_qubits_st, n_layers_st, seed_st
from tno_compiler.brickwall import random_brickwall
from tno_compiler.compile_state import (
    circuit_to_state_mps_arrays,
    compile_state,
    state_mps_to_target_arrays,
)


# --- MPS conversion: ⟨ψ_quimb_mps|·⟩ matches qiskit Statevector ---


@given(n=n_qubits_st, d=n_layers_st, seed=seed_st)
@settings(max_examples=10, deadline=15000)
def test_state_mps_matches_qiskit_statevector(n, d, seed):
    qc = random_brickwall(n, d, seed=seed)
    arrays, _bond = circuit_to_state_mps_arrays(qc, max_bond=2 ** n, cutoff=0.0)

    # Our arrays standardize to 3D (1, 2, br), (bl, 2, br), (bl, 2, 1)
    # in 'lpr' order. quimb's MatrixProductState defaults to 'lrp' and
    # wants 2D end sites, so squeeze + permute.
    quimb_arrays = [arrays[0][0], *arrays[1:-1], arrays[-1][:, :, 0]]
    psi_quimb = qtn.MatrixProductState(
        quimb_arrays, shape="lpr"
    ).to_dense().reshape(-1)
    psi_qiskit = Statevector(qc).data

    # quimb MPS to_dense flattens with site 0 = MSB. We mapped qiskit qubit q
    # to quimb site (n-1-q), so site 0 = qubit (n-1) = qiskit's MSB. That
    # matches qiskit's bit-ordering, so the dense vectors should be equal
    # element-by-element (up to a possible global phase, but circuits acting
    # on |0⟩ produce a fixed phase).
    overlap = np.vdot(psi_qiskit, psi_quimb)
    assert abs(abs(overlap) - 1.0) < 1e-9, f"|overlap|={abs(overlap)}"
    # phase should also match (no extra global phase introduced)
    assert abs(overlap - 1.0) < 1e-9, f"overlap={overlap}"


# --- |ψ⟩⟨0| MPO embedding gives the right amplitude ---


def test_target_mpo_amplitude_matches_state_amplitude():
    """For target_arrays = adjoint(|ψ⟩⟨0|), Tr(I·target_arrays) should equal
    ⟨ψ|0⟩^* (i.e., the (0...0) component of |ψ⟩, conjugated)."""
    np.random.seed(0)
    n = 4
    qc = random_brickwall(n, 2, seed=0)
    psi_arrays, _ = circuit_to_state_mps_arrays(qc, max_bond=2 ** n, cutoff=0.0)
    target_arrays = state_mps_to_target_arrays(psi_arrays)

    # Contract Tr(I · target) = sum over k=b of target[*, k, b, *].
    # Identity MPO is diag in (k,b). For our rank-1 target only b=0 is
    # nonzero, so this picks out target[*, 0, 0, *] = conj(MPS[*, 0, *]).
    val = np.array([[1.0]], dtype=complex)  # (br_prev=1, br_prev=1)
    for T in target_arrays:
        # T: (bl, k, b, br); trace k==b -> (bl, br)
        T_trace = np.einsum("ikkr->ir", T)
        val = val @ T_trace
    val = complex(val[0, 0])

    # ⟨0|ψ⟩
    psi_full = Statevector(qc).data
    expected = np.conj(psi_full[0])
    assert abs(val - expected) < 1e-10, f"got {val}, expected {expected}"


# --- End-to-end on a small, easy case: at depth = circuit's own depth,
# compile_state should reach (near-)perfect state fidelity. ---


def test_compile_state_recovers_easy_target():
    """A 4-qubit, 2-layer brickwall target should compile back to itself
    (state-fidelity > 1 - 1e-4) with an ansatz of equal depth."""
    n = 4
    qc = random_brickwall(n, 2, seed=42, first_odd=True)
    _, info = compile_state(
        qc,
        ansatz_depth=2,
        first_odd=True,
        max_iter=30,
        method="polar",
    )
    assert info["state_infidelity"] < 1e-4, info["state_infidelity"]
