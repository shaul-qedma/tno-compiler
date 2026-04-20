"""qiskit ↔ quimb conversion: forward and round-trip consistency.

Forward: `circuit_to_mpo → to_dense` of a qiskit circuit equals
`qiskit.Operator(qc).data`. This requires a qubit-ordering fix in
`circuit_to_quimb_tn` (see that function's docstring and
`memory/project_qiskit_quimb_convention.md`): two endianness gaps —
local within-gate MSB/LSB and global site-to-index flattening —
collapse into one transform, `sites = tuple(n-1-q for q in reversed(qubits))`.

Round-trip: qiskit → quimb → qiskit via `gates_to_circuit` preserves
`qiskit.Operator` when the intermediate gate tensors are placed at
ansatz positions matching the compile's convention.
"""

import numpy as np
from hypothesis import given, settings
from qiskit.circuit import QuantumCircuit
from qiskit.circuit.library import UnitaryGate
from qiskit.quantum_info import Operator, random_unitary

from conftest import n_qubits_st, n_layers_st, seed_st
from tno_compiler.brickwall import (
    brickwall_ansatz_gates, circuit_to_mpo, gates_to_circuit,
    random_brickwall,
)


# --- Forward: qiskit circuit → MPO → dense ----------------------------------

@given(n=n_qubits_st, d=n_layers_st, seed=seed_st)
@settings(max_examples=10, deadline=30000)
def test_mpo_matches_qiskit_on_random_brickwall(n, d, seed):
    qc = random_brickwall(n, d, seed=seed)
    mpo, _ = circuit_to_mpo(qc)
    V_mpo = np.asarray(mpo.to_dense())
    V_qiskit = np.asarray(Operator(qc).data)
    assert np.allclose(V_mpo, V_qiskit, atol=1e-10)


def test_mpo_matches_qiskit_with_1q_and_2q_gates():
    """1-qubit gates interleaved with 2-qubit gates. Tests that
    `contract=False` keeps 1-qubit gates as separate tensors rather
    than absorbing them into quimb's initial state."""
    qc = QuantumCircuit(4)
    qc.h(0)
    qc.append(UnitaryGate(random_unitary(4, seed=0).data), [0, 1])
    qc.append(UnitaryGate(random_unitary(4, seed=1).data), [2, 3])
    qc.append(UnitaryGate(random_unitary(4, seed=2).data), [1, 2])
    qc.h(3)
    mpo, _ = circuit_to_mpo(qc, tol=0.0)
    V_mpo = np.asarray(mpo.to_dense())
    V_qiskit = np.asarray(Operator(qc).data)
    assert np.allclose(V_mpo, V_qiskit, atol=1e-10)


# --- Round-trip: qiskit → gate tensors → qiskit via gates_to_circuit --------

def _extract_gates_in_ansatz_order(qc, ansatz):
    """Within each brickwall layer, qiskit's qubit-pair ordering is the
    reverse of the compile ansatz's site-pair ordering (because the
    compile uses site s ↔ qubit n-1-s). Extract qiskit gates layer by
    layer and reverse within each."""
    qiskit_gates = [
        np.asarray(instr.operation.to_matrix()).reshape(2, 2, 2, 2)
        for instr in qc.data if instr.operation.num_qubits == 2
    ]
    out, idx = [], 0
    for _, pairs in ansatz:
        layer = qiskit_gates[idx:idx + len(pairs)]
        out.extend(layer[::-1])
        idx += len(pairs)
    return out


@given(n=n_qubits_st, d=n_layers_st, seed=seed_st)
@settings(max_examples=10, deadline=30000)
def test_round_trip_preserves_operator(n, d, seed):
    qc_in = random_brickwall(n, d, first_odd=True, seed=seed)
    V_in = np.asarray(Operator(qc_in).data)

    ansatz = brickwall_ansatz_gates(n, d, first_odd=True)
    tensors = _extract_gates_in_ansatz_order(qc_in, ansatz)

    qc_out = gates_to_circuit(tensors, n, ansatz)
    V_out = np.asarray(Operator(qc_out).data)

    assert np.allclose(V_in, V_out, atol=1e-10)
