"""Transverse-Field Ising (TFI) Trotter circuit generation.

H = J Σ Z_i Z_{i+1} + g Σ X_i + h Σ Z_i

First-order Trotter step: exp(-i dt H) ≈ ZZ_odd · ZZ_even · X_all · Z_all
"""

import numpy as np
from qiskit.circuit import QuantumCircuit


def tfi_trotter_circuit(n_qubits, J, g, h, dt, steps, order=1):
    """Generate a Trotterized TFI circuit.

    Args:
        n_qubits: number of qubits.
        J: ZZ coupling (nearest-neighbor).
        g: transverse field (X).
        h: longitudinal field (Z).
        dt: Trotter step size.
        steps: number of Trotter steps.
        order: Trotter order (1 supported, 2/4 TODO).

    Returns:
        QuantumCircuit implementing exp(-i H t) with t = dt * steps.
    """
    if order != 1:
        raise NotImplementedError(f"Trotter order {order} not yet implemented (TODO)")
    qc = QuantumCircuit(n_qubits)
    for _ in range(steps):
        _trotter_step(qc, n_qubits, J, g, h, dt)
    return qc


def _trotter_step(qc, n, J, g, h, dt):
    """Append one first-order Trotter step to qc."""
    # ZZ interactions: odd pairs (0,1),(2,3),...
    for i in range(0, n - 1, 2):
        qc.rzz(2 * J * dt, i, i + 1)
    # ZZ interactions: even pairs (1,2),(3,4),...
    for i in range(1, n - 1, 2):
        qc.rzz(2 * J * dt, i, i + 1)
    # Single-qubit X rotations
    for i in range(n):
        qc.rx(2 * g * dt, i)
    # Single-qubit Z rotations
    for i in range(n):
        qc.rz(2 * h * dt, i)
