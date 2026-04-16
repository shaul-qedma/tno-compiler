"""End-to-end Kalloor ensemble pipeline.

Target: a QuantumCircuit at any depth.
Output: a weighted ensemble of shallower brickwall circuits with a
certified diamond distance bound.
"""

import numpy as np
from qiskit.quantum_info import Operator

from .brickwall import random_brickwall, circuit_to_mpo
from .compiler import compile_circuit
from .ensemble import ensemble_qp


def compile_ensemble(target, ansatz_depth, n_circuits=5,
                     tol=1e-2, compress_fraction=0.0, max_bond=None,
                     max_iter=200, lr=5e-3, first_odd=True, seed=0):
    """Compile an ensemble of brickwall circuits approximating a target.

    Args:
        target: qiskit QuantumCircuit.
        ansatz_depth: depth of each compiled circuit.
        n_circuits: number of circuits in the ensemble.
        tol, compress_fraction, max_bond: passed to compile_circuit.
        max_iter, lr: optimizer parameters.
        seed: base seed for random initialization.

    Returns dict with weights, circuits, diamond_bound, etc.
    """
    n = target.num_qubits

    # Compile M circuits from different random initializations
    circuits = []
    gate_tensors_list = []
    compile_errors = []
    compress_error = 0.0
    for i in range(n_circuits):
        # Perturbed identity init: small random rotation for diversity
        init_tensors = _perturbed_identity(n, ansatz_depth, first_odd,
                                           scale=0.01, seed=seed + 1000 * i)
        compiled, info = compile_circuit(
            target, ansatz_depth, compress_fraction=compress_fraction,
            tol=tol, max_bond=max_bond, max_iter=max_iter, lr=lr,
            first_odd=first_odd, init_gates=init_tensors, callback=None)
        circuits.append(compiled)
        gate_tensors_list.append(info['gate_tensors'])
        compile_errors.append(info['compile_error'])
        compress_error = info['compress_error']

    # Gram matrix and target overlaps (dense, small n only)
    V = Operator(target).data
    Us = [Operator(c).data for c in circuits]
    M = len(Us)
    gram = np.zeros((M, M))
    overlaps = np.zeros(M)
    for i in range(M):
        overlaps[i] = np.trace(Us[i].conj().T @ V).real
        for j in range(M):
            gram[i, j] = np.trace(Us[i].conj().T @ Us[j]).real

    # Solve QP
    weights, qp_val = ensemble_qp(gram, overlaps)

    # Certification
    d = 2 ** n
    ensemble_frob = np.sqrt(max(qp_val + d, 0))
    individual_frobs = [np.sqrt(max(2 * d - 2 * overlaps[i], 0)) for i in range(M)]
    R = max(individual_frobs[i] for i in range(M) if weights[i] > 1e-10)
    delta_ens = ensemble_frob + compress_error
    R_total = R + compress_error

    return {
        'weights': weights,
        'circuits': circuits,
        'delta_ens': delta_ens,
        'R': R_total,
        'compress_error': compress_error,
        'diamond_bound': 2 * delta_ens + R_total ** 2,
        'individual_frobs': individual_frobs,
        'qp_value': qp_val,
    }


def find_min_depth(target, tol, max_depth=20, **kwargs):
    """Binary search for minimum ansatz_depth achieving diamond_bound ≤ tol."""
    lo, hi = 1, max_depth
    best = None
    while lo <= hi:
        mid = (lo + hi) // 2
        result = compile_ensemble(target, mid, tol=tol, **kwargs)
        if result['diamond_bound'] <= tol:
            best = (mid, result)
            hi = mid - 1
        else:
            lo = mid + 1
    if best is None:
        return max_depth, compile_ensemble(target, max_depth, tol=tol, **kwargs)
    return best


def _perturbed_identity(n_qubits, n_layers, first_odd, scale=0.1, seed=0):
    """Identity gates with small random perturbation for ensemble diversity."""
    rng = np.random.RandomState(seed)
    from .brickwall import brickwall_ansatz_gates
    structure = brickwall_ansatz_gates(n_qubits, n_layers, first_odd)
    tensors = []
    for _, pairs in structure:
        for _ in pairs:
            # Small anti-Hermitian perturbation → near-identity unitary
            A = scale * (rng.randn(4, 4) + 1j * rng.randn(4, 4))
            A = A - A.conj().T  # anti-Hermitian
            from scipy.linalg import expm
            U = expm(A)
            tensors.append(U.reshape(2, 2, 2, 2))
    return tensors


def _qc_to_gate_tensors(qc):
    """Extract (2,2,2,2) gate tensors from a QuantumCircuit."""
    tensors = []
    for instruction in qc.data:
        gate = instruction.operation
        mat = np.array(gate.to_matrix())
        if mat.shape == (4, 4):
            tensors.append(mat.reshape(2, 2, 2, 2))
    return tensors
