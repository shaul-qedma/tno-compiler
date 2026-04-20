"""End-to-end Kalloor ensemble pipeline.

Target: a QuantumCircuit at any depth.
Output: a weighted ensemble of shallower brickwall circuits with a
certified diamond distance bound.
"""

import numpy as np
from tqdm import tqdm
from .brickwall import random_brickwall, circuit_to_mpo
from .compiler import compile_circuit
from .ensemble import ensemble_qp
from .mpo_ops import mpo_to_arrays, mpo_overlap


def compile_ensemble(target, ansatz_depth, n_circuits=5,
                     tol=1e-2, max_bond=256,
                     max_iter=200, lr=2e-2, first_odd=True, seed=0,
                     drop_rate=0.0):
    """Compile an ensemble of brickwall circuits approximating a target.

    Args:
        target: qiskit QuantumCircuit.
        ansatz_depth: depth of each compiled circuit.
        n_circuits: number of circuits in the ensemble.
        tol, max_bond: passed to compile_circuit.
        max_iter, lr: optimizer parameters.
        seed: base seed for random initialization.
        drop_rate: per-gate polar-sweep dropout probability passed to
            compile_circuit. Each candidate uses a distinct drop_seed
            derived from `seed` to decorrelate ensemble members.

    Returns dict with weights, circuits, diamond_bound, etc.
    """
    n = target.num_qubits
    print(f"[ensemble] n={n}, ansatz_depth={ansatz_depth}, "
          f"{n_circuits} circuits, {max_iter} iters each", flush=True)

    # Compile M circuits from different random initializations
    circuits = []
    gate_tensors_list = []
    compile_errors = []
    compress_error = 0.0
    for i in tqdm(range(n_circuits), desc="Compiling circuits"):
        init_qc = random_brickwall(n, ansatz_depth, first_odd, seed=seed + 1000 * i)
        init_tensors = _qc_to_gate_tensors(init_qc)
        compiled, info = compile_circuit(
            target, ansatz_depth,
            tol=tol, max_bond=max_bond, max_iter=max_iter, lr=lr,
            first_odd=first_odd, init_gates=init_tensors, callback=None,
            drop_rate=drop_rate, seed=seed + 1000 * i)
        circuits.append(compiled)
        gate_tensors_list.append(info['gate_tensors'])
        compile_errors.append(info['compile_error'])
        compress_error = info['compress_error']
        print(f"  circuit {i}: cost={info['compile_error']:.2e}", flush=True)

    # Target overlaps from compile costs (already computed, exact)
    print(f"[ensemble] Computing overlaps...", flush=True)
    d = 2.0 ** n
    M = len(circuits)
    overlaps = np.array([(1.0 - compile_errors[i] / 2.0) * d for i in range(M)])

    # Pairwise overlaps via MPO contraction
    # Compiled circuits approximate a low-bond target, so modest bond suffices
    overlap_bond = max(32, max_bond)
    U_arrays = [mpo_to_arrays(circuit_to_mpo(c, max_bond=overlap_bond, tol=tol)[0])
                for c in tqdm(circuits, desc="Converting to MPO")]
    gram = np.zeros((M, M))
    for i in range(M):
        for j in range(i, M):
            gram[i, j] = mpo_overlap(U_arrays[i], U_arrays[j]).real
            gram[j, i] = gram[i, j]
    print(f"[ensemble] Overlaps done. Solving QP...", flush=True)

    # Solve QP
    weights, qp_val = ensemble_qp(gram, overlaps)
    print(f"[ensemble] QP solved. Certifying...", flush=True)

    # Certification
    d = 2.0 ** n
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
