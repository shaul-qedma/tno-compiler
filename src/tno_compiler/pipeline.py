"""End-to-end Kalloor-style ensemble pipeline.

Given a target `QuantumCircuit` V and an ansatz depth d, compile B
independent brickwall approximations Uᵢ in a single batched polar
sweep, solve the Kalloor QP for optimal ensemble weights pᵢ over the
U(V†U) overlap Gram matrix, and return a weighted channel
`E = Σᵢ pᵢ Uᵢ · Uᵢ†` together with a diamond-norm error bound.
"""

import numpy as np
from tqdm import tqdm

from .brickwall import (
    brickwall_ansatz_gates, circuit_to_mpo, gates_to_circuit,
    random_brickwall,
)
from .compiler import build_target_arrays
from .ensemble import ensemble_qp
from .mpo_ops import mpo_overlap, mpo_to_arrays
from .optim import polar_sweeps


def compile_ensemble(target, ansatz_depth, n_circuits=5,
                      tol=1e-2, max_bond=256,
                      max_iter=200, lr=2e-2, first_odd=True, seed=0,
                      drop_rate=0.0):
    """Compile a weighted ensemble of brickwall circuits approximating V.

    All `n_circuits` members are optimized in a single batched call to
    `polar_sweeps` — one target-MPO build, one JIT trace, one XLA
    execution across the batch dim. Each member starts from an
    independent Haar-random brickwall init.

    Args:
        target: qiskit QuantumCircuit.
        ansatz_depth: brickwall depth for every member.
        n_circuits: ensemble size B.
        tol, max_bond: passed to MPO compression.
        max_iter: polar sweeps per member.
        lr: unused (polar method); kept for ADAM-compat signature.
        drop_rate: per-gate polar-sweep dropout probability.
        seed: master RNG seed; derives per-member init seeds.

    Returns dict with keys:
        weights, circuits, delta_ens, R, compress_error, diamond_bound,
        individual_frobs, qp_value.
    """
    n = target.num_qubits
    print(f"[ensemble] n={n}, ansatz_depth={ansatz_depth}, "
          f"{n_circuits} circuits, {max_iter} iters each (batched)",
          flush=True)

    target_arrays, compress_error, actual_bond = build_target_arrays(
        target, max_bond=max_bond, tol=tol)

    # Haar-random init per member. The `seed + 1000*i` spacing is
    # arbitrary — any deterministic, distinct-per-member mapping works.
    init_gates_list = [
        _qc_to_gate_tensors(random_brickwall(
            n, ansatz_depth, first_odd, seed=seed + 1000 * i))
        for i in range(n_circuits)
    ]

    opt_gates_list, histories = polar_sweeps(
        init_gates_list, max_iter=max_iter,
        target_arrays=target_arrays, n_qubits=n, n_layers=ansatz_depth,
        max_bond=actual_bond, first_odd=first_odd,
        drop_rate=drop_rate, seed=seed)

    ansatz = brickwall_ansatz_gates(n, ansatz_depth, first_odd)
    circuits = [gates_to_circuit(g, n, ansatz) for g in opt_gates_list]
    compile_errors = [h[-1] for h in histories]
    for i, err in enumerate(compile_errors):
        print(f"  circuit {i}: cost={err:.2e}", flush=True)

    # Target overlaps: the final polar-sweep cost gives
    # `cost = 2 − 2·Re Tr(V†Uᵢ)/d`, so `Re Tr(V†Uᵢ) = (1 − cost/2)·d`.
    # No need to re-contract — these are exact within the compile's
    # own convention.
    print("[ensemble] Computing pairwise overlaps...", flush=True)
    d = 2.0 ** n
    M = len(circuits)
    overlaps = np.array([(1.0 - compile_errors[i] / 2.0) * d for i in range(M)])

    # Pairwise Gram via MPO transfer-matrix contraction. The compiled
    # circuits are shallow (depth d ≪ target depth), so bond≈32 is a
    # safe floor; max_bond is an upper bound only used if the
    # approximation genuinely needs it.
    overlap_bond = max(32, max_bond)
    U_arrays = [mpo_to_arrays(
                    circuit_to_mpo(c, max_bond=overlap_bond, tol=tol)[0])
                for c in tqdm(circuits, desc="Converting to MPO")]
    gram = np.zeros((M, M))
    for i in range(M):
        for j in range(i, M):
            gram[i, j] = mpo_overlap(U_arrays[i], U_arrays[j]).real
            gram[j, i] = gram[i, j]

    weights, qp_val = ensemble_qp(gram, overlaps)

    # Certification. `delta_ens = ‖Σ pᵢUᵢ − V‖_F`, `R = maxᵢ ‖Uᵢ − V‖_F`
    # in absolute Frobenius. The Mixing Lemma (Kalloor Lemma 1) gives
    # `‖E − V‖_◇ ≤ 2·δ_ens + R²`. See docs/candidate_generation.md for
    # why this bound is vacuous at large n without partitioning.
    ensemble_frob = np.sqrt(max(qp_val + d, 0))
    individual_frobs = [np.sqrt(max(2 * d - 2 * overlaps[i], 0))
                        for i in range(M)]
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
    """Binary search for the minimum `ansatz_depth` with diamond_bound ≤ tol."""
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


def _qc_to_gate_tensors(qc):
    """Extract 2-qubit gates from a QuantumCircuit as (2,2,2,2) tensors."""
    tensors = []
    for instruction in qc.data:
        mat = np.asarray(instruction.operation.to_matrix())
        if mat.shape == (4, 4):
            tensors.append(mat.reshape(2, 2, 2, 2))
    return tensors
