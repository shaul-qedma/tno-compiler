"""End-to-end Kalloor-style ensemble pipeline.

Given a target `QuantumCircuit` V and an ansatz depth d, compile B
independent brickwall approximations Uᵢ in a single batched polar
sweep, solve the Kalloor QP for optimal ensemble weights pᵢ over the
U(V†U) overlap Gram matrix, and return a weighted channel
`E = Σᵢ pᵢ Uᵢ · Uᵢ†` together with a diamond-norm error bound.
"""

import numpy as np
from scipy.linalg import expm
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
                      repel_lambda=0.0, top_k=None, n_pairs=0,
                      perturb_scale=0.1):
    """Compile a weighted ensemble of brickwall circuits approximating V.

    All `n_circuits` members are optimized in a single batched call to
    `polar_sweeps` — one target-MPO build, one JIT trace, one XLA
    execution across the batch dim. Each member starts from an
    independent Haar-random brickwall init.

    Optional paired-opposite perturbation (Kalloor methods §Ensemble
    Generation) — if `n_pairs > 0`, after the initial compile we
    select the `top_k` best members (by compile cost), generate
    `n_pairs` paired perturbations around each, and run the QP on the
    expanded set `{top_k seeds} ∪ {2·n_pairs·top_k perturbed members}`.
    The pairs are symmetric around each seed (`G·exp(+sH)`, `G·exp(-sH)`
    per gate), so the convex hull contains V's neighborhood rather
    than just scattered basins — the geometric condition Kalloor's
    quadratic reduction actually needs.

    Args:
        target: qiskit QuantumCircuit.
        ansatz_depth: brickwall depth for every member.
        n_circuits: ensemble size B for the initial Haar-random compile.
        tol, max_bond: passed to MPO compression.
        max_iter: polar sweeps per member.
        lr: unused (polar method); kept for ADAM-compat signature.
        repel_lambda: diversity regularization strength in the batched
            polar sweep. Each gate is pushed away from the mean of its
            siblings' gates at the same position. 0 disables.
        seed: master RNG seed; derives per-member init seeds.
        top_k: if set (and `n_pairs > 0`), select the top-K compiled
            members by final cost to seed perturbation. Default None
            means "don't perturb".
        n_pairs: number of paired perturbations per selected seed.
            Each pair contributes 2 members to the QP.
        perturb_scale: ε for the perturbation `exp(±ε·H)` per gate,
            where H is a random anti-Hermitian 4×4.

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
        seed=seed, repel_lambda=repel_lambda)

    ansatz = brickwall_ansatz_gates(n, ansatz_depth, first_odd)
    compile_errors = [h[-1] for h in histories]
    for i, err in enumerate(compile_errors):
        print(f"  circuit {i}: cost={err:.2e}", flush=True)

    # Optional paired-opposite perturbation of the top-K compiled members.
    if n_pairs > 0 and top_k is not None and top_k > 0:
        top_idx = sorted(range(len(opt_gates_list)),
                         key=lambda i: compile_errors[i])[:top_k]
        print(f"[ensemble] Expanding top-{top_k} seeds with {n_pairs} "
              f"pair(s) each (scale={perturb_scale}) → "
              f"{top_k + top_k * 2 * n_pairs} total members.",
              flush=True)
        seed_gate_sets = [opt_gates_list[i] for i in top_idx]
        rng = np.random.default_rng(seed + 9999)
        all_gate_sets = list(seed_gate_sets)
        for gates in seed_gate_sets:
            for _ in range(n_pairs):
                plus, minus = _perturb_gates_paired(gates, perturb_scale, rng)
                all_gate_sets.append(plus)
                all_gate_sets.append(minus)
    else:
        all_gate_sets = opt_gates_list

    circuits = [gates_to_circuit(g, n, ansatz) for g in all_gate_sets]

    # Gram + target overlaps via MPO transfer-matrix contraction. The
    # compiled circuits are shallow (depth d ≪ target depth), so
    # bond ≈ 32 is a safe floor; max_bond bounds genuine bond growth.
    print("[ensemble] Computing overlaps...", flush=True)
    d = 2.0 ** n
    overlap_bond = max(32, max_bond)
    V_arrays = mpo_to_arrays(
        circuit_to_mpo(target, max_bond=overlap_bond, tol=tol)[0])
    U_arrays = [mpo_to_arrays(
                    circuit_to_mpo(c, max_bond=overlap_bond, tol=tol)[0])
                for c in tqdm(circuits, desc="Converting to MPO")]
    M = len(circuits)
    overlaps = np.array([mpo_overlap(U_arrays[i], V_arrays).real
                         for i in range(M)])
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


def compile_ensemble_optimal(
    target, threshold, n_circuits=20, *,
    search_n_seeds=3, search_lo=1, search_hi=24, search_max_iter=100,
    search_tol=None, search_max_bond=None,
    tol=1e-2, max_bond=256, max_iter=200, lr=2e-2,
    first_odd=True, seed=0,
    repel_lambda=0.0, top_k=None, n_pairs=0, perturb_scale=0.1,
):
    """Two-stage compile: cheap-search the smallest depth meeting a
    single-circuit Frobenius threshold, then run the full ensemble at
    that depth.

    Stage 1 uses `compile_circuit_optimal` with a small search batch
    (`search_n_seeds` members per probe) to find `D*` cheaply.
    Stage 2 runs `compile_ensemble` at `D*` with `n_circuits` members
    (typically ≥20 — required to see real Kalloor-QP ensemble gains).

    Args:
        target: qiskit QuantumCircuit.
        threshold: single-circuit Frobenius `compile_error` target for
            the binary search.
        n_circuits: ensemble size B for the final QP (Stage 2).
        search_n_seeds: B_search for each probe in Stage 1.
        search_lo, search_hi: depth bounds for the binary search.
        search_max_iter: polar sweeps per Stage-1 probe.
        search_tol, search_max_bond: optional separate MPO compression
            settings for Stage 1; default to (tol, max_bond).
        Other args: forwarded to `compile_ensemble` for Stage 2.

    Returns the same dict as `compile_ensemble`, with extra keys:
        optimal_depth: D* (or None if no D in [lo, hi] meets threshold —
            in that case Stage 2 runs at the highest probed depth).
        search: per-probe per-seed Frobenius errors from Stage 1.
    """
    from .compiler import compile_circuit_optimal

    s_tol = search_tol if search_tol is not None else tol
    s_max_bond = search_max_bond if search_max_bond is not None else max_bond
    print(
        f"[ensemble-opt] Stage 1: binary-search D in [{search_lo}, {search_hi}] "
        f"for threshold {threshold:.2e}  (B_search={search_n_seeds}, "
        f"max_iter={search_max_iter})",
        flush=True,
    )
    D_opt, _, info, search = compile_circuit_optimal(
        target, threshold,
        lo=search_lo, hi=search_hi, n_seeds=search_n_seeds,
        tol=s_tol, max_bond=s_max_bond, max_iter=search_max_iter,
        first_odd=first_odd, seed=seed,
    )
    if D_opt is None:
        # Pick the depth with the lowest best-error seen — Stage 2 still
        # gives the user *something* (a useful ensemble at the closest
        # we got), and they can read off `optimal_depth=None` to know
        # the threshold wasn't met.
        chosen = info["depth"]
        print(
            f"[ensemble-opt] Stage 1: NO depth in [{search_lo}, {search_hi}] "
            f"reached {threshold:.2e}; falling back to D={chosen} "
            f"(best error seen = {info['compile_error']:.2e})",
            flush=True,
        )
    else:
        chosen = D_opt
        print(
            f"[ensemble-opt] Stage 1: D* = {D_opt} meets threshold "
            f"({info['compile_error']:.2e} ≤ {threshold:.2e})",
            flush=True,
        )

    print(
        f"[ensemble-opt] Stage 2: compile_ensemble at D={chosen} "
        f"with {n_circuits} circuits (max_iter={max_iter})",
        flush=True,
    )
    result = compile_ensemble(
        target, ansatz_depth=chosen, n_circuits=n_circuits,
        tol=tol, max_bond=max_bond, max_iter=max_iter, lr=lr,
        first_odd=first_odd, seed=seed,
        repel_lambda=repel_lambda, top_k=top_k, n_pairs=n_pairs,
        perturb_scale=perturb_scale,
    )
    result["optimal_depth"] = D_opt
    result["chosen_depth"] = chosen
    result["search"] = search
    return result


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


def _perturb_gates_paired(gate_tensors, scale, rng):
    """Generate a ±-paired perturbation of a brickwall gate set.

    For each 2-qubit gate `G`, draw a random anti-Hermitian 4×4 `H`
    and produce the two members `G · exp(+s·H)` and `G · exp(-s·H)`.
    The pair's arithmetic mean is `G · cosh(s·H) ≈ G` for small `s`,
    so its convex hull sits symmetrically around the seed — the
    geometric precondition for Kalloor's quadratic ReWEE reduction.
    Different gates get independent `H` draws; different pair indices
    get independent draws too.
    """
    plus, minus = [], []
    for G in gate_tensors:
        G_mat = np.asarray(G).reshape(4, 4)
        X = (rng.standard_normal((4, 4))
             + 1j * rng.standard_normal((4, 4)))
        H = X - X.conj().T  # anti-Hermitian generator
        step = scale * H
        plus.append((G_mat @ expm(step)).reshape(2, 2, 2, 2))
        minus.append((G_mat @ expm(-step)).reshape(2, 2, 2, 2))
    return plus, minus
