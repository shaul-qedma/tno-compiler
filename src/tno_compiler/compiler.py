"""Compile a target `QuantumCircuit` into a shallower brickwall circuit.

The target is converted to an MPO (adjoint-reindexed into the
`rqcopt` convention) and then one of two optimizers runs to find the
brickwall gate set minimizing `‖U − V‖_F`:

- `method="polar"` (default) — Gibbs-Cincio polar-decomposition sweep.
  Uses the batched `polar_sweeps` with B=1 so the code path matches
  what `compile_ensemble` uses for B>1.
- `method="adam"` — Riemannian ADAM over U(4)^N.
"""

import numpy as np

from .brickwall import (
    brickwall_ansatz_gates, circuit_to_quimb_tn, gates_to_circuit,
    random_brickwall,
)
from .compress import tn_to_mpo
from .gradient import compute_cost_and_grad
from .mpo_ops import mpo_to_arrays
from .optim import polar_sweeps, riemannian_adam


def _qc_to_gate_tensors_local(qc):
    tensors = []
    for instruction in qc.data:
        mat = np.asarray(instruction.operation.to_matrix())
        if mat.shape == (4, 4):
            tensors.append(mat.reshape(2, 2, 2, 2))
    return tensors


def _perturbed_identity_gates(n_gates: int, scale: float, seed: int) -> list:
    """Build n_gates init gates as `exp(scale · H)` for random anti-Hermitian H.

    With scale=0 → all identities (V = I, V|0⟩ = |0⟩).
    With scale ~ 0.1 → small spread around identity, retains the
    "near-identity" basin while giving each seed a different starting point.
    Better than Haar-random for state-prep where identity is informative.
    """
    from scipy.linalg import expm
    rng = np.random.default_rng(seed)
    gates = []
    for _ in range(n_gates):
        Z = rng.standard_normal((4, 4)) + 1j * rng.standard_normal((4, 4))
        H = Z - Z.conj().T   # anti-Hermitian generator
        U = expm(scale * H)
        gates.append(U.reshape(2, 2, 2, 2).astype(complex))
    return gates


def build_target_arrays(target, max_bond=256, tol=1e-2):
    """Compress `target` to an MPO and produce the adjoint-reindexed
    arrays the compile's cost function expects.

    Returns (target_arrays, compress_error, actual_bond).
    `actual_bond` is the MPO's realized bond dimension after
    compression — always ≤ `max_bond` but often much smaller.
    """
    n_qubits = target.num_qubits
    tn = circuit_to_quimb_tn(target)
    mpo, compress_error = tn_to_mpo(
        tn, n_qubits, max_bond=max_bond,
        tol=tol * 0.1, norm="frobenius")
    actual_bond = max(mpo.bond_sizes())
    # Adjoint for rqcopt convention: the cost function expects V† contracted
    # against the circuit, so we swap bra/ket labels on the MPO.
    reindex_map = {f"k{i}": f"b{i}" for i in range(n_qubits)}
    reindex_map.update({f"b{i}": f"k{i}" for i in range(n_qubits)})
    target_arrays = mpo_to_arrays(mpo.conj().reindex(reindex_map))
    return target_arrays, compress_error, actual_bond


def compile_circuit(target, ansatz_depth, tol=1e-2,
                     max_bond=256, max_iter=500, lr=1e-3,
                     method="polar", first_odd=True,
                     init_gates=None, callback=None, seed=0):
    """Compile `target` to a brickwall of depth `ansatz_depth`.

    Args:
        target: qiskit QuantumCircuit defining V.
        ansatz_depth: number of brickwall layers in the compiled circuit.
        tol: Frobenius error budget (10% allocated to MPO compression,
            90% remaining for the optimizer).
        max_bond: hard cap on MPO bond dimension during compression
            and envelope merging.
        max_iter: number of polar / ADAM iterations.
        lr: only used for `method="adam"`.
        method: "polar" (default) or "adam".
        first_odd: whether the brickwall ansatz starts with an odd-pair
            layer (gates at (0,1), (2,3), …).
        init_gates: optional list of (2,2,2,2) initial gate tensors.
            If None, an identity init is used.
        callback: optional callable(step, cost) for progress reporting.
        seed: master RNG seed (init randomness is the caller's
            responsibility via `init_gates`).

    Returns:
        compiled: `QuantumCircuit` implementing the compiled brickwall.
        info: dict with keys `cost_history`, `compress_error`,
            `compile_error` (final cost), `gate_tensors` (list of
            (2,2,2,2) final gates).
    """
    n_qubits = target.num_qubits
    ansatz = brickwall_ansatz_gates(n_qubits, ansatz_depth, first_odd)

    target_arrays, compress_error, actual_bond = build_target_arrays(
        target, max_bond=max_bond, tol=tol)

    if init_gates is None:
        ng = sum(len(pairs) for _, pairs in ansatz)
        init_gates = [np.eye(4, dtype=complex).reshape(2, 2, 2, 2)] * ng

    if method == "polar":
        # B=1 wrapping: the batched path handles single-member compiles
        # uniformly, so there's one code path to maintain and test.
        opt_gates_list, hist_list = polar_sweeps(
            [init_gates], max_iter=max_iter, callback=callback,
            target_arrays=target_arrays, n_qubits=n_qubits,
            n_layers=ansatz_depth, max_bond=actual_bond,
            first_odd=first_odd, seed=seed)
        opt_gates = opt_gates_list[0]
        cost_history = hist_list[0]
    elif method == "adam":
        def cost_grad_fn(gates):
            return compute_cost_and_grad(
                target_arrays, gates, n_qubits, ansatz_depth,
                max_bond=actual_bond, first_odd=first_odd)

        opt_gates, cost_history = riemannian_adam(
            cost_grad_fn, init_gates, max_iter=max_iter, lr=lr,
            callback=callback)
    else:
        raise ValueError(f"Unknown method: {method}. Use 'polar' or 'adam'.")

    compiled = gates_to_circuit(opt_gates, n_qubits, ansatz)
    return compiled, {
        'cost_history': cost_history,
        'compress_error': compress_error,
        'compile_error': cost_history[-1] if cost_history else float('inf'),
        'gate_tensors': opt_gates,
    }


def _gates_for_depth(n_qubits: int, depth: int, first_odd: bool) -> int:
    odd = first_odd
    total = 0
    for _ in range(depth):
        total += (n_qubits // 2) if odd else ((n_qubits - 1) // 2)
        odd = not odd
    return total


def _warm_start_init(prev_depth: int, prev_gates: list, target_depth: int,
                       n_qubits: int, first_odd: bool) -> list:
    """Build an init for `target_depth` from a previous probe's optimized
    gates. Pads with identity (target_depth > prev_depth) or truncates
    (target_depth < prev_depth) at the brickwall layer boundary."""
    n_target = _gates_for_depth(n_qubits, target_depth, first_odd)
    if target_depth == prev_depth:
        return list(prev_gates)
    if target_depth > prev_depth:
        identity = np.eye(4, dtype=complex).reshape(2, 2, 2, 2)
        n_extra = n_target - len(prev_gates)
        return list(prev_gates) + [identity] * n_extra
    # target_depth < prev_depth: truncate to first target_depth layers'
    # worth of gates. Layer counts honor first_odd.
    return list(prev_gates[:n_target])


def compile_circuit_optimal(target, threshold, *, lo=1, hi=24, n_seeds=3,
                              tol=1e-2, max_bond=256, max_iter=200,
                              first_odd=True, seed=0, warm_start=True,
                              init_perturb_scale=0.1):
    """Binary-search the smallest brickwall depth `D*` such that the best
    of `n_seeds` polar compiles at depth `D*` reaches Frobenius
    ``compile_error <= threshold``.

    The n_seeds runs at each probe depth are dispatched as ONE batched
    `polar_sweeps` call (B=n_seeds), so they share the JIT cache and
    target-MPO build.

    With ``warm_start=True`` (default), one of the n_seeds slots at each
    probe is initialized from the BEST result of the closest previously
    probed depth (extended with identity layers if going deeper, truncated
    if going shallower). The other slots stay Haar-random for diversity.
    This eliminates the random-init lottery that otherwise inflates D*
    at tight tolerances.

    Returns:
        (D_opt, compiled, info, search) — see compile_circuit_optimal docs.
    """
    n_qubits = target.num_qubits
    target_arrays, compress_error, actual_bond = build_target_arrays(
        target, max_bond=max_bond, tol=tol)

    search: dict[int, dict] = {}  # depth → {seeds, gate_lists, histories}
    best_so_far: dict[int, list] = {}  # depth → best gates seen at that depth

    def _make_warm_start(d: int):
        """Return warm-start init derived from the closest previously-best
        probe, or None if no previous probe."""
        if not warm_start or not best_so_far:
            return None
        closest = min(best_so_far.keys(), key=lambda x: abs(x - d))
        return _warm_start_init(
            closest, best_so_far[closest], d, n_qubits, first_odd,
        )

    def probe(d: int):
        if d in search:
            return search[d]
        # Build init gate lists. Slot 0 may be the warm-start (if available);
        # remaining slots are Haar-random for diversity.
        init_gates_list = []
        ws = _make_warm_start(d)
        if ws is not None:
            init_gates_list.append(ws)
        n_gates_d = _gates_for_depth(n_qubits, d, first_odd)
        seeds_needed = n_seeds - len(init_gates_list)
        for s in range(seeds_needed):
            init_gates_list.append(_perturbed_identity_gates(
                n_gates_d, init_perturb_scale, seed + 1000 * d + s,
            ))
        # ONE batched polar call for all seeds.
        opt_gates_list, hist_list = polar_sweeps(
            init_gates_list, max_iter=max_iter,
            target_arrays=target_arrays, n_qubits=n_qubits,
            n_layers=d, max_bond=actual_bond, first_odd=first_odd,
            seed=seed + d)
        per_seed = []
        for s, hist in enumerate(hist_list):
            per_seed.append({
                'seed_idx': s,
                'is_warm_start': (s == 0 and ws is not None),
                'compile_error': float(hist[-1]) if hist else float('inf'),
            })
        record = {
            'depth': d,
            'gate_lists': opt_gates_list,
            'histories': hist_list,
            'per_seed': per_seed,
        }
        search[d] = record
        # Cache best gates at this depth for future warm-starts.
        best_idx = min(range(len(per_seed)),
                       key=lambda s: per_seed[s]['compile_error'])
        best_so_far[d] = opt_gates_list[best_idx]
        return record

    def best_error(record):
        return min(r['compile_error'] for r in record['per_seed'])

    optimal = None
    while lo <= hi:
        mid = (lo + hi) // 2
        rec = probe(mid)
        if best_error(rec) <= threshold:
            optimal = mid
            hi = mid - 1
        else:
            lo = mid + 1

    chosen = optimal if optimal is not None else min(search, key=lambda d: best_error(search[d]))
    rec = search[chosen]
    best_idx = min(range(len(rec['per_seed'])),
                   key=lambda s: rec['per_seed'][s]['compile_error'])
    best_gates = rec['gate_lists'][best_idx]
    best_hist = rec['histories'][best_idx]
    ansatz = brickwall_ansatz_gates(n_qubits, chosen, first_odd)
    compiled = gates_to_circuit(best_gates, n_qubits, ansatz)
    info = {
        'cost_history': best_hist,
        'compress_error': float(compress_error),
        'compile_error': float(best_hist[-1]) if best_hist else float('inf'),
        'gate_tensors': best_gates,
        'best_seed_idx': int(best_idx),
        'depth': chosen,
    }
    search_summary = {d: r['per_seed'] for d, r in sorted(search.items())}
    return optimal, compiled, info, search_summary
