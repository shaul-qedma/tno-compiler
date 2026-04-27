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
                     init_gates=None, callback=None,
                     drop_rate=0.0, drop_rate_schedule=None, seed=0):
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
        drop_rate: per-gate polar-sweep dropout probability (0 disables).
            Polar method only.
        drop_rate_schedule: optional sweep-wise dropout schedule for the
            polar method. Supported kinds: `linear`, `cosine`, `constant`.
        seed: master RNG seed; drives dropout (init randomness is the
            caller's responsibility via `init_gates`).

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
            first_odd=first_odd,
            drop_rate=drop_rate, drop_rate_schedule=drop_rate_schedule,
            seed=seed)
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


def compile_circuit_optimal(target, threshold, *, lo=1, hi=24, n_seeds=3,
                              tol=1e-2, max_bond=256, max_iter=200,
                              first_odd=True, seed=0):
    """Binary-search the smallest brickwall depth `D*` such that the best
    of `n_seeds` polar compiles at depth `D*` reaches Frobenius
    ``compile_error <= threshold``.

    The n_seeds runs at each probe depth are dispatched as ONE batched
    `polar_sweeps` call (B=n_seeds), so they share the JIT cache and
    target-MPO build — much faster than n_seeds sequential
    `compile_circuit` calls.

    Args:
        target: qiskit QuantumCircuit defining V.
        threshold: target Frobenius `compile_error`. The "optimal" D is
            the smallest in [lo, hi] whose best (min over seeds) run is
            ≤ threshold.
        lo, hi: search bounds on brickwall depth.
        n_seeds: number of independent Haar init runs per probe (batched).
        Other args: forwarded to the underlying `polar_sweeps` /
            `build_target_arrays`.

    Returns:
        (D_opt, compiled, info, search) where:
          D_opt: optimal depth (int) or None if no D in [lo, hi] meets threshold
          compiled, info: best polar result at D_opt (or at the best
              probed depth if D_opt is None). info has keys
              `cost_history`, `compress_error`, `compile_error`, `gate_tensors`.
          search: dict mapping each probed depth → list of n_seeds
              {seed_idx, compile_error, compress_error}.
    """
    n_qubits = target.num_qubits
    target_arrays, compress_error, actual_bond = build_target_arrays(
        target, max_bond=max_bond, tol=tol)

    search: dict[int, dict] = {}  # depth → {seeds, gate_lists, histories}

    def probe(d: int):
        if d in search:
            return search[d]
        # Build n_seeds Haar-random init gate lists.
        init_gates_list = [
            _qc_to_gate_tensors_local(
                random_brickwall(n_qubits, d, first_odd=first_odd,
                                 seed=seed + 1000 * d + s))
            for s in range(n_seeds)
        ]
        # ONE batched polar call for all n_seeds.
        opt_gates_list, hist_list = polar_sweeps(
            init_gates_list, max_iter=max_iter,
            target_arrays=target_arrays, n_qubits=n_qubits,
            n_layers=d, max_bond=actual_bond, first_odd=first_odd,
            seed=seed + d)
        per_seed = []
        for s, hist in enumerate(hist_list):
            per_seed.append({
                'seed_idx': s,
                'compile_error': float(hist[-1]) if hist else float('inf'),
            })
        record = {
            'depth': d,
            'gate_lists': opt_gates_list,
            'histories': hist_list,
            'per_seed': per_seed,
        }
        search[d] = record
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
