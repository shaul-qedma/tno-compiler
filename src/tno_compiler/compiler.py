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
                              tol=1e-2, max_bond=256, max_iter=200, lr=1e-3,
                              method="polar", first_odd=True, drop_rate=0.0,
                              seed=0):
    """Binary-search the smallest brickwall depth `D*` such that
    ``compile_circuit(target, D*)`` reaches ``compile_error <= threshold``.

    Per probe runs `n_seeds` independent Haar-init compiles and takes the
    one with the lowest Frobenius error — this gives the strongest claim
    that "depth D can achieve `threshold`". Each compile uses the
    underlying `compile_circuit` (polar sweeps by default).

    Args:
        target: qiskit QuantumCircuit defining V.
        threshold: target Frobenius `compile_error`. The "optimal" D is
            the smallest in [lo, hi] whose best run is ≤ threshold.
        lo, hi: search bounds on brickwall depth.
        n_seeds: number of independent Haar init runs per probe.
        Other args: forwarded to `compile_circuit`.

    Returns:
        (D_opt, compiled, info, search) where:
          D_opt: optimal depth (int) or None if no D in [lo, hi] meets threshold
          compiled, info: best `compile_circuit` result at D_opt (or at the
              best probed depth if D_opt is None)
          search: dict mapping each probed depth → list of per-seed results.
    """
    n_qubits = target.num_qubits
    search: dict[int, list[dict]] = {}

    def best_at(d: int) -> dict:
        if d in search:
            return min(search[d], key=lambda r: r['compile_error'])
        runs = []
        for s in range(n_seeds):
            init = _qc_to_gate_tensors_local(
                random_brickwall(n_qubits, d, first_odd=first_odd,
                                 seed=seed + 1000 * d + s))
            compiled, info = compile_circuit(
                target, d, tol=tol, max_bond=max_bond, max_iter=max_iter,
                lr=lr, method=method, first_odd=first_odd,
                init_gates=init, drop_rate=drop_rate, seed=seed + d * n_seeds + s)
            runs.append({
                'seed_idx': s,
                'compile_error': float(info['compile_error']),
                'compress_error': float(info['compress_error']),
                'compiled': compiled,
                'info': info,
            })
        search[d] = runs
        return min(runs, key=lambda r: r['compile_error'])

    optimal = None
    while lo <= hi:
        mid = (lo + hi) // 2
        b = best_at(mid)
        if b['compile_error'] <= threshold:
            optimal = mid
            hi = mid - 1
        else:
            lo = mid + 1

    chosen = optimal if optimal is not None else min(
        search, key=lambda d: min(r['compile_error'] for r in search[d]))
    chosen_best = best_at(chosen)
    search_summary = {
        d: [{k: v for k, v in r.items() if k not in ('compiled', 'info')}
            for r in runs]
        for d, runs in sorted(search.items())
    }
    return optimal, chosen_best['compiled'], chosen_best['info'], search_summary
