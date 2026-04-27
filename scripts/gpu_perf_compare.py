"""Performance comparison: tno-compiler `compile_state` vs AQC-Tensor on
the same target circuit, fixed brickwall ansatz, fixed iteration budget.

Both methods are JAX-backed (tno via jax_ops, AQC-Tensor via quimb's
TNOptimizer with autodiff_backend="jax"), so the same machine runs both
on whatever JAX device is available (CPU or CUDA GPU).

Target options:
  - `notebook_steps5`/`steps10`/`steps20` — chronological-step truncations
    of `circ_qasm2_qiskit.qasm` (the 37-active-qubit notebook circuit,
    padded to 38 for even parity).
  - `tfi:N:STEPS` — synthetic TFI Trotter on N qubits, STEPS first-order
    Trotter steps, J=1, g=1, h=0.5.

Timings reported (wall):
  - target MPS build
  - one warm-up polar/L-BFGS iter (incl. JIT compile)
  - steady-state polar/L-BFGS iter
  - total optimization (max_iter sweeps/iters)
  - final state fidelity / overlap

GPU usage:
  Set `JAX_PLATFORMS=cuda` (or unset for default device order). The script
  prints `jax.default_backend()` at the start. For a real GPU run install
  `jaxlib` with CUDA wheels — see https://jax.readthedocs.io/en/latest/installation.html.

Usage:
  # CPU baseline:
  uv run python gpu_perf_compare.py --target tfi:14:8 --depth 6 --iters 20

  # GPU run (after installing jax[cuda12]):
  JAX_PLATFORMS=cuda uv run python gpu_perf_compare.py \
      --target notebook_steps5 --depth 8 --iters 20

  # apples-to-apples both methods, full circuit truncation:
  uv run python gpu_perf_compare.py --target notebook_steps10 --depth 10 \
      --iters 30 --max-bond 64 --n-seeds 1
"""

from __future__ import annotations

import argparse
import json
import os
import time
from functools import partial
from pathlib import Path

import numpy as np


# ---------- Diagnostics ----------


def _print_jax_diag() -> None:
    """Print JAX backend info up-front so the user sees CPU vs GPU."""
    import jax
    print(
        f"[jax] default_backend={jax.default_backend()!r}  "
        f"devices={jax.devices()}  jax_version={jax.__version__}",
        flush=True,
    )


# ---------- Target loading ----------


def load_target(spec: str):
    """Return (qiskit_circuit, padded_to_even, label).

    - "notebook_stepsN" → load circ_qasm2_qiskit_stepsN.qasm
      from the repo root, restrict to the active 37 qubits, pad to 38.
    - "tfi:N:STEPS" → tfi_trotter_circuit(N, J=1, g=1, h=0.5, dt=0.1, STEPS).
    """
    from qiskit.circuit import QuantumCircuit

    if spec.startswith("notebook_steps"):
        steps = spec.replace("notebook_steps", "")
        path = Path(
            f"/Users/shaulbarkan/Qedma/Code/transpile-gd/circ_qasm2_qiskit_steps{steps}.qasm"
        )
        from aqc_lib import load_active_circuit
        from tno_state_prep import pad_to_even
        qc = load_active_circuit(str(path))
        qc = pad_to_even(qc)
        return qc, f"notebook_steps{steps}"

    if spec.startswith("tfi:"):
        from tno_compiler.tfi import tfi_trotter_circuit
        _, n_str, steps_str = spec.split(":")
        n, steps = int(n_str), int(steps_str)
        qc = tfi_trotter_circuit(n, J=1.0, g=1.0, h=0.5, dt=0.1, steps=steps, order=1)
        return qc, f"tfi:n={n}:s={steps}"

    raise ValueError(f"unknown target spec: {spec}")


# ---------- tno-compiler timing ----------


def time_tno(qc, depth: int, iters: int, max_bond: int, n_seeds: int,
              seed: int) -> dict:
    from tno_compiler.brickwall import random_brickwall
    from tno_compiler.compile_state import (
        circuit_to_state_mps_arrays,
        state_mps_to_target_arrays_general,
    )
    from tno_compiler.optim import polar_sweeps
    from tno_compiler.pipeline import _qc_to_gate_tensors

    n = qc.num_qubits
    print(f"  [tno] n={n}  depth={depth}  iters={iters}  bond={max_bond}  B={n_seeds}",
          flush=True)

    # Stage 1: target MPS
    t0 = time.time()
    target_mps, target_bond = circuit_to_state_mps_arrays(
        qc, max_bond=max_bond, cutoff=1e-10,
    )
    t_mps = time.time() - t0
    print(f"  [tno] mps build: {t_mps:.2f}s  (realized bond {target_bond})",
          flush=True)

    target_arrays = state_mps_to_target_arrays_general(target_mps, None)

    init_gates_list = [
        _qc_to_gate_tensors(random_brickwall(n, depth, first_odd=True, seed=seed + s))
        for s in range(n_seeds)
    ]

    # Warm-up: 1 iter (JIT compile)
    t0 = time.time()
    polar_sweeps(init_gates_list, max_iter=1,
                  target_arrays=target_arrays, n_qubits=n,
                  n_layers=depth, max_bond=max_bond, first_odd=True, seed=seed)
    t_warmup = time.time() - t0
    print(f"  [tno] warm-up (1 iter incl JIT): {t_warmup:.2f}s", flush=True)

    # Steady-state run
    t0 = time.time()
    opt_gates_list, hist_list = polar_sweeps(
        init_gates_list, max_iter=iters,
        target_arrays=target_arrays, n_qubits=n, n_layers=depth,
        max_bond=max_bond, first_odd=True, seed=seed,
    )
    t_total = time.time() - t0
    per_iter = t_total / iters
    final_costs = [float(h[-1]) if h else float("inf") for h in hist_list]
    print(
        f"  [tno] {iters} iters: {t_total:.2f}s  ({per_iter*1000:.0f} ms/iter)  "
        f"final cost min={min(final_costs):.3e} max={max(final_costs):.3e}",
        flush=True,
    )

    # State fidelity
    from tno_compiler.brickwall import brickwall_ansatz_gates
    from tno_compiler.compile_state import _compute_state_overlap
    ansatz = brickwall_ansatz_gates(n, depth, first_odd=True)
    best_idx = min(range(len(opt_gates_list)),
                   key=lambda i: final_costs[i])
    overlap = _compute_state_overlap(opt_gates_list[best_idx], ansatz, target_mps, None)
    state_fid = float(abs(overlap) ** 2)
    print(f"  [tno] best state-fid: {state_fid:.6f}", flush=True)

    return {
        "method": "tno",
        "n_qubits": n,
        "depth": depth,
        "iters": iters,
        "max_bond": max_bond,
        "n_seeds": n_seeds,
        "target_bond": int(target_bond),
        "t_mps_s": t_mps,
        "t_warmup_s": t_warmup,
        "t_total_s": t_total,
        "ms_per_iter": per_iter * 1000,
        "state_fid_best": state_fid,
        "final_costs": final_costs,
    }


# ---------- AQC-Tensor timing ----------


def time_aqc(qc, depth: int, iters: int, max_bond: int, seed: int) -> dict:
    """AQC-Tensor: build random brickwall template, generate_ansatz_from_circuit,
    optimize via L-BFGS-B with `MaximizeStateFidelity` (jax-backed quimb)."""
    import quimb.tensor as qtn
    from qiskit.circuit import QuantumCircuit
    from qiskit.quantum_info import random_unitary
    from qiskit_addon_aqc_tensor import generate_ansatz_from_circuit
    from qiskit_addon_aqc_tensor.objective import MaximizeStateFidelity
    from qiskit_addon_aqc_tensor.simulation import tensornetwork_from_circuit
    from qiskit_addon_aqc_tensor.simulation.quimb import QuimbSimulator
    from scipy.optimize import minimize

    n = qc.num_qubits
    print(f"  [aqc] n={n}  depth={depth}  iters={iters}  bond={max_bond}",
          flush=True)

    sim = QuimbSimulator(
        partial(qtn.CircuitMPS, max_bond=max_bond, cutoff=1e-10),
        autodiff_backend="jax",
    )

    # Stage 1: target MPS
    t0 = time.time()
    target_circ = tensornetwork_from_circuit(qc, sim)
    t_mps = time.time() - t0
    print(f"  [aqc] mps build: {t_mps:.2f}s", flush=True)

    # Build a 1D brickwall template via random gates so generate_ansatz_from_circuit
    # produces a meaningful KAK+ZXZ ansatz with non-trivial init.
    rng = np.random.default_rng(seed)
    template = QuantumCircuit(n)
    odd = True
    for _ in range(depth):
        start = 0 if odd else 1
        for i in range(start, n - 1, 2):
            from qiskit.circuit.library import UnitaryGate
            template.append(
                UnitaryGate(np.asarray(random_unitary(4, seed=int(rng.integers(2**31))).data)),
                [i, i + 1],
            )
        odd = not odd
    ansatz, init_params = generate_ansatz_from_circuit(template, qubits_initially_zero=True)
    init_params = np.asarray(init_params, dtype=float)

    objective = MaximizeStateFidelity(target_circ, ansatz, sim)

    # Warm-up: one loss + grad
    t0 = time.time()
    v0, _ = objective.loss_function(init_params)
    t_warmup = time.time() - t0
    print(
        f"  [aqc] warm-up (loss+grad incl JIT): {t_warmup:.2f}s  init fid={1-float(v0):.4f}",
        flush=True,
    )

    # Optimize
    last = {"v": None}
    def f(x):
        v, g = objective.loss_function(x)
        last["v"] = float(v)
        return v, g

    t0 = time.time()
    res = minimize(
        f, init_params, method="L-BFGS-B", jac=True,
        options={"maxiter": iters, "ftol": 1e-12, "gtol": 1e-12},
    )
    t_total = time.time() - t0
    per_iter = t_total / max(res.nit, 1)
    final_fid = 1.0 - float(res.fun)
    print(
        f"  [aqc] {res.nit} L-BFGS iters (limit {iters}): {t_total:.2f}s  "
        f"({per_iter*1000:.0f} ms/iter)  final fid={final_fid:.6f}  msg={res.message}",
        flush=True,
    )

    return {
        "method": "aqc",
        "n_qubits": n,
        "depth": depth,
        "iters_limit": iters,
        "iters_actual": int(res.nit),
        "max_bond": max_bond,
        "t_mps_s": t_mps,
        "t_warmup_s": t_warmup,
        "t_total_s": t_total,
        "ms_per_iter": per_iter * 1000,
        "state_fid": final_fid,
        "scipy_msg": str(res.message),
    }


# ---------- Main ----------


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--target", required=True,
        help="Target circuit. e.g. 'tfi:14:8' or 'notebook_steps10'",
    )
    ap.add_argument("--depth", type=int, default=8,
                    help="Brickwall ansatz depth")
    ap.add_argument("--iters", type=int, default=20,
                    help="Iter budget for both methods")
    ap.add_argument("--max-bond", type=int, default=64,
                    help="Max MPS bond for both target and (tno) merging")
    ap.add_argument("--n-seeds", type=int, default=1,
                    help="Batch size for tno (AQC-Tensor is single-instance)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--methods", nargs="+", default=["tno", "aqc"],
        choices=["tno", "aqc"],
    )
    ap.add_argument("--out", default="gpu_perf_compare.json")
    args = ap.parse_args()

    _print_jax_diag()

    qc, label = load_target(args.target)
    target_2q_depth = qc.depth(filter_function=lambda ci: ci.operation.num_qubits >= 2)
    print(
        f"\n[target] {label}  qubits={qc.num_qubits}  ops={len(qc.data)}  "
        f"2q-depth={target_2q_depth}",
        flush=True,
    )

    results = []
    if "tno" in args.methods:
        print("\n[run] tno-compiler", flush=True)
        results.append(time_tno(qc, args.depth, args.iters, args.max_bond,
                                  args.n_seeds, args.seed))
    if "aqc" in args.methods:
        print("\n[run] AQC-Tensor", flush=True)
        try:
            results.append(time_aqc(qc, args.depth, args.iters, args.max_bond,
                                      args.seed))
        except Exception as e:
            print(f"  [aqc] FAILED: {type(e).__name__}: {str(e)[:300]}",
                  flush=True)
            results.append({"method": "aqc", "error": f"{type(e).__name__}: {e}"})

    # Comparison summary
    print("\n[summary]", flush=True)
    for r in results:
        if r.get("error"):
            print(f"  {r['method']:5s}: FAILED ({r['error'][:80]})", flush=True)
            continue
        print(
            f"  {r['method']:5s}  mps={r['t_mps_s']:.1f}s  warmup={r['t_warmup_s']:.1f}s  "
            f"total={r['t_total_s']:.1f}s  per-iter={r['ms_per_iter']:.0f}ms  "
            f"fid={r.get('state_fid', r.get('state_fid_best', float('nan'))):.4f}",
            flush=True,
        )

    Path(args.out).write_text(json.dumps({
        "target_spec": args.target,
        "target_label": label,
        "n_qubits": qc.num_qubits,
        "target_2q_depth": int(target_2q_depth),
        "depth": args.depth,
        "iters": args.iters,
        "max_bond": args.max_bond,
        "n_seeds": args.n_seeds,
        "jax_backend": __import__("jax").default_backend(),
        "results": results,
    }, indent=2))
    print(f"[save] {args.out}", flush=True)


if __name__ == "__main__":
    main()
