"""Segment the notebook circuit and compile each chunk via `compile_ensemble`,
running the chunks **in parallel** across worker processes.

Why segment: the full notebook circuit (38 qubits, 2q-depth ~hundreds at
steps≥40) is too deep to compile to a single shallow brickwall with a
useful fidelity. Slicing it into K shorter time-windows gives K easier
compile problems whose composition reproduces the original action.

Why parallel: each segment's compile_ensemble call is itself batched
(B = ensemble size) inside one process. Running K segments in K parallel
processes gives an additional layer of concurrency. On a single GPU the
processes share cusolver, so parallel speedup tops out around 2-4 workers
before contention dominates; on multi-GPU or CPU multi-core the linear
scaling holds further.

Usage:
  uv run python scripts/compile_notebook_parallel_segments.py \\
      --steps 20 --n-segments 4 --depth 6 --ensemble-size 8 \\
      --max-iter 60 --workers 4
"""

from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

# Reuse the loader/inliner from gpu_perf_compare.py.
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from gpu_perf_compare import _load_notebook_active, _NOTEBOOK_DIR, _pad_to_even


def _qc_to_qasm(qc) -> str:
    """qiskit ≥1.0 dropped `QuantumCircuit.qasm()`; use the new exporter."""
    from qiskit.qasm2 import dumps
    return dumps(qc)


def slice_circuit(qc, n_segments: int):
    """Split a qiskit circuit into `n_segments` consecutive op-window chunks
    operating on the same qubit register. Each chunk is a QuantumCircuit
    on the same n_qubits.

    Every segment gets a leading `rz(0, q)` on each qubit so even a chunk
    that touches no ops on some qubit still tags every site in the TN
    (quimb's MPO compress fails otherwise with `KeyError: 'I{q}'`).
    """
    from qiskit import QuantumCircuit
    n = qc.num_qubits
    n_ops = len(qc.data)
    boundaries = [int(round(i * n_ops / n_segments)) for i in range(n_segments + 1)]
    segments = []
    for k in range(n_segments):
        seg = QuantumCircuit(n)
        for q in range(n):
            seg.rz(0.0, q)
        for ci in qc.data[boundaries[k]:boundaries[k + 1]]:
            new_qs = [seg.qubits[qc.find_bit(q).index] for q in ci.qubits]
            seg.append(ci.operation, new_qs, [])
        segments.append(seg)
    return segments


def _compile_one_segment(seg_qasm: str, idx: int, depth: int,
                          ensemble_size: int, max_iter: int,
                          max_bond: int, tol: float, seed: int) -> dict:
    """Worker entry point. Takes a qasm string (so the segment is
    serializable across the process boundary), compiles it via
    `compile_ensemble`, returns a JSON-able summary dict and the
    compiled circuit qasm."""
    from qiskit import QuantumCircuit
    from tno_compiler.pipeline import compile_ensemble

    qc = QuantumCircuit.from_qasm_str(seg_qasm)
    t0 = time.time()
    result = compile_ensemble(
        qc, ansatz_depth=depth, n_circuits=ensemble_size,
        tol=tol, max_bond=max_bond, max_iter=max_iter,
        first_odd=True, seed=seed + 1009 * idx,
    )
    elapsed = time.time() - t0

    best_circ = result["circuits"][0]  # weighted ensemble's first elt; user can pick best
    return {
        "segment_idx": idx,
        "n_qubits": qc.num_qubits,
        "n_ops": len(qc.data),
        "compile_s": elapsed,
        "delta_ens": float(result.get("delta_ens", float("nan"))),
        "compress_error": float(result["compress_error"]),
        "individual_frobs": [float(x) for x in result["individual_frobs"]],
        "weights": [float(w) for w in result["weights"]],
        "best_circuit_qasm": _qc_to_qasm(best_circ),
        "ensemble_qasms": [_qc_to_qasm(c) for c in result["circuits"]],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=20,
                    help="Notebook target: circ_qasm2_qiskit_steps{steps}.qasm")
    ap.add_argument("--qasm", default=None,
                    help="Override the bundled notebook qasm path")
    ap.add_argument("--n-segments", type=int, default=4)
    ap.add_argument("--depth", type=int, default=6,
                    help="Brickwall depth per segment compile")
    ap.add_argument("--ensemble-size", type=int, default=8,
                    help="n_circuits for compile_ensemble per segment")
    ap.add_argument("--max-iter", type=int, default=60)
    ap.add_argument("--max-bond", type=int, default=64)
    ap.add_argument("--tol", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=1,
                    help="Process-level parallelism. >1 contends on single GPU.")
    ap.add_argument("--out", default="notebook_parallel_segments.json")
    args = ap.parse_args()

    qasm_path = (Path(args.qasm) if args.qasm
                 else _NOTEBOOK_DIR / f"circ_qasm2_qiskit_steps{args.steps}.qasm")
    if not qasm_path.exists():
        raise FileNotFoundError(qasm_path)

    qc = _pad_to_even(_load_notebook_active(qasm_path))
    n = qc.num_qubits
    target_2q = qc.depth(filter_function=lambda ci: ci.operation.num_qubits >= 2)
    print(
        f"[load] {qasm_path.name}  qubits={n}  ops={len(qc.data)}  "
        f"2q-depth={target_2q}",
        flush=True,
    )

    segments = slice_circuit(qc, args.n_segments)
    seg_specs = [
        (_qc_to_qasm(s), i, args.depth, args.ensemble_size, args.max_iter,
         args.max_bond, args.tol, args.seed)
        for i, s in enumerate(segments)
    ]
    for i, s in enumerate(segments):
        seg_2q = s.depth(filter_function=lambda ci: ci.operation.num_qubits >= 2)
        print(f"  segment {i}: ops={len(s.data)}  2q-depth={seg_2q}", flush=True)

    print(
        f"\n[compile] depth={args.depth}  ensemble={args.ensemble_size}  "
        f"max_iter={args.max_iter}  workers={args.workers}",
        flush=True,
    )

    t0 = time.time()
    results: list[dict] = [None] * args.n_segments
    if args.workers <= 1:
        for spec in seg_specs:
            r = _compile_one_segment(*spec)
            results[r["segment_idx"]] = r
            print(
                f"  segment {r['segment_idx']}: {r['compile_s']:.1f}s  "
                f"delta_ens={r['delta_ens']:.3e}  "
                f"min(frob)={min(r['individual_frobs']):.2e}",
                flush=True,
            )
    else:
        # Limit per-process GPU memory so K workers can coexist on one GPU.
        os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION",
                               f"{0.9 / args.workers:.3f}")
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futs = {pool.submit(_compile_one_segment, *spec): spec[1]
                    for spec in seg_specs}
            for fut in as_completed(futs):
                r = fut.result()
                results[r["segment_idx"]] = r
                print(
                    f"  segment {r['segment_idx']}: {r['compile_s']:.1f}s  "
                    f"delta_ens={r['delta_ens']:.3e}  "
                    f"min(frob)={min(r['individual_frobs']):.2e}",
                    flush=True,
                )
    total_wall = time.time() - t0
    sum_seg = sum(r["compile_s"] for r in results)
    print(
        f"\n[done] wall={total_wall:.1f}s  sum-of-segs={sum_seg:.1f}s  "
        f"speedup={sum_seg / max(total_wall, 1e-9):.2f}x",
        flush=True,
    )

    Path(args.out).write_text(json.dumps({
        "config": vars(args),
        "n_qubits": n,
        "target_2q_depth": int(target_2q),
        "total_wall_s": total_wall,
        "sum_segment_s": sum_seg,
        "segments": results,
    }, indent=2))
    print(f"[save] {args.out}", flush=True)


if __name__ == "__main__":
    main()
