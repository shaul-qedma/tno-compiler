"""Segment the notebook circuit and compile **all chunks together** as one
batched polar_sweeps call.

Each segment becomes one ensemble member-group within a single
B = K_segments × n_seeds batched optimization. Per-segment target
MPOs are zero-padded to a common bond and stacked along the leading
B dim, so every cusolver SVD / merge / contract op runs once per
iter for the whole stack — no multiprocessing, no GPU contention,
one JIT graph for K segments at once.

Usage:
  uv run python scripts/compile_notebook_batched_segments.py \\
      --steps 20 --n-segments 4 --depth 6 --n-seeds 4 \\
      --max-iter 60 --max-bond 64
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path


def _assert_jax_backend() -> None:
    """Print backend + devices, and hard-fail if `JAX_PLATFORMS` was set
    to something we couldn't actually load. JAX's default behavior is
    to fall back silently to CPU when a requested platform is missing —
    that's the worst kind of bug for a perf benchmark.
    """
    import jax
    requested = os.environ.get("JAX_PLATFORMS")
    actual = jax.default_backend()
    devices = jax.devices()
    print(
        f"[jax] requested={requested!r}  actual={actual!r}  "
        f"devices={devices}  jax={jax.__version__}",
        flush=True,
    )
    if requested and requested != "" and requested.lower() != actual.lower():
        # JAX reports 'gpu' for cuda; treat them as equivalent.
        equiv = {"cuda": "gpu", "gpu": "cuda"}
        if equiv.get(requested.lower()) != actual.lower():
            raise RuntimeError(
                f"JAX_PLATFORMS={requested!r} but jax.default_backend() "
                f"resolved to {actual!r}. Backend probably failed to load "
                f"silently — check `uv pip list | grep jax` for version "
                f"mismatch between jaxlib and jax-cuda12-plugin."
            )

# Reuse loader/inliner from gpu_perf_compare.py.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from gpu_perf_compare import _load_notebook_active, _NOTEBOOK_DIR, _pad_to_even
from compile_notebook_parallel_segments import slice_circuit  # same slicer

from tno_compiler.pipeline import compile_targets_batched


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--qasm", default=None)
    ap.add_argument("--n-segments", type=int, default=4)
    ap.add_argument("--depth", type=int, default=6,
                    help="Brickwall depth per segment (uniform across all)")
    ap.add_argument("--n-seeds", type=int, default=4,
                    help="Inits per segment. Total batch B = n_segments * n_seeds.")
    ap.add_argument("--max-iter", type=int, default=60)
    ap.add_argument("--max-bond", type=int, default=64)
    ap.add_argument("--tol", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="notebook_batched_segments.json")
    args = ap.parse_args()

    _assert_jax_backend()

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
    for i, s in enumerate(segments):
        seg_2q = s.depth(filter_function=lambda ci: ci.operation.num_qubits >= 2)
        print(f"  segment {i}: ops={len(s.data)}  2q-depth={seg_2q}", flush=True)

    t0 = time.time()
    results = compile_targets_batched(
        segments, ansatz_depth=args.depth,
        n_seeds_per_target=args.n_seeds,
        tol=args.tol, max_bond=args.max_bond,
        max_iter=args.max_iter, first_odd=True, seed=args.seed,
    )
    wall = time.time() - t0

    print(f"\n[done] wall={wall:.1f}s for B={args.n_segments * args.n_seeds}", flush=True)
    for r in results:
        print(
            f"  segment {r['target_idx']}: best_cost={r['compile_error']:.3e}  "
            f"all_costs={[f'{c:.2e}' for c in r['all_costs']]}",
            flush=True,
        )

    summary = {
        "config": vars(args),
        "n_qubits": n,
        "target_2q_depth": int(target_2q),
        "wall_s": wall,
        "segments": [
            {k: v for k, v in r.items()
             if k not in ("compiled", "gate_tensors")}
            for r in results
        ],
    }
    Path(args.out).write_text(json.dumps(summary, indent=2, default=str))
    print(f"[save] {args.out}", flush=True)


if __name__ == "__main__":
    main()
