"""Optimal brickwall depth vs Frobenius tolerance, sweep across (n, d_target).

For each (n_qubits, target_depth) and each tolerance ε in `tolerances`,
binary-search the smallest brickwall depth `D*` whose best-of-`n_seeds`
polar compile reaches Frobenius compile_error ≤ ε.

Targets are Haar-random brickwalls (n_qubits, target_depth, first_odd=True).

Output: JSON with `{(n, d_target, tol)}` → D*, plus a PNG plot.

Usage:
  uv run python scripts/optimal_depth_vs_tol.py
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from tno_compiler.brickwall import random_brickwall
from tno_compiler.compiler import compile_circuit_optimal


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-qubits", type=int, nargs="+", default=[10, 12, 14])
    ap.add_argument("--target-depths", type=int, nargs="+", default=[8, 16])
    ap.add_argument(
        "--tolerances", type=float, nargs="+",
        default=[1e-1, 5e-2, 1e-2, 5e-3, 1e-3, 5e-4, 1e-4],
    )
    ap.add_argument("--lo", type=int, default=1)
    ap.add_argument("--hi", type=int, default=32)
    ap.add_argument("--n-seeds", type=int, default=3)
    ap.add_argument("--max-iter", type=int, default=200)
    ap.add_argument("--mpo-tol", type=float, default=1e-4,
                    help="MPO compression tolerance (must be << min compile tol)")
    ap.add_argument("--mpo-max-bond", type=int, default=512)
    ap.add_argument("--target-seed", type=int, default=42)
    ap.add_argument("--out-json", default="scripts/optimal_depth_vs_tol.json")
    ap.add_argument("--out-png", default="scripts/optimal_depth_vs_tol.png")
    args = ap.parse_args()

    results = []  # list of {n, d_target, tol, D_opt, best_compile_err, elapsed_s}
    sorted_tols = sorted(args.tolerances, reverse=True)  # loose → tight (warm-start hi)

    for n in args.n_qubits:
        for d_t in args.target_depths:
            target = random_brickwall(
                n, d_t, first_odd=True, seed=args.target_seed,
            )
            print(
                f"\n=== n={n}  target_depth={d_t}  "
                f"(seed={args.target_seed}) ===",
                flush=True,
            )
            current_hi = args.hi  # tighten as tolerances tighten
            current_lo = args.lo
            last_D_opt = None
            for tol in sorted_tols:
                t0 = time.time()
                D_opt, _, info, search = compile_circuit_optimal(
                    target, threshold=tol,
                    lo=current_lo, hi=current_hi,
                    n_seeds=args.n_seeds, tol=args.mpo_tol,
                    max_bond=args.mpo_max_bond, max_iter=args.max_iter,
                    first_odd=True, seed=args.target_seed,
                )
                elapsed = time.time() - t0
                # Best compile_error across all probed depths (informative
                # even if we found a D_opt at the threshold).
                best_err_seen = min(
                    min(r["compile_error"] for r in runs)
                    for runs in search.values()
                )
                print(
                    f"  tol={tol:.0e}  D*={D_opt}  best_err={best_err_seen:.2e}  "
                    f"info.compile_error={info['compile_error']:.2e}  "
                    f"probes={list(search.keys())}  ({elapsed:.1f}s)",
                    flush=True,
                )
                results.append({
                    "n": n,
                    "d_target": d_t,
                    "tol": tol,
                    "D_opt": D_opt,
                    "compile_error": float(info["compile_error"]),
                    "best_err_seen": float(best_err_seen),
                    "probes": {str(k): v for k, v in search.items()},
                    "elapsed_s": elapsed,
                })
                # Warm-start the next (tighter) tol's search bounds.
                if D_opt is not None:
                    current_lo = D_opt   # any solution at tol_next has D ≥ D_opt
                    last_D_opt = D_opt
                # Persist incrementally.
                Path(args.out_json).write_text(json.dumps({
                    "config": vars(args),
                    "results": results,
                }, indent=2, default=str))

    # ----- Plot -----
    fig, ax = plt.subplots(figsize=(7, 5))
    by_config: dict[tuple[int, int], list[tuple[float, int | None]]] = {}
    for r in results:
        key = (r["n"], r["d_target"])
        by_config.setdefault(key, []).append((r["tol"], r["D_opt"]))

    cmap = plt.cm.viridis(np.linspace(0, 1, len(by_config)))
    for color, ((n, d_t), pts) in zip(cmap, sorted(by_config.items())):
        pts_sorted = sorted(pts)
        xs = [p[0] for p in pts_sorted]
        ys = [p[1] if p[1] is not None else np.nan for p in pts_sorted]
        ax.plot(
            xs, ys, "o-", color=color,
            label=f"n={n}, target d={d_t}",
        )
        # Mark unreachable tols with red x at the top of the plot.
        for tol, d_opt in pts_sorted:
            if d_opt is None:
                ax.plot(tol, args.hi + 0.5, "x", color="red", markersize=8)

    ax.set_xscale("log")
    ax.set_xlabel("Frobenius tolerance ε")
    ax.set_ylabel("Optimal brickwall depth D*")
    ax.set_title(
        f"compile_circuit_optimal: D* vs ε  (target = random brickwall, "
        f"n_seeds={args.n_seeds})"
    )
    ax.invert_xaxis()  # left = loose, right = tight (intuitive direction)
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    fig.savefig(args.out_png, dpi=120)
    print(f"\n[save] {args.out_png}", flush=True)


if __name__ == "__main__":
    main()
