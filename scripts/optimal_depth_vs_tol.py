"""Optimal brickwall depth vs Frobenius tolerance, sweep across (n, steps).

For each (n_qubits, steps) and each tolerance ε in `tolerances`,
binary-search the smallest brickwall depth `D*` whose best-of-`n_seeds`
polar compile reaches Frobenius compile_error ≤ ε.

Targets are TFI Trotter circuits (`tfi_trotter_circuit`) — physically
meaningful, structured, and Trotter-compressible. NOT Haar-random
brickwalls (those have no compressible structure beyond their depth).

Default TFI: J=1, g=1, h=0, dt=0.1.

Output: JSON with `{(n, steps, tol)}` → D*, plus a PNG plot.

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

from tno_compiler.compiler import compile_circuit_optimal
from tno_compiler.tfi import tfi_trotter_circuit


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-qubits", type=int, nargs="+", default=[10, 12, 14])
    ap.add_argument("--steps", type=int, nargs="+", default=[4, 8, 16],
                    help="number of TFI Trotter steps in the target")
    ap.add_argument(
        "--tolerances", type=float, nargs="+",
        default=[5e-3, 1e-3, 5e-4, 1e-4],
        help="Frobenius targets, processed loose→tight to warm-start hi.",
    )
    ap.add_argument("--J", type=float, default=1.0)
    ap.add_argument("--g", type=float, default=1.0)
    ap.add_argument("--h", type=float, default=0.0)
    ap.add_argument("--dt", type=float, default=0.1)
    ap.add_argument("--lo", type=int, default=1)
    ap.add_argument("--hi", type=int, default=32)
    ap.add_argument("--n-seeds", type=int, default=8)
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
        for steps in args.steps:
            target = tfi_trotter_circuit(
                n, args.J, args.g, args.h, args.dt, steps, order=1,
            )
            target_depth = target.depth(
                filter_function=lambda ci: ci.operation.num_qubits >= 2
            )
            print(
                f"\n=== n={n}  target depth={target_depth} (= {steps} TFI steps)  "
                f"[J={args.J}, g={args.g}, h={args.h}, dt={args.dt}] ===",
                flush=True,
            )
            current_hi = args.hi  # tighten as tolerances tighten
            current_lo = args.lo
            last_D_opt = None
            # Shared warm-start cache across thresholds on this target.
            # Each compile_circuit_optimal call updates it with best gates
            # seen at every probe; subsequent calls reuse those for any
            # depth they re-probe.
            warm_cache: dict[int, list] = {}
            for tol in sorted_tols:
                t0 = time.time()
                D_opt, _, info, search = compile_circuit_optimal(
                    target, threshold=tol,
                    lo=current_lo, hi=current_hi,
                    n_seeds=args.n_seeds, tol=args.mpo_tol,
                    max_bond=args.mpo_max_bond, max_iter=args.max_iter,
                    first_odd=True, seed=args.target_seed,
                    warm_start_cache=warm_cache,
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
                # Per-probe per-seed spread — diversity check.
                for d_probed, runs in sorted(search.items()):
                    errs = [r["compile_error"] for r in runs]
                    e_min, e_max = min(errs), max(errs)
                    e_mean = sum(errs) / len(errs)
                    print(
                        f"      probe d={d_probed:2d}  errs: "
                        f"min={e_min:.2e} max={e_max:.2e} mean={e_mean:.2e}  "
                        f"spread={e_max/max(e_min, 1e-30):.1f}×",
                        flush=True,
                    )
                results.append({
                    "n": n,
                    "steps": steps,
                    "target_depth": int(target_depth),
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
    by_config: dict[tuple[int, int, int], list[tuple[float, int | None]]] = {}
    for r in results:
        key = (r["n"], r["target_depth"], r["steps"])
        by_config.setdefault(key, []).append((r["tol"], r["D_opt"]))

    cmap = plt.cm.viridis(np.linspace(0, 1, len(by_config)))
    for color, ((n, td, steps), pts) in zip(cmap, sorted(by_config.items())):
        pts_sorted = sorted(pts)
        xs = [p[0] for p in pts_sorted]
        ys = [p[1] if p[1] is not None else np.nan for p in pts_sorted]
        ax.plot(
            xs, ys, "o-", color=color,
            label=f"n={n}, depth={td} ({steps} TFI steps)",
        )
        # Mark unreachable tols with red x at the top of the plot.
        for tol, d_opt in pts_sorted:
            if d_opt is None:
                ax.plot(tol, args.hi + 0.5, "x", color="red", markersize=8)

    ax.set_xscale("log")
    ax.set_xlabel("Frobenius tolerance ε")
    ax.set_ylabel("Optimal brickwall depth D*")
    ax.set_title(
        f"compile_circuit_optimal: D* vs ε  "
        f"(target = TFI Trotter J={args.J} g={args.g} h={args.h} dt={args.dt}, "
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
