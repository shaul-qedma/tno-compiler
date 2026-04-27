"""Sweep `gpu_perf_compare` over (target, depth) and dump a CSV.

For each (n, steps, depth) tuple, runs both tno and aqc and records
per-iter wall, total wall, MPS-build wall, JIT warm-up, state fidelity.
Use this on the GPU host to get a full perf×accuracy×depth comparison.

Usage:
  uv run python scripts/gpu_perf_sweep.py \
      --tfi-n 10 14 20 \
      --tfi-steps 4 8 \
      --depths 2 4 6 8 10 \
      --iters 30 --max-bond 64

  # AQC-only or tno-only:
  uv run python scripts/gpu_perf_sweep.py --methods tno --tfi-n 14 --depths 4 8

Each row written incrementally to --out (CSV) so partial sweeps survive
crashes. JSON dump of full per-config results goes to --out-json.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

from gpu_perf_compare import _print_jax_diag, load_target, time_aqc, time_tno


CSV_FIELDS = [
    "target", "n_qubits", "target_2q_depth", "depth", "method",
    "iters_limit", "iters_actual", "max_bond", "n_seeds",
    "t_mps_s", "t_warmup_s", "t_total_s", "ms_per_iter",
    "state_fid", "error",
]


def _row_for(method: str, target: str, qc, depth: int,
             iters: int, res: dict) -> dict:
    target_2q_depth = qc.depth(
        filter_function=lambda ci: ci.operation.num_qubits >= 2)
    base = {
        "target": target,
        "n_qubits": qc.num_qubits,
        "target_2q_depth": int(target_2q_depth),
        "depth": depth,
        "method": method,
        "iters_limit": iters,
        "max_bond": res.get("max_bond"),
        "n_seeds": res.get("n_seeds"),
        "t_mps_s": res.get("t_mps_s"),
        "t_warmup_s": res.get("t_warmup_s"),
        "t_total_s": res.get("t_total_s"),
        "ms_per_iter": res.get("ms_per_iter"),
        "iters_actual": res.get("iters_actual", iters),
        "state_fid": res.get("state_fid", res.get("state_fid_best")),
        "error": res.get("error"),
    }
    return base


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tfi-n", type=int, nargs="*", default=[10, 14, 20],
                    help="TFI qubit counts. Pass `--tfi-n` (no args) to skip TFI.")
    ap.add_argument("--tfi-steps", type=int, nargs="*", default=[4, 8],
                    help="TFI Trotter step counts. Pass `--tfi-steps` (no args) to skip TFI.")
    ap.add_argument(
        "--notebook-steps", type=int, nargs="+", default=[],
        help="Sweep notebook_stepsN targets (bundled in data/notebook/). "
             "Combines with --tfi-* (both kinds run if both given).",
    )
    ap.add_argument(
        "--qasm-template", default=None,
        help="Override the bundled notebook qasm path (use {steps} placeholder)",
    )
    ap.add_argument("--depths", type=int, nargs="+", default=[2, 4, 6, 8, 10])
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--max-bond", type=int, default=64)
    ap.add_argument("--n-seeds", type=int, default=1,
                    help="Batch size for tno (aqc is single-instance)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--methods", nargs="+", default=["tno", "aqc"],
                    choices=["tno", "aqc"])
    ap.add_argument("--out", default="gpu_perf_sweep.csv")
    ap.add_argument("--out-json", default="gpu_perf_sweep.json")
    args = ap.parse_args()

    _print_jax_diag()

    rows: list[dict] = []
    full: list[dict] = []

    csv_path = Path(args.out)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()

        configs: list[tuple[str, str | None]] = []  # (target_spec, qasm_path)
        for n in args.tfi_n:
            for steps in args.tfi_steps:
                configs.append((f"tfi:{n}:{steps}", None))
        for steps in args.notebook_steps:
            qpath = (args.qasm_template.format(steps=steps)
                     if args.qasm_template else None)
            configs.append((f"notebook_steps{steps}", qpath))

        for target, qasm_path in configs:
            qc, label = load_target(target, qasm_path=qasm_path)
            target_2q = qc.depth(
                filter_function=lambda ci: ci.operation.num_qubits >= 2)
            print(
                f"\n=== {label}  qubits={qc.num_qubits}  2q-depth={target_2q} ===",
                flush=True,
            )

            for depth in args.depths:
                print(f"\n--- depth={depth} ---", flush=True)
                if "tno" in args.methods:
                    try:
                        t0 = time.time()
                        r = time_tno(qc, depth, args.iters, args.max_bond,
                                      args.n_seeds, args.seed)
                        r["wall_s"] = time.time() - t0
                    except Exception as e:
                        r = {"method": "tno",
                             "error": f"{type(e).__name__}: {str(e)[:300]}"}
                        print(f"  [tno] FAILED: {r['error']}", flush=True)
                    full.append({"target": target, "depth": depth, **r})
                    row = _row_for("tno", target, qc, depth, args.iters, r)
                    writer.writerow(row); f.flush()
                    rows.append(row)

                if "aqc" in args.methods:
                    try:
                        t0 = time.time()
                        r = time_aqc(qc, depth, args.iters, args.max_bond,
                                      args.seed)
                        r["wall_s"] = time.time() - t0
                    except Exception as e:
                        r = {"method": "aqc",
                             "error": f"{type(e).__name__}: {str(e)[:300]}"}
                        print(f"  [aqc] FAILED: {r['error']}", flush=True)
                    full.append({"target": target, "depth": depth, **r})
                    row = _row_for("aqc", target, qc, depth, args.iters, r)
                    writer.writerow(row); f.flush()
                    rows.append(row)

                Path(args.out_json).write_text(json.dumps({
                    "config": vars(args),
                    "results": full,
                }, indent=2, default=str))

    # Final compact table
    print("\n[summary]", flush=True)
    print(
        f"  {'target':18s} {'D':>3s} {'method':6s} "
        f"{'mps':>6s} {'warm':>6s} {'total':>7s} {'ms/it':>7s} {'fid':>8s}",
        flush=True,
    )
    for r in rows:
        if r.get("error"):
            print(f"  {r['target']:18s} {r['depth']:3d} {r['method']:6s} "
                  f"FAILED: {r['error'][:60]}", flush=True)
            continue
        print(
            f"  {r['target']:18s} {r['depth']:3d} {r['method']:6s} "
            f"{r['t_mps_s']:6.1f} {r['t_warmup_s']:6.1f} {r['t_total_s']:7.1f} "
            f"{r['ms_per_iter']:7.0f} {r['state_fid']:8.4f}",
            flush=True,
        )

    print(f"\n[save] {args.out}\n[save] {args.out_json}", flush=True)


if __name__ == "__main__":
    main()
