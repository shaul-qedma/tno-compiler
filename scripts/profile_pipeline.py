"""Profile each stage of the compilation pipeline.

Usage: uv run python scripts/profile_pipeline.py
"""

import cProfile
import pstats
import io
import time
import numpy as np
from tno_compiler.brickwall import random_haar_gates, target_mpo, total_gates
from tno_compiler.gradient import compute_cost_and_grad


def profile_fn(fn):
    """Run fn under cProfile. Return (result, seconds, stats_string)."""
    pr = cProfile.Profile()
    t0 = time.perf_counter()
    pr.enable()
    result = fn()
    pr.disable()
    elapsed = time.perf_counter() - t0
    s = io.StringIO()
    pstats.Stats(pr, stream=s).sort_stats('cumulative').print_stats(10)
    return result, elapsed, s.getvalue()


# Only test-relevant sizes
configs = [(4, 1), (4, 2), (4, 4), (6, 1), (6, 2), (6, 4), (8, 1), (8, 2)]

print(f"{'n':>3} {'d':>3} {'gates':>5} {'target_s':>9} {'grad_s':>8}")
print("-" * 35)

for n, d in configs:
    tg = random_haar_gates(n, d, seed=42)
    cg = random_haar_gates(n, d, seed=1042)

    ta, t1, _ = profile_fn(lambda: target_mpo(tg, n, d))
    _, t2, _ = profile_fn(lambda: compute_cost_and_grad(ta, cg, n, d))

    print(f"{n:3d} {d:3d} {total_gates(n,d):5d} {t1:9.4f} {t2:8.4f}", flush=True)

# Detailed breakdown for n=8, d=2
print("\n--- Detailed: compute_cost_and_grad(n=8, d=2) ---")
tg8 = random_haar_gates(8, 2, seed=42)
cg8 = random_haar_gates(8, 2, seed=1042)
ta8 = target_mpo(tg8, 8, 2)
_, _, stats = profile_fn(lambda: compute_cost_and_grad(ta8, cg8, 8, 2))
print(stats)
