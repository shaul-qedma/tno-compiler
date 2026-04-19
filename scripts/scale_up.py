"""Scale up qubit count with the O(L) Gibbs-Cincio sweep.

TFI (J=1, g=0.75, h=0.6, dt=0.1, 2 steps, depth 8), 2× compression.
20 polar sweeps. Report cost, ||U-V||_F, time.

Usage: uv run python scripts/scale_up.py
"""

import time
import numpy as np
from tno_compiler.tfi import tfi_trotter_circuit
from tno_compiler.compiler import compile_circuit

J, g, h, dt, steps = 1.0, 0.75, 0.6, 0.1, 2

print(f"{'n':>4} | {'cost':>10} {'||U-V||_F':>10} {'time':>8}")
print("-" * 40)

for n in [4, 6, 8, 10, 12, 16, 20, 30, 40, 50]:
    target = tfi_trotter_circuit(n, J, g, h, dt, steps)
    td = target.depth()
    ad = max(1, td // 2)  # 2× compression

    t0 = time.perf_counter()
    compiled, info = compile_circuit(target, ad, method="polar", max_iter=20)
    elapsed = time.perf_counter() - t0

    cost = info['compile_error']
    frob = np.sqrt(max(2**n * cost, 0))

    print(f"{n:4d} | {cost:10.2e} {frob:10.2e} {elapsed:8.1f}s", flush=True)
