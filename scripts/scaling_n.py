"""Verify compilation quality is stable with n at fixed TFI parameters.

Fixed: J=1, g=0.75, h=0.6, dt=0.1, steps=2 (depth 8).
Sweep: n={4,6,8,10,12,14,16} × frac={0.25,0.5,1.0}.
Report Frobenius cost (computable at any n via MPO).
Superop norm only for n≤6.

Usage: uv run python scripts/scaling_n.py
"""

import time
import numpy as np
from tno_compiler.tfi import tfi_trotter_circuit
from tno_compiler.brickwall import circuit_to_mpo
from tno_compiler.compiler import compile_circuit

J, g, h, dt, steps = 1.0, 0.75, 0.6, 0.1, 2

print(f"{'n':>4} {'depth':>5} {'frac':>5} {'ad':>4} | "
      f"{'cost':>10} {'||U-V||_F':>10} {'superop':>10} {'time':>6}")
print("-" * 70)

for n in [4, 6, 8, 10, 12, 14, 16]:
    target = tfi_trotter_circuit(n, J, g, h, dt, steps)
    td = target.depth()
    d = 2 ** n

    for frac in [0.25, 0.5, 1.0]:
        ad = max(1, int(round(td * frac)))
        t0 = time.perf_counter()
        compiled, info = compile_circuit(target, ad, max_iter=1000, lr=2e-2)
        cost = info['compile_error']
        frob = np.sqrt(max(d * cost, 0))
        elapsed = time.perf_counter() - t0

        # Superop norm only feasible for n≤6
        if n <= 6:
            V = np.array(circuit_to_mpo(target, tol=0.0)[0].to_dense())
            U = np.array(circuit_to_mpo(compiled, tol=0.0)[0].to_dense())
            superop = np.linalg.norm(
                np.kron(U.conj(), U) - np.kron(V.conj(), V), ord=2)
            superop_str = f"{superop:10.2e}"
        else:
            superop_str = f"{'n/a':>10}"

        print(f"{n:4d} {td:5d} {frac:5.2f} {ad:4d} | "
              f"{cost:10.2e} {frob:10.2e} {superop_str} {elapsed:6.0f}s",
              flush=True)
