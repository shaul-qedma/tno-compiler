"""Single compilation quality at n=6 across depths.

Identity init, lr=2e-2, 1000 iter. No ensemble -- just how good
can one circuit get?

Usage: uv run python scripts/single_n6.py
"""

import time
import numpy as np
from qiskit.quantum_info import Operator
from tno_compiler.tfi import tfi_trotter_circuit
from tno_compiler.compiler import compile_circuit

n = 6
J, g, h, dt = 1.0, 0.75, 0.6, 0.1

print(f"{'steps':>5} {'depth':>5} {'frac':>5} {'ad':>4} | "
      f"{'cost':>10} {'superop':>10} {'time':>6}")
print("-" * 60)

for steps in [1, 2, 4, 8]:
    target = tfi_trotter_circuit(n, J, g, h, dt, steps)
    td = target.depth()
    V = Operator(target).data
    d = 2 ** n

    for frac in [0.25, 0.5, 0.75, 1.0]:
        ad = max(1, int(round(td * frac)))
        t0 = time.perf_counter()
        compiled, info = compile_circuit(target, ad, max_iter=1000, lr=2e-2)
        cost = info['compile_error']

        U = Operator(compiled).data
        S_diff = np.kron(U.conj(), U) - np.kron(V.conj(), V)
        superop = np.linalg.norm(S_diff, ord=2)
        elapsed = time.perf_counter() - t0

        marker = ""
        if superop < 0.05:
            marker = " ***"
        elif superop < 0.5:
            marker = " *"

        print(f"{steps:5d} {td:5d} {frac:5.2f} {ad:4d} | "
              f"{cost:10.2e} {superop:10.2e} {elapsed:6.0f}s{marker}",
              flush=True)
