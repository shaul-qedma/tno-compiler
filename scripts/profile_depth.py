"""Profile time vs depth at fixed n=10, same-depth compilation, 5 sweeps."""

import time
from tno_compiler.tfi import tfi_trotter_circuit
from tno_compiler.compiler import compile_circuit

n = 10
print(f"{'steps':>5} {'depth':>5} {'gates':>5} | {'cost':>10} {'time':>8}")
print("-" * 45)

for steps in [1, 2, 4, 8]:
    target = tfi_trotter_circuit(n, 1.0, 0.75, 0.6, 0.1, steps)
    td = target.depth()
    ng = sum(1 for inst in target.data if len(inst.qubits) == 2)

    t0 = time.perf_counter()
    _, info = compile_circuit(target, td, method="polar", max_iter=5)
    elapsed = time.perf_counter() - t0

    print(f"{steps:5d} {td:5d} {ng:5d} | {info['compile_error']:10.2e} {elapsed:8.1f}s",
          flush=True)
