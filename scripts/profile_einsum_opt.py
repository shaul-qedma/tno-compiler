"""Compare einsum with and without optimize='optimal' for the 6-tensor contraction."""

import time
import numpy as np
from tno_compiler.brickwall import random_haar_gates, target_mpo
from tno_compiler.mpo_ops import matrix_to_mpo, identity_mpo

n, d = 8, 2
tg = random_haar_gates(n, d, seed=42)
ta = target_mpo(tg, n, d)
lower = identity_mpo(n)

A1, A2 = ta[2], ta[3]
B1, B2 = lower[2], lower[3]
gate = random_haar_gates(n, d, seed=100)[0]
R = np.eye(1, dtype=complex)

print(f"Tensor shapes: A1={A1.shape} A2={A2.shape} gate={gate.shape} B1={B1.shape} B2={B2.shape} R={R.shape}")

expr = 'abcd,defg,cfhk,ihbj,jkel,gl->ai'

# No optimization
t0 = time.perf_counter()
for _ in range(5):
    np.einsum(expr, A1, A2, gate, B1, B2, R)
t_none = (time.perf_counter() - t0) / 5

# optimize=True
t0 = time.perf_counter()
for _ in range(5):
    np.einsum(expr, A1, A2, gate, B1, B2, R, optimize=True)
t_opt = (time.perf_counter() - t0) / 5

# optimize='optimal'
t0 = time.perf_counter()
for _ in range(5):
    np.einsum(expr, A1, A2, gate, B1, B2, R, optimize='optimal')
t_optimal = (time.perf_counter() - t0) / 5

print(f"No opt:    {t_none:.4f}s")
print(f"opt=True:  {t_opt:.4f}s")
print(f"opt=optimal: {t_optimal:.4f}s")
print(f"Speedup (True): {t_none/t_opt:.1f}x")
