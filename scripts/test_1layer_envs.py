"""Debug _layer_partial_derivatives for the 1-layer case."""

import numpy as np
from tno_compiler.brickwall import random_haar_gates, gates_to_unitary, target_mpo
from tno_compiler.mpo_ops import matrix_to_mpo, identity_mpo
from tno_compiler.gradient import _layer_partial_derivatives

n, d = 4, 1
tg = random_haar_gates(n, d, seed=0)
cg = random_haar_gates(n, d, seed=100)

V = gates_to_unitary(tg, n, d)
U = gates_to_unitary(cg, n, d)
exact = np.trace(V.conj().T @ U)
print(f"Exact Tr(V†U) = {exact}")

upper = target_mpo(tg, n, d)  # V†
lower = identity_mpo(n)

# Layer is odd, 2 gates
grads = _layer_partial_derivatives(cg, True, upper, lower)
print(f"grad shape: {grads.shape}")

# Overlap from first gate
ov = np.einsum('abcd,abcd->', grads[0].conj(), cg[0])
print(f"Overlap from env: {ov}")
print(f"Match: {np.allclose(ov, exact, atol=1e-6)}")

# The rqcopt convention: overlap = Tr(env†_0.conj() @ gate_0)
# = Tr(grad[0] @ gate_0) since grad is already conjugated
ov2 = np.einsum('abcd,abcd->', grads[0], cg[0])
print(f"Alt overlap (no conj): {ov2}")
