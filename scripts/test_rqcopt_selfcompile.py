"""Find the correct convention by self-compilation: circuit = target adjoint.

In rqcopt, if target is V and circuit is V†, then cost ≈ 0.
This means: target MPO = V, circuit gates = V† gates.

Let's generate V as a Haar brickwall, compute V† gates, and verify.
"""

import sys
sys.path.insert(0, '_reference_rqcopt')

import numpy as np
import jax.numpy as jnp
from jax import config
config.update("jax_enable_x64", True)

from rqcopt_mpo.tn_brickwall_methods import compute_full_gradient

from tno_compiler.brickwall import (
    random_haar_gates, gates_to_unitary, partition_gates, layer_structure,
)
from tno_compiler.mpo_ops import matrix_to_mpo

n, d = 4, 2

# Target V as a brickwall circuit
target_gates = random_haar_gates(n, d, seed=0)
V = gates_to_unitary(target_gates, n, d)

# V as MPO
V_mpo = [jnp.asarray(a) for a in matrix_to_mpo(V)]

# V† as brickwall: reverse layer order, adjoint each gate
# If V = L2 L1 (layer 1 then layer 2), then V† = L1† L2†
# The adjoint of a gate G (2,2,2,2) as a matrix is G.conj().T
# As a tensor: G†[k0,k1,b0,b1] = G*[b0,b1,k0,k1]
structure = layer_structure(n, d)
target_layers = partition_gates(target_gates, n, d)

# Build V† gates: reverse layer order, conjugate-transpose each gate
vdag_gates = []
for layer_gates in reversed(target_layers):
    for g in layer_gates:
        g_mat = g.reshape(4, 4)
        g_adj = g_mat.conj().T
        vdag_gates.append(g_adj.reshape(2, 2, 2, 2))

# These V† gates have reversed layer structure
# For d=2 layers: original is odd,even -> reversed is even,odd
reversed_is_odd = [s[0] for s in reversed(structure)]
vdag_layers = partition_gates(vdag_gates, n, d, first_odd=reversed_is_odd[0])
vdag_gl_jax = [jnp.asarray(gl) for gl in vdag_layers]

_, ov = compute_full_gradient(V_mpo, vdag_gl_jax, reversed_is_odd, 128, compute_overlap=True)
cost = 2 - 2 * float(ov.real) / (2**n)
print(f"Self-compile (V† as circuit): cost={cost:.10f}, overlap={ov}")
print(f"If correct, cost should be ~0 and overlap should be ~{2**n}")

# Alternative: just use the SAME gates (not reversed) as circuit
# This would be Tr(V · circuit_formed_by_same_gates)
gl_same = [jnp.asarray(gl) for gl in target_layers]
is_odd_same = [s[0] for s in structure]
_, ov2 = compute_full_gradient(V_mpo, gl_same, is_odd_same, 128, compute_overlap=True)
cost2 = 2 - 2 * float(ov2.real) / (2**n)
print(f"\nSame gates as circuit: cost={cost2:.10f}, overlap={ov2}")
print(f"Expected: Tr(V · V-circuit) = Tr(V²) ≠ 2^n")

# What if we pass the same gates WITHOUT reversing?
# The circuit formed by gates g0..g9 in order is U = g9*g8*...*g0
# If target is V = g9*g8*...*g0 (same gates same order),
# then Tr(V · U†) = Tr(V V†) = 2^n
# But the rqcopt code forms the circuit by contracting gates layer by layer
# into the MPO from below. Let me check: does it form U or U†?

# The merge with gate_is_left=True contracts the gate below the MPO.
# If the circuit layers are applied in order (layer 0 first, bottom),
# then the circuit unitary is U = layer_{d-1} ... layer_1 layer_0.
# The merge builds V · layer_{d-1} · ... · layer_0 from top to bottom.
# So the overlap is Tr(V · U†) if the layers represent U†,
# or Tr(V · U) if they represent U directly.

# Actually, in rqcopt, the Trotter gates with -H represent exp(+iHt) = V†.
# These are passed as the circuit. The merge computes something like
# Tr(target_MPO · product_of_circuit_gates).
# When target=V and circuit=V†, the result is Tr(V · V†) = Tr(I·2^n) = 2^n.
# But it's not Tr(V·V†) = 2^n in general -- it depends on HOW the gates
# are contracted.
