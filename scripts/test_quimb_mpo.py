"""Verify the new quimb-based circuit_to_mpo and target_mpo."""

import numpy as np
from tno_compiler.brickwall import (
    random_haar_gates, circuit_to_mpo, target_mpo, circuit_to_tn,
)

n, d = 6, 4
gates = random_haar_gates(n, d, seed=42)

# Build MPO
mpo = circuit_to_mpo(gates, n, d)
print(f"MPO: L={mpo.L}, bonds={mpo.bond_sizes()}, max_bond={max(mpo.bond_sizes())}")

# Build target (V†)
tmpo = target_mpo(gates, n, d)
print(f"Target MPO: L={tmpo.L}, bonds={tmpo.bond_sizes()}")

# Check: V† @ V should be ~identity (trace = 2^n)
# Use quimb overlap: mpo1.overlap(mpo2) = conj(Tr(mpo1† mpo2))
# So tmpo.overlap(mpo) = conj(Tr(V @ V)) -- not what we want
# Tr(V† V) = Tr(I) * 2^n... actually Tr(V† V) = 2^n for unitary V
# mpo represents V, tmpo represents V†
# Tr(V† V) via quimb: we need Tr(tmpo @ mpo) as operators

# Simpler: just check the trace and overlap
print(f"\nTr(V) via MPO: {mpo.trace()}")
print(f"Tr(V†) via target MPO: {tmpo.trace()}")
print(f"conj(Tr(V)) = {np.conj(mpo.trace())}")
print(f"Match: {np.allclose(tmpo.trace(), np.conj(mpo.trace()), atol=1e-6)}")

# Check adjoint is correct: V†.to_dense() should be conj(V.to_dense()).T
V = np.array(mpo.to_dense())
Vd = np.array(tmpo.to_dense())
print(f"\nV† matches conj(V).T: {np.allclose(Vd, V.conj().T, atol=1e-8)}")
print(f"V is unitary: {np.allclose(V @ V.conj().T, np.eye(2**n), atol=1e-8)}")

# Check fidelity of compressed MPO vs exact TN
tn = circuit_to_tn(gates, n, d)
dist = mpo.distance_normalized(tn)
print(f"\nNormalized distance (MPO vs exact TN): {dist:.2e}")
