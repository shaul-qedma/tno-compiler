"""Explore quimb's circuit-to-MPO pipeline end to end."""

import quimb.tensor as qtn
import quimb as qu
import numpy as np

# Build a brickwall circuit in quimb
n, d = 8, 4
circ = qtn.Circuit(n)
rng = np.random.default_rng(42)
for layer in range(d):
    start = 0 if layer % 2 == 0 else 1
    for i in range(start, n - 1, 2):
        gate = qu.rand_uni(4, seed=int(rng.integers(0, 2**31)))
        circ.apply_gate_raw(gate, (i, i + 1), gate_round=layer, contract="split-gate")

print("Circuit:", circ)
print("Num tensors:", circ.psi.num_tensors)
print("Site tags:", circ.psi.site_tags)

# Get unitary TN
tn_uni = circ.get_uni()
print("\nUnitary TN:", type(tn_uni))
print("Num tensors:", tn_uni.num_tensors)
print("Site tags:", tn_uni.site_tags)

# Method 1: Direct contraction per site then compress
tn1 = tn_uni.copy()
for site in tn1.site_tags:
    tn1 ^= site
tn1.fuse_multibonds_()
print("\nAfter site contraction, bond sizes:", [t.shape for t in tn1])

tn1.view_as_(qtn.MatrixProductOperator, cyclic=False, L=n)
print("MPO max bond:", max(tn1.bond_sizes()))
tn1.compress(max_bond=32, cutoff=1e-6, cutoff_mode="rel")
print("After compress:", tn1.bond_sizes())

# Method 2: tensor_network_1d_compress (fit method)
tn_uni2 = circ.get_uni()
tn2 = qtn.tensor_network_1d_compress(
    tn_uni2, max_bond=32, cutoff=0.0, method="dm",
)
print("\nDM compress bonds:", tn2.bond_sizes() if hasattr(tn2, 'bond_sizes') else "N/A")

# Method 3: fit method
tn_uni3 = circ.get_uni()
tn3 = qtn.tensor_network_1d_compress(
    tn_uni3, max_bond=32, cutoff=0.0, method="fit",
    bsz=2, max_iterations=50, tol=1e-6,
)
print("Fit compress bonds:", tn3.bond_sizes() if hasattr(tn3, 'bond_sizes') else "N/A")

# Compare fidelities
tn_exact = circ.get_uni()
for m, tnc in [("direct+compress", tn1), ("dm", tn2), ("fit", tn3)]:
    dist = tnc.distance_normalized(tn_exact)
    print(f"  {m}: normalized distance = {dist:.6e}")
