"""Find the two quimb APIs we need:
1) Build a 2D tensor network from a brickwall circuit (gates as vertices)
2) Compress any TN into an MPO with target error tolerance
"""

import quimb.tensor as qtn
import quimb as qu
import numpy as np

# ============================================================
# Part 1: Circuit as a raw tensor network
# ============================================================

# Can we just build a TN by hand from gate tensors?
n = 6

tn = qtn.TensorNetwork([])

# Add a gate on qubits (0,1) with named indices
gate01 = qu.rand_uni(4, seed=0).reshape(2, 2, 2, 2)
# Indices: k0_out, k1_out, k0_in, k1_in
tn.add_tensor(qtn.Tensor(
    data=gate01,
    inds=("k0_layer0", "k1_layer0", "b0", "b1_layer0"),
    tags={"GATE", "L0", "Q0", "Q1"},
))

gate23 = qu.rand_uni(4, seed=1).reshape(2, 2, 2, 2)
tn.add_tensor(qtn.Tensor(
    data=gate23,
    inds=("k2_layer0", "k3_layer0", "b2", "b3_layer0"),
    tags={"GATE", "L0", "Q2", "Q3"},
))

# Even layer: gate on (1,2)
gate12 = qu.rand_uni(4, seed=2).reshape(2, 2, 2, 2)
tn.add_tensor(qtn.Tensor(
    data=gate12,
    inds=("k1", "k2", "b1_layer0", "b2_layer0_out"),
    tags={"GATE", "L1", "Q1", "Q2"},
))

print("Manual TN:", tn)
print("Tensors:", tn.num_tensors)
print()

# ============================================================
# Better: use qtn.Circuit which handles index naming for us
# ============================================================

circ = qtn.Circuit(n)
rng = np.random.default_rng(42)

for layer in range(4):
    start = 0 if layer % 2 == 0 else 1
    for i in range(start, n - 1, 2):
        gate = qu.rand_uni(4, seed=int(rng.integers(0, 2**31)))
        # contract=False keeps gates as separate tensors (the 2D TN)
        circ.apply_gate_raw(gate, (i, i + 1), gate_round=layer, contract=False)

print("Circuit (contract=False):", circ)
print("Num tensors in psi:", circ.psi.num_tensors)

# Extract just the unitary part (remove initial state tensors)
tn_uni = circ.get_uni()
print("Unitary TN type:", type(tn_uni))
print("Num tensors:", tn_uni.num_tensors)
print("Site tags:", tn_uni.site_tags)
print()

# ============================================================
# Part 2: Compress TN to MPO
# ============================================================

# Method A: tensor_network_1d_compress (requires site tags)
print("=== tensor_network_1d_compress methods ===")
for method in ["direct", "dm", "zipup", "fit"]:
    try:
        tnc = qtn.tensor_network_1d_compress(
            tn_uni.copy(),
            max_bond=16,
            cutoff=1e-6,
            method=method,
        )
        print(f"  {method}: type={type(tnc).__name__}, "
              f"num_tensors={tnc.num_tensors}")
        if hasattr(tnc, 'bond_sizes'):
            print(f"    bonds={tnc.bond_sizes()}")
    except Exception as e:
        print(f"  {method}: FAILED - {e}")

# Method B: contract_compressed (arbitrary TN)
print("\n=== contract_compressed ===")
try:
    tnc2 = tn_uni.copy().contract_compressed(
        optimize="auto-hq",
        max_bond=16,
        output_inds=[f"k{i}" for i in range(n)] + [f"b{i}" for i in range(n)],
    )
    print(f"  type={type(tnc2).__name__}, num_tensors={tnc2.num_tensors}")
except Exception as e:
    print(f"  FAILED - {e}")

# Check what output_inds the unitary TN has
print("\n=== TN index structure ===")
outer = tn_uni.outer_inds()
print(f"Outer indices: {outer}")
