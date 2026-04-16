"""Check the index ordering of quimb MPO tensors."""

from tno_compiler.brickwall import random_haar_gates, target_mpo

n, d = 4, 2
gates = random_haar_gates(n, d, seed=0)
mpo = target_mpo(gates, n, d)

for i in range(mpo.L):
    t = mpo[i]
    print(f"site {i}: shape={t.shape}, inds={t.inds}")
