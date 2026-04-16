"""Verify: is the compiled circuit in the right qubit convention?

If the Frobenius cost is ~0 but the superop norm is large, we have
an endianness mismatch. Check by computing the superop norm using
the MPO dense matrices instead of Qiskit Operator.
"""

import numpy as np
from qiskit.quantum_info import Operator
from tno_compiler.tfi import tfi_trotter_circuit
from tno_compiler.brickwall import circuit_to_mpo
from tno_compiler.compiler import compile_circuit

n = 6
target = tfi_trotter_circuit(n, 1.0, 0.75, 0.6, 0.1, 1)
compiled, info = compile_circuit(target, target.depth(), max_iter=500, lr=2e-2)
cost = info['compile_error']
print(f"Frobenius cost: {cost:.2e}")
print(f"||U-V||_F estimate: {np.sqrt(2**n * cost):.4f}")

d = 2**n

# Method 1: Qiskit Operator (little-endian)
V_qiskit = Operator(target).data
U_qiskit = Operator(compiled).data
S_diff_qiskit = np.kron(U_qiskit.conj(), U_qiskit) - np.kron(V_qiskit.conj(), V_qiskit)
superop_qiskit = np.linalg.norm(S_diff_qiskit, ord=2)
print(f"\nQiskit superop norm: {superop_qiskit:.4e}")

# Method 2: MPO dense (big-endian, same as optimizer)
V_mpo = np.array(circuit_to_mpo(target, tol=0.0)[0].to_dense())
U_mpo = np.array(circuit_to_mpo(compiled, tol=0.0)[0].to_dense())
S_diff_mpo = np.kron(U_mpo.conj(), U_mpo) - np.kron(V_mpo.conj(), V_mpo)
superop_mpo = np.linalg.norm(S_diff_mpo, ord=2)
print(f"MPO superop norm:   {superop_mpo:.4e}")

# Method 3: Frobenius operator distance directly from MPOs
frob_mpo = np.linalg.norm(U_mpo - V_mpo, 'fro')
print(f"\n||U-V||_F from MPO: {frob_mpo:.4e}")
print(f"2*||U-V||_F bound: {2*frob_mpo:.4e}")
