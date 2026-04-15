"""Test if qubit ordering mismatch causes the overlap discrepancy.

Qiskit uses little-endian (qubit 0 is least significant).
The MPO decomposition from matrix_to_mpo might use big-endian.
"""

import sys
sys.path.insert(0, '_reference_rqcopt')

import numpy as np
import jax.numpy as jnp
from jax import config
config.update("jax_enable_x64", True)

from rqcopt_mpo.tn_brickwall_methods import compute_full_gradient
from tno_compiler.brickwall import random_haar_gates, partition_gates, layer_structure
from tno_compiler.mpo_ops import matrix_to_mpo
from qiskit.quantum_info import random_unitary, Operator
from qiskit.circuit import QuantumCircuit
from qiskit.circuit.library import UnitaryGate

n = 4

# Build a simple circuit: one gate on (0,1)
G01 = random_unitary(4, seed=0).data

# Qiskit circuit
qc = QuantumCircuit(n)
qc.append(UnitaryGate(G01), [0, 1])
U_qiskit = Operator(qc).data

# Manual tensor product: G01 on qubits (0,1), identity on (2,3)
# Big-endian: U = G01 ⊗ I
U_big = np.kron(G01, np.eye(4))
# Little-endian: U = I ⊗ G01 (Qiskit convention)
U_little = np.kron(np.eye(4), G01)

print("Qiskit matches big-endian:", np.allclose(U_qiskit, U_big, atol=1e-10))
print("Qiskit matches little-endian:", np.allclose(U_qiskit, U_little, atol=1e-10))

# Check what matrix_to_mpo assumes
# If we do matrix_to_mpo(G01 ⊗ I), the first site should be G01
mpo = matrix_to_mpo(U_big)
# Reconstruct
A = mpo[0]
for B in mpo[1:]:
    C = np.einsum('iabj,jcdk->iacbdk', A, B)
    s = C.shape
    A = C.reshape(s[0], s[1]*s[2], s[3]*s[4], s[-1])
U_recon = np.einsum('iabj->ab', A)
print("MPO roundtrip of big-endian:", np.allclose(U_recon, U_big, atol=1e-10))

# Now: Qiskit's gates_to_unitary uses Operator(qc).data which is little-endian.
# But matrix_to_mpo decomposes in big-endian order.
# When we do target_mpo -> matrix_to_mpo(Operator(qc).data), the MPO has
# little-endian qubit order. But the circuit gates from partition_gates
# assign gate[0] to qubits (0,1) which in the MPO is... which ordering?

# The MPO site 0 corresponds to the FIRST qubit in the matrix row/col index.
# For big-endian: site 0 = most significant qubit
# For Qiskit (little-endian): qubit 0 = least significant

# So if we build the MPO from Qiskit's unitary, site 0 of the MPO corresponds
# to qubit (n-1) in Qiskit's convention!

# This means: when we put gate[0] on (0,1) in Qiskit, the corresponding MPO
# sites are (n-2, n-1) = (2, 3) for n=4. And gate on (2,3) in Qiskit maps
# to MPO sites (0, 1).

# This is the qubit ordering mismatch!
print("\n=== QUBIT ORDERING MISMATCH ===")
print("Qiskit: qubit 0 is least significant (rightmost in tensor product)")
print("matrix_to_mpo: site 0 is most significant (leftmost in tensor product)")
print("Gate on qubits (0,1) in Qiskit -> MPO sites (n-2, n-1)")
print("Gate on qubits (2,3) in Qiskit -> MPO sites (0, 1)")

# Fix: either reverse the qubit order in gates_to_unitary, or reverse the
# gate assignment in partition_gates.
