"""Channel-level verification via sampled trace distance.

Implements the worst-case-channel-error proxy used by Kalloor et al.
2025 (arXiv:2510.18000, results §Demonstration): sample K random pure
input states, compute the trace distance between the ensemble-channel
output and the target-unitary output on each, return the max.

For pure target output |φ⟩ = V|ψ⟩ and ensemble output σ = Σᵢ pᵢ Uᵢ|ψ⟩⟨ψ|Uᵢ†,
the trace distance is exactly

    D(|φ⟩⟨φ|, σ) = ½ Σ |λ_i|

where {λ_i} are the eigenvalues of (|φ⟩⟨φ| − σ). This operator has
support only on the (M+1)-dim subspace spanned by {|φ⟩, U₁|ψ⟩, …, Uₘ|ψ⟩},
so the eigendecomposition is an (M+1)×(M+1) problem regardless of n.

Scaling: statevector simulation — O(gates · 2ⁿ) per sample, total
O(K(M+1) · 2ⁿ) memory during compute, independent of 2ⁿ×2ⁿ matrices.
Works up to ~n=20 on a laptop (with K·(M+1) small).

max_i d_i is a lower bound on (½)·‖E - V‖_◇.
"""

import numpy as np
from qiskit.quantum_info import Statevector


def _random_pure_states(n_qubits, n_samples, seed):
    rng = np.random.default_rng(seed)
    d = 2 ** n_qubits
    psi = (rng.standard_normal((n_samples, d))
           + 1j * rng.standard_normal((n_samples, d)))
    psi /= np.linalg.norm(psi, axis=1, keepdims=True)
    return psi


def _evolve(psi_vec, circuit):
    """Apply `circuit` to pure state `psi_vec` (ndarray) via statevector sim."""
    return np.asarray(Statevector(psi_vec).evolve(circuit).data)


def _trace_distance_pure_vs_mixed(phi, u_states, weights):
    """Trace distance D(|φ⟩⟨φ|, Σᵢ pᵢ |uᵢ⟩⟨uᵢ|) via rank-limited eigendecomposition.

    phi: (d,) pure state.
    u_states: (M, d) ensemble pure states.
    weights: (M,) nonneg weights summing to 1.
    """
    vecs = np.vstack([phi[np.newaxis, :], u_states]).T  # (d, M+1)
    # Orthonormal basis for span(vecs) via QR
    Q, R = np.linalg.qr(vecs, mode='reduced')  # Q: (d, k), R: (k, M+1)
    # Coordinates of each vector in the Q-basis
    phi_c = R[:, 0]
    u_c = R[:, 1:].T  # (M, k)
    # Build (|φ⟩⟨φ| − Σ pᵢ |uᵢ⟩⟨uᵢ|) in the Q-basis (k×k, k ≤ M+1)
    A = np.outer(phi_c, phi_c.conj())
    for p, u in zip(weights, u_c):
        if p < 1e-15:
            continue
        A -= p * np.outer(u, u.conj())
    A = 0.5 * (A + A.conj().T)  # enforce Hermitian against roundoff
    evals = np.linalg.eigvalsh(A)
    return 0.5 * float(np.sum(np.abs(evals)))


def sampled_max_trace_distance(target, circuits, weights,
                                n_samples=10, seed=0):
    """Max trace distance between ensemble-channel and target-channel
    outputs over K Haar-random pure input states.

    Scales with 2ⁿ in memory (one statevector at a time), not 2ⁿ×2ⁿ.

    Args:
        target: qiskit QuantumCircuit defining V.
        circuits: list of QuantumCircuit, ensemble members Uᵢ.
        weights: (M,) probabilities.
        n_samples: number of random input states.
        seed: RNG seed.

    Returns:
        dict with keys: max_td, mean_td, trace_distances (list).
    """
    n = target.num_qubits
    weights = np.asarray(weights, dtype=float)

    psi_batch = _random_pure_states(n, n_samples, seed)

    tds = []
    for psi in psi_batch:
        phi = _evolve(psi, target)
        u_states = np.stack([_evolve(psi, qc) for qc in circuits])
        td = _trace_distance_pure_vs_mixed(phi, u_states, weights)
        tds.append(td)

    return {
        'max_td': max(tds),
        'mean_td': float(np.mean(tds)),
        'trace_distances': tds,
    }
