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
from qiskit.quantum_info import Operator, Statevector, SuperOp, diamond_norm


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


def circuit_superop_matrix(circuit):
    """Dense superoperator matrix for a unitary circuit channel.

    The returned matrix represents `rho -> U rho U†` in Qiskit's
    vectorization convention. It has shape `(4**n, 4**n)`.
    """
    return np.asarray(SuperOp(Operator(circuit)).data, dtype=complex)


def ensemble_superop_matrix(circuits, weights):
    """Dense superoperator for `sum_i weights[i] U_i rho U_i†`."""
    if not circuits:
        raise ValueError("circuits must be non-empty")
    weights = np.asarray(weights, dtype=float)
    if len(circuits) != len(weights):
        raise ValueError("circuits and weights must have the same length")

    n = circuits[0].num_qubits
    dim2 = 4 ** n
    S = np.zeros((dim2, dim2), dtype=complex)
    for circuit, weight in zip(circuits, weights):
        if weight < 1e-15:
            continue
        S += weight * circuit_superop_matrix(circuit)
    return S


def exact_diamond_distance(target, circuits, weights, **diamond_kwargs):
    """Exact diamond distance between an ensemble channel and a target unitary.

    Returns both conventions:
      - `diamond_norm`: `||E - V||_diamond`
      - `diamond_distance`: `0.5 * ||E - V||_diamond`

    `diamond_distance` matches the convention used in Kalloor Methods:
    `d_diamond(E, V) = 1/2 ||E - V||_diamond`.

    This is dense and SDP-based. It is intended for small n only.
    """
    n = target.num_qubits
    S_ens = ensemble_superop_matrix(circuits, weights)
    S_tgt = circuit_superop_matrix(target)
    diff = SuperOp(
        S_ens - S_tgt,
        input_dims=(2,) * n,
        output_dims=(2,) * n,
    )
    raw = float(diamond_norm(diff, **diamond_kwargs))
    return {
        'diamond_norm': raw,
        'diamond_distance': 0.5 * raw,
    }


def unitary_channel_diamond_distance_from_matrices(U, V):
    """Exact diamond distance between two unitary channels.

    For channels `rho -> U rho U†` and `rho -> V rho V†`, the diamond
    distance is determined by the distance from 0 to the convex hull of
    the eigenvalues of `U†V`. This avoids an SDP for single-circuit
    comparisons.

    Returns the same two conventions as `exact_diamond_distance`.
    """
    W = np.asarray(U).conj().T @ np.asarray(V)
    eigvals = np.linalg.eigvals(W)
    pts = np.column_stack([eigvals.real, eigvals.imag])

    def dist_to_segment(a, b):
        ab = b - a
        denom = float(np.dot(ab, ab))
        if denom < 1e-30:
            return float(np.linalg.norm(a))
        t = float(np.clip(-np.dot(a, ab) / denom, 0.0, 1.0))
        return float(np.linalg.norm(a + t * ab))

    nu = min(float(abs(z)) for z in eigvals)
    for i in range(len(pts)):
        for j in range(i + 1, len(pts)):
            nu = min(nu, dist_to_segment(pts[i], pts[j]))
    nu = float(np.clip(nu, 0.0, 1.0))
    raw = 2.0 * np.sqrt(max(1.0 - nu * nu, 0.0))
    return {
        'diamond_norm': raw,
        'diamond_distance': 0.5 * raw,
        'convex_hull_origin_distance': nu,
    }
