"""Compile a state |ψ⟩ = U|0⟩ into a shallower brickwall V with V|0⟩ ≈ |ψ⟩.

Companion to `compiler.compile_circuit` (which does *unitary* compression
‖U − V‖_F via Tr(V†U)/2ⁿ). This module reuses the same polar-sweep /
Riemannian-ADAM machinery but minimizes a *state*-distance cost:

    cost = 2 − 2·Re⟨ψ|V|0⟩         (≡ ‖V|0⟩ − |ψ⟩‖² when both normalized)

State infidelity 1 − |⟨ψ|V|0⟩|² is reported alongside.

How it plugs into the existing engine
-------------------------------------
The polar-sweep cost is derived from the Frobenius identity
    ‖V − U‖² = 2ⁿ · 2 − 2 Re Tr(V†U)
with U presented as an MPO. We build a rank-1 MPO

    M = |ψ⟩⟨0| ,    M[bond_l, k, b, bond_r] = MPS[bond_l, k, bond_r] · δ_{b,0}

so that
    Tr(V†·M) = ⟨0|V†|ψ⟩ = ⟨ψ|V|0⟩^*

drops out of the same einsum as the unitary case. The /2ⁿ inside the
existing cost reporter has the wrong normalization for a rank-1 target
(no operator normalization) and is corrected inside `compile_state`'s
returned info, but it does NOT change the polar update — gate-by-gate
the polar factor of the environment is the same regardless of the
overall scale.

Convention
----------
The MPS is built via quimb with the same `qubit → site (n-1-q)` and
within-gate qubit reversal that `circuit_to_quimb_tn` uses. Verified
in `tests/test_state_conversion.py`.
"""

from __future__ import annotations

import numpy as np
import quimb.tensor as qtn

from .brickwall import brickwall_ansatz_gates, gates_to_circuit
from .gradient import compute_cost_and_grad
from .optim import polar_sweeps, riemannian_adam


# -----------------------------------------------------------------------------
# State-MPS construction (qiskit circuit → tno-convention MPS arrays)
# -----------------------------------------------------------------------------


def circuit_to_state_mps_arrays(
    target_circuit, max_bond=None, cutoff=1e-10
):
    """Simulate `target_circuit · |0…0⟩` to an MPS in tno-compiler convention.

    Convention: qiskit qubit `q` ↔ quimb site `n-1-q`, with within-gate
    qubit reversal (matches `brickwall.circuit_to_quimb_tn`).

    Returns
    -------
    arrays : list of complex ndarray
        Length-n list of MPS site tensors, site i has shape
        (bond_l, 2, bond_r). Site 0 has bond_l=1, site n-1 has bond_r=1.
        Site ordering matches tno-compiler's "site 0 first" convention.
    bond : int
        Largest realized bond dimension across all cuts.
    """
    n = target_circuit.num_qubits
    kwargs = {}
    if max_bond is not None:
        kwargs["max_bond"] = max_bond
    kwargs["cutoff"] = cutoff
    circ = qtn.CircuitMPS(N=n, **kwargs)
    for instruction in target_circuit.data:
        gate = instruction.operation
        qubits = [target_circuit.find_bit(q).index for q in instruction.qubits]
        mat = np.array(gate.to_matrix())
        sites = tuple(n - 1 - q for q in reversed(qubits))
        circ.apply_gate_raw(mat, sites)

    # Standardize index order to (left, phys, right) on middle sites.
    psi = circ.psi.copy()
    psi.permute_arrays("lpr")

    arrays: list[np.ndarray] = []
    raw = psi.arrays  # site 0: (p, r); middle: (l, p, r); site n-1: (l, p)
    for i, a in enumerate(raw):
        if i == 0:
            data = np.asarray(a, dtype=complex)[np.newaxis, :, :]   # (1, 2, br)
        elif i == n - 1:
            data = np.asarray(a, dtype=complex)[:, :, np.newaxis]   # (bl, 2, 1)
        else:
            data = np.asarray(a, dtype=complex)                       # (bl, 2, br)
        arrays.append(data)

    bond = max(a.shape[0] for a in arrays[1:])
    return arrays, bond


# -----------------------------------------------------------------------------
# Rank-1 MPO embedding |ψ⟩⟨0| in adjoint-reindexed form
# -----------------------------------------------------------------------------


def state_mps_to_target_arrays(state_mps_arrays):
    """Embed an MPS |ψ⟩ as the adjoint-reindexed rank-1 MPO arrays for
    |ψ⟩⟨0|, in the layout the polar-sweep engine consumes.

    The engine expects target_arrays = conj(M).reindex(k↔b), see
    `compiler.build_target_arrays`.

    For M = |ψ⟩⟨0|: M[k, b] = ψ_k · δ_{b, 0}. Then
        conj(M)[k, b] = conj(ψ_k) · δ_{b, 0}
    and after the k↔b swap
        target[k, b] = conj(M)[b, k] = δ_{k, 0} · conj(ψ_b)

    So the output array is nonzero only on the k=0 slice, with
        out[bl, 0, b, br] = conj(mps[bl, b, br])

    Equivalent identity check (computing Tr(target_arrays·I_phys)):
    sum_{k=b} out[bl, k, b, br] = out[bl, 0, 0, br] = conj(mps[bl, 0, br]),
    which is the conjugate of the |0…0⟩-amplitude of |ψ⟩, as it should be.
    """
    out: list[np.ndarray] = []
    for site in state_mps_arrays:
        bl, _, br = site.shape
        T = np.zeros((bl, 2, 2, br), dtype=complex)
        # k fixed to 0; b indexes the MPS physical leg (conjugated).
        T[:, 0, :, :] = np.conj(site)
        out.append(T)
    return out


# -----------------------------------------------------------------------------
# Top-level entry point
# -----------------------------------------------------------------------------


def compile_state(
    target_circuit,
    ansatz_depth,
    *,
    target_state_mps=None,
    state_max_bond=256,
    state_cutoff=1e-10,
    max_bond=256,
    max_iter=200,
    method="polar",
    first_odd=True,
    init_gates=None,
    callback=None,
    drop_rate=0.0,
    seed=0,
    lr=1e-3,
):
    """Compile a state |ψ⟩ = target_circuit·|0⟩ into a brickwall V with V|0⟩ ≈ |ψ⟩.

    Args:
        target_circuit: qiskit QuantumCircuit defining U; the target state
            is U|0…0⟩.
        ansatz_depth: number of brickwall layers in V.
        target_state_mps: optional pre-computed MPS arrays in
            tno-compiler convention (skips the simulation step).
        state_max_bond / state_cutoff: MPS truncation when simulating the
            target circuit.
        max_bond: MPO bond cap during envelope merging in polar sweeps.
            For state compression the merged envelopes have bond at most
            (target MPS bond) × (gate ansatz width); set to a value
            comparable to state_max_bond.
        max_iter, method, first_odd, init_gates, callback, drop_rate, seed,
            lr: passed through to `polar_sweeps` / `riemannian_adam`.

    Returns:
        compiled: QuantumCircuit implementing the brickwall V.
        info: dict with keys
            cost_history: per-iter polar cost (= 2 - 2·Re⟨ψ|V|0⟩, NOT
                normalized by 2ⁿ — see module docstring).
            state_overlap: complex ⟨ψ|V|0⟩ at the optimum.
            state_infidelity: 1 - |⟨ψ|V|0⟩|².
            state_l2_squared: ‖V|0⟩ - |ψ⟩‖² = 2 - 2·Re⟨ψ|V|0⟩.
            target_state_bond: realized MPS bond of the target state.
            gate_tensors: list of (2,2,2,2) optimized gates.
    """
    n_qubits = target_circuit.num_qubits

    if target_state_mps is None:
        target_state_mps, target_bond = circuit_to_state_mps_arrays(
            target_circuit, max_bond=state_max_bond, cutoff=state_cutoff
        )
    else:
        target_bond = max(a.shape[0] for a in target_state_mps[1:])

    target_arrays = state_mps_to_target_arrays(target_state_mps)

    ansatz = brickwall_ansatz_gates(n_qubits, ansatz_depth, first_odd)

    if init_gates is None:
        n_gates = sum(len(pairs) for _, pairs in ansatz)
        init_gates = [np.eye(4, dtype=complex).reshape(2, 2, 2, 2)] * n_gates

    if method == "polar":
        opt_gates_list, hist_list = polar_sweeps(
            [init_gates],
            max_iter=max_iter,
            callback=callback,
            target_arrays=target_arrays,
            n_qubits=n_qubits,
            n_layers=ansatz_depth,
            max_bond=max_bond,
            first_odd=first_odd,
            drop_rate=drop_rate,
            seed=seed,
        )
        opt_gates = opt_gates_list[0]
        cost_history = hist_list[0]
    elif method == "adam":
        def cost_grad_fn(gates):
            return compute_cost_and_grad(
                target_arrays, gates, n_qubits, ansatz_depth,
                max_bond=max_bond, first_odd=first_odd,
            )
        opt_gates, cost_history = riemannian_adam(
            cost_grad_fn, init_gates, max_iter=max_iter, lr=lr, callback=callback
        )
    else:
        raise ValueError(f"Unknown method: {method}. Use 'polar' or 'adam'.")

    overlap = _compute_state_overlap(opt_gates, ansatz, target_state_mps)
    state_infidelity = 1.0 - float(abs(overlap) ** 2)
    state_l2_squared = 2.0 - 2.0 * float(overlap.real)

    compiled = gates_to_circuit(opt_gates, n_qubits, ansatz)
    info = {
        "cost_history": cost_history,
        "state_overlap": complex(overlap),
        "state_infidelity": state_infidelity,
        "state_l2_squared": state_l2_squared,
        "target_state_bond": int(target_bond),
        "gate_tensors": opt_gates,
    }
    return compiled, info


def _compute_state_overlap(gate_tensors, ansatz_structure, target_state_mps):
    """Compute ⟨ψ_target | V | 0⟩ for the optimized brickwall V.

    Builds V as a quimb circuit acting on |0⟩, then takes the inner
    product with the target MPS (also a quimb MPS). Both live in
    tno-compiler's site-ordering convention.
    """
    n = len(target_state_mps)
    circ = qtn.CircuitMPS(N=n)
    idx = 0
    for _, pairs in ansatz_structure:
        for s1, s2 in pairs:
            mat = np.asarray(gate_tensors[idx]).reshape(4, 4)
            circ.apply_gate_raw(mat, (s1, s2))
            idx += 1
    psi_v = circ.psi

    # target_state_mps is in 'lpr' order with 3D end sites (1,2,br) / (bl,2,1).
    # quimb's MatrixProductState wants 2D end sites; squeeze + pass shape.
    quimb_arrays = [
        target_state_mps[0][0],
        *target_state_mps[1:-1],
        target_state_mps[-1][:, :, 0],
    ]
    psi_target = qtn.MatrixProductState(quimb_arrays, shape="lpr")
    return complex((psi_target.H & psi_v) ^ ...)
