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

from .brickwall import brickwall_ansatz_gates, gates_to_circuit, random_brickwall
from .compiler import (
    _gates_for_depth, _perturbed_identity_gates,
    _qc_to_gate_tensors_local, _warm_start_init,
)
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

    Equivalent to ``state_mps_to_target_arrays_general(state_mps_arrays, None)``
    but takes the |0⟩-bond-1 fast path (no MPO bond growth).
    """
    out: list[np.ndarray] = []
    for site in state_mps_arrays:
        bl, _, br = site.shape
        T = np.zeros((bl, 2, 2, br), dtype=complex)
        # k fixed to 0; b indexes the MPS physical leg (conjugated).
        T[:, 0, :, :] = np.conj(site)
        out.append(T)
    return out


def state_mps_to_target_arrays_general(target_mps_arrays, initial_mps_arrays=None):
    """Embed |target⟩⟨initial| as adjoint-reindexed rank-1 MPO arrays.

    The engine cost is ``2 - 2·Re Tr(V†·M)/2ⁿ``. With
    M = |target⟩⟨initial|, this becomes
        Tr(V†·M) = ⟨initial|V†|target⟩ = ⟨target|V|initial⟩^*
    so the polar sweep maximizes Re⟨target|V|initial⟩.

    Per-site tensor:
        T[(bl_t, bl_i), k, b, (br_t, br_i)] =
            conj(target[bl_t, k, br_t]) · initial[bl_i, b, br_i]
    so MPO bond per cut = (target bond) × (initial bond).

    If ``initial_mps_arrays is None`` we take |initial⟩ = |0…0⟩ (bond 1)
    and the bond is unchanged from the target's bond — equivalent to
    `state_mps_to_target_arrays`.
    """
    if initial_mps_arrays is None:
        return state_mps_to_target_arrays(target_mps_arrays)
    if len(target_mps_arrays) != len(initial_mps_arrays):
        raise ValueError(
            f"target ({len(target_mps_arrays)} sites) and initial "
            f"({len(initial_mps_arrays)} sites) must agree"
        )
    out: list[np.ndarray] = []
    for t_site, i_site in zip(target_mps_arrays, initial_mps_arrays):
        bl_t, _, br_t = t_site.shape
        bl_i, _, br_i = i_site.shape
        # Per derivation:
        #   target_arrays[bl, k, b, br] = conj(target[bl_t, b, br_t]) · initial[bl_i, k, br_i]
        # i.e., k slot ← initial's phys, b slot ← target's phys (conj).
        # einsum "apc,bqd->abqpcd": a=bl_t, p=target_phys, c=br_t,
        #                           b=bl_i, q=initial_phys, d=br_i;
        # output axes (a, b, q, p, c, d) → (bl_t, bl_i, k, b_mpo, br_t, br_i).
        T = np.einsum(
            "apc,bqd->abqpcd", np.conj(t_site), i_site,
            dtype=complex,
        )
        T = T.reshape(bl_t * bl_i, 2, 2, br_t * br_i)
        out.append(T.astype(complex, copy=False))
    return out


# -----------------------------------------------------------------------------
# Top-level entry point
# -----------------------------------------------------------------------------


def compile_state(
    target_circuit,
    ansatz_depth,
    *,
    target_state_mps=None,
    initial_state_mps=None,
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
        max_iter, method, first_odd, init_gates, callback, seed, lr:
            passed through to `polar_sweeps` / `riemannian_adam`.

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

    target_arrays = state_mps_to_target_arrays_general(
        target_state_mps, initial_state_mps,
    )

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

    overlap = _compute_state_overlap(
        opt_gates, ansatz, target_state_mps, initial_state_mps,
    )
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


def _arrays_to_quimb_mps(arrays):
    """Convert tno-convention 3D 'lpr' arrays (with shape (1, 2, br) /
    (bl, 2, br) / (bl, 2, 1)) into a quimb MatrixProductState."""
    quimb_arrays = [arrays[0][0], *arrays[1:-1], arrays[-1][:, :, 0]]
    return qtn.MatrixProductState(quimb_arrays, shape="lpr")


def _compute_state_overlap(
    gate_tensors, ansatz_structure, target_state_mps,
    initial_state_mps=None,
):
    """Compute ⟨ψ_target | V | φ_initial⟩ for the optimized brickwall V.

    Default initial = |0…0⟩. With an explicit `initial_state_mps`,
    builds the initial state as a quimb MPS and starts V's circuit
    from that state.
    """
    n = len(target_state_mps)
    if initial_state_mps is None:
        circ = qtn.CircuitMPS(N=n)
    else:
        psi0 = _arrays_to_quimb_mps(initial_state_mps)
        circ = qtn.CircuitMPS(N=n, psi0=psi0)
    idx = 0
    for _, pairs in ansatz_structure:
        for s1, s2 in pairs:
            mat = np.asarray(gate_tensors[idx]).reshape(4, 4)
            circ.apply_gate_raw(mat, (s1, s2))
            idx += 1
    psi_v = circ.psi
    psi_target = _arrays_to_quimb_mps(target_state_mps)
    return complex((psi_target.H & psi_v) ^ ...)


# -----------------------------------------------------------------------------
# Optimal-depth state-prep compile (binary search + warm start, batched)
# -----------------------------------------------------------------------------


def compile_state_optimal(
    target_circuit, threshold, *,
    target_state_mps=None, initial_state_mps=None,
    state_max_bond=64, state_cutoff=1e-10,
    lo=1, hi=24, n_seeds=3, max_iter=100,
    max_bond=64, first_odd=True, seed=0, warm_start=True,
    init_perturb_scale=0.1, drop_rate=0.0,
):
    """Binary-search the smallest brickwall depth `D*` such that the best
    of `n_seeds` polar compiles at `D*` reaches state-infidelity ≤ `threshold`.

    State-infidelity = ``1 - |⟨ψ_target | V | φ_initial⟩|²``. Default
    initial = |0⟩ (specify `initial_state_mps` for non-trivial).

    Same warm-start + batched-polar machinery as `compile_circuit_optimal`,
    just with state-prep cost (compile_state's rank-1 MPO embedding under
    the hood) and state-fidelity threshold.

    Returns:
        (D_opt, compiled, info, search) where:
          D_opt: smallest depth in [lo, hi] meeting threshold (or None)
          compiled: best compiled QuantumCircuit at chosen depth
          info: dict with state_overlap, state_infidelity, gate_tensors,
              best_seed_idx, depth, cost_history.
          search: depth → list of {seed_idx, is_warm_start, state_infidelity}
    """
    n_qubits = target_circuit.num_qubits

    # Build target state MPS once.
    if target_state_mps is None:
        target_state_mps, target_bond = circuit_to_state_mps_arrays(
            target_circuit, max_bond=state_max_bond, cutoff=state_cutoff,
        )
    else:
        target_bond = max(a.shape[0] for a in target_state_mps[1:])

    # Build the rank-1 MPO target arrays (handles initial=None or MPS).
    target_arrays_mpo = state_mps_to_target_arrays_general(
        target_state_mps, initial_state_mps,
    )

    search: dict[int, dict] = {}
    best_so_far: dict[int, list] = {}

    def _make_warm_start(d: int):
        if not warm_start or not best_so_far:
            return None
        closest = min(best_so_far.keys(), key=lambda x: abs(x - d))
        return _warm_start_init(
            closest, best_so_far[closest], d, n_qubits, first_odd,
        )

    def probe(d: int):
        if d in search:
            return search[d]
        init_gates_list = []
        ws = _make_warm_start(d)
        if ws is not None:
            init_gates_list.append(ws)
        n_gates_d = _gates_for_depth(n_qubits, d, first_odd)
        seeds_needed = n_seeds - len(init_gates_list)
        for s in range(seeds_needed):
            init_gates_list.append(_perturbed_identity_gates(
                n_gates_d, init_perturb_scale, seed + 1000 * d + s,
            ))
        # ONE batched polar over all seeds.
        opt_gates_list, hist_list = polar_sweeps(
            init_gates_list, max_iter=max_iter,
            target_arrays=target_arrays_mpo, n_qubits=n_qubits,
            n_layers=d, max_bond=max_bond, first_odd=first_odd,
            drop_rate=drop_rate, seed=seed + d,
        )
        # Compute state-infidelity per member.
        ansatz_struct = brickwall_ansatz_gates(n_qubits, d, first_odd)
        per_seed = []
        for s, gates in enumerate(opt_gates_list):
            ov = _compute_state_overlap(
                gates, ansatz_struct, target_state_mps, initial_state_mps,
            )
            per_seed.append({
                'seed_idx': s,
                'is_warm_start': (s == 0 and ws is not None),
                'state_infidelity': 1.0 - float(abs(ov) ** 2),
                'overlap_abs': float(abs(ov)),
            })
        record = {
            'depth': d,
            'gate_lists': opt_gates_list,
            'histories': hist_list,
            'per_seed': per_seed,
        }
        search[d] = record
        best_idx = min(range(len(per_seed)),
                       key=lambda s: per_seed[s]['state_infidelity'])
        best_so_far[d] = opt_gates_list[best_idx]
        return record

    optimal = None
    while lo <= hi:
        mid = (lo + hi) // 2
        rec = probe(mid)
        best_inf = min(r['state_infidelity'] for r in rec['per_seed'])
        if best_inf <= threshold:
            optimal = mid
            hi = mid - 1
        else:
            lo = mid + 1

    chosen = (optimal if optimal is not None
              else min(search,
                       key=lambda d: min(r['state_infidelity']
                                         for r in search[d]['per_seed'])))
    rec = search[chosen]
    best_idx = min(range(len(rec['per_seed'])),
                   key=lambda s: rec['per_seed'][s]['state_infidelity'])
    best_gates = rec['gate_lists'][best_idx]
    best_hist = rec['histories'][best_idx]
    ov = _compute_state_overlap(
        best_gates, brickwall_ansatz_gates(n_qubits, chosen, first_odd),
        target_state_mps, initial_state_mps,
    )
    compiled = gates_to_circuit(
        best_gates, n_qubits,
        brickwall_ansatz_gates(n_qubits, chosen, first_odd),
    )
    info = {
        'state_overlap': complex(ov),
        'state_infidelity': 1.0 - float(abs(ov) ** 2),
        'overlap_abs': float(abs(ov)),
        'gate_tensors': best_gates,
        'best_seed_idx': int(best_idx),
        'depth': chosen,
        'target_state_bond': int(target_bond),
        'cost_history': best_hist,
    }
    search_summary = {d: r['per_seed'] for d, r in sorted(search.items())}
    return optimal, compiled, info, search_summary
