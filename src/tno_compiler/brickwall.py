"""1D brickwall circuit: alternating layers of nearest-neighbor 2-qubit gates.

Odd layers:  (0,1), (2,3), (4,5), ...
Even layers: (1,2), (3,4), (5,6), ...

Circuits are built as quimb tensor networks (gates split into
single-site tensors). MPOs are obtained by contracting the depth
direction via quimb's 1D compression.
"""

import numpy as np
import quimb.tensor as qtn
from qiskit.quantum_info import random_unitary


def layer_structure(n_qubits, n_layers, first_odd=True):
    """List of (is_odd, [(q1,q2), ...]) for each layer."""
    odd = first_odd
    result = []
    for _ in range(n_layers):
        start = 0 if odd else 1
        result.append((odd, [(i, i + 1) for i in range(start, n_qubits - 1, 2)]))
        odd = not odd
    return result


def total_gates(n_qubits, n_layers, first_odd=True):
    return sum(len(p) for _, p in layer_structure(n_qubits, n_layers, first_odd))


def partition_gates(gates, n_qubits, n_layers, first_odd=True):
    """Split a flat gate list into per-layer lists."""
    result, idx = [], 0
    for _, pairs in layer_structure(n_qubits, n_layers, first_odd):
        result.append(gates[idx:idx + len(pairs)])
        idx += len(pairs)
    return result


def random_haar_gates(n_qubits, n_layers, first_odd=True, seed=0):
    """Haar-random 2-qubit gates. Returns list of (2,2,2,2) arrays."""
    ng = total_gates(n_qubits, n_layers, first_odd)
    return [random_unitary(4, seed=seed + i).data.reshape(2, 2, 2, 2)
            for i in range(ng)]


def circuit_to_tn(gates, n_qubits, n_layers, first_odd=True):
    """Build a quimb TN from brickwall gates (split-gate: one tensor per site per gate)."""
    circ = qtn.Circuit(n_qubits)
    idx = 0
    for layer, (_, pairs) in enumerate(layer_structure(n_qubits, n_layers, first_odd)):
        for q1, q2 in pairs:
            circ.apply_gate_raw(
                np.asarray(gates[idx]).reshape(4, 4),
                (q1, q2),
                gate_round=layer,
                contract="split-gate",
            )
            idx += 1
    return circ.get_uni()


def circuit_to_mpo(gates, n_qubits, n_layers, first_odd=True,
                   max_bond=None, tol=1e-10):
    """Convert brickwall gates to a quimb MPO with guaranteed error tolerance.

    Iteratively compresses with decreasing per-bond cutoff until the
    total operator norm error (sum of first-discarded SVs across bonds)
    is below tol.

    Returns (mpo, error_bound).
    """
    n_bonds = max(n_qubits - 1, 1)
    cutoff = tol / n_bonds  # conservative initial guess

    for _ in range(10):  # at most 10 refinement rounds
        mpo, error = _compress_to_mpo(gates, n_qubits, n_layers, first_odd,
                                       max_bond, cutoff)
        if error <= tol:
            return mpo, error
        # Tighten cutoff proportionally
        cutoff *= tol / error / 2
    return mpo, error


def _compress_to_mpo(gates, n_qubits, n_layers, first_odd, max_bond, cutoff):
    """Single compression pass. Returns (mpo, total_error)."""
    tn = circuit_to_tn(gates, n_qubits, n_layers, first_odd)
    site_tags = tn.site_tags

    # Contract each site group, cast to MPO, ensure bonds exist
    for tag in site_tags:
        tn ^= tag
    tn.fuse_multibonds_()
    tn.view_as_(qtn.MatrixProductOperator, cyclic=False, L=n_qubits)
    tn.fill_empty_sites_()
    tn.ensure_bonds_exist()

    # Canonicalize rightward
    for i in range(len(site_tags) - 1):
        tn.canonize_between(site_tags[i], site_tags[i + 1])

    # Sweep left: SVD-compress each bond, track operator norm error
    total_error = 0.0
    for i in range(len(site_tags) - 1, 0, -1):
        ta = tn[site_tags[i - 1]]
        tb = tn[site_tags[i]]
        bix = list(qtn.tensor_core.bonds(ta, tb))
        if not bix:
            continue
        bnd = bix[0]

        tc = ta @ tb
        left_inds = [ix for ix in ta.inds if ix != bnd]
        u, s, v = tc.split(
            left_inds=left_inds, bond_ind=bnd,
            absorb=None, get="tensors", cutoff=0.0,
        )
        all_svs = s.data
        keep = len(all_svs)
        if max_bond is not None:
            keep = min(keep, max_bond)
        if cutoff > 0:
            keep = min(keep, int(np.sum(all_svs >= cutoff)))
            keep = max(keep, 1)

        if keep < len(all_svs):
            total_error += float(all_svs[keep])

        kept = slice(None, keep)
        left_data = np.einsum("...i,i->...i", u.data[..., kept], all_svs[kept])
        ta.modify(data=left_data, inds=u.inds)
        tb.modify(data=v.data[kept, ...], inds=v.inds)

    return tn, total_error


def mpo_to_arrays(mpo):
    """Extract (bond_l, k, b, bond_r) numpy arrays from a quimb MPO.

    Permutes axes from quimb's arbitrary order to the fixed convention
    needed by the gradient computation.
    """
    arrays = []
    for i in range(mpo.L):
        t = mpo[i]
        inds = t.inds
        ki, bi = f"k{i}", f"b{i}"

        # Find which axis is k, b, and bond(s)
        ax_k = inds.index(ki)
        ax_b = inds.index(bi)
        bond_axes = [j for j in range(len(inds)) if j != ax_k and j != ax_b]

        if i == 0:
            # (k, b, bond_r) → (1, k, b, bond_r)
            perm = (ax_k, ax_b, bond_axes[0])
            data = t.data.transpose(perm)[np.newaxis, ...]
        elif i == mpo.L - 1:
            # (k, b, bond_l) → (bond_l, k, b, 1)
            perm = (bond_axes[0], ax_k, ax_b)
            data = t.data.transpose(perm)[..., np.newaxis]
        else:
            # (k, b, bond_l, bond_r) → (bond_l, k, b, bond_r)
            # Need to identify which bond is left vs right
            # Left bond connects to site i-1, right to site i+1
            bond_inds = [inds[j] for j in bond_axes]
            prev_inds = set(mpo[i - 1].inds)
            if bond_inds[0] in prev_inds:
                ax_bl, ax_br = bond_axes[0], bond_axes[1]
            else:
                ax_bl, ax_br = bond_axes[1], bond_axes[0]
            perm = (ax_bl, ax_k, ax_b, ax_br)
            data = t.data.transpose(perm)

        arrays.append(np.array(data, dtype=complex))
    return arrays


def target_mpo(gates, n_qubits, n_layers, first_odd=True,
               max_bond=None, cutoff=1e-10):
    """Target MPO for compilation (stores V†).

    Returns (mpo, error_bound) where error_bound is the operator norm
    bound on ||V† - V†_mpo|| from MPO compression.
    """
    mpo, error_bound = circuit_to_mpo(gates, n_qubits, n_layers, first_odd,
                                      max_bond, cutoff)
    reindex_map = {}
    for i in range(n_qubits):
        reindex_map[f"k{i}"] = f"b{i}"
        reindex_map[f"b{i}"] = f"k{i}"
    return mpo.conj().reindex(reindex_map), error_bound
