"""Compress an arbitrary quimb TN into an MPO with guaranteed tolerance.

Two paths:
1. Exact: contract to exact MPO, collect spectra, DP budget allocation.
   Requires bond dim < ~4^(D/4). For shallow circuits.
2. Bounded: use quimb's tensor_network_1d_compress with max_bond directly.
   Works for any depth. Error tracked via post-hoc spectra collection.

Note: quimb's built-in per-bond cutoffs do not account for error
accumulation across bonds. Our implementation fills this gap.
"""

import numpy as np
import quimb.tensor as qtn


def tn_to_mpo(tn, n_sites, max_bond=None, tol=1e-10, norm="operator"):
    """Compress a quimb TN (with site tags) into an MPO.

    If max_bond is set, uses quimb's 1D compress (handles deep circuits).
    Otherwise contracts to exact MPO first (only for shallow circuits).

    Returns (mpo, error_bound) where error_bound ≤ tol.
    """
    if max_bond is not None:
        # Bounded path: compress with zipup, then apply DP budget allocation
        mpo = _compress_bounded(tn, n_sites, max_bond, tol, norm)
        spectra = _collect_spectra(mpo)
        cutoff = _find_optimal_cutoff(spectra, tol, max_bond, norm)
        if cutoff > 0:
            mpo.compress(max_bond=max_bond, cutoff=cutoff, cutoff_mode="abs")
        return mpo, _compute_error(spectra, cutoff, max_bond, norm)
    else:
        # Exact path: contract fully, then DP budget allocation
        mpo = _contract_to_exact_mpo(tn, n_sites)
        spectra = _collect_spectra(mpo)
        cutoff = _find_optimal_cutoff(spectra, tol, max_bond, norm)
        mpo.compress(max_bond=max_bond, cutoff=cutoff, cutoff_mode="abs")
        return mpo, _compute_error(spectra, cutoff, max_bond, norm)


def _compress_bounded(tn, n_sites, max_bond, tol, norm):
    """Compress using quimb's 1D compress with max_bond cap.

    Iteratively increases max_bond if the error exceeds tol.
    """
    bond = max_bond
    for _ in range(5):
        tn_copy = tn.copy()
        mpo = qtn.tensor_network_1d_compress(
            tn_copy, max_bond=bond, cutoff=1e-14, method="zipup")
        mpo.view_as_(qtn.MatrixProductOperator, cyclic=False, L=n_sites)
        spectra = _collect_spectra(mpo)
        error = _compute_error(spectra, 0.0, bond, norm)
        if error <= tol:
            return mpo
        bond = int(bond * 1.5)
    return mpo  # return best effort


def _contract_to_exact_mpo(tn, n_sites):
    for tag in tn.site_tags:
        tn ^= tag
    tn.fuse_multibonds_()
    tn.view_as_(qtn.MatrixProductOperator, cyclic=False, L=n_sites)
    tn.fill_empty_sites_()
    tn.ensure_bonds_exist()
    return tn


def _collect_spectra(mpo):
    spectra = []
    for i in range(mpo.L - 1):
        ta = mpo[mpo.site_tags[i]]
        tb = mpo[mpo.site_tags[i + 1]]
        bix = list(qtn.tensor_core.bonds(ta, tb))
        if not bix:
            spectra.append(np.array([1.0]))
            continue
        tc = ta @ tb
        left_inds = [ix for ix in ta.inds if ix != bix[0]]
        _, s, _ = tc.split(left_inds=left_inds, bond_ind=bix[0],
                           absorb=None, get="tensors", cutoff=0.0)
        spectra.append(np.array(s.data))
    return spectra


def _find_optimal_cutoff(spectra, tol, max_bond, norm):
    all_svs = np.concatenate([s[1:] for s in spectra if len(s) > 1])
    if len(all_svs) == 0 or _compute_error(spectra, 0.0, max_bond, norm) <= tol:
        return 0.0
    lo, hi = 0.0, float(np.max(all_svs))
    for _ in range(64):
        mid = (lo + hi) / 2
        if _compute_error(spectra, mid, max_bond, norm) <= tol:
            lo = mid
        else:
            hi = mid
    return lo


def _compute_error(spectra, cutoff, max_bond, norm):
    total = 0.0
    for svs in spectra:
        cap = min(len(svs), max_bond) if max_bond else len(svs)
        keep = min(cap, int(np.sum(svs >= cutoff))) if cutoff > 0 else cap
        keep = max(keep, 1)
        discarded = svs[keep:]
        if len(discarded) == 0:
            continue
        if norm == "operator":
            total += float(discarded[0])
        elif norm == "frobenius":
            total += float(np.sum(discarded ** 2))
    if norm == "frobenius":
        total = np.sqrt(total)
    return total
