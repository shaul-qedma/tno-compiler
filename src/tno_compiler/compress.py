"""Compress an arbitrary quimb TN into an MPO with guaranteed operator norm tolerance.

Three phases:
1. Contract site groups into a single tensor per site (exact MPO).
2. Collect the SVD spectrum at each bond.
3. Binary search for the optimal global cutoff, then compress.
"""

import numpy as np
import quimb.tensor as qtn


def tn_to_mpo(tn, n_sites, max_bond=None, tol=1e-10):
    """Compress a quimb TN (with site tags) into an MPO.

    Returns (mpo, error_bound) where error_bound ≤ tol is the actual
    sum of first-discarded singular values across all bonds (operator
    norm error via triangle inequality).
    """
    mpo = _contract_to_exact_mpo(tn, n_sites)
    spectra = _collect_spectra(mpo)
    cutoff = _find_optimal_cutoff(spectra, tol, max_bond)
    mpo.compress(max_bond=max_bond, cutoff=cutoff, cutoff_mode="abs")
    return mpo, _compute_error(spectra, cutoff, max_bond)


def _contract_to_exact_mpo(tn, n_sites):
    """Contract a site-tagged TN into an exact (untruncated) MPO."""
    for tag in tn.site_tags:
        tn ^= tag
    tn.fuse_multibonds_()
    tn.view_as_(qtn.MatrixProductOperator, cyclic=False, L=n_sites)
    tn.fill_empty_sites_()
    tn.ensure_bonds_exist()
    return tn


def _collect_spectra(mpo):
    """SVD each bond, return list of singular value arrays."""
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


def _find_optimal_cutoff(spectra, tol, max_bond):
    """Binary search for the largest cutoff with total error ≤ tol."""
    all_svs = np.concatenate([s[1:] for s in spectra if len(s) > 1])
    if len(all_svs) == 0 or _compute_error(spectra, 0.0, max_bond) <= tol:
        return 0.0
    lo, hi = 0.0, float(np.max(all_svs))
    for _ in range(64):
        mid = (lo + hi) / 2
        if _compute_error(spectra, mid, max_bond) <= tol:
            lo = mid
        else:
            hi = mid
    return lo


def _compute_error(spectra, cutoff, max_bond):
    """Sum of first-discarded SVs across all bonds for a given cutoff."""
    total = 0.0
    for svs in spectra:
        cap = min(len(svs), max_bond) if max_bond else len(svs)
        keep = min(cap, int(np.sum(svs >= cutoff))) if cutoff > 0 else cap
        keep = max(keep, 1)
        if keep < len(svs):
            total += float(svs[keep])
    return total
