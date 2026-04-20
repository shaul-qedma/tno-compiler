"""Batched (vmap'd) JAX kernels must agree with the per-element
unbatched versions. No correctness freedom — just vectorization.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from tno_compiler import jax_ops as jo


rng = np.random.default_rng(0)


def _rand(shape):
    return rng.standard_normal(shape) + 1j * rng.standard_normal(shape)


B = 3  # batch size for the vmap'd tests


def test_contract_R_batched_matches_serial():
    bond = 4
    args = [
        _rand((B, 1, 2, 2, bond)),    # upper_i
        _rand((B, bond, 2, 2, 1)),    # upper_ip1
        _rand((B, 2, 2, 2, 2)),       # gate
        _rand((B, 1, 2, 2, bond)),    # lower_i
        _rand((B, bond, 2, 2, 1)),    # lower_ip1
        _rand((B, 1, 1)),             # R
    ]
    batched = jo.contract_R_batched(*args)
    for b in range(B):
        single = jo.contract_R(*[a[b] for a in args])
        assert jnp.allclose(batched[b], single, atol=1e-10)


def test_env_and_polar_update_batched_matches_serial():
    L = _rand((B, 2, 2))
    R = _rand((B, 2, 2))
    A1 = _rand((B, 2, 2, 2, 3))
    A2 = _rand((B, 3, 2, 2, 2))
    B1 = _rand((B, 2, 2, 2, 3))
    B2 = _rand((B, 3, 2, 2, 2))
    batched = jo.env_and_polar_update_batched(L, A1, A2, B1, B2, R)
    for b in range(B):
        single = jo.env_and_polar_update(L[b], A1[b], A2[b], B1[b], B2[b], R[b])
        assert jnp.allclose(batched[b], single, atol=1e-10)


def test_contract_L_batched_matches_serial():
    bond = 3
    L = _rand((B, 1, 1))
    upper_i = _rand((B, 1, 2, 2, bond))
    upper_ip1 = _rand((B, bond, 2, 2, 1))
    gate = _rand((B, 2, 2, 2, 2))
    lower_i = _rand((B, 1, 2, 2, bond))
    lower_ip1 = _rand((B, bond, 2, 2, 1))
    batched = jo.contract_L_batched(L, upper_i, upper_ip1, gate, lower_i, lower_ip1)
    for b in range(B):
        single = jo.contract_L(L[b], upper_i[b], upper_ip1[b], gate[b],
                               lower_i[b], lower_ip1[b])
        assert jnp.allclose(batched[b], single, atol=1e-10)


def test_init_L_R_batched_matches_serial():
    upper_0 = _rand((B, 1, 2, 2, 4))
    lower_0 = _rand((B, 1, 2, 2, 4))
    batched_L = jo.init_L_batched(upper_0, lower_0)
    upper_last = _rand((B, 4, 2, 2, 1))
    lower_last = _rand((B, 4, 2, 2, 1))
    batched_R = jo.init_R_batched(upper_last, lower_last)
    for b in range(B):
        assert jnp.allclose(batched_L[b], jo.init_L(upper_0[b], lower_0[b]),
                            atol=1e-10)
        assert jnp.allclose(batched_R[b], jo.init_R(upper_last[b], lower_last[b]),
                            atol=1e-10)


def test_absorb_R_batched_matches_serial():
    R = _rand((B, 4, 4))
    mpo_i = _rand((B, 4, 2, 2, 5))
    left = jo.absorb_R_left_batched(R, mpo_i)
    for b in range(B):
        assert jnp.allclose(left[b], jo.absorb_R_left(R[b], mpo_i[b]), atol=1e-10)
    mpo_i2 = _rand((B, 5, 2, 2, 4))
    right = jo.absorb_R_right_batched(mpo_i2, R)
    for b in range(B):
        assert jnp.allclose(right[b], jo.absorb_R_right(mpo_i2[b], R[b]),
                            atol=1e-10)


def test_canonicalize_batched_matches_serial():
    # Left canonicalize needs mat.shape[0] >= mat.shape[-1]; use (bond_l, k, b, bond_r)
    T = _rand((B, 4, 2, 2, 3))
    Q_b, R_b = jo.canonicalize_tensor_batched(T, left=True)
    for b in range(B):
        Q_s, R_s = jo.canonicalize_tensor(T[b], left=True)
        # QR is unique up to sign of diagonal, compare Q R reconstruction
        assert jnp.allclose(
            Q_b[b].reshape(-1, Q_b.shape[-1]) @ R_b[b],
            Q_s.reshape(-1, Q_s.shape[-1]) @ R_s, atol=1e-10)


def test_merge_gate_with_mpo_pair_batched_matches_serial():
    mpo1 = _rand((B, 1, 2, 2, 3))
    mpo2 = _rand((B, 3, 2, 2, 1))
    gate = _rand((B, 2, 2, 2, 2))
    out_left = jo.merge_gate_with_mpo_pair_batched(gate, mpo1, mpo2, True)
    for b in range(B):
        assert jnp.allclose(
            out_left[b],
            jo.merge_gate_with_mpo_pair(gate[b], mpo1[b], mpo2[b], True),
            atol=1e-10)
    out_right = jo.merge_gate_with_mpo_pair_batched(gate, mpo1, mpo2, False)
    for b in range(B):
        assert jnp.allclose(
            out_right[b],
            jo.merge_gate_with_mpo_pair(gate[b], mpo1[b], mpo2[b], False),
            atol=1e-10)


def test_split_merged_tensor_batched_matches_serial():
    # T has 6 legs after reshape in the unbatched case: (bl, k1, b1, k2, b2, br).
    # Batched adds leading dim so 7 total.
    T = _rand((B, 2, 2, 2, 2, 2, 2))
    T1_b, T2_b = jo.split_merged_tensor_batched(T, canonical='left', max_bond=4)
    for b in range(B):
        T1_s, T2_s = jo.split_merged_tensor(T[b], canonical='left', max_bond=4)
        # Split is unique up to gauge on the bond; check reconstruction.
        # Need to contract along the new internal bond.
        recon_b = jnp.einsum('abcD,Defg->abcefg', T1_b[b], T2_b[b])
        recon_s = jnp.einsum('abcD,Defg->abcefg', T1_s, T2_s)
        assert jnp.allclose(recon_b, recon_s, atol=1e-10)
