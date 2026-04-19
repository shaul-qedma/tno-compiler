"""JAX-accelerated operations for the compilation hot path.

Replaces numpy operations in gradient.py with JIT-compiled JAX equivalents.
Key wins:
- JIT caches einsum contraction paths (eliminates 5.5s of path-finding per run)
- Fused environment contraction + polar update
- Truncated randomized SVD for large bond dimensions
"""

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update('jax_enable_x64', True)


# --- Constants ---

_EYE1 = jnp.eye(1, dtype=complex)
_ID_TENSOR = jnp.eye(2, dtype=complex)[jnp.newaxis, :, :, jnp.newaxis]


# --- MPO primitives ---

def identity_mpo(n_sites):
    return [_ID_TENSOR] * n_sites


@jax.jit
def _merge_left(mpo1, mpo2, gate):
    return jnp.einsum('iabc,cdef,begh->iadghf', mpo1, mpo2, gate, optimize=True)

@jax.jit
def _merge_right(gate, mpo1, mpo2):
    return jnp.einsum('abcd,icef,fdgh->iabegh', gate, mpo1, mpo2, optimize=True)

def merge_gate_with_mpo_pair(gate, mpo1, mpo2, gate_is_left=True):
    if gate_is_left:
        return _merge_left(mpo1, mpo2, gate)
    else:
        return _merge_right(gate, mpo1, mpo2)


def split_merged_tensor(T, canonical='left', max_bond=128):
    """Split (bl,k1,b1,k2,b2,br) -> T1, T2 with truncated SVD."""
    m_phys = T.shape[0] * T.shape[1] * T.shape[2]
    n_phys = T.shape[3] * T.shape[4] * T.shape[5]
    if min(m_phys, n_phys) > max_bond * 2:
        return _split_randomized(T, canonical, max_bond)
    else:
        return _split_full(T, canonical, max_bond)


@partial(jax.jit, static_argnums=(1, 2))
def _split_full(T, canonical='left', max_bond=128):
    """Split via full SVD + truncation (JIT'd)."""
    A = jnp.moveaxis(T, 3, 2)
    shape = A.shape
    mat = A.reshape(shape[0] * shape[1] * shape[2],
                    shape[3] * shape[4] * shape[5])
    k = min(mat.shape[0], mat.shape[1], max_bond)
    u, s, vh = jnp.linalg.svd(mat, full_matrices=False)
    u, s, vh = u[:, :k], s[:k], vh[:k]
    if canonical == 'left':
        T1 = u.reshape(shape[:3] + (k,))
        T2 = (jnp.diag(s) @ vh).reshape((k,) + shape[3:])
    else:
        T1 = (u @ jnp.diag(s)).reshape(shape[:3] + (k,))
        T2 = vh.reshape((k,) + shape[3:])
    return T1, T2


@partial(jax.jit, static_argnums=(1, 2))
def _split_randomized(T, canonical='left', max_bond=128):
    """Split via randomized SVD for large matrices (JIT'd)."""
    A = jnp.moveaxis(T, 3, 2)
    shape = A.shape
    mat = A.reshape(shape[0] * shape[1] * shape[2],
                    shape[3] * shape[4] * shape[5])
    k = min(mat.shape[0], mat.shape[1], max_bond)
    u, s, vh = _randomized_svd_impl(mat, k)
    if canonical == 'left':
        T1 = u.reshape(shape[:3] + (k,))
        T2 = (jnp.diag(s) @ vh).reshape((k,) + shape[3:])
    else:
        T1 = (u @ jnp.diag(s)).reshape(shape[:3] + (k,))
        T2 = vh.reshape((k,) + shape[3:])
    return T1, T2


_RSVD_KEY = jax.random.PRNGKey(42)
_RSVD_KEY_FOLD = jax.random.fold_in(_RSVD_KEY, 1)


def _randomized_svd_impl(mat, k, p=10, q=2):
    """Halko-Martinsson-Tropp randomized SVD with subspace iteration."""
    m, n = mat.shape
    Omega = (jax.random.normal(_RSVD_KEY, (n, k + p)) +
             1j * jax.random.normal(_RSVD_KEY_FOLD, (n, k + p)))

    Y = mat @ Omega
    Q, _ = jnp.linalg.qr(Y)
    for _ in range(q):
        Z = mat.conj().T @ Q
        Qz, _ = jnp.linalg.qr(Z)
        Y = mat @ Qz
        Q, _ = jnp.linalg.qr(Y)

    B = Q.conj().T @ mat
    uh, s, vh = jnp.linalg.svd(B, full_matrices=False)
    u = Q @ uh
    return u[:, :k], s[:k], vh[:k]


def canonicalize_tensor(T, left=True):
    if left:
        return _canonicalize_left(T)
    else:
        return _canonicalize_right(T)


@jax.jit
def _canonicalize_left(T):
    shape = T.shape
    mat = T.reshape(-1, shape[-1])
    Q, R = jnp.linalg.qr(mat)
    return Q.reshape(shape[:-1] + (Q.shape[-1],)), R


@jax.jit
def _canonicalize_right(T):
    shape = T.shape
    mat = T.reshape(shape[0], -1)
    Qt, Rt = jnp.linalg.qr(mat.conj().T)
    R, Q = Rt.conj().T, Qt.conj().T
    return Q.reshape((Q.shape[0],) + shape[1:]), R


# --- Environment contractions (JIT'd, path cached after first call) ---

@jax.jit
def contract_R(upper_i, upper_ip1, gate, lower_i, lower_ip1, R):
    return jnp.einsum('abcd,defg,cfhk,ihbj,jkel,gl->ai',
                       upper_i, upper_ip1, gate,
                       lower_i, lower_ip1, R, optimize=True)

@jax.jit
def contract_L(L, upper_i, upper_ip1, gate, lower_i, lower_ip1):
    return jnp.einsum('ai,abcd,defg,cfhk,ihbj,jkel->gl',
                       L, upper_i, upper_ip1, gate,
                       lower_i, lower_ip1, optimize=True)

@jax.jit
def compute_gate_env(L, A1, A2, B1, B2, R):
    return jnp.einsum('ab,acde,efgh,bick,kjfl,hl->dgij',
                       L, A1, A2, B1, B2, R, optimize=True).conj()

@jax.jit
def env_and_polar_update(L, A1, A2, B1, B2, R):
    """Fused: compute environment + polar decomposition update."""
    env = jnp.einsum('ab,acde,efgh,bick,kjfl,hl->dgij',
                      L, A1, A2, B1, B2, R, optimize=True)
    u, _, vh = jnp.linalg.svd(env.conj().reshape(4, 4), full_matrices=False)
    return (u @ vh).reshape(2, 2, 2, 2)


# --- Absorb / init helpers ---

@jax.jit
def absorb_R_left(R, mpo_i):
    return jnp.einsum('ij,jabk->iabk', R, mpo_i)

@jax.jit
def absorb_R_right(mpo_i, R):
    return jnp.einsum('iabj,jk->iabk', mpo_i, R)

@jax.jit
def init_L(upper_0, lower_0):
    return jnp.einsum('abcd,acbe->de', upper_0, lower_0)

@jax.jit
def init_R(upper_last, lower_last):
    return jnp.einsum('abcd,ecbd->ae', upper_last, lower_last)
