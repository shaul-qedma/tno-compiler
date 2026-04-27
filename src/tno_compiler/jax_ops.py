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


def identity_mpo_batched(n_sites, B):
    """Identity MPO broadcast to batch size B. Shape per site:
    (B, 1, 2, 2, 1)."""
    t = jnp.broadcast_to(_ID_TENSOR, (B,) + _ID_TENSOR.shape)
    return [t] * n_sites


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
    """Split (bl,k1,b1,k2,b2,br) -> T1, T2 with truncated SVD.

    Routes to randomized SVD whenever the matrix is meaningfully larger
    than `max_bond` (target rank). Lower threshold → rSVD covers more
    of the truncated path, where its k+p<<min(m,n) cost wins on GPU.
    """
    m_phys = T.shape[0] * T.shape[1] * T.shape[2]
    n_phys = T.shape[3] * T.shape[4] * T.shape[5]
    if min(m_phys, n_phys) > max_bond + 16:
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


def _randomized_svd_impl(mat, k, p=10, q=2):
    """Halko-Martinsson-Tropp randomized SVD with subspace iteration.

    The sketch key folds in `(m, n, k)`, so different shapes use different
    Ω. Same-shape calls reuse the same Ω — that's intentional: stable
    truncations across iters of the same compile, and rSVD doesn't need
    fresh randomness for accuracy (the Halko bounds depend on `p`/`q`).
    """
    m, n = mat.shape
    key_re = jax.random.fold_in(_RSVD_KEY, m * 1009 + n * 31 + k)
    key_im = jax.random.fold_in(key_re, 1)
    Omega = (jax.random.normal(key_re, (n, k + p)) +
             1j * jax.random.normal(key_im, (n, k + p)))

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
def gate_env_for_polar(L, A1, A2, B1, B2, R):
    """Environment tensor in the shape `polar_from_env` expects —
    unconjugated, i.e., the same thing `env_and_polar_update` computes
    internally before its SVD step. Separated so callers can modify
    the env (e.g., add a diversity regularizer) before taking the
    polar factor.
    """
    return jnp.einsum('ab,acde,efgh,bick,kjfl,hl->dgij',
                       L, A1, A2, B1, B2, R, optimize=True)

_POLAR_NEWTON_ITER = 7


@jax.jit
def _polar_newton_4x4(M):
    """Polar factor of 4×4 M via Newton iteration with closed-form 4×4 inverse.

    Iteration:  X_{k+1} = 0.5 · (X_k + (X_k^H)^{-1})
    Quadratically convergent for nonsingular M; ~5 iters reach machine eps
    for moderately conditioned env. The matmul + jnp.linalg.inv pattern
    fuses with the surrounding sweep ops in XLA, which is the launch-bound
    win on GPU vs the opaque `jnp.linalg.svd` polar (one cusolver call
    per update that doesn't fuse with anything).

    Followed by one cubic Newton-Schulz polish step (pure matmul) to
    clean any residual non-unitarity.

    Selected via env var `TNO_POLAR_METHOD=newton`. Default is `svd`
    because Newton diverges on near-singular env (early in compile when
    init gates are random and target is identity); SVD is unconditionally
    robust. Use `newton` once you've confirmed env is well-conditioned
    in your workload.
    """
    norm = jnp.sqrt(jnp.sum(jnp.real(M.conj() * M)) + 1e-30)
    X = M / jnp.sqrt(norm)
    for _ in range(_POLAR_NEWTON_ITER):
        Xh_inv = jnp.linalg.inv(X.conj().T)
        X = 0.5 * (X + Xh_inv)
    XH_X = X.conj().T @ X
    X = X @ (1.5 * jnp.eye(4, dtype=M.dtype) - 0.5 * XH_X)
    return X


@jax.jit
def _polar_svd_4x4(M):
    """Polar factor via SVD — unconditionally robust, default."""
    u, _, vh = jnp.linalg.svd(M, full_matrices=False)
    return u @ vh


def _select_polar():
    """Pick the polar implementation once at import based on env var."""
    import os
    method = os.environ.get("TNO_POLAR_METHOD", "svd").lower()
    if method == "svd":
        return _polar_svd_4x4
    if method == "newton":
        return _polar_newton_4x4
    raise ValueError(
        f"TNO_POLAR_METHOD must be 'svd' or 'newton', got {method!r}")


_polar_4x4 = _select_polar()


@jax.jit
def env_and_polar_update(L, A1, A2, B1, B2, R):
    """Fused: compute environment + polar decomposition update."""
    env = jnp.einsum('ab,acde,efgh,bick,kjfl,hl->dgij',
                      L, A1, A2, B1, B2, R, optimize=True)
    return _polar_4x4(env.conj().reshape(4, 4)).reshape(2, 2, 2, 2)


@jax.jit
def polar_from_env(env):
    """Polar factor of a (2,2,2,2) environment tensor.

    Separate from `env_and_polar_update` so the caller can modify
    the env between its computation and the polar step — used by the
    diversity-regularized batched sweep.
    """
    return _polar_4x4(env.conj().reshape(4, 4)).reshape(2, 2, 2, 2)


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


# --- Batched (vmap'd) variants ---
#
# Each takes a leading batch dimension on every JAX-array argument.
# Static / non-array arguments (e.g. `canonical`, `max_bond`) have
# `in_axes=None`. Behavior is identical to the unbatched version, just
# vectorized over the batch dim so the XLA compiler can fuse/parallelize.

contract_R_batched = jax.vmap(contract_R, in_axes=0)
contract_L_batched = jax.vmap(contract_L, in_axes=0)
compute_gate_env_batched = jax.vmap(compute_gate_env, in_axes=0)
env_and_polar_update_batched = jax.vmap(env_and_polar_update, in_axes=0)
polar_from_env_batched = jax.vmap(polar_from_env, in_axes=0)
gate_env_for_polar_batched = jax.vmap(gate_env_for_polar, in_axes=0)

absorb_R_left_batched = jax.vmap(absorb_R_left, in_axes=0)
absorb_R_right_batched = jax.vmap(absorb_R_right, in_axes=0)
init_L_batched = jax.vmap(init_L, in_axes=0)
init_R_batched = jax.vmap(init_R, in_axes=0)

_merge_left_batched = jax.vmap(_merge_left, in_axes=0)
_merge_right_batched = jax.vmap(_merge_right, in_axes=0)

_canonicalize_left_batched = jax.vmap(_canonicalize_left, in_axes=0)
_canonicalize_right_batched = jax.vmap(_canonicalize_right, in_axes=0)

# For split/canonicalize, `canonical` and `max_bond` are static
# (already marked via static_argnums in the unbatched jits). vmap
# passes them through unchanged via `in_axes=None`.
_split_full_batched = jax.vmap(_split_full, in_axes=(0, None, None))
_split_randomized_batched = jax.vmap(_split_randomized, in_axes=(0, None, None))


def merge_gate_with_mpo_pair_batched(gate, mpo1, mpo2, gate_is_left=True):
    if gate_is_left:
        return _merge_left_batched(mpo1, mpo2, gate)
    else:
        return _merge_right_batched(gate, mpo1, mpo2)


def canonicalize_tensor_batched(T, left=True):
    if left:
        return _canonicalize_left_batched(T)
    else:
        return _canonicalize_right_batched(T)


def split_merged_tensor_batched(T, canonical='left', max_bond=128):
    """Batched split: T has leading batch dim. The branching on
    matrix size uses T.shape[1:] (unbatched shape).

    Lower threshold than the original `> max_bond * 2` so randomized SVD
    fires whenever the input matrix has meaningful slack above the target
    rank — that is the regime where it actually helps.
    """
    m_phys = T.shape[1] * T.shape[2] * T.shape[3]
    n_phys = T.shape[4] * T.shape[5] * T.shape[6]
    if min(m_phys, n_phys) > max_bond + 16:
        return _split_randomized_batched(T, canonical, max_bond)
    else:
        return _split_full_batched(T, canonical, max_bond)
