"""Optimizers for brickwall circuit compilation on U(4)^N.

Two methods:

- `polar_sweeps` (primary): alternating-direction polar decomposition
  sweeps (Gibbs & Cincio 2025). Batched over an ensemble of B members;
  a single-circuit compile just passes B=1. Per-sweep cost is O(L),
  and convergence is typically in a handful of sweeps from Trotter
  init or a few tens from Haar-random init.

- `riemannian_adam`: unbatched gradient-based optimization on U(4)^N,
  kept as a reference path and used when an explicit gradient is
  preferred over the polar analytic step.
"""

import jax.numpy as jnp
import numpy as np


# =============================================================================
# Polar decomposition sweeps (Gibbs & Cincio 2025) — batched primary path
# =============================================================================

def polar_sweeps(gates_init_list, max_iter=100, callback=None,
                  target_arrays=None, n_qubits=None, n_layers=None,
                  max_bond=128, first_odd=True,
                  drop_rate=0.0,
                  seed=0, repel_lambda=0.0):
    """Batched polar sweeps: optimize B ensemble members in parallel.

    Args:
        gates_init_list: list of B per-member init-gate lists. Each
            inner element is a list of (2, 2, 2, 2) gate tensors. For a
            single-circuit compile pass `[gates]` to get B=1.
        target_arrays: unbatched target MPO (list of ndarrays). Gets
            broadcast to the batch dim internally.
        max_bond: MPO bond-dim cap during envelope merging.
        drop_rate: per-gate, per-batch-element probability of skipping
            the polar update on each visit (0 disables). Independent
            coins per batch element — members' dropout trajectories
            decorrelate, which helps ensemble diversity.
        drop_rate_schedule: optional schedule for varying `drop_rate`
            across sweep iterations. Supported kinds:
            `{"kind": "linear"|"cosine"|"constant", "start": ..., "end": ...}`.
            If omitted, the fixed `drop_rate` is used for every sweep.
        seed: master RNG seed driving dropout.
        repel_lambda: diversity regularization strength. At each gate
            update, each member is pushed away from the mean of the
            other members' gates at that position. 0 disables. Small
            positive values (0.01–0.1) encourage member separation
            during optimization without destroying accuracy.

    Returns:
        opt_gates_list: list of B optimized gate lists (numpy arrays).
        history_per_member: list of B per-iteration cost histories.
    """
    from .gradient import polar_sweep_batched, polar_sweep_step_jit

    B = len(gates_init_list)
    n_gates = len(gates_init_list[0])

    # Stack members along leading dim once (avoids per-iter conversions).
    gates = [jnp.stack([jnp.asarray(gates_init_list[b][g])
                         for b in range(B)])
             for g in range(n_gates)]
    target_jax = [jnp.broadcast_to(jnp.asarray(a), (B,) + a.shape)
                   for a in target_arrays]

    rng = np.random.default_rng(seed) if drop_rate > 0.0 else None
    per_iter_costs = []  # (B,) array per iter

    # Fast path: no dropout / no repulsion → fuse the whole sweep into one
    # JIT graph per (n_qubits, n_layers, max_bond) shape. Avoids the
    # per-iter Python dispatch overhead that dominates on GPU at B=1.
    use_jit = (drop_rate == 0.0) and (repel_lambda == 0.0)

    for t in range(1, max_iter + 1):
        if use_jit:
            gates_t, cost = polar_sweep_step_jit(
                tuple(target_jax), tuple(gates), n_qubits, n_layers,
                max_bond=max_bond, first_odd=first_odd,
            )
            gates = list(gates_t)
        else:
            cost = polar_sweep_batched(
                target_jax, gates, n_qubits, n_layers,
                max_bond, first_odd,
                drop_rate=drop_rate, rng=rng,
                repel_lambda=repel_lambda,
            )
        per_iter_costs.append(np.asarray(cost))
        if callback:
            callback(t, per_iter_costs[-1])

    opt_gates_list = [[np.asarray(gates[g][b]) for g in range(n_gates)]
                      for b in range(B)]
    history_per_member = [[float(per_iter_costs[t][b])
                           for t in range(max_iter)]
                          for b in range(B)]
    return opt_gates_list, history_per_member


# =============================================================================
# Riemannian ADAM — unbatched, gradient-based (from INMLe/rqcopt-mpo)
# =============================================================================

def _project(U, Z):
    """Project Z onto the tangent space of U(d) at U."""
    UdZ = U.conj().T @ Z
    return Z - U @ (0.5 * (UdZ + UdZ.conj().T))


def _retract(U, eta):
    """Retract U+eta back onto U(d) via polar factor of (U+eta)."""
    u, _, vh = np.linalg.svd(U + eta, full_matrices=False)
    return u @ vh


def riemannian_adam(cost_grad_fn, gates_init, max_iter=1000, lr=1e-3,
                     beta1=0.9, beta2=0.99, eps=1e-8, callback=None,
                     **kwargs):
    """Minimize cost over U(4)^N via Riemannian ADAM.

    Ported from rqcopt (INMLe/rqcopt-mpo). Unbatched — takes a single
    gate list and returns a single optimized gate list.

    Args:
        cost_grad_fn: callable (gates: list of (2,2,2,2)) →
            (cost: float, grad: ndarray (N, 2, 2, 2, 2)).
    """
    N = len(gates_init)
    gates = np.stack([np.asarray(g).reshape(4, 4) for g in gates_init])
    m = np.zeros_like(gates)
    v = np.zeros(N)
    history = []

    for t in range(1, max_iter + 1):
        cost, grad_raw = cost_grad_fn(list(gates.reshape(N, 2, 2, 2, 2)))
        grad = grad_raw.reshape(N, 4, 4)
        history.append(float(cost))
        if callback:
            callback(t, float(cost))

        rg = np.stack([-_project(gates[i], grad[i]) for i in range(N)])

        m = beta1 * m + (1 - beta1) * rg
        v = beta2 * v + (1 - beta2) * np.array(
            [np.trace(rg[i].conj().T @ rg[i]).real for i in range(N)])
        lr_t = lr * np.sqrt(1 - beta2 ** t) / (1 - beta1 ** t)
        step = -lr_t * m / (np.sqrt(v) + eps)[:, None, None]

        gates = np.stack([_retract(gates[i], step[i]) for i in range(N)])
        m = np.stack([_project(gates[i], m[i]) for i in range(N)])

    return list(gates.reshape(N, 2, 2, 2, 2)), history
