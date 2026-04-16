"""Riemannian ADAM on U(4)^N."""

import numpy as np


def _project(U, Z):
    """Project Z onto the tangent space at U ∈ U(4)."""
    UdZ = U.conj().T @ Z
    return Z - U @ (0.5 * (UdZ + UdZ.conj().T))


def _retract(U, eta):
    """SVD retraction: polar factor of U + eta."""
    u, _, vh = np.linalg.svd(U + eta, full_matrices=False)
    return u @ vh


def riemannian_adam(cost_grad_fn, gates_init, max_iter=1000, lr=1e-3,
                    beta1=0.9, beta2=0.99, eps=1e-8, callback=None):
    """Minimize cost over U(4)^N via Riemannian ADAM.

    cost_grad_fn(gates) -> (cost, grad) where gates is a list of (2,2,2,2).
    Returns (optimized_gates, cost_history).
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

        # Riemannian gradient (negated for descent)
        rg = np.stack([-_project(gates[i], grad[i]) for i in range(N)])

        # ADAM update
        m = beta1 * m + (1 - beta1) * rg
        v = beta2 * v + (1 - beta2) * np.array(
            [np.trace(rg[i].conj().T @ rg[i]).real for i in range(N)])
        lr_t = lr * np.sqrt(1 - beta2 ** t) / (1 - beta1 ** t)
        step = -lr_t * m / (np.sqrt(v) + eps)[:, None, None]

        gates = np.stack([_retract(gates[i], step[i]) for i in range(N)])
        m = np.stack([_project(gates[i], m[i]) for i in range(N)])

    return list(gates.reshape(N, 2, 2, 2, 2)), history
