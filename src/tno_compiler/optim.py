"""Riemannian ADAM optimizer on the product manifold U(4)^N.

Ported from rqcopt-mpo/util.py and rqcopt-mpo/adam.py.
"""

import numpy as np


def project_tangent(U, Z):
    """Project Z onto the tangent space of U(4) at U.

    Pi_U(Z) = Z - U * sym(U†Z), where sym(A) = (A + A†)/2.
    """
    UdZ = U.conj().T @ Z
    sym = 0.5 * (UdZ + UdZ.conj().T)
    return Z - U @ sym


def retract(U, eta):
    """SVD retraction: map U + eta back to U(4)."""
    u, _, vh = np.linalg.svd(U + eta, full_matrices=False)
    return u @ vh


def riemannian_adam(cost_grad_fn, gates_init, max_iter=1000, lr=1e-3,
                    beta1=0.9, beta2=0.99, eps=1e-8, callback=None):
    """Optimize gates via Riemannian ADAM.

    Args:
        cost_grad_fn: callable(gates_4x4) -> (cost, grad_4x4)
            where gates_4x4 is (N, 4, 4) and grad_4x4 is (N, 4, 4).
        gates_init: initial gates, shape (N, 2, 2, 2, 2).
        max_iter: max iterations.
        lr: learning rate.
        callback: optional callable(step, cost).

    Returns:
        gates: optimized gates, shape (N, 2, 2, 2, 2).
        cost_history: list of costs.
    """
    N = len(gates_init)
    gates = np.array([g.reshape(4, 4) for g in gates_init])

    m = np.zeros_like(gates)
    v = np.zeros(N)
    cost_history = []

    for t in range(1, max_iter + 1):
        gates_tn = np.array([g.reshape(2, 2, 2, 2) for g in gates])
        cost, grad = cost_grad_fn(gates_tn)
        grad = np.array([g.reshape(4, 4) for g in grad])

        # Project Euclidean gradient to Riemannian (negate for descent)
        rie_grad = np.array([
            -project_tangent(gates[i], grad[i]) for i in range(N)])

        cost_history.append(float(cost))
        if callback:
            callback(t, float(cost))

        # ADAM
        m = beta1 * m + (1 - beta1) * rie_grad
        v_new = np.array([np.trace(rie_grad[i].conj().T @ rie_grad[i]).real
                          for i in range(N)])
        v = beta2 * v + (1 - beta2) * v_new

        lr_t = lr * np.sqrt(1 - beta2 ** t) / (1 - beta1 ** t)
        step = np.array([
            -lr_t * m[i] / (np.sqrt(v[i]) + eps) for i in range(N)])

        gates_new = np.array([retract(gates[i], step[i]) for i in range(N)])

        # Parallel transport momentum
        m = np.array([project_tangent(gates_new[i], m[i]) for i in range(N)])
        gates = gates_new

    return np.array([g.reshape(2, 2, 2, 2) for g in gates]), cost_history
