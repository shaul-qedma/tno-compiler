"""Optimizers for circuit compilation on U(4)^N.

Two methods:
- Polar decomposition sweeps (Gibbs & Cincio 2025): analytic
  optimal update per gate via MPO environments.
- Riemannian ADAM (rqcopt, INMLe/rqcopt-mpo): gradient-based
  optimization on the unitary manifold.
"""

import numpy as np


# --- Polar decomposition sweeps ---

def polar_sweeps(cost_grad_fn, gates_init, max_iter=100, callback=None):
    """Optimize gates via sequential polar decomposition sweeps.

    Each sweep updates gates one at a time, recomputing all environments
    after each single-gate update. This is Gauss-Seidel style, matching
    Gibbs & Cincio (2025).

    Slower per sweep than Jacobi (N environment computations per sweep
    instead of 1), but converges reliably.
    """
    gates = list(gates_init)
    history = []

    for t in range(1, max_iter + 1):
        # Update each gate sequentially
        for i in range(len(gates)):
            _, grad = cost_grad_fn(gates)
            env = grad[i].reshape(4, 4)
            u, _, vh = np.linalg.svd(env, full_matrices=False)
            gates[i] = (u @ vh).reshape(2, 2, 2, 2)

        cost, _ = cost_grad_fn(gates)
        history.append(float(cost))
        if callback:
            callback(t, float(cost))

    return gates, history


# --- Riemannian ADAM (from rqcopt, INMLe/rqcopt-mpo) ---

def _project(U, Z):
    UdZ = U.conj().T @ Z
    return Z - U @ (0.5 * (UdZ + UdZ.conj().T))


def _retract(U, eta):
    u, _, vh = np.linalg.svd(U + eta, full_matrices=False)
    return u @ vh


def riemannian_adam(cost_grad_fn, gates_init, max_iter=1000, lr=1e-3,
                    beta1=0.9, beta2=0.99, eps=1e-8, callback=None):
    """Minimize cost over U(4)^N via Riemannian ADAM.

    Ported from rqcopt (INMLe/rqcopt-mpo), adapted from Qiskit's
    ADAM optimizer to the Riemannian setting on U(4).
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
