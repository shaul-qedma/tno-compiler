"""`_optimize_layer_inplace_batched` with B independent inputs must
produce the same per-member results as running the serial version B
times. drop_rate=0 ensures determinism."""

import jax.numpy as jnp
import numpy as np

from tno_compiler.gradient import (
    _optimize_layer_inplace, _optimize_layer_inplace_batched,
)


rng = np.random.default_rng(1)


def _rand_complex(shape):
    return jnp.asarray(rng.standard_normal(shape) + 1j * rng.standard_normal(shape))


def _build_mpo(n, bond=3):
    """n-site MPO: first/last with unit bond, middles with `bond`."""
    out = []
    for i in range(n):
        bl = 1 if i == 0 else bond
        br = 1 if i == n - 1 else bond
        out.append(_rand_complex((bl, 2, 2, br)))
    return out


def _random_unitary(seed):
    rs = np.random.default_rng(seed)
    X = rs.standard_normal((4, 4)) + 1j * rs.standard_normal((4, 4))
    Q, _ = np.linalg.qr(X)
    return jnp.asarray(Q).reshape(2, 2, 2, 2)


def test_batched_layer_equals_serial_runs():
    B = 3
    n = 4  # 4-site MPO → 2 gates in an odd layer
    odd = True

    # Build B independent upper/lower MPO envelopes
    uppers = [_build_mpo(n) for _ in range(B)]
    lowers = [_build_mpo(n) for _ in range(B)]
    init_gates_per_b = [[_random_unitary(100 * b + g) for g in range(2)]
                         for b in range(B)]

    # --- Serial: run B independent unbatched optimizations ---
    serial_outputs = []
    for b in range(B):
        gates = [g for g in init_gates_per_b[b]]
        _optimize_layer_inplace(gates, odd, uppers[b], lowers[b], n_inner=2)
        serial_outputs.append(gates)

    # --- Batched: stack everything along leading dim, run once ---
    batched_gates = [jnp.stack([init_gates_per_b[b][g] for b in range(B)])
                      for g in range(2)]
    batched_upper = [jnp.stack([uppers[b][i] for b in range(B)])
                      for i in range(n)]
    batched_lower = [jnp.stack([lowers[b][i] for b in range(B)])
                      for i in range(n)]
    _optimize_layer_inplace_batched(
        batched_gates, odd, batched_upper, batched_lower, n_inner=2)

    # --- Compare per-element ---
    for g in range(2):
        for b in range(B):
            assert jnp.allclose(batched_gates[g][b], serial_outputs[b][g],
                                atol=1e-9), \
                f"gate {g}, batch {b}: mismatch"


def test_batched_layer_dropout_independent_per_batch():
    """With drop_rate>0, each batch element should receive independent
    coin flips, so B batched elements starting from the same init may
    diverge. (We don't check exact values — only that the batched API
    runs and produces unitary-like (2,2,2,2) tensors.)"""
    B = 3
    n = 4
    odd = True
    uppers = [_build_mpo(n) for _ in range(B)]
    lowers = [_build_mpo(n) for _ in range(B)]

    # All three start from the SAME init to isolate dropout effect
    shared_init = [_random_unitary(g) for g in range(2)]
    batched_gates = [jnp.stack([shared_init[g]] * B) for g in range(2)]
    batched_upper = [jnp.stack([uppers[b][i] for b in range(B)])
                      for i in range(n)]
    batched_lower = [jnp.stack([lowers[b][i] for b in range(B)])
                      for i in range(n)]

    rng_drop = np.random.default_rng(7)
    _optimize_layer_inplace_batched(
        batched_gates, odd, batched_upper, batched_lower, n_inner=2,
        drop_rate=0.3, rng=rng_drop)

    # Sanity: result is finite, right shape.
    for g in range(2):
        assert batched_gates[g].shape == (B, 2, 2, 2, 2)
        assert jnp.all(jnp.isfinite(batched_gates[g]))
