"""`polar_sweep_batched` on B independent inits must agree per-element
with B serial `polar_sweep` runs.

Uses a real compile context (target MPO from a random brickwall) so
every path through the function is exercised.
"""

import jax.numpy as jnp
import numpy as np

from tno_compiler.brickwall import (
    circuit_to_quimb_tn, brickwall_ansatz_gates, random_brickwall,
)
from tno_compiler.compress import tn_to_mpo
from tno_compiler.mpo_ops import mpo_to_arrays
from tno_compiler.gradient import polar_sweep, polar_sweep_batched


def _target_arrays(target, max_bond):
    tn = circuit_to_quimb_tn(target)
    mpo, _ = tn_to_mpo(tn, target.num_qubits, max_bond=max_bond,
                        tol=1e-12, norm="frobenius")
    n = target.num_qubits
    reidx = {f"k{i}": f"b{i}" for i in range(n)}
    reidx.update({f"b{i}": f"k{i}" for i in range(n)})
    return mpo_to_arrays(mpo.conj().reindex(reidx)), max(mpo.bond_sizes())


def _init_gates(n, ansatz_depth, seed):
    qc = random_brickwall(n, ansatz_depth, first_odd=True, seed=seed)
    out = []
    for instr in qc.data:
        mat = np.asarray(instr.operation.to_matrix())
        if mat.shape == (4, 4):
            out.append(mat.reshape(2, 2, 2, 2))
    return out


def test_polar_sweep_batched_equals_serial():
    n = 4
    ansatz_depth = 2
    B = 3
    target = random_brickwall(n, ansatz_depth, seed=0)
    target_arrays, actual_bond = _target_arrays(target, max_bond=16)

    # B independent init gate sets
    inits = [[jnp.asarray(g) for g in _init_gates(n, ansatz_depth, s)]
             for s in [1, 2, 3]]

    # --- Serial: run B independent polar_sweep ---
    serial_gates = [list(g) for g in inits]
    serial_costs = []
    tgt_jax = [jnp.asarray(a) for a in target_arrays]
    for b in range(B):
        cost = polar_sweep(tgt_jax, serial_gates[b], n, ansatz_depth,
                           max_bond=actual_bond, first_odd=True)
        serial_costs.append(float(cost))

    # --- Batched: stack inputs, run once ---
    batched_gates = [jnp.stack([inits[b][g] for b in range(B)])
                     for g in range(len(inits[0]))]
    target_batched = [jnp.broadcast_to(t, (B,) + t.shape) for t in tgt_jax]
    batched_cost = polar_sweep_batched(
        target_batched, batched_gates, n, ansatz_depth,
        max_bond=actual_bond, first_odd=True)

    # Per-element comparison
    for g in range(len(batched_gates)):
        for b in range(B):
            assert jnp.allclose(batched_gates[g][b], serial_gates[b][g],
                                atol=1e-8), \
                f"gate {g}, batch {b}: mismatch"
    for b in range(B):
        assert abs(float(batched_cost[b]) - serial_costs[b]) < 1e-8, \
            f"cost batch {b}: {float(batched_cost[b])} vs {serial_costs[b]}"
