"""Hierarchical (multi-scale) brickwall compile.

Goal: fix the GPU launch-overhead problem by replacing the single-pass
2-qubit brickwall optimization with two stages:

  Stage 1 (coarse): brickwall of K-qubit tile gates (K ∈ {3, 4, 5}).
      Each tile is a 2^K × 2^K unitary, optimized via polar sweep with
      tile-sized SVDs. Bigger SVD per launch ⇒ better cusolver
      arithmetic intensity ⇒ higher GPU util.

  Stage 2 (fine): each Stage-1 tile is a 2^K × 2^K unitary; decompose
      each independently into a brickwall of 2-qubit gates via the
      existing compiler. K independent compiles, perfectly batchable
      via `compile_targets_batched` from pipeline.py.

Total work is similar to the flat compile but the hot kernels operate
on bigger matrices, which is what the GPU actually rewards.

Status: SCAFFOLDING. Core K-qubit polar update + K-qubit brickwall
ansatz machinery are TODO. Wired only to make the design concrete and
let downstream consumers be written against the eventual API.

Implementation plan (in order):
  1. K-qubit brickwall ansatz layout (analog of brickwall_ansatz_gates).
  2. K-qubit gate ↔ MPO merge ops (jax_ops.py: merge_gate_with_mpo_K).
  3. K-qubit polar update (jax_ops.py: env_and_polar_update_K_batched).
  4. K-qubit polar_sweep_batched (gradient.py).
  5. Stage-2 decomposition: pass each K-qubit tile as a target to
     compile_targets_batched with a small 2-qubit brickwall ansatz.
"""

from __future__ import annotations

import numpy as np


def hierarchical_compile(target,
                          tile_qubits: int = 4,
                          coarse_layers: int = 4,
                          fine_layers: int = 6,
                          n_seeds: int = 8,
                          tol: float = 1e-3,
                          max_bond: int = 64,
                          max_iter_coarse: int = 60,
                          max_iter_fine: int = 60,
                          seed: int = 0):
    """Two-stage hierarchical brickwall compile.

    Args:
        target: qiskit QuantumCircuit on n qubits.
        tile_qubits: K — width of the Stage-1 tile gates. Pick the
            largest K such that 2^K × 2^K SVD fits the GPU's cusolver
            sweet spot (typically K ∈ {3, 4, 5}).
        coarse_layers: number of K-qubit brickwall layers in Stage 1.
        fine_layers: number of 2-qubit brickwall layers per tile in Stage 2.
        n_seeds: ensemble size for the Stage-1 batched polar sweep.
        tol, max_bond: target MPO compression knobs (Stage 1 only).
        max_iter_coarse / max_iter_fine: polar sweep iters per stage.
        seed: master RNG seed.

    Returns:
        compiled: qiskit QuantumCircuit equivalent to a brickwall of
            2-qubit gates (post Stage-2 decomposition). Composition of
            (coarse_layers × fine_layers × ... ) 2-qubit gates.
        info: dict with stage timings, costs, and per-tile breakdowns.

    Raises:
        NotImplementedError: until the K-qubit kernels land. See module
            docstring for implementation order.
    """
    raise NotImplementedError(
        "Hierarchical compile is scaffolded but not yet implemented. "
        "See `src/tno_compiler/hierarchical.py` module docstring for "
        "the four-step implementation plan. Until it lands, use "
        "`compile_circuit` (flat, n_seeds≥64 on GPU) or "
        "`compile_targets_batched` (K independent compiles in one batch)."
    )


def _coarse_stage_K_qubit_brickwall(target, tile_qubits, n_layers, n_seeds,
                                      tol, max_bond, max_iter, seed):
    """Stage 1: optimize a K-qubit brickwall against the target MPO.

    TODO: requires K-qubit analogues of:
      - brickwall_ansatz_gates (with K-stride pairs)
      - merge_gate_with_mpo_pair (K consecutive MPO sites)
      - env_and_polar_update_batched (2^K × 2^K SVD, batched)
    """
    raise NotImplementedError


def _fine_stage_decompose_tiles(tile_unitaries, tile_qubits, n_layers,
                                  n_seeds, max_iter, seed):
    """Stage 2: decompose each K-qubit tile into a 2-qubit brickwall.

    Each tile is a 2^K × 2^K unitary (well-conditioned by construction
    from Stage 1). Build a small synthetic K-qubit "target circuit"
    that implements the unitary, then dispatch all K_tiles through
    `compile_targets_batched` so they share a single batched polar
    sweep with B = K_tiles × n_seeds.

    TODO: building a K-qubit synthetic target from a unitary needs a
    QSD or similar — could also just embed the unitary as a multi-qubit
    gate and let `circuit_to_mpo` do the rest.
    """
    raise NotImplementedError
