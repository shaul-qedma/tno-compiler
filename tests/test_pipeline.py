"""Tests for the end-to-end compilation pipeline.

Accuracy is verified by sampled max trace distance (the same metric
Kalloor et al. use for their benchmarks, results §Demonstration), so
these tests scale to the same n as statevector simulation — currently
up to ~n=20 on a laptop. The suite is structured to surface where the
compile fails at scale rather than only validate it at small n.
"""

import numpy as np
import pytest

from tno_compiler.brickwall import random_brickwall
from tno_compiler.pipeline import compile_ensemble
from tno_compiler.tfi import tfi_trotter_circuit
from tno_compiler.verify import sampled_max_trace_distance


# Basic structural checks (cheap)

def test_ensemble_produces_valid_weights():
    target = random_brickwall(4, 2, seed=42)
    result = compile_ensemble(target, 2, n_circuits=3, max_iter=30)
    assert np.all(result['weights'] >= -1e-10)
    assert abs(np.sum(result['weights']) - 1) < 1e-8


def test_diamond_bound_is_finite():
    target = random_brickwall(4, 2, seed=42)
    result = compile_ensemble(target, 2, n_circuits=3, max_iter=50)
    assert np.isfinite(result['diamond_bound'])
    assert result['diamond_bound'] >= 0


# Compile-convergence at same-depth on random brickwall — each seed
# should recover the target exactly. Tests the optimizer itself, not
# the ensemble.

_SAME_DEPTH_CASES = [
    # label, n, depth — samples small / medium scale. Comprehensive
    # scaling data lives in docs/data/compression_grid*.csv.
    ("random_n4",  4, 2),
    ("random_n8",  8, 2),
    ("random_n12", 12, 2),
]


@pytest.mark.parametrize("label,n,depth", _SAME_DEPTH_CASES,
                         ids=[c[0] for c in _SAME_DEPTH_CASES])
def test_pipeline_same_depth(label, n, depth):
    target = random_brickwall(n, depth, seed=42)
    result = compile_ensemble(
        target, depth, n_circuits=3, max_iter=200, seed=42)
    out = sampled_max_trace_distance(
        target, result['circuits'], result['weights'],
        n_samples=10, seed=0)
    compile_errs = [f"{c:.2e}" for c in result.get('individual_frobs', [])]
    assert out['max_td'] < 0.05, (
        f"{label}: max_td={out['max_td']:.3e}. compile_errs={compile_errs}")


# Compression on TFI Trotter targets. Random brickwalls at depth D have
# MPO bond ~4^min(k, D-k); compressing to depth D/2 is information-
# theoretically lossy. The right compression benchmark is a structured
# target with bounded entanglement: gapped TFI dynamics at short time.
#
# TFI Hamiltonian H = J·ΣZZ + g·ΣX + h·ΣZ. Gapped regime: |g/J| ≠ 1.

_TFI_COMPRESS_CASES = [
    # label, n, dt, steps, ansatz_depth, max_td_bound
    # Small sample covering: short-time easy, moderate-time 2× compression,
    # scale-up at moderate depth. Comprehensive landscape data is in
    # docs/data/compression_grid*.csv; these tests only guard against
    # regression of the regime where we *expect* the compiler to succeed.
    ("tfi_n8_s2_d2",   8, 0.1,  2, 2, 0.1),
    ("tfi_n8_s4_d2",   8, 0.1,  4, 2, 0.1),
    ("tfi_n8_s12_d4",  8, 0.1, 12, 4, 0.1),
    ("tfi_n12_s4_d2", 12, 0.1,  4, 2, 0.1),
]


@pytest.mark.parametrize("label,n,dt,steps,ansatz_d,max_td_bound",
                         _TFI_COMPRESS_CASES,
                         ids=[c[0] for c in _TFI_COMPRESS_CASES])
def test_pipeline_tfi_compression(label, n, dt, steps, ansatz_d,
                                   max_td_bound):
    target = tfi_trotter_circuit(n, J=-1.0, g=0.5, h=0.0,
                                  dt=dt, steps=steps)
    result = compile_ensemble(
        target, ansatz_d, n_circuits=3, max_iter=200, seed=42)
    out = sampled_max_trace_distance(
        target, result['circuits'], result['weights'],
        n_samples=10, seed=0)
    compile_errs = [f"{c:.2e}" for c in result.get('individual_frobs', [])]
    tgt_depth = target.depth()
    assert out['max_td'] < max_td_bound, (
        f"{label}: target_depth={tgt_depth}, ansatz_d={ansatz_d}, "
        f"max_td={out['max_td']:.3e}. compile_errs={compile_errs}")
