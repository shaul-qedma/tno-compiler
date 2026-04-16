"""Tests for the end-to-end compilation pipeline."""

import numpy as np

from conftest import seed_st
from tno_compiler.brickwall import random_brickwall
from tno_compiler.pipeline import compile_ensemble


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


def test_compression_works():
    """Compile a depth-4 target to a depth-2 ansatz."""
    target = random_brickwall(4, 4, seed=42)
    result = compile_ensemble(target, 2, n_circuits=3, max_iter=100)
    assert np.isfinite(result['diamond_bound'])
    assert len(result['circuits']) == 3
