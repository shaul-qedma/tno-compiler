"""Tests for compile_circuit_optimal binary-search."""

import numpy as np
import pytest

from tno_compiler.brickwall import random_brickwall
from tno_compiler.compiler import compile_circuit, compile_circuit_optimal


def test_finds_minimal_depth_for_loose_threshold():
    """A depth-3 random brickwall should be compilable to itself at
    its own depth → optimal depth ≤ 3 for a moderate threshold."""
    target = random_brickwall(4, 3, first_odd=True, seed=0)
    D_opt, _compiled, info, search = compile_circuit_optimal(
        target, threshold=1e-2, lo=1, hi=6, n_seeds=2,
        max_iter=80, tol=1e-3,
    )
    assert D_opt is not None
    assert D_opt <= 3, f"expected D* ≤ 3, got {D_opt}"
    assert info['compile_error'] <= 1e-2


def test_returns_none_for_unreachable_threshold():
    """An impossibly tight threshold should return D_opt=None and
    still hand back the best probe seen."""
    target = random_brickwall(4, 3, first_odd=True, seed=1)
    D_opt, _compiled, info, search = compile_circuit_optimal(
        target, threshold=1e-12, lo=1, hi=2, n_seeds=1,
        max_iter=20, tol=1e-3,
    )
    assert D_opt is None
    # Even unreachable, must return a non-empty search summary.
    assert len(search) > 0
    assert info['compile_error'] > 1e-12


def test_search_explores_log2_probes():
    """Binary search on [1, 8] makes at most log2(8) + 1 ≈ 4 unique
    probes, not 8."""
    target = random_brickwall(4, 2, first_odd=True, seed=2)
    D_opt, _, _, search = compile_circuit_optimal(
        target, threshold=1e-2, lo=1, hi=8, n_seeds=1,
        max_iter=40, tol=1e-3,
    )
    assert D_opt is not None
    assert len(search) <= 4, f"binary search should make ≤4 probes, made {len(search)}"
