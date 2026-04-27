"""Tests for compile_ensemble_optimal: Stage-1 search + Stage-2 ensemble."""

import numpy as np

from tno_compiler.brickwall import random_brickwall
from tno_compiler.pipeline import compile_ensemble_optimal


def test_finds_optimal_then_runs_ensemble():
    """A loose threshold should find some D* and Stage 2 returns a valid
    ensemble result."""
    target = random_brickwall(4, 3, first_odd=True, seed=0)
    result = compile_ensemble_optimal(
        target, threshold=1e-1, n_circuits=4,
        search_n_seeds=2, search_lo=1, search_hi=6,
        search_max_iter=40, max_iter=60, tol=1e-3,
    )
    assert "optimal_depth" in result
    assert result["optimal_depth"] is not None
    assert result["optimal_depth"] <= 3
    # Stage-2 ensemble keys.
    for k in (
        "weights", "circuits", "delta_ens", "R", "diamond_bound",
        "search", "chosen_depth",
    ):
        assert k in result, f"missing key {k}"
    assert result["chosen_depth"] == result["optimal_depth"]
    assert len(result["circuits"]) == 4
    # Weights sum to 1.
    assert abs(np.sum(result["weights"]) - 1.0) < 1e-6


def test_unreachable_threshold_falls_back():
    """An impossibly tight threshold returns optimal_depth=None but
    still hands back a Stage-2 ensemble at the best probed depth."""
    target = random_brickwall(4, 3, first_odd=True, seed=1)
    result = compile_ensemble_optimal(
        target, threshold=1e-12, n_circuits=3,
        search_n_seeds=1, search_lo=1, search_hi=2,
        search_max_iter=20, max_iter=20, tol=1e-3,
    )
    assert result["optimal_depth"] is None
    assert result["chosen_depth"] in (1, 2)
    assert len(result["circuits"]) == 3
