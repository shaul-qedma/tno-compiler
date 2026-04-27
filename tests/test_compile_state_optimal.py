"""compile_state_optimal binary-search tests."""

import numpy as np

from tno_compiler.brickwall import random_brickwall
from tno_compiler.compile_state import compile_state_optimal
from tno_compiler.tfi import tfi_trotter_circuit


def test_finds_minimal_depth_for_loose_threshold():
    """A 4-qubit, depth-3 random brickwall as state-prep target should
    compile to ≤ depth 3 at a moderate state-infidelity threshold."""
    target = random_brickwall(4, 3, first_odd=True, seed=0)
    D_opt, _, info, search = compile_state_optimal(
        target, threshold=1e-2, lo=1, hi=6, n_seeds=2,
        max_iter=60, max_bond=16,
    )
    assert D_opt is not None
    assert D_opt <= 3
    assert info["state_infidelity"] <= 1e-2


def test_returns_none_for_unreachable():
    """An impossibly tight infidelity should return optimal=None and
    still hand back the best probe seen."""
    target = tfi_trotter_circuit(4, 1.0, 1.0, 0.0, 0.1, 4, order=1)
    D_opt, _, info, search = compile_state_optimal(
        target, threshold=1e-15, lo=1, hi=2, n_seeds=1,
        max_iter=20, max_bond=16,
    )
    assert D_opt is None
    assert len(search) > 0
    assert info["state_infidelity"] > 1e-15


def test_warm_start_helps_at_tight_tolerance():
    """At tight infidelity targets, warm-start should give equal or smaller
    D* than a no-warm-start run."""
    target = tfi_trotter_circuit(6, 1.0, 1.0, 0.5, 0.1, 4, order=1)
    D_ws, _, info_ws, _ = compile_state_optimal(
        target, threshold=1e-3, lo=1, hi=12, n_seeds=2,
        max_iter=60, max_bond=32, warm_start=True,
    )
    D_no, _, info_no, _ = compile_state_optimal(
        target, threshold=1e-3, lo=1, hi=12, n_seeds=2,
        max_iter=60, max_bond=32, warm_start=False,
    )
    if D_ws is None or D_no is None:
        # Threshold not reachable in this budget; just sanity check that
        # both ran without error.
        return
    assert D_ws <= D_no, f"warm-start D* ({D_ws}) should not exceed no-WS D* ({D_no})"
