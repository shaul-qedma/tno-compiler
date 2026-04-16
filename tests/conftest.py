"""Shared hypothesis strategies for the test suite."""

from hypothesis import strategies as st

n_qubits_st = st.sampled_from([4, 6])
n_layers_st = st.integers(1, 3)
seed_st = st.integers(0, 9999)
