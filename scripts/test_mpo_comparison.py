"""Compare our matrix_to_mpo with rqcopt's get_mpo_from_matrix."""

import sys
sys.path.insert(0, '_reference_rqcopt')

import numpy as np
import jax.numpy as jnp
from jax import config
config.update("jax_enable_x64", True)
from qiskit.quantum_info import random_unitary

from rqcopt_mpo.tn_helpers import get_mpo_from_matrix
from tno_compiler.mpo_ops import matrix_to_mpo

for n in [2, 4]:
    U = random_unitary(2**n, seed=42).data
    mpo_rqcopt = get_mpo_from_matrix(jnp.asarray(U))
    mpo_ours = matrix_to_mpo(U)

    print(f"\nn={n}:")
    for i in range(n):
        r = np.array(mpo_rqcopt[i])
        o = mpo_ours[i]
        print(f"  site {i}: rqcopt={r.shape}, ours={o.shape}, match={np.allclose(r, o, atol=1e-10)}")
