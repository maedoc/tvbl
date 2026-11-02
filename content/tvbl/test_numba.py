"""Validate numba works in this environment.

Runs both natively (regular CPython) and through the wasm path
(pyjs/emscripten-wasm32 via wasm-tests/run.sh), so keep the jit workloads
small and dependency-light.
"""

import numpy as np
import numba
import pytest


def test_numba_version():
    assert numba.__version__


def test_njit_matches_numpy():
    """A jitted saxpy must equal the numpy expression and actually be compiled."""

    @numba.njit
    def saxpy(a, x, y):
        out = np.empty_like(x)
        for i in range(x.shape[0]):
            out[i] = a * x[i] + y[i]
        return out

    x = np.arange(64, dtype=np.float64)
    y = np.linspace(0.0, 1.0, 64)

    np.testing.assert_allclose(saxpy(2.5, x, y), 2.5 * x + y)
    # `signatures` is only populated once a machine function was compiled,
    # i.e. it proves we did not silently fall back to plain python
    assert saxpy.signatures, "numba did not compile a function"


def test_njit_reduction_matches_python():
    @numba.njit
    def total(xs):
        acc = 0.0
        for v in xs:
            acc += v
        return acc

    xs = np.random.default_rng(0).standard_normal(1000)
    assert total(xs) == pytest.approx(float(xs.sum()), rel=1e-10)
    assert total.signatures
