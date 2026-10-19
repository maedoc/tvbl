"""High-level Jansen-Rit parameter-recovery test for MDN and MAF.

Two JR parameters are varied (A and a), the output trace is summarized by
mean/variance/PSD features, and the CDE posterior median is checked against
held-out true parameters. Thresholds are deliberately loose enough to be
stable across random seeds while still requiring genuine recovery.
"""

import numpy as np
import pytest
import scipy.signal
from scipy.integrate import odeint

from . import cde


V_MAX, R = 5.0, 0.56
PARAMS = {"A": (3.25, 0.3), "a": (100.0, 10.0)}


def _sigmoid(x, v0):
    return V_MAX / (1.0 + np.exp(R * (v0 - x)))


def _jr_ode(y, t, A, a, B=22.0, b=50.0, v0=6.0, C=135.0):
    y0, y1, y2, y3, y4, y5 = y
    C1 = C2 = C
    C3, C4 = 0.8 * C, 0.25 * C
    p_mu = 220.0
    return [
        y3, y4, y5,
        A * a * _sigmoid(y1 - y2, v0) - 2 * a * y3 - a * a * y0,
        A * a * (p_mu + C2 * _sigmoid(C1 * y0, v0)) - 2 * a * y4 - a * a * y1,
        B * b * C4 * _sigmoid(C3 * y0, v0) - 2 * b * y5 - b * b * y2,
    ]


def _simulate(A, a, fs=200, duration=1.0):
    t = np.arange(0.0, duration, 1.0 / fs)
    sol = odeint(_jr_ode, np.zeros(6), t, args=(A, a), rtol=1e-5, atol=1e-7)
    return sol[:, 1] - sol[:, 2]


def _features(sims, fs=200):
    mean = sims.mean(axis=1, keepdims=True)
    var = sims.var(axis=1, keepdims=True)
    freqs, psd = scipy.signal.welch(sims, fs=fs, nperseg=64)
    keep = (freqs > 0) & (freqs <= 80)
    return np.hstack([mean, var, np.log10(psd[:, keep] + 1e-10)])


def _generate(n, seed, fs=200):
    rng = np.random.RandomState(seed)
    A = rng.randn(n) * PARAMS["A"][1] + PARAMS["A"][0]
    a = rng.randn(n) * PARAMS["a"][1] + PARAMS["a"][0]
    sims = np.array([_simulate(A[i], a[i], fs=fs) for i in range(n)])
    x = _features(sims, fs=fs)
    y = np.c_[(A - PARAMS["A"][0]) / PARAMS["A"][1],
              (a - PARAMS["a"][0]) / PARAMS["a"][1]]
    return y.astype("f"), x.astype("f")


@pytest.fixture(scope="module")
def jr_data():
    y_train, x_train = _generate(400, seed=1)
    y_test, x_test = _generate(24, seed=2)
    mean, std = x_train.mean(0), x_train.std(0)
    std[std < 1e-6] = 1.0
    return (
        y_train,
        ((x_train - mean) / std).astype("f"),
        y_test,
        ((x_test - mean) / std).astype("f"),
    )


@pytest.mark.parametrize("kind", ["mdn", "maf"])
def test_jr_parameter_recovery(jr_data, kind):
    y_train, x_train, y_test, x_test = jr_data
    if kind == "mdn":
        model = cde.MDNEstimator(
            param_dim=2, feature_dim=x_train.shape[1],
            n_components=5, hidden_sizes=(32, 32),
        )
    else:
        model = cde.MAFEstimator(
            param_dim=2, feature_dim=x_train.shape[1],
            n_flows=3, hidden_units=32,
        )

    model.train(y_train, x_train, n_iter=200, learning_rate=1e-3,
                seed=0, use_tqdm=False)
    assert model.loss_history[-1] < model.loss_history[0]

    samples = model.sample(x_test, 300, np.random.RandomState(10))
    assert samples.shape == (len(x_test), 300, 2)
    assert np.all(np.isfinite(samples))

    posterior_median = np.median(samples, axis=1)
    mae = np.mean(np.abs(posterior_median - y_test), axis=0)
    corr = np.array([
        np.corrcoef(posterior_median[:, j], y_test[:, j])[0, 1]
        for j in range(2)
    ])

    assert np.all(mae < 0.5), f"{kind} MAE (prior std units): {mae}"
    assert np.all(corr > 0.7), f"{kind} true-vs-posterior correlation: {corr}"
