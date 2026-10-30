"""Tests for the SBI evaluation harness in :mod:`tvbl.sbi_eval`."""

import numpy as np
import pytest

from . import cde, sbi_eval


def _toy_estimator(seed=0):
    est = cde.MAFEstimator(param_dim=2, feature_dim=3, n_flows=2,
                           hidden_units=8)
    return est


def _toy_data(n=48, seed=1):
    rng = np.random.RandomState(seed)
    x = rng.randn(n, 3)
    y = np.stack([x[:, 0] + 0.1 * rng.randn(n),
                  x[:, 1] - x[:, 2] + 0.1 * rng.randn(n)], axis=1)
    return x, y


def test_fit_reduces_loss_and_returns_history():
    est = _toy_estimator()
    x, y = _toy_data()
    history, seconds = sbi_eval.fit(est, x, y, n_iter=60, batch_size=16,
                                    lr=1e-2, seed=0)
    assert history.shape == (60,)
    assert seconds > 0
    assert np.all(np.isfinite(history))
    assert history[-10:].mean() < history[:10].mean()


def test_fit_raises_on_non_finite_loss():
    est = _toy_estimator()
    x, y = _toy_data()
    x_bad = x.copy()
    x_bad[0] = np.inf
    with pytest.raises(RuntimeError, match="non-finite loss"):
        sbi_eval.fit(est, x_bad, y, n_iter=5, batch_size=48, seed=0)


def test_fit_clip_zero_disables_and_clip_changes_trajectory():
    # Saturated alpha units give huge raw gradients; the clip path must be
    # exercised (clip>0) or skipped (clip=0) without breaking the run.
    rng = np.random.RandomState(1)
    x = rng.randn(48, 3) * 1e3
    y = rng.randn(48, 2) * 1e3
    finals = {}
    for clip in (0.0, 5.0):
        est = _toy_estimator()
        history, _ = sbi_eval.fit(est, x, y, n_iter=30, batch_size=16,
                                  lr=3e-3, seed=0, clip=clip)
        assert np.all(np.isfinite(history))
        assert all(np.all(np.isfinite(v)) for v in est.weights.values())
        finals[clip] = history[-1]
    assert finals[0.0] != pytest.approx(finals[5.0])


def test_evaluate_reports_calibration_keys():
    est = _toy_estimator()
    est.weights = est._initialize_weights(np.random.RandomState(0))
    x, y = _toy_data()
    m = sbi_eval.evaluate(est, x, y, n_posterior=32, seed=0)
    for key in ('nll', 'z_std', 'z_bias', 'coverage', 'shrink'):
        assert key in m
        assert np.isfinite(m[key])
    assert m['coverage'] >= 0.0 and m['coverage'] <= 1.0


def test_evaluate_tight_posterior_beats_prior_on_easy_data():
    est = _toy_estimator()
    x, y = _toy_data(n=128)
    sbi_eval.fit(est, x, y, n_iter=200, batch_size=32, lr=5e-3, seed=0)
    m = sbi_eval.evaluate(est, x, y, n_posterior=64, seed=1)
    assert m['shrink'] > 0.5
    assert m['z_std'] > 0.2  # not collapsed


def test_n_params_counts_scalars():
    est = _toy_estimator()
    est.weights = est._initialize_weights(np.random.RandomState(0))
    expect = sum(v.size for v in est.weights.values())
    assert sbi_eval.n_params(est) == int(expect)


def test_trace_features_shape():
    rng = np.random.RandomState(0)
    t = rng.randn(5, 256, 4) + 5 * np.sin(
        2 * np.pi * np.arange(256)[:, None] * 0.05)
    feat = sbi_eval.trace_features(t, fs=100.0, n_bands=6, f_max=20.0)
    assert feat.shape == (5, 4, 3 + 6)
    assert np.all(np.isfinite(feat))


def test_max_corr_and_cross_talk_detect_own_signal():
    rng = np.random.RandomState(0)
    n, nodes = 200, 3
    params = rng.randn(n, nodes)
    # Node j's feature 0 is exactly its own parameter -> own corr ~ 1.
    feat = np.zeros((n, nodes, 2))
    for j in range(nodes):
        feat[:, j, 0] = params[:, j]
        feat[:, j, 1] = rng.randn(n)
    C = sbi_eval.max_corr(feat, params)
    assert C.shape == (nodes, nodes)
    assert np.allclose(np.diag(C), 1.0, atol=1e-6)
    own, cross, ratio = sbi_eval.cross_talk(feat, params)
    assert own == pytest.approx(1.0, abs=1e-6)
    assert cross < 0.2
    assert ratio < 0.2


def test_cross_talk_high_when_features_mix_nodes():
    rng = np.random.RandomState(0)
    n, nodes = 200, 2
    params = rng.randn(n, nodes)
    # Every node's feature tracks *both* parameters equally: graph dominates.
    feat = np.zeros((n, nodes, 1))
    feat[:, :, 0] = params.mean(axis=1)[:, None]
    own, cross, ratio = sbi_eval.cross_talk(feat, params)
    assert ratio > 0.9
