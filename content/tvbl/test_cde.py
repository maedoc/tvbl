"""Regression tests for known issues in :mod:`tvbl.cde`.

Each test in this module encodes the *desired* behaviour for one issue found
during review.  The tests deliberately do **not** contain fixes: against the
current implementation they should fail, and each failure message points back
to the underlying bug.

Run from the repository root with::

    PYTHONPATH=content pytest content/tvbl/test_cde.py -v

Issue index
-----------
1. ``MAFEstimator.sample`` cannot sample more than one conditioning input.
2. ``MAFEstimator`` cannot be initialised for ``param_dim == 1``.
3. ``MDNEstimator`` adds ``1e-9`` to mixture weights before taking logs.
4. ``MDNEstimator`` does not bound the exponential precision diagonal.
5. ``ConditionalDensityEstimator.train`` performs two forward passes/iter.
6. ``train`` downcasts data to float32, losing gradient precision.
7. ``MAFEstimator`` initialises the second-layer weights to zero, so the
   first layer receives exactly zero gradient on the first step.
8. ``MAFEstimator`` hard-clips ``alpha`` to ``[-7, 7]``, zeroing gradients
   for saturated scale units.
9. ``MDNEstimator`` duplicates its density implementation in
   ``_loss_function`` and ``log_prob``.
10. ``train`` can partially update weights when a later gradient is
    non-finite.
"""

import inspect
import warnings

import numpy as np
import pytest
from scipy.special import logsumexp

from . import cde


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _as_float64(weights):
    """Return a copy of a weight dict with every array in float64."""
    return {key: np.asarray(value, dtype=np.float64) for key, value in weights.items()}


def _mdn_reference_log_prob(estimator, weights, features, params):
    """Reference MDN log-density using ``log(alpha)``, not ``log(alpha + eps)``."""
    alpha, mu, l_prec, l_prec_log_diag = estimator._forward_pass(weights, features)

    delta = params[:, np.newaxis, :] - mu
    z = np.einsum("nkij,nkj->nki", l_prec, delta)
    quad_term = -0.5 * np.sum(z ** 2, axis=2)
    log_det_term = np.sum(l_prec_log_diag, axis=2)
    log_probs_k = (
        quad_term
        + log_det_term
        - 0.5 * estimator.param_dim * np.log(2 * np.pi)
    )

    # This is the intended formulation: the log mixture weights should be the
    # log-softmax values directly, with no additive epsilon.
    return logsumexp(np.log(alpha) + log_probs_k, axis=1)


class _Float64MDN(cde.MDNEstimator):
    """MDN whose initialization is float64, used to test dtype preservation."""

    def _initialize_weights(self, rng):
        weights = super()._initialize_weights(rng)
        return _as_float64(weights)


class _RecordingMDN(cde.MDNEstimator):
    """MDN that snapshots its initialization before training mutates it."""

    def _initialize_weights(self, rng):
        weights = super()._initialize_weights(rng)
        self.initial_weights = {
            key: np.array(value, copy=True) for key, value in weights.items()
        }
        return weights


# ---------------------------------------------------------------------------
# 1. MAF sampling with multiple conditioning rows
# ---------------------------------------------------------------------------

def test_maf_sample_supports_multiple_conditioning_rows():
    """MAFEstimator.sample must support ``n_conditions > 1``.

    Regression guard for the former shape mismatch where features were
    repeated to ``n_cond * n_samples`` rows while ``z`` had only ``n_samples``.
    """
    estimator = cde.MAFEstimator(param_dim=2, feature_dim=1, n_flows=2, hidden_units=8)
    estimator.weights = estimator._initialize_weights(np.random.RandomState(0))

    features = np.array([[0.0], [1.0]], dtype=np.float32)
    n_samples = 7

    try:
        samples = estimator.sample(features, n_samples, np.random.RandomState(1))
    except Exception as exc:  # noqa: BLE001 - we want the original failure text
        pytest.fail(
            "ISSUE 1: MAFEstimator.sample() cannot sample multiple conditions. "
            "features has n_cond*n_samples rows while z has only n_samples rows; "
            "expected output shape (2, 7, 2). Original error: "
            f"{type(exc).__name__}: {exc}"
        )

    assert samples.shape == (2, n_samples, 2), (
        "ISSUE 1: MAFEstimator.sample() returned the wrong shape for multiple "
        f"conditions: {samples.shape}, expected {(2, n_samples, 2)}."
    )
    assert np.all(np.isfinite(samples)), (
        "ISSUE 1: MAFEstimator.sample() produced non-finite samples for multiple "
        "conditions."
    )


# ---------------------------------------------------------------------------
# 2. MAF initialization with param_dim == 1
# ---------------------------------------------------------------------------

def test_maf_initializes_with_scalar_parameter_dimension():
    """MAFEstimator must be constructible with ``param_dim == 1``.

    Regression guard for the former ``rng.randint(1, 1, size=H)`` failure.
    """
    estimator = cde.MAFEstimator(param_dim=1, feature_dim=1, n_flows=1, hidden_units=4)

    try:
        estimator._initialize_weights(np.random.RandomState(0))
    except ValueError as exc:
        pytest.fail(
            "ISSUE 2: MAFEstimator cannot be initialised with param_dim=1. "
            "The MADE mask generation calls rng.randint(1, D, size=H) with "
            f"D=1. Original error: {exc}"
        )


# ---------------------------------------------------------------------------
# 3. MDN mixture weights should use log_alpha directly
# ---------------------------------------------------------------------------

def test_mdn_log_prob_does_not_add_epsilon_to_mixture_weights():
    """MDN log-probability should use ``log_alpha``, not ``log(alpha + 1e-9)``.

    Adding epsilon changes the normalized mixture density and, more
    importantly, removes the mixture-weight gradient for components whose
    ``alpha`` underflows.  The reference below uses ``log(alpha)`` directly.
    The target point is deliberately far in the tail so the epsilon bias is
    numerically visible.
    """
    estimator = cde.MDNEstimator(
        param_dim=1,
        feature_dim=1,
        n_components=2,
        hidden_sizes=(),
    )
    weights = _as_float64(estimator._initialize_weights(np.random.RandomState(0)))

    # Make one component have a tiny mixture weight but a very large density
    # at the evaluation point, and the other a large weight but a much smaller
    # density.  This amplifies the additive-epsilon bias in the mixture
    # normaliser (with uniform weights the bias cancels to K*eps and is
    # invisible at practical tolerances).
    for value in weights.values():
        value[:] = 0.0
    weights["b_alpha"][:] = [0.0, -20.0]          # alpha ~= [1.0, 2.1e-9]
    weights["b_mu"][:] = [5.0, 0.0]                # dominant mean away from y
    weights["b_L_prec_log_diag"][:] = [0.0, 10.0]  # second component very peaked
    estimator.weights = weights

    features = np.zeros((1, 1), dtype=np.float64)
    params = np.array([[0.0]], dtype=np.float64)

    actual = np.asarray(estimator.log_prob(features, params), dtype=np.float64)
    expected = np.asarray(
        _mdn_reference_log_prob(estimator, weights, features, params),
        dtype=np.float64,
    )

    assert np.allclose(actual, expected, atol=1e-8, rtol=0.0), (
        "ISSUE 3: MDNEstimator.log_prob() uses log(alpha + 1e-9) instead of "
        "log(alpha). The additive epsilon biases the mixture density and zeroes "
        f"mixture-weight gradients under underflow. actual={actual}, "
        f"expected={expected}, difference={actual - expected}."
    )


# ---------------------------------------------------------------------------
# 4. MDN precision diagonal must be numerically bounded
# ---------------------------------------------------------------------------

def test_mdn_precision_diagonal_remains_finite_for_extreme_log_diagonal():
    """MDN precision diagonal must not overflow for extreme log-diagonal values.

    Regression guard for unconstrained ``exp(L_prec_log_diag)``.
    """
    estimator = cde.MDNEstimator(
        param_dim=1,
        feature_dim=1,
        n_components=1,
        hidden_sizes=(),
    )
    weights = _as_float64(estimator._initialize_weights(np.random.RandomState(0)))

    # Force the single precision diagonal to exp(1000), which overflows even
    # float64.  A bounded parameterisation (e.g. a clip) should keep the
    # density finite.
    weights["W_L_prec_log_diag"][:] = 1000.0
    weights["b_L_prec_log_diag"][:] = 0.0
    estimator.weights = weights

    features = np.ones((1, 1), dtype=np.float64)
    params = np.zeros((1, 1), dtype=np.float64)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        log_prob = np.asarray(estimator.log_prob(features, params))

    assert np.all(np.isfinite(log_prob)), (
        "ISSUE 4: MDNEstimator does not bound exp(L_prec_log_diag). "
        "An extreme log-diagonal overflows the precision diagonal and makes "
        f"log_prob non-finite (got {log_prob}). The density/gradient should "
        "remain finite under a bounded parameterisation."
    )


# ---------------------------------------------------------------------------
# 5. Training performs two forward passes per iteration
# ---------------------------------------------------------------------------

def test_train_evaluates_loss_once_per_iteration():
    """Training evaluates loss+grad once per iteration on the NumPy path."""
    estimator = cde.MDNEstimator(
        param_dim=1,
        feature_dim=1,
        n_components=2,
        hidden_sizes=(4,),
    )

    calls = {"n": 0}
    original = estimator._loss_and_grad_np

    def counted(weights, features, params):
        calls["n"] += 1
        return original(weights, features, params)

    estimator._loss_and_grad_np = counted

    rng = np.random.RandomState(0)
    params = rng.randn(32, 1).astype(np.float32)
    features = rng.randn(32, 1).astype(np.float32)
    n_iter = 3

    estimator.train(
        params,
        features,
        n_iter=n_iter,
        learning_rate=1e-3,
        use_tqdm=False,
    )

    assert calls["n"] == n_iter, (
        f"train() evaluated loss+grad {calls['n']} times for {n_iter} "
        "iterations; expected once per iteration."
    )


# ---------------------------------------------------------------------------
# 6. Training should not silently downcast gradients to float32
# ---------------------------------------------------------------------------

def test_train_preserves_float64_precision_for_gradients():
    """Training must not downcast user-provided float64 data to float32."""
    estimator = _Float64MDN(
        param_dim=2,
        feature_dim=2,
        n_components=2,
        hidden_sizes=(4,),
    )

    seen_dtypes = []
    original = estimator._loss_and_grad_np

    def dtype_spy(weights, features, params):
        seen_dtypes.append((np.asarray(features).dtype, np.asarray(params).dtype))
        return original(weights, features, params)

    estimator._loss_and_grad_np = dtype_spy

    rng = np.random.RandomState(0)
    params = rng.randn(32, 2).astype(np.float64)
    features = rng.randn(32, 2).astype(np.float64)

    estimator.train(
        params,
        features,
        n_iter=1,
        learning_rate=1e-3,
        use_tqdm=False,
    )

    assert seen_dtypes, "ISSUE 6: training never evaluated the loss."

    for feature_dtype, param_dtype in seen_dtypes:
        assert feature_dtype == np.float64 and param_dtype == np.float64, (
            "ISSUE 6: train() downcasts data to float32 even when weights are "
            f"float64 (features={feature_dtype}, params={param_dtype}). This "
            "loses gradient precision; measured float32-vs-float64 gradient "
            "error was ~3e-4 in the review."
        )


# ---------------------------------------------------------------------------
# 7. MAF first layer receives no gradient at initialization
# ---------------------------------------------------------------------------

def test_maf_first_layer_receives_gradient_at_initialization():
    """The first MAF layer should receive a non-zero gradient on step one.

    Regression guard for zero initialisation of ``W2``/``W2c``.
    """
    estimator = cde.MAFEstimator(
        param_dim=2,
        feature_dim=1,
        n_flows=1,
        hidden_units=4,
    )
    weights = _as_float64(estimator._initialize_weights(np.random.RandomState(0)))

    rng = np.random.RandomState(1)
    features = rng.randn(16, 1)
    params = rng.randn(16, 2)

    _, gradients = estimator._loss_and_grad_np(weights, features, params)

    first_layer_y = np.asarray(gradients["W1y_0"])
    first_layer_c = np.asarray(gradients["W1c_0"])

    assert np.any(np.abs(first_layer_y) > 0.0), (
        "ISSUE 7: MAFEstimator's first autoregressive layer receives exactly "
        "zero gradient at initialization because W2/W2c are initialised to "
        "zero. Initialise the output layer non-trivially so training starts "
        "on step one."
    )
    assert np.any(np.abs(first_layer_c) > 0.0), (
        "ISSUE 7: MAFEstimator's first conditional layer receives exactly zero "
        "gradient at initialization because W2/W2c are initialised to zero."
    )


# ---------------------------------------------------------------------------
# 8. MAF alpha hard-clipping zeroes gradients
# ---------------------------------------------------------------------------

def test_maf_saturated_alpha_still_receives_gradient():
    """Saturated MAF scale units should still receive a gradient.

    Regression guard for the former hard clip, which zeroed saturated-unit
    gradients.
    """
    estimator = cde.MAFEstimator(
        param_dim=2,
        feature_dim=1,
        n_flows=1,
        hidden_units=4,
    )
    weights = _as_float64(estimator._initialize_weights(np.random.RandomState(0)))

    # Give the output layer non-zero weights and push the scale pre-activations
    # far outside the hard clip range.
    rng = np.random.RandomState(2)
    weights["W2_0"] = 0.1 * rng.randn(*weights["W2_0"].shape)
    weights["b2_0"][estimator.param_dim:] = 100.0

    features = rng.randn(16, 1)
    params = rng.randn(16, 2)

    _, gradients = estimator._loss_and_grad_np(weights, features, params)

    scale_w2_rows = np.asarray(gradients["W2_0"][estimator.param_dim:])
    scale_b2 = np.asarray(gradients["b2_0"][estimator.param_dim:])

    assert np.any(scale_w2_rows != 0.0) or np.any(scale_b2 != 0.0), (
        "ISSUE 8: MAFEstimator hard-clips alpha to [-7, 7], so units whose "
        "pre-activation is outside that interval receive exactly zero "
        "gradient. Use a soft bound or otherwise preserve a gradient for "
        "saturated scale units."
    )


# ---------------------------------------------------------------------------
# 9. MDN duplicates density math between loss and log_prob
# ---------------------------------------------------------------------------

def test_mdn_loss_and_log_prob_share_one_density_implementation():
    """MDN training and evaluation should share one density implementation.

    Regression guard against duplicating the mixture-density calculation in
    ``_loss_function`` and ``log_prob``.
    """
    loss_source = inspect.getsource(cde.MDNEstimator._loss_function)
    log_prob_source = inspect.getsource(cde.MDNEstimator.log_prob)

    duplicated_tokens = [
        token
        for token in ("quad_term", "log_det_term", "logsumexp")
        if token in loss_source and token in log_prob_source
    ]

    assert not duplicated_tokens, (
        "ISSUE 9: MDNEstimator._loss_function and MDNEstimator.log_prob each "
        "contain a separate copy of the mixture-density calculation "
        f"(duplicated tokens: {duplicated_tokens}). Extract the density into "
        "one helper so training and evaluation cannot drift."
    )


# ---------------------------------------------------------------------------
# 10. Training must not partially apply updates on non-finite gradients
# ---------------------------------------------------------------------------

def test_train_does_not_partially_update_on_nonfinite_gradient(monkeypatch):
    """A non-finite gradient in any weight should abort before any update.

    Regression guard against validating gradients one key at a time and
    leaving a partially updated model.
    """
    estimator = _RecordingMDN(
        param_dim=1,
        feature_dim=1,
        n_components=1,
        hidden_sizes=(),
    )

    def fake_loss_and_grad(weights, features, params):
        keys = list(weights)
        gradients = {key: np.ones_like(value) for key, value in weights.items()}
        gradients[keys[-1]] = np.full_like(weights[keys[-1]], np.inf)
        return np.float32(1.0), gradients

    monkeypatch.setattr(estimator, "_loss_and_grad", fake_loss_and_grad)

    rng = np.random.RandomState(0)
    params = rng.randn(16, 1).astype(np.float32)
    features = rng.randn(16, 1).astype(np.float32)

    estimator.train(
        params,
        features,
        n_iter=1,
        learning_rate=1e-3,
        use_tqdm=False,
    )

    for key, initial_value in estimator.initial_weights.items():
        assert np.array_equal(estimator.weights[key], initial_value), (
            "ISSUE 10: train() partially updated weights before discovering a "
            f"non-finite gradient in a later weight ('{key}' was modified). "
            "Validate all gradients before applying any Adam update."
        )
