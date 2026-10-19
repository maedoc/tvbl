"""Tests for the pure-NumPy MADE forward and backward passes.

Part 1 covers the forward pass of ``MAFEstimator._made_forward_np`` and its
backward cache. Part 2 covers the explicit, hand-written backward pass
``MAFEstimator._made_backward_np``, whose gradients are checked against
``autograd`` (via the original ``_made_forward``) on float64 inputs.

Run from the repository root with::

    PYTHONPATH=content python3 -m pytest content/tvbl/test_maf_numpy.py -q
"""

import numpy as np
import pytest

try:
    import autograd
    import autograd.numpy as anp
    HAS_AUTOGRAD = True
except ImportError:
    autograd = anp = None
    HAS_AUTOGRAD = False

requires_autograd = pytest.mark.skipif(not HAS_AUTOGRAD, reason="autograd not installed")

from . import cde


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_estimator(param_dim=2, feature_dim=3, n_flows=1, hidden_units=5, seed=0):
    """Build a float64 MAFEstimator with freshly initialised constants."""
    est = cde.MAFEstimator(
        param_dim=param_dim,
        feature_dim=feature_dim,
        n_flows=n_flows,
        hidden_units=hidden_units,
    )
    rng = np.random.RandomState(seed)
    est._initialize_weights(rng)
    return est


def _random_weights(est, seed=1):
    """Return a float64 weight dict with non-trivial (non-zero) values."""
    rng = np.random.RandomState(seed)
    weights = est._initialize_weights(np.random.RandomState(seed))
    for key, value in list(weights.items()):
        if key.startswith('W1y') or key.startswith('W1c'):
            scale = 0.5
        elif key.startswith('W2') or key.startswith('b2'):
            scale = 0.7
        else:  # b1
            scale = 0.3
        new = rng.randn(*value.shape) * scale
        if key.startswith('W1c') or key.startswith('W2c'):
            # Make sure the context path is genuinely exercised.
            new = new + 0.4
        weights[key] = np.asarray(new, dtype=np.float64)
    return weights


def _inputs(n=17, param_dim=2, feature_dim=3, seed=2):
    rng = np.random.RandomState(seed)
    y = rng.randn(n, param_dim).astype(np.float64)
    ctx = rng.randn(n, feature_dim).astype(np.float64)
    return y, ctx


def _ref_made_forward(est, y, ctx, layer_const, k, weights):
    M1, M2 = layer_const['M1'], layer_const['M2']
    W1y, W1c, b1 = weights[f'W1y_{k}'], weights[f'W1c_{k}'], weights[f'b1_{k}']
    W2, W2c, b2 = weights[f'W2_{k}'], weights[f'W2c_{k}'], weights[f'b2_{k}']
    y_h = anp.dot(y, (W1y * M1).T)
    c_h = anp.dot(ctx, W1c.T) if est.feature_dim > 0 else 0.0
    h = anp.tanh(y_h + c_h + b1)
    M2_t = anp.concatenate([M2, M2], axis=0)
    out = anp.dot(h, (W2 * M2_t).T)
    if est.feature_dim > 0:
        out = out + anp.dot(ctx, W2c.T)
    out = out + b2
    return out[:, :est.param_dim], 7.0 * anp.tanh(out[:, est.param_dim:] / 7.0)


def _ref_get_log_prob(est, weights, features, params):
    u = params
    log_det = anp.zeros(params.shape[0])
    for k, layer_const in enumerate(est.model_constants['layers']):
        u = u[:, layer_const['perm']]
        mu, alpha = _ref_made_forward(est, u, features, layer_const, k, weights)
        u = (u - mu) * anp.exp(-alpha)
        log_det -= anp.sum(alpha, axis=1)
    base = -0.5 * anp.sum(u ** 2, axis=1) - 0.5 * est.param_dim * anp.log(2.0 * anp.pi)
    return base + log_det


def _ref_loss_function(est, weights, features, params):
    return -anp.mean(_ref_get_log_prob(est, weights, features, params))


# ---------------------------------------------------------------------------
# 1. Numerical equivalence with the existing autograd implementation
# ---------------------------------------------------------------------------

@requires_autograd
def test_mu_alpha_match_made_forward():
    est = _make_estimator()
    weights = _random_weights(est)
    y, ctx = _inputs()
    layer_const = est.model_constants['layers'][0]

    mu_ref, alpha_ref = _ref_made_forward(est, y, ctx, layer_const, 0, weights)
    mu_np, alpha_np, _ = est._made_forward_np(weights, y, ctx, layer_const, 0)

    assert mu_np.shape == (y.shape[0], est.param_dim)
    assert alpha_np.shape == (y.shape[0], est.param_dim)
    np.testing.assert_allclose(mu_np, np.asarray(mu_ref), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(alpha_np, np.asarray(alpha_ref), rtol=1e-12, atol=1e-12)


@requires_autograd
def test_mu_alpha_match_made_forward_multiple_layers():
    """Same equivalence for a later layer index (different weight keys)."""
    est = _make_estimator(n_flows=3, seed=3)
    weights = _random_weights(est, seed=4)
    y, ctx = _inputs(n=9, seed=5)

    for k, layer_const in enumerate(est.model_constants['layers']):
        y_perm = y[:, layer_const['perm']]
        mu_ref, alpha_ref = _ref_made_forward(est, y_perm, ctx, layer_const, k, weights)
        mu_np, alpha_np, _ = est._made_forward_np(weights, y_perm, ctx, layer_const, k)
        np.testing.assert_allclose(mu_np, np.asarray(mu_ref), rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(alpha_np, np.asarray(alpha_ref), rtol=1e-12, atol=1e-12)


# ---------------------------------------------------------------------------
# 2. Cache entries against independently recomputed expected values
# ---------------------------------------------------------------------------

def test_cache_matches_independent_recomputation():
    est = _make_estimator()
    D, C, H = est.param_dim, est.feature_dim, est.hidden_units
    weights = _random_weights(est)
    y, ctx = _inputs()
    layer_const = est.model_constants['layers'][0]
    M1, M2 = layer_const['M1'], layer_const['M2']

    mu, alpha, cache = est._made_forward_np(weights, y, ctx, layer_const, 0)

    W1y = weights['W1y_0']
    W1c = weights['W1c_0']
    b1 = weights['b1_0']
    W2 = weights['W2_0']
    W2c = weights['W2c_0']
    b2 = weights['b2_0']

    # Masked weights.
    W1y_masked = W1y * M1
    M2_tiled = np.concatenate([M2, M2], axis=0)
    W2_masked = W2 * M2_tiled

    # Hidden path.
    y_h = y @ W1y_masked.T
    c_h = ctx @ W1c.T
    hidden_pre = y_h + c_h + b1
    h = np.tanh(hidden_pre)

    # Output path.
    out = h @ W2_masked.T + ctx @ W2c.T + b2
    out_mu = out[:, :D]
    out_alpha_preclip = out[:, D:]
    expected_alpha = 7.0 * np.tanh(out_alpha_preclip / 7.0)
    expected_deriv = 1.0 - (expected_alpha / 7.0) ** 2

    np.testing.assert_allclose(cache['p'], y, rtol=0, atol=0)
    np.testing.assert_allclose(cache['ctx'], ctx, rtol=0, atol=0)
    np.testing.assert_allclose(cache['M1'], M1, rtol=0, atol=0)
    np.testing.assert_allclose(cache['M2'], M2, rtol=0, atol=0)
    np.testing.assert_allclose(cache['W1y_masked'], W1y_masked, rtol=0, atol=0)
    np.testing.assert_allclose(cache['W2_masked'], W2_masked, rtol=0, atol=0)
    # The masked weights really are masked (not accidentally dense).
    assert np.all(cache['W1y_masked'][M1 == 0] == 0.0)
    assert np.all(cache['W2_masked'][M2_tiled == 0] == 0.0)
    assert np.any(cache['W1y_masked'] != 0.0)
    assert np.any(cache['W2_masked'] != 0.0)

    np.testing.assert_allclose(cache['y_h'], y_h, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(cache['c_h'], c_h, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(cache['hidden_pre'], hidden_pre, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(cache['h'], h, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(cache['c_out'], ctx @ W2c.T, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(cache['out'], out, rtol=1e-12, atol=1e-12)

    assert cache['out'].shape == (y.shape[0], 2 * D)
    np.testing.assert_allclose(cache['out_alpha_preclip'], out_alpha_preclip,
                               rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(cache['mu'], out_mu, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(cache['alpha'], expected_alpha,
                               rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(cache['alpha_preclip'], out_alpha_preclip,
                               rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(cache['alpha_deriv'], expected_deriv,
                               rtol=1e-12, atol=1e-12)
    assert cache['param_dim'] == D

    # Returned values agree with the cache.
    np.testing.assert_allclose(mu, cache['mu'], rtol=0, atol=0)
    np.testing.assert_allclose(alpha, cache['alpha'], rtol=0, atol=0)
    np.testing.assert_allclose(alpha, expected_alpha, rtol=0, atol=0)

    # Shapes of intermediate quantities.
    assert cache['W1y_masked'].shape == (H, D)
    assert cache['W2_masked'].shape == (2 * D, H)
    assert cache['y_h'].shape == (y.shape[0], H)
    assert cache['c_h'].shape == (y.shape[0], H)
    assert cache['h'].shape == (y.shape[0], H)
    assert cache['out_alpha_preclip'].shape == (y.shape[0], D)
    assert cache['alpha_deriv'].shape == (y.shape[0], D)


# ---------------------------------------------------------------------------
# 3. Saturated alpha -> clip mask must be False exactly where clipped
# ---------------------------------------------------------------------------

@requires_autograd
def test_alpha_derivative_marks_saturated_entries():
    # param_dim=3 so the three alpha columns can be driven to the lower
    # saturation, left unsaturated, and driven to the upper saturation.
    est = _make_estimator(param_dim=3, seed=9)
    D = est.param_dim
    weights = _random_weights(est)
    y, ctx = _inputs(n=13, param_dim=D, seed=10)
    layer_const = est.model_constants['layers'][0]

    # Make the alpha pre-activation exactly equal to b2 (per sample) so the
    # expected clip pattern is fully deterministic.
    weights['W2_0'] = weights['W2_0'].copy()
    weights['W2_0'][D:, :] = 0.0
    weights['W2c_0'] = np.zeros_like(weights['W2c_0'])
    weights['b2_0'] = weights['b2_0'].copy()
    weights['b2_0'][D:] = np.array([-40.0, 0.0, 40.0])

    mu, alpha, cache = est._made_forward_np(weights, y, ctx, layer_const, 0)

    preclip = cache['out_alpha_preclip']
    expected_alpha = 7.0 * np.tanh(preclip / 7.0)
    expected_deriv = 1.0 - (expected_alpha / 7.0) ** 2

    # One saturated-low, one unsaturated, one saturated-high alpha column.
    assert np.all(preclip[:, 0] == -40.0)
    assert np.all(preclip[:, 1] == 0.0)
    assert np.all(preclip[:, 2] == 40.0)
    np.testing.assert_allclose(alpha, expected_alpha, rtol=0, atol=0)
    np.testing.assert_allclose(cache['alpha_deriv'], expected_deriv,
                               rtol=0, atol=0)
    assert np.all(alpha[:, 0] > -7.0)
    assert np.all(alpha[:, 2] < 7.0)
    assert np.all(expected_deriv[:, 0] < 1e-4)
    assert np.all(expected_deriv[:, 1] == 1.0)
    assert np.all(expected_deriv[:, 2] < 1e-4)

    # Saturated comparisons still agree with the autograd implementation.
    mu_ref, alpha_ref = _ref_made_forward(est, y, ctx, layer_const, 0, weights)
    np.testing.assert_allclose(mu, np.asarray(mu_ref), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(alpha, np.asarray(alpha_ref), rtol=1e-12, atol=1e-12)


# ---------------------------------------------------------------------------
# 4. feature_dim == 0 (no-context branch)
# ---------------------------------------------------------------------------

@requires_autograd
def test_no_context_branch_matches_made_forward():
    est = _make_estimator(feature_dim=0, seed=6)
    weights = _random_weights(est, seed=7)
    y, _ = _inputs(n=11, feature_dim=0, seed=8)
    ctx = np.zeros((y.shape[0], 0), dtype=np.float64)
    layer_const = est.model_constants['layers'][0]

    mu_ref, alpha_ref = _ref_made_forward(est, y, ctx, layer_const, 0, weights)
    mu_np, alpha_np, cache = est._made_forward_np(weights, y, ctx, layer_const, 0)

    np.testing.assert_allclose(mu_np, np.asarray(mu_ref), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(alpha_np, np.asarray(alpha_ref), rtol=1e-12, atol=1e-12)

    # Mirror the scalar-0.0 context branch of ``_made_forward``.
    assert cache['c_h'] == 0.0
    assert cache['c_out'] is None

    D = est.param_dim
    expected_h = np.tanh(y @ (weights['W1y_0'] * layer_const['M1']).T + weights['b1_0'])
    M2_tiled = np.concatenate([layer_const['M2'], layer_const['M2']], axis=0)
    expected_out = expected_h @ (weights['W2_0'] * M2_tiled).T + weights['b2_0']
    np.testing.assert_allclose(cache['h'], expected_h, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(cache['out'], expected_out, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(cache['mu'], expected_out[:, :D], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(cache['out_alpha_preclip'], expected_out[:, D:],
                               rtol=1e-12, atol=1e-12)


# ---------------------------------------------------------------------------
# 5. Plain NumPy only: no autograd boxes / tracers involved
# ---------------------------------------------------------------------------

@requires_autograd
def test_all_outputs_are_plain_numpy_arrays():
    est = _make_estimator()
    weights = _random_weights(est)
    y, ctx = _inputs()
    layer_const = est.model_constants['layers'][0]

    mu, alpha, cache = est._made_forward_np(weights, y, ctx, layer_const, 0)

    for name, value in [('mu', mu), ('alpha', alpha)] + list(cache.items()):
        if name == 'param_dim':
            assert isinstance(value, int)
            continue
        if name == 'c_out' and value is None:
            continue
        if name == 'c_h' and np.isscalar(value):
            assert isinstance(value, float)
            continue
        assert type(value) is np.ndarray, f"{name} is {type(value)}, not np.ndarray"
        assert not hasattr(value, '_value'), f"{name} looks like an autograd box"


@requires_autograd
def test_does_not_touch_autograd():
    """Guard: forward must work when the autograd box machinery is bypassed."""
    est = _make_estimator()
    weights = _random_weights(est)
    y, ctx = _inputs()
    layer_const = est.model_constants['layers'][0]

    from autograd.numpy import numpy_boxes as boxes

    original = boxes.ArrayBox.__init__

    def _forbidden(self, *args, **kwargs):  # pragma: no cover - guard only
        raise AssertionError("_made_forward_np constructed an autograd ArrayBox")

    boxes.ArrayBox.__init__ = _forbidden
    try:
        mu, alpha, cache = est._made_forward_np(weights, y, ctx, layer_const, 0)
    finally:
        boxes.ArrayBox.__init__ = original

    assert type(mu) is np.ndarray
    assert type(alpha) is np.ndarray
    assert type(cache['h']) is np.ndarray


# ===========================================================================
# Part 2: explicit backward pass (``_made_backward_np``)
# ===========================================================================

GRAD_NAMES = ('W1y', 'W1c', 'b1', 'W2', 'W2c', 'b2')


def _grad_keys(k):
    return [f'{name}_{k}' for name in GRAD_NAMES]


def _made_objective(est, y, ctx, layer_const, k, r_mu, r_alpha):
    """Scalar objective for autograd: <mu, r_mu> + <alpha, r_alpha>."""

    def objective(weights):
        mu, alpha = _ref_made_forward(est, y, ctx, layer_const, k, weights)
        return anp.sum(mu * r_mu) + anp.sum(alpha * r_alpha)

    return objective


def _parity_setup(k, param_dim=3, feature_dim=3, n_flows=3, hidden_units=6,
                  n=11, seeds=(11, 12, 13, 14)):
    est = _make_estimator(param_dim=param_dim, feature_dim=feature_dim,
                          n_flows=n_flows, hidden_units=hidden_units,
                          seed=seeds[0])
    weights = _random_weights(est, seed=seeds[1])
    y, ctx = _inputs(n=n, param_dim=param_dim, feature_dim=feature_dim,
                     seed=seeds[2])
    layer_const = est.model_constants['layers'][k]
    rng = np.random.RandomState(seeds[3])
    r_mu = rng.randn(*y.shape)
    r_alpha = rng.randn(*y.shape)
    return est, weights, y, ctx, layer_const, r_mu, r_alpha


# ---------------------------------------------------------------------------
# 1. Weight-gradient parity against autograd
# ---------------------------------------------------------------------------

@requires_autograd
@pytest.mark.parametrize('k', [0, 2])
def test_weight_gradients_match_autograd(k):
    est, weights, y, ctx, layer_const, r_mu, r_alpha = _parity_setup(k)

    objective = _made_objective(est, y, ctx, layer_const, k, r_mu, r_alpha)
    g_ref = autograd.grad(objective)(weights)

    mu, alpha, cache = est._made_forward_np(weights, y, ctx, layer_const, k)
    # The clip must be inactive for a clean parity comparison.
    assert np.all(np.isfinite(alpha))
    assert np.all(np.abs(alpha) < 7.0 - 1e-6)
    assert np.all(cache['alpha_deriv'] > 0.0)

    grads, bar_p, bar_ctx = est._made_backward_np(
        weights, cache, k, r_mu, r_alpha)

    assert set(grads) == set(_grad_keys(k))
    for name in _grad_keys(k):
        assert grads[name].shape == weights[name].shape
        np.testing.assert_allclose(grads[name], np.asarray(g_ref[name]),
                                   rtol=1e-9, atol=1e-9)

    # The masks must be honoured exactly (masked-out weights keep zero grad).
    M1, M2 = layer_const['M1'], layer_const['M2']
    M2_tiled = np.concatenate([M2, M2], axis=0)
    assert np.all(grads[f'W1y_{k}'][M1 == 0] == 0.0)
    assert np.all(grads[f'W2_{k}'][M2_tiled == 0] == 0.0)
    # ... and the surviving entries are genuinely exercised.
    assert np.any(grads[f'W1y_{k}'][M1 != 0] != 0.0)
    assert np.any(grads[f'W2_{k}'][M2_tiled != 0] != 0.0)


@requires_autograd
def test_weight_gradients_match_autograd_k0_single_flow():
    """k = 0 of a single-flow estimator (the simplest weight naming case)."""
    est, weights, y, ctx, layer_const, r_mu, r_alpha = _parity_setup(
        0, n_flows=1, seeds=(31, 32, 33, 34))
    objective = _made_objective(est, y, ctx, layer_const, 0, r_mu, r_alpha)
    g_ref = autograd.grad(objective)(weights)

    mu, alpha, cache = est._made_forward_np(weights, y, ctx, layer_const, 0)
    assert np.all(np.abs(alpha) < 7.0 - 1e-6)
    grads, _, _ = est._made_backward_np(weights, cache, 0, r_mu, r_alpha)

    for name in _grad_keys(0):
        np.testing.assert_allclose(grads[name], np.asarray(g_ref[name]),
                                   rtol=1e-9, atol=1e-9)


# ---------------------------------------------------------------------------
# 2. Input-gradient parity (bar_p, bar_ctx) against autograd
# ---------------------------------------------------------------------------

@requires_autograd
def test_input_gradients_match_autograd():
    k = 1
    est, weights, y, ctx, layer_const, r_mu, r_alpha = _parity_setup(k)

    mu, alpha, cache = est._made_forward_np(weights, y, ctx, layer_const, k)
    assert np.all(np.abs(alpha) < 7.0 - 1e-6)
    grads, bar_p, bar_ctx = est._made_backward_np(
        weights, cache, k, r_mu, r_alpha)

    def objective_y(y_in):
        mu_y, alpha_y = _ref_made_forward(est, y_in, ctx, layer_const, k, weights)
        return anp.sum(mu_y * r_mu) + anp.sum(alpha_y * r_alpha)

    def objective_ctx(ctx_in):
        mu_c, alpha_c = _ref_made_forward(est, y, ctx_in, layer_const, k, weights)
        return anp.sum(mu_c * r_mu) + anp.sum(alpha_c * r_alpha)

    g_y = autograd.grad(objective_y)(y)
    g_ctx = autograd.grad(objective_ctx)(ctx)

    assert bar_p.shape == y.shape
    assert bar_ctx.shape == ctx.shape
    np.testing.assert_allclose(bar_p, np.asarray(g_y), rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(bar_ctx, np.asarray(g_ctx), rtol=1e-9, atol=1e-9)


def test_input_gradient_split_matches_individual_paths():
    """bar_ctx is the sum of the hidden-context and output-context paths."""
    k = 0
    est, weights, y, ctx, layer_const, r_mu, r_alpha = _parity_setup(k)
    mu, alpha, cache = est._made_forward_np(weights, y, ctx, layer_const, k)
    _, bar_p, bar_ctx = est._made_backward_np(
        weights, cache, k, r_mu, r_alpha)

    bar_alpha = r_alpha * cache['alpha_deriv']
    bar_out = np.concatenate([r_mu, bar_alpha], axis=1)
    bar_h = bar_out @ cache['W2_masked']
    bar_a = bar_h * (1.0 - cache['h'] ** 2)

    expected_bar_p = bar_a @ cache['W1y_masked']
    expected_bar_ctx = bar_a @ cache['W1c'] + bar_out @ cache['W2c']

    np.testing.assert_allclose(bar_p, expected_bar_p, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(bar_ctx, expected_bar_ctx, rtol=1e-12, atol=1e-12)


# ---------------------------------------------------------------------------
# 3. Clip handling: saturated alpha columns carry exactly zero gradient
# ---------------------------------------------------------------------------

@requires_autograd
def test_clip_saturates_alpha_gradients():
    D = 3
    est = _make_estimator(param_dim=D, feature_dim=2, n_flows=1,
                          hidden_units=5, seed=41)
    weights = _random_weights(est, seed=42)
    y, ctx = _inputs(n=9, param_dim=D, feature_dim=2, seed=43)
    layer_const = est.model_constants['layers'][0]

    # Force the alpha pre-activations to be exactly [-40, 0, +40] per sample
    # so the clip pattern is deterministic: one saturated-low column, one
    # unsaturated column and one saturated-high column.
    weights['W2_0'] = weights['W2_0'].copy()
    weights['W2_0'][D:, :] = 0.0
    weights['W2c_0'] = np.zeros_like(weights['W2c_0'])
    weights['b2_0'] = weights['b2_0'].copy()
    weights['b2_0'][D:] = np.array([-40.0, 0.0, 40.0])

    rng = np.random.RandomState(44)
    r_mu = rng.randn(y.shape[0], D)
    r_alpha = rng.randn(y.shape[0], D)

    mu, alpha, cache = est._made_forward_np(weights, y, ctx, layer_const, 0)
    deriv = cache['alpha_deriv']
    assert np.all(deriv[:, 0] < 1e-4)
    assert np.all(deriv[:, 1] == 1.0)
    assert np.all(deriv[:, 2] < 1e-4)
    np.testing.assert_allclose(alpha[:, 0], -7.0, atol=1e-3)
    np.testing.assert_allclose(alpha[:, 1], 0.0, atol=1e-12)
    np.testing.assert_allclose(alpha[:, 2], 7.0, atol=1e-3)

    grads, bar_p, bar_ctx = est._made_backward_np(
        weights, cache, 0, r_mu, r_alpha)

    for value in list(grads.values()) + [bar_p, bar_ctx]:
        assert np.all(np.isfinite(value))

    # Saturated columns still carry a finite gradient.
    for value in grads.values():
        assert np.all(np.isfinite(value))
    assert np.any(grads['W2_0'][D + 1] != 0.0)
    assert grads['b2_0'][D + 1] != 0.0

    # Large adjoints in the saturated columns stay finite.
    r_alpha_huge = r_alpha.copy()
    r_alpha_huge[:, 0] = 1e6
    r_alpha_huge[:, 2] = -1e6
    grads_huge, bar_p_huge, bar_ctx_huge = est._made_backward_np(
        weights, cache, 0, r_mu, r_alpha_huge)
    for value in list(grads_huge.values()) + [bar_p_huge, bar_ctx_huge]:
        assert np.all(np.isfinite(value))

    # Autograd agrees with the smooth bounded transform.
    objective = _made_objective(est, y, ctx, layer_const, 0, r_mu, r_alpha)
    g_ref = autograd.grad(objective)(weights)
    for name in _grad_keys(0):
        np.testing.assert_allclose(grads[name], np.asarray(g_ref[name]),
                                   rtol=1e-9, atol=1e-9)


# ---------------------------------------------------------------------------
# 4. feature_dim == 0 (no context path)
# ---------------------------------------------------------------------------

@requires_autograd
def test_backward_without_features():
    D = 2
    est = _make_estimator(param_dim=D, feature_dim=0, n_flows=2,
                          hidden_units=4, seed=51)
    weights = _random_weights(est, seed=52)
    y, _ = _inputs(n=7, param_dim=D, feature_dim=0, seed=53)
    ctx = np.zeros((y.shape[0], 0), dtype=np.float64)
    layer_const = est.model_constants['layers'][1]
    k = 1

    rng = np.random.RandomState(54)
    r_mu = rng.randn(y.shape[0], D)
    r_alpha = rng.randn(y.shape[0], D)

    mu, alpha, cache = est._made_forward_np(weights, y, ctx, layer_const, k)
    assert np.all(np.abs(alpha) < 7.0 - 1e-6)
    grads, bar_p, bar_ctx = est._made_backward_np(
        weights, cache, k, r_mu, r_alpha)

    # Shapes follow the weight shapes, including the empty context weights.
    assert weights[f'W1c_{k}'].shape == (est.hidden_units, 0)
    assert weights[f'W2c_{k}'].shape == (2 * D, 0)
    for name in _grad_keys(k):
        assert grads[name].shape == weights[name].shape
    assert grads[f'W1c_{k}'].shape == (est.hidden_units, 0)
    assert grads[f'W2c_{k}'].shape == (2 * D, 0)
    assert bar_p.shape == (y.shape[0], D)
    assert bar_ctx.shape == (y.shape[0], 0)

    # Parity with autograd still holds for the non-empty paths.
    objective = _made_objective(est, y, ctx, layer_const, k, r_mu, r_alpha)
    g_ref = autograd.grad(objective)(weights)
    for name in _grad_keys(k):
        np.testing.assert_allclose(grads[name], np.asarray(g_ref[name]),
                                   rtol=1e-9, atol=1e-9)

    def objective_y(y_in):
        mu_y, alpha_y = _ref_made_forward(est, y_in, ctx, layer_const, k, weights)
        return anp.sum(mu_y * r_mu) + anp.sum(alpha_y * r_alpha)

    g_y = autograd.grad(objective_y)(y)
    np.testing.assert_allclose(bar_p, np.asarray(g_y), rtol=1e-9, atol=1e-9)


# ---------------------------------------------------------------------------
# 5. Everything returned by the backward pass is plain NumPy
# ---------------------------------------------------------------------------

@requires_autograd
def test_backward_returns_plain_numpy_arrays():
    est, weights, y, ctx, layer_const, r_mu, r_alpha = _parity_setup(0)
    mu, alpha, cache = est._made_forward_np(weights, y, ctx, layer_const, 0)

    grads, bar_p, bar_ctx = est._made_backward_np(
        weights, cache, 0, r_mu, r_alpha)

    assert isinstance(grads, dict)
    for name, value in grads.items():
        assert type(value) is np.ndarray, f"{name} is {type(value)}"
        assert value.dtype == np.float64
        assert not hasattr(value, '_value'), f"{name} looks like an autograd box"
    for name, value in [('bar_p', bar_p), ('bar_ctx', bar_ctx)]:
        assert type(value) is np.ndarray, f"{name} is {type(value)}"
        assert value.dtype == np.float64
        assert not hasattr(value, '_value'), f"{name} looks like an autograd box"


@requires_autograd
def test_backward_does_not_touch_autograd():
    """Guard: the backward pass must not build autograd ArrayBoxes."""
    from autograd.numpy import numpy_boxes as boxes

    est, weights, y, ctx, layer_const, r_mu, r_alpha = _parity_setup(0)
    mu, alpha, cache = est._made_forward_np(weights, y, ctx, layer_const, 0)

    original = boxes.ArrayBox.__init__

    def _forbidden(self, *args, **kwargs):  # pragma: no cover - guard only
        raise AssertionError("_made_backward_np constructed an autograd ArrayBox")

    boxes.ArrayBox.__init__ = _forbidden
    try:
        grads, bar_p, bar_ctx = est._made_backward_np(
            weights, cache, 0, r_mu, r_alpha)
    finally:
        boxes.ArrayBox.__init__ = original

    assert type(bar_p) is np.ndarray
    assert type(bar_ctx) is np.ndarray
    assert all(type(v) is np.ndarray for v in grads.values())


# ===========================================================================
# Part 3: full multi-flow forward/backward pass (``_get_log_prob_np``,
# ``_flow_backward_np`` and ``_loss_and_grad_np``)
# ===========================================================================

FLOW_CONFIGS = [
    # (param_dim, feature_dim, n_flows, seed)
    (1, 0, 1, 201),
    (1, 2, 1, 202),
    (1, 0, 2, 203),
    (1, 2, 2, 204),
    (1, 0, 4, 205),
    (1, 2, 4, 206),
    (3, 0, 1, 207),
    (3, 2, 1, 208),
    (3, 0, 2, 209),
    (3, 2, 2, 210),
    (3, 0, 4, 211),
    (3, 2, 4, 212),
]
FLOW_CONFIG_IDS = [f'D{d}_C{c}_K{k}' for d, c, k, _ in FLOW_CONFIGS]


def _make_d1_estimator_and_weights(feature_dim, n_flows, seed,
                                   hidden_units=6):
    """Manual D=1 setup.

    ``_initialize_weights`` uses ``rng.randint(1, D, size=H)``, which has no
    valid range for ``D == 1`` (the existing implementation was never meant
    for a scalar target).  Build an equivalent D=1 MADE configuration directly
    here, without touching the production code.
    """
    est = cde.MAFEstimator(
        param_dim=1,
        feature_dim=feature_dim,
        n_flows=n_flows,
        hidden_units=hidden_units,
    )
    rng = np.random.RandomState(seed)
    D, C, H = 1, feature_dim, hidden_units
    layers = []
    weights = {}

    for k in range(n_flows):
        m_in = np.arange(1, D + 1)
        # For D == 1 the original range ``[1, D)`` is empty.  Use hidden
        # ranks 0 and 1: these keep the mask hierarchy valid and exercise both
        # the hidden and output paths.
        m_hidden = (np.arange(H) % 2).astype(int)
        M1 = (m_in[None, :] <= m_hidden[:, None]).astype('f')
        m_out = m_in.copy()
        M2 = (m_hidden[None, :] < m_out[:, None]).astype('f')
        perm = rng.permutation(D)
        inv_perm = np.empty(D, dtype=int)
        inv_perm[perm] = np.arange(D)
        layers.append({'M1': M1, 'M2': M2, 'perm': perm,
                       'inv_perm': inv_perm})

        weights[f'W1y_{k}'] = rng.randn(H, D) * 0.5
        if C > 0:
            weights[f'W1c_{k}'] = rng.randn(H, C) * 0.5 + 0.4
            weights[f'W2c_{k}'] = rng.randn(2 * D, C) * 0.7 + 0.4
        else:
            weights[f'W1c_{k}'] = np.zeros((H, C))
            weights[f'W2c_{k}'] = np.zeros((2 * D, C))
        weights[f'b1_{k}'] = rng.randn(H) * 0.3
        weights[f'W2_{k}'] = rng.randn(2 * D, H) * 0.7
        weights[f'b2_{k}'] = rng.randn(2 * D) * 0.7

    est.model_constants = {'layers': layers}
    return est, weights


def _flow_setup(param_dim, feature_dim, n_flows, seed, n=13, hidden_units=6):
    """Build an estimator, float64 weights, features and parameters."""
    if param_dim == 1:
        est, weights = _make_d1_estimator_and_weights(
            feature_dim, n_flows, seed + 1, hidden_units=hidden_units)
    else:
        est = _make_estimator(param_dim=param_dim, feature_dim=feature_dim,
                              n_flows=n_flows, hidden_units=hidden_units,
                              seed=seed)
        weights = _random_weights(est, seed=seed + 1)
    params, features = _inputs(n=n, param_dim=param_dim,
                               feature_dim=feature_dim, seed=seed + 2)
    return est, weights, features, params


@requires_autograd
@pytest.mark.parametrize('param_dim,feature_dim,n_flows,seed', FLOW_CONFIGS,
                         ids=FLOW_CONFIG_IDS)
def test_get_log_prob_np_matches_autograd(param_dim, feature_dim, n_flows,
                                          seed):
    est, weights, features, params = _flow_setup(
        param_dim, feature_dim, n_flows, seed)
    params_before = params.copy()

    log_prob_ref = _ref_get_log_prob(est, weights, features, params)
    log_prob, caches = est._get_log_prob_np(weights, features, params)

    # The input parameters are never mutated.
    np.testing.assert_array_equal(params, params_before)

    assert log_prob.shape == (params.shape[0],)
    assert len(caches) == n_flows
    np.testing.assert_allclose(log_prob, np.asarray(log_prob_ref),
                               rtol=1e-12, atol=1e-12)

    # Per-flow cache invariants.
    u_prev = params_before
    identity = np.arange(param_dim)
    for c in caches:
        np.testing.assert_array_equal(c['u_in'], u_prev)
        np.testing.assert_array_equal(c['p'], c['u_in'][:, c['perm']])
        np.testing.assert_array_equal(c['block']['p'], c['p'])
        np.testing.assert_array_equal(c['inv_perm'][c['perm']], identity)
        np.testing.assert_array_equal(c['perm'][c['inv_perm']], identity)
        np.testing.assert_allclose(
            c['exp_neg_alpha'], np.exp(-c['block']['alpha']),
            rtol=0, atol=0)
        np.testing.assert_allclose(
            c['u_out'],
            (c['p'] - c['block']['mu']) * c['exp_neg_alpha'],
            rtol=1e-12, atol=1e-12)
        u_prev = c['u_out']

    # The last flow output is exactly what the base density consumes.
    base_logp = (-0.5 * np.sum(caches[-1]['u_out'] ** 2, axis=1)
                 - 0.5 * param_dim * np.log(2.0 * np.pi))
    log_det = -np.sum([np.sum(c['block']['alpha'], axis=1) for c in caches],
                      axis=0)
    np.testing.assert_allclose(log_prob, base_logp + log_det,
                               rtol=1e-12, atol=1e-12)


@requires_autograd
@pytest.mark.parametrize('param_dim,feature_dim,n_flows,seed', FLOW_CONFIGS,
                         ids=FLOW_CONFIG_IDS)
def test_loss_and_grad_np_match_autograd(param_dim, feature_dim, n_flows, seed):
    est, weights, features, params = _flow_setup(
        param_dim, feature_dim, n_flows, seed)

    loss, grads = est._loss_and_grad_np(weights, features, params)
    loss_ref = _ref_loss_function(est, weights, features, params)
    g_ref = autograd.grad(lambda w: _ref_loss_function(est, w, features, params))(weights)

    assert isinstance(loss, float)
    np.testing.assert_allclose(loss, float(np.asarray(loss_ref)),
                               rtol=1e-12, atol=1e-12)

    assert set(grads) == set(weights)
    for key in weights:
        assert grads[key].shape == weights[key].shape
        np.testing.assert_allclose(grads[key], np.asarray(g_ref[key]),
                                   rtol=1e-9, atol=1e-9)


@requires_autograd
def test_gradient_parity_with_nonidentity_permutations():
    """A concrete configuration whose per-flow permutations are non-trivial."""
    est, weights, features, params = _flow_setup(3, 2, 4, seed=301)
    identity = np.arange(est.param_dim)
    perms = [np.asarray(layer['perm'])
             for layer in est.model_constants['layers']]

    # Explicitly assert the setup really exercises permutations.
    assert any(not np.array_equal(perm, identity) for perm in perms)
    for layer in est.model_constants['layers']:
        np.testing.assert_array_equal(np.sort(layer['perm']), identity)
        np.testing.assert_array_equal(
            layer['inv_perm'][layer['perm']], identity)

    loss, grads = est._loss_and_grad_np(weights, features, params)
    g_ref = autograd.grad(lambda w: _ref_loss_function(est, w, features, params))(weights)
    for key in weights:
        np.testing.assert_allclose(grads[key], np.asarray(g_ref[key]),
                                   rtol=1e-9, atol=1e-9)


@requires_autograd
def test_loss_gradient_sign_is_negative_log_prob():
    """The returned gradient belongs to the loss, not to the log probability."""
    est, weights, features, params = _flow_setup(3, 2, 2, seed=351)
    loss, grads = est._loss_and_grad_np(weights, features, params)

    def positive_log_prob(w):
        return anp.mean(_ref_get_log_prob(est, w, features, params))

    g_logp = autograd.grad(positive_log_prob)(weights)
    g_loss = autograd.grad(lambda w: _ref_loss_function(est, w, features, params))(weights)

    total_dot_loss = 0.0
    for key in weights:
        np.testing.assert_allclose(grads[key], -np.asarray(g_logp[key]),
                                   rtol=1e-9, atol=1e-9)
        np.testing.assert_allclose(grads[key], np.asarray(g_loss[key]),
                                   rtol=1e-9, atol=1e-9)
        total_dot_loss += float(np.sum(grads[key] * np.asarray(g_loss[key])))

    # Same direction as the loss gradient (and non-zero), i.e. the opposite
    # direction of the positive log-probability gradient.
    assert total_dot_loss > 0.0


@requires_autograd
def test_saturated_alpha_gradient_finite_and_masked():
    D = 3
    est = _make_estimator(param_dim=D, feature_dim=2, n_flows=2,
                          hidden_units=5, seed=401)
    weights = _random_weights(est, seed=402)
    params, features = _inputs(n=9, param_dim=D, feature_dim=2, seed=403)

    # Drive the alpha pre-activations of every flow to [-40, 0, +40] so the
    # clip pattern is deterministic: saturated-low, unsaturated, saturated-high.
    for k in range(est.n_flows):
        weights[f'W2_{k}'] = weights[f'W2_{k}'].copy()
        weights[f'W2_{k}'][D:, :] = 0.0
        weights[f'W2c_{k}'] = np.zeros_like(weights[f'W2c_{k}'])
        weights[f'b2_{k}'] = weights[f'b2_{k}'].copy()
        weights[f'b2_{k}'][D:] = np.array([-40.0, 0.0, 40.0])

    log_prob, caches = est._get_log_prob_np(weights, features, params)
    for c in caches:
        deriv = c['block']['alpha_deriv']
        assert np.all(deriv[:, 0] < 1e-4)
        assert np.all(deriv[:, 1] == 1.0)
        assert np.all(deriv[:, 2] < 1e-4)
        np.testing.assert_allclose(c['block']['alpha'][:, 0], -7.0, atol=1e-3)
        np.testing.assert_allclose(c['block']['alpha'][:, 1], 0.0, atol=1e-12)
        np.testing.assert_allclose(c['block']['alpha'][:, 2], 7.0, atol=1e-3)

    loss, grads = est._loss_and_grad_np(weights, features, params)
    assert np.isfinite(loss)
    for value in grads.values():
        assert np.all(np.isfinite(value))

    for k in range(est.n_flows):
        # Saturated columns receive finite gradients.
        for value in grads.values():
            assert np.all(np.isfinite(value))
        assert np.any(grads[f'W2_{k}'][D + 1, :] != 0.0)
        assert grads[f'b2_{k}'][D + 1] != 0.0

    # Autograd agrees with the smooth bounded transform.
    g_ref = autograd.grad(lambda w: _ref_loss_function(est, w, features, params))(weights)
    for key in weights:
        np.testing.assert_allclose(grads[key], np.asarray(g_ref[key]),
                                   rtol=1e-9, atol=1e-9)


@requires_autograd
def test_flow_feature_dim_zero_parity():
    est, weights, features, params = _flow_setup(3, 0, 2, seed=451)
    assert features.shape == (params.shape[0], 0)

    log_prob_ref = _ref_get_log_prob(est, weights, features, params)
    log_prob, caches = est._get_log_prob_np(weights, features, params)
    np.testing.assert_allclose(log_prob, np.asarray(log_prob_ref),
                               rtol=1e-12, atol=1e-12)
    for c in caches:
        assert c['block']['c_h'] == 0.0
        assert c['block']['c_out'] is None

    loss, grads = est._loss_and_grad_np(weights, features, params)
    loss_ref = _ref_loss_function(est, weights, features, params)
    g_ref = autograd.grad(lambda w: _ref_loss_function(est, w, features, params))(weights)
    np.testing.assert_allclose(loss, float(np.asarray(loss_ref)),
                               rtol=1e-12, atol=1e-12)
    for key in weights:
        assert grads[key].shape == weights[key].shape
        np.testing.assert_allclose(grads[key], np.asarray(g_ref[key]),
                                   rtol=1e-9, atol=1e-9)
    for k in range(est.n_flows):
        assert grads[f'W1c_{k}'].shape == (est.hidden_units, 0)
        assert grads[f'W2c_{k}'].shape == (2 * est.param_dim, 0)


@requires_autograd
def test_loss_and_grad_np_returns_plain_numpy():
    est, weights, features, params = _flow_setup(2, 2, 2, seed=501)

    log_prob, caches = est._get_log_prob_np(weights, features, params)
    assert type(log_prob) is np.ndarray
    assert log_prob.dtype == np.float64
    assert not hasattr(log_prob, '_value')
    for c in caches:
        assert type(c['u_in']) is np.ndarray
        assert type(c['p']) is np.ndarray
        assert type(c['exp_neg_alpha']) is np.ndarray
        assert type(c['u_out']) is np.ndarray

    loss, grads = est._loss_and_grad_np(weights, features, params)
    assert type(loss) is float
    assert set(grads) == set(weights)
    for key, value in grads.items():
        assert type(value) is np.ndarray, f"{key} is {type(value)}"
        assert value.dtype == np.float64
        assert not hasattr(value, '_value'), f"{key} looks like an autograd box"


# ---------------------------------------------------------------------------
# 4. Train-time backend wiring
# ---------------------------------------------------------------------------

def _float32_setup(param_dim=3, feature_dim=2, n_flows=3, seed=601, n=13,
                   hidden_units=6, weight_scale=0.25):
    """Same as :func:`_flow_setup` but with float32 weights/data.

    ``weight_scale`` keeps the setup in a well-conditioned regime (|gradients|
    of order 1).  The raw :func:`_random_weights` scales produce gradients up
    to ~1e8 for three stacked flows, where float32 cancellation between
    individual terms (not the algorithm) dominates the residual error.
    """
    est = _make_estimator(param_dim=param_dim, feature_dim=feature_dim,
                          n_flows=n_flows, hidden_units=hidden_units,
                          seed=seed)
    weights = {
        key: np.asarray(value * weight_scale, dtype=np.float32)
        for key, value in _random_weights(est, seed=seed + 1).items()}
    params, features = _inputs(n=n, param_dim=param_dim,
                               feature_dim=feature_dim, seed=seed + 2)
    return (est, weights, features.astype(np.float32),
            params.astype(np.float32))


def _train_data(n=200, param_dim=2, feature_dim=2, seed=700):
    """A small, genuinely conditional float32 training set."""
    rng = np.random.RandomState(seed)
    params = rng.randn(n, param_dim).astype(np.float32)
    features = rng.randn(n, feature_dim).astype(np.float32)
    if param_dim > 1:
        params[:, 1] = params[:, 0] ** 2 + 0.1 * params[:, 1]
    features[:, 0] = features[:, 1] + 0.5 * params[:, 0]
    return params, features


@requires_autograd
def test_float32_loss_and_grad_parity_with_autograd():
    """The NumPy path stays in float32 and agrees with autograd."""
    est, weights, features, params = _float32_setup()
    for key, value in weights.items():
        assert value.dtype == np.float32, key

    loss, grads = est._loss_and_grad_np(weights, features, params)
    loss_ref = _ref_loss_function(est, weights, features, params)
    g_ref = autograd.grad(lambda w: _ref_loss_function(est, w, features, params))(weights)

    print(f"float32 loss parity |numpy - autograd| = "
          f"{abs(loss - float(np.asarray(loss_ref))):.3e}")
    np.testing.assert_allclose(loss, float(np.asarray(loss_ref)),
                               rtol=1e-6, atol=1e-6)

    assert set(grads) == set(weights)
    max_abs_err = 0.0
    for key in weights:
        assert grads[key].shape == weights[key].shape
        # Not silently promoted to float64 by the NumPy path.
        assert grads[key].dtype == np.float32, (
            f"{key}: gradient dtype {grads[key].dtype}, expected float32")
        ref = np.asarray(g_ref[key], dtype=np.float32)
        max_abs_err = max(
            max_abs_err,
            float(np.max(np.abs(grads[key] - ref), initial=0.0)))
        np.testing.assert_allclose(grads[key], ref, rtol=1e-5, atol=1e-5)

    print(f"float32 gradient parity max abs error = {max_abs_err:.3e}")
    assert max_abs_err < 1e-3
