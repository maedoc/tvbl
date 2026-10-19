"""Parity tests for the plain-NumPy MDN forward pass and log-prob helper.

Run from the repository root with::

    PYTHONPATH=content python3 -m pytest content/tvbl/test_mdn_numpy.py -q
"""

import itertools

import numpy as np
import pytest

try:
    import autograd
    import autograd.numpy as anp
    from autograd.scipy.special import logsumexp as ag_logsumexp
    HAS_AUTOGRAD = True
except ImportError:
    autograd = anp = ag_logsumexp = None
    HAS_AUTOGRAD = False

requires_autograd = pytest.mark.skipif(not HAS_AUTOGRAD, reason="autograd not installed")

from . import cde


GRID = [
    case for case in itertools.product(
        [1, 2, 4],          # param_dim
        [0, 3],             # feature_dim
        [2, 4],             # n_components
        [(), (8,), (8, 8)],  # hidden_sizes
        [np.float32, np.float64],
    )
    # _initialize_weights divides by feature_dim for the first hidden layer.
    if not (case[1] == 0 and case[3] != ())
]


def _case(param_dim, feature_dim, n_components, hidden_sizes, dtype, seed=0):
    est = cde.MDNEstimator(
        param_dim=param_dim,
        feature_dim=feature_dim,
        n_components=n_components,
        hidden_sizes=hidden_sizes,
    )
    rng = np.random.RandomState(seed)
    base = est._initialize_weights(np.random.RandomState(seed))
    weights = {}
    for key, value in base.items():
        scale = 0.3 if key.startswith('b') else 0.5
        new = rng.randn(*value.shape) * scale + 0.2
        weights[key] = np.asarray(new, dtype=dtype)
    est.weights = weights

    n = 9
    features = rng.randn(n, feature_dim).astype(dtype)
    params = rng.randn(n, param_dim).astype(dtype)
    return est, weights, features, params


def _tol(dtype):
    return (1e-5, 1e-5) if dtype is np.float32 else (1e-12, 1e-12)


def _ref_forward_pass(est, weights, features):
    h = features
    for i in range(len(est.hidden_sizes)):
        h = anp.tanh(h @ weights[f'W{i}'] + weights[f'b{i}'])
    K, D = est.n_components, est.param_dim
    log_alpha = h @ weights['W_alpha'] + weights['b_alpha']
    alpha = anp.exp(log_alpha - ag_logsumexp(log_alpha, axis=1, keepdims=True))
    mu = (h @ weights['W_mu'] + weights['b_mu']).reshape(-1, K, D)
    ld_raw = (h @ weights['W_L_prec_log_diag'] + weights['b_L_prec_log_diag']).reshape(-1, K, D)
    # Mirror the implementation's bound on the precision log-diagonal.
    ld = anp.clip(ld_raw, -cde.MDN_LOG_DIAG_LIMIT, cde.MDN_LOG_DIAG_LIMIT)
    L = anp.einsum('nki,ij->nkij', anp.exp(ld), anp.eye(D, dtype='f'))
    M = est._offdiag_rows.size
    if M:
        off = (h @ weights['W_L_prec_offdiag'] + weights['b_L_prec_offdiag']).reshape(-1, K, M)
        L = L + anp.einsum('nkl,lij->nkij', off, est._offdiag_basis)
    return alpha, mu, L, ld, log_alpha


def _ref_loss_function(est, weights, features, params):
    alpha, mu, L, ld, log_alpha = _ref_forward_pass(est, weights, features)
    delta = params[:, None, :] - mu
    z = anp.einsum('nkij,nkj->nki', L, delta)
    lp = -0.5 * anp.sum(z ** 2, axis=2) + anp.sum(ld, axis=2) - 0.5 * est.param_dim * anp.log(2 * np.pi)
    total = ag_logsumexp(log_alpha + lp, axis=1)
    return -anp.mean(total)


@requires_autograd
@pytest.mark.parametrize("param_dim,feature_dim,n_components,hidden_sizes,dtype", GRID)
def test_forward_pass_np_matches_autograd(param_dim, feature_dim, n_components, hidden_sizes, dtype):
    est, weights, features, _ = _case(param_dim, feature_dim, n_components, hidden_sizes, dtype)

    ref = _ref_forward_pass(est, weights, features)
    out = est._forward_pass_np(weights, features)

    alpha, mu, l_prec, ld, cache = out
    rtol, atol = _tol(dtype)
    assert alpha.shape == (features.shape[0], n_components)
    assert mu.shape == (features.shape[0], n_components, param_dim)
    assert l_prec.shape == (features.shape[0], n_components, param_dim, param_dim)
    assert ld.shape == mu.shape

    for got, want in zip(out[:4], ref[:4]):
        np.testing.assert_allclose(np.asarray(got, dtype=np.float64),
                                   np.asarray(want, dtype=np.float64),
                                   rtol=rtol, atol=atol)

    assert cache['h'] is cache['hs'][-1]
    assert len(cache['hs']) == len(hidden_sizes) + 1
    np.testing.assert_allclose(alpha.sum(axis=1), np.ones(features.shape[0]),
                               rtol=rtol, atol=atol)
    log_alpha = cache['log_alpha']
    lse = np.log(np.exp(log_alpha).sum(axis=1, keepdims=True))
    np.testing.assert_allclose(np.log(alpha), log_alpha - lse, rtol=1e-4, atol=1e-4)


def np_logsumexp_ref(alpha):
    return np.log(alpha.sum(axis=1, keepdims=True))


@requires_autograd
@pytest.mark.parametrize("param_dim,feature_dim,n_components,hidden_sizes,dtype", GRID)
def test_log_prob_from_cache_matches_autograd(param_dim, feature_dim, n_components, hidden_sizes, dtype):
    est, weights, features, params = _case(param_dim, feature_dim, n_components, hidden_sizes, dtype)

    alpha, mu, l_prec, ld, cache = est._forward_pass_np(weights, features)
    log_prob, resp, delta, z = est._mdn_log_prob_from_cache(cache['log_alpha'], mu, l_prec, ld, params)

    loss_ref = _ref_loss_function(est, weights, features, params)
    lp_ref = est.log_prob(features, params)

    rtol, atol = _tol(dtype)
    assert log_prob.shape == (features.shape[0],)
    assert resp.shape == (features.shape[0], n_components)
    assert delta.shape == mu.shape
    assert z.shape == mu.shape

    np.testing.assert_allclose(log_prob.astype(np.float64), np.asarray(lp_ref, dtype=np.float64),
                               rtol=rtol, atol=atol)
    np.testing.assert_allclose(-np.mean(log_prob, dtype=np.float64), float(loss_ref),
                               rtol=rtol, atol=atol)
    np.testing.assert_allclose(resp.sum(axis=1), np.ones(features.shape[0]),
                               rtol=rtol, atol=atol)
    assert np.all(resp >= 0)


@requires_autograd
def test_specific_d1_and_c0_cases():
    for param_dim, feature_dim, hidden_sizes in [(1, 0, ()), (1, 3, (8,)), (3, 0, ())]:
        est, weights, features, params = _case(param_dim, feature_dim, 3, hidden_sizes, np.float64)
        assert (est._offdiag_rows.size == 0) == (param_dim == 1)
        alpha, mu, l_prec, ld, cache = est._forward_pass_np(weights, features)
        assert (cache['off'] is None) == (param_dim == 1)
        ref = _ref_forward_pass(est, weights, features)
        for got, want in zip((alpha, mu, l_prec, ld), ref[:4]):
            np.testing.assert_allclose(got, np.asarray(want, dtype=np.float64), rtol=1e-12, atol=1e-12)
        log_prob, resp, _, _ = est._mdn_log_prob_from_cache(cache['log_alpha'], mu, l_prec, ld, params)
        np.testing.assert_allclose(log_prob, est.log_prob(features, params), rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(resp.sum(axis=1), np.ones(features.shape[0]), rtol=1e-12, atol=1e-12)


def test_offdiag_populated_and_features_not_mutated():
    est, weights, features, params = _case(4, 3, 3, (8,), np.float64)
    before = features.copy()
    alpha, mu, l_prec, ld, cache = est._forward_pass_np(weights, features)
    np.testing.assert_array_equal(features, before)
    assert cache['off'] is not None
    np.testing.assert_allclose(l_prec[:, :, est._offdiag_rows, est._offdiag_cols], cache['off'])
    est._mdn_log_prob_from_cache(cache['log_alpha'], mu, l_prec, ld, params)
    np.testing.assert_array_equal(features, before)


# ---------------------------------------------------------------------------
# M2: explicit plain-NumPy loss + gradient
# ---------------------------------------------------------------------------

def _autograd_grads(est, weights, features, params):
    return autograd.grad(lambda w: _ref_loss_function(est, w, features, params))(weights)


def _grad_tol(dtype):
    return (1e-5, 1e-5) if dtype is np.float32 else (1e-10, 1e-12)


def _max_rel_err(got, want):
    got = np.asarray(got, dtype=np.float64)
    want = np.asarray(want, dtype=np.float64)
    if got.size == 0:
        return 0.0
    return float(np.max(np.abs(got - want)) / max(np.max(np.abs(want)), 1e-12))


@requires_autograd
@pytest.mark.parametrize("param_dim,feature_dim,n_components,hidden_sizes,dtype", GRID)
def test_loss_and_grad_np_matches_autograd(param_dim, feature_dim, n_components, hidden_sizes, dtype):
    est, weights, features, params = _case(param_dim, feature_dim, n_components, hidden_sizes, dtype)

    _, grads = est._loss_and_grad_np(weights, features, params)
    ref = _autograd_grads(est, weights, features, params)

    rtol, atol = _grad_tol(dtype)
    worst = 0.0
    for key in weights:
        got = np.asarray(grads[key], dtype=np.float64)
        want = np.asarray(ref[key], dtype=np.float64)
        np.testing.assert_allclose(got, want, rtol=rtol, atol=atol, err_msg=key)
        worst = max(worst, _max_rel_err(got, want))
    print(f"max rel grad err {np.dtype(dtype).name}: {worst:.3e}")


@requires_autograd
@pytest.mark.parametrize("param_dim,feature_dim,n_components,hidden_sizes,dtype", GRID)
def test_loss_parity(param_dim, feature_dim, n_components, hidden_sizes, dtype):
    est, weights, features, params = _case(param_dim, feature_dim, n_components, hidden_sizes, dtype)

    loss, _ = est._loss_and_grad_np(weights, features, params)
    ref = float(_ref_loss_function(est, weights, features, params))

    alpha, mu, l_prec, ld, cache = est._forward_pass_np(weights, features)
    log_prob, _, _, _ = est._mdn_log_prob_from_cache(cache['log_alpha'], mu, l_prec, ld, params)
    manual = float(-np.mean(log_prob, dtype=np.float64))

    rtol, atol = _tol(dtype)
    assert isinstance(loss, float)
    np.testing.assert_allclose(loss, ref, rtol=rtol, atol=atol)
    np.testing.assert_allclose(loss, manual, rtol=rtol, atol=atol)


@pytest.mark.parametrize("param_dim,feature_dim,n_components,hidden_sizes,dtype", GRID)
def test_grad_keys_and_shapes(param_dim, feature_dim, n_components, hidden_sizes, dtype):
    est, weights, features, params = _case(param_dim, feature_dim, n_components, hidden_sizes, dtype)

    _, grads = est._loss_and_grad_np(weights, features, params)

    assert set(grads) == set(weights)
    for key, value in grads.items():
        assert value.shape == weights[key].shape, key
    assert ('W_L_prec_offdiag' in grads) == (param_dim > 1)


@pytest.mark.parametrize("param_dim,feature_dim,n_components,hidden_sizes",
                         [case[:4] for case in GRID if case[4] is np.float32])
def test_float32_grads_stay_float32(param_dim, feature_dim, n_components, hidden_sizes):
    est, weights, features, params = _case(param_dim, feature_dim, n_components, hidden_sizes, np.float32)

    loss, grads = est._loss_and_grad_np(weights, features, params)

    assert isinstance(loss, float)
    for key, value in grads.items():
        assert value.dtype == np.float32, (key, value.dtype)


@requires_autograd
def test_finite_difference_spot_check():
    est, weights, features, params = _case(3, 2, 3, (5,), np.float64)
    _, grads = est._loss_and_grad_np(weights, features, params)

    rng = np.random.RandomState(7)
    keys = sorted(weights)
    eps = 1e-6
    for _ in range(8):
        key = keys[rng.randint(len(keys))]
        index = tuple(rng.randint(n) for n in weights[key].shape)
        plus = {k: v.copy() for k, v in weights.items()}
        minus = {k: v.copy() for k, v in weights.items()}
        plus[key][index] += eps
        minus[key][index] -= eps
        fd = (float(_ref_loss_function(est, plus, features, params))
              - float(_ref_loss_function(est, minus, features, params))) / (2 * eps)
        np.testing.assert_allclose(grads[key][index], fd, rtol=1e-6, atol=1e-8,
                                   err_msg=f"{key}{index}")


def test_no_mutation_of_inputs():
    est, weights, features, params = _case(3, 3, 3, (6,), np.float64)
    w_before = {k: v.copy() for k, v in weights.items()}
    f_before = features.copy()
    p_before = params.copy()

    est._loss_and_grad_np(weights, features, params)

    assert set(weights) == set(w_before)
    for key, value in w_before.items():
        np.testing.assert_array_equal(weights[key], value)
    np.testing.assert_array_equal(features, f_before)
    np.testing.assert_array_equal(params, p_before)


# ---------------------------------------------------------------------------
# M3: training wiring (plain-NumPy backend)
# ---------------------------------------------------------------------------

def _train_data(n=64, param_dim=2, feature_dim=1, seed=1):
    rng = np.random.RandomState(seed)
    features = rng.randn(n, feature_dim).astype(np.float32)
    params = (rng.randn(n, param_dim) * 0.5
              + np.tanh(features[:, :1])).astype(np.float32)
    return params, features


def _adam(step, weights, n_iter, lr=1e-2):
    """Minimal Adam mirroring ``ConditionalDensityEstimator.train``."""
    weights = {k: v.copy() for k, v in weights.items()}
    m = {k: np.zeros_like(v) for k, v in weights.items()}
    v = {k: np.zeros_like(v) for k, v in weights.items()}
    beta1, beta2, epsilon = 0.9, 0.999, 1e-8
    losses = []
    for i in range(n_iter):
        loss, g = step(weights)
        losses.append(float(loss))
        assert np.all(np.isfinite(loss))
        for key in weights:
            gk = np.asarray(g[key], dtype=weights[key].dtype)
            m[key] = beta1 * m[key] + (1 - beta1) * gk
            v[key] = beta2 * v[key] + (1 - beta2) * gk**2
            m_hat = m[key] / (1 - beta1**(i + 1))
            v_hat = v[key] / (1 - beta2**(i + 1))
            weights[key] = weights[key] - lr * m_hat / (np.sqrt(v_hat) + epsilon)
    return losses, weights


@requires_autograd
def test_mdn_training_matches_autograd_reference():
    est = cde.MDNEstimator(param_dim=2, feature_dim=1, n_components=3, hidden_sizes=(8,))
    params, features = _train_data()
    init = {k: v.copy() for k, v in
            est._initialize_weights(np.random.RandomState(0)).items()}
    init = {k: v.astype(np.float32) for k, v in init.items()}

    def np_step(w):
        return est._loss_and_grad_np(w, features, params)

    def ag_step(w):
        loss = _ref_loss_function(est, w, features, params)
        g = autograd.grad(lambda ww: _ref_loss_function(est, ww, features, params))(w)
        return loss, g

    np_losses, np_weights = _adam(np_step, init, 20)
    ag_losses, ag_weights = _adam(ag_step, init, 20)

    loss_diff = max(abs(a - b) for a, b in zip(np_losses, ag_losses))
    weight_diff = max(float(np.max(np.abs(np_weights[k] - ag_weights[k])))
                      for k in init)
    weight_scale = max(float(np.max(np.abs(ag_weights[k]))) for k in init)
    print(f"training parity: max |dLoss|={loss_diff:.3e}, "
          f"max |dW|={weight_diff:.3e} (scale {weight_scale:.3e})")

    np.testing.assert_allclose(np_losses, ag_losses, rtol=1e-4, atol=1e-4)
    for key in init:
        np.testing.assert_allclose(np_weights[key], ag_weights[key],
                                   rtol=1e-3, atol=1e-4, err_msg=key)
    assert np_losses[-1] < np_losses[0]


def test_mdn_log_prob_matches_reference_after_training():
    mdn = cde.MDNEstimator(param_dim=2, feature_dim=1, n_components=3, hidden_sizes=(8,))
    params, features = _train_data()
    mdn.train(params, features, n_iter=10, learning_rate=1e-2, use_tqdm=False)

    lp = np.asarray(mdn.log_prob(features, params), dtype=np.float64)
    ref_mean_loss = float(mdn._loss_function(mdn.weights, features, params))
    np.testing.assert_allclose(-np.mean(lp), ref_mean_loss, rtol=1e-5, atol=1e-5)

    alpha, mu, l_prec, ld, cache = mdn._forward_pass_np(mdn.weights, features)
    lp_np, _, _, _ = mdn._mdn_log_prob_from_cache(cache['log_alpha'], mu, l_prec, ld, params)
    np.testing.assert_allclose(lp, np.asarray(lp_np, dtype=np.float64),
                               rtol=1e-6, atol=1e-6)
    assert np.all(np.isfinite(lp))
