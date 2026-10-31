"""Gradient and contract tests for :mod:`tvbl.attn`.

The hand-written backward pass is checked against central finite differences
along random directions, which touches every entry of every weight array with
two loss evaluations per direction:

    <dL/dw, v>  ~=  (L(w + eps v) - L(w - eps v)) / (2 eps)

For a quadratic this is exact; for our losses with ``eps = 1e-6`` in float64
agreement to ~1e-7 relative is expected and a wrong adjoint anywhere shows up
immediately.

Run from the repository root with::

    PYTHONPATH=content python3 -m pytest content/tvbl/test_attn.py -q
"""

import numpy as np
import pytest

from . import attn, cde


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _f64(weights):
    return {k: np.asarray(v, dtype=np.float64) for k, v in weights.items()}


def _estimator(n_tokens=4, token_dim=3, param_dim=2, n_blocks=2, seed=0):
    est = attn.AttnMAFEstimator(
        param_dim=param_dim, n_tokens=n_tokens, token_dim=token_dim,
        d_model=8, n_heads=2, n_blocks=n_blocks, ffn_mult=2,
        n_flows=2, hidden_units=6,
    )
    weights = _f64(est._initialize_weights(np.random.RandomState(seed)))
    # Non-trivial weights so no branch is silently inactive.
    rng = np.random.RandomState(seed + 1)
    for k, v in weights.items():
        if k.endswith('_g') or '_g_' in k or k == 'e_lnf_g':
            weights[k] = np.asarray(v, dtype=np.float64)  # keep LN gains at 1
        else:
            weights[k] = rng.randn(*v.shape) * 0.4
    return est, weights


def _data(est, n=11, seed=2):
    rng = np.random.RandomState(seed)
    tokens = rng.randn(n, est.n_tokens, est.token_dim).astype(np.float64)
    params = rng.randn(n, est.param_dim).astype(np.float64)
    return tokens.reshape(n, -1), params


def _directional_check(loss_fn, weights, grads, n_dirs=3, eps=1e-6, seed=0,
                       rtol=1e-6):
    """Compare <grad, v> with a central difference along ``v``."""
    rng = np.random.RandomState(seed)
    missing = set(weights) - set(grads)
    assert not missing, f"no gradient for {sorted(missing)}"
    for d in range(n_dirs):
        # ``randn()`` with a 0-d shape returns a Python float, which breaks the
        # ``.sum()`` below -- the graph-bias scale is such a scalar.
        direction = {k: np.asarray(rng.randn(*np.shape(w))).reshape(np.shape(w))
                     for k, w in weights.items()}
        norm = np.sqrt(sum((v ** 2).sum() for v in direction.values()))
        analytic = sum((grads[k] * direction[k]).sum() for k in weights) / norm
        plus = {k: weights[k] + eps * direction[k] for k in weights}
        minus = {k: weights[k] - eps * direction[k] for k in weights}
        numeric = (loss_fn(plus) - loss_fn(minus)) / (2.0 * eps) / norm
        assert np.isfinite(numeric) and np.isfinite(analytic), "non-finite grad"
        assert abs(analytic - numeric) <= rtol * max(1.0, abs(numeric)), (
            f"direction {d}: analytic {analytic!r} vs numeric {numeric!r}")


# ---------------------------------------------------------------------------
# Encoder alone
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n_blocks", [0, 1, 2])
def test_encoder_gradients_match_finite_differences(n_blocks):
    est, weights = _estimator(n_blocks=n_blocks)
    # The encoder only owns the ``e_*`` slice of the flat weight dict.
    weights = {k: v for k, v in weights.items() if k.startswith('e_')}
    tokens = np.random.RandomState(3).randn(7, est.n_tokens, est.token_dim)
    bar_c = np.random.RandomState(4).randn(7, est.d_model)

    _, cache = est.encoder.forward(weights, tokens)
    grads = est.encoder.backward(weights, cache, bar_c)

    def loss(w):
        c, _ = est.encoder.forward(w, tokens)
        return float((c * bar_c).sum())

    _directional_check(loss, weights, grads, seed=5)


def test_encoder_backward_returns_float64_for_float64_input():
    est, weights = _estimator()
    tokens = np.random.RandomState(3).randn(7, est.n_tokens, est.token_dim)
    _, cache = est.encoder.forward(weights, tokens)
    grads = est.encoder.backward(weights, cache, np.ones((7, est.d_model)))
    assert all(np.asarray(g).dtype == np.float64 for g in grads.values())


def test_encoder_rejects_wrong_token_shape():
    est, weights = _estimator()
    with pytest.raises(ValueError, match="expected tokens of shape"):
        est.encoder.forward(weights, np.zeros((2, est.n_tokens + 1, est.token_dim)))


def test_encoder_is_not_permutation_equivariant_over_time():
    """The learned positional embedding must make token order matter."""
    est, weights = _estimator()
    tokens = np.random.RandomState(3).randn(4, est.n_tokens, est.token_dim)
    c_a, _ = est.encoder.forward(weights, tokens)
    c_b, _ = est.encoder.forward(weights, tokens[:, ::-1])
    assert not np.allclose(c_a, c_b)


# ---------------------------------------------------------------------------
# Full attention-conditioned MAF
# ---------------------------------------------------------------------------

def test_attn_maf_gradients_match_finite_differences():
    est, weights = _estimator(n_blocks=1)
    features, params = _data(est)

    loss, grads = est._loss_and_grad_np(weights, features, params)

    def loss_fn(w):
        return float(est._loss_and_grad_np(w, features, params)[0])

    assert abs(loss - loss_fn(weights)) < 1e-12
    _directional_check(loss_fn, weights, grads, seed=7)


def test_attn_maf_gradients_reach_the_encoder():
    """Every encoder weight must receive a non-zero gradient end to end."""
    est, weights = _estimator(n_blocks=1)
    features, params = _data(est)
    _, grads = est._loss_and_grad_np(weights, features, params)
    encoder_keys = [k for k in weights if k.startswith('e_')]
    assert encoder_keys
    for k in encoder_keys:
        assert np.any(np.abs(grads[k]) > 0), f"zero gradient for encoder weight {k}"


def test_attn_maf_weight_keys_are_disjoint():
    est = attn.AttnMAFEstimator(param_dim=2, n_tokens=4, token_dim=3,
                                d_model=8, n_heads=2, n_blocks=1,
                                n_flows=2, hidden_units=6)
    weights = est._initialize_weights(np.random.RandomState(0))
    encoder = [k for k in weights if k.startswith('e_')]
    flow = [k for k in weights if not k.startswith('e_')]
    assert set(encoder).isdisjoint(flow)
    assert set(flow) == set(cde.MAFEstimator(param_dim=2, feature_dim=8,
                                             n_flows=2, hidden_units=6)
                            ._initialize_weights(np.random.RandomState(0)))


def test_attn_maf_log_prob_and_sample_shapes():
    est, weights = _estimator()
    est.weights = weights
    features, params = _data(est, n=5)

    lp = est.log_prob(features, params)
    assert lp.shape == (5,) and np.all(np.isfinite(lp))

    samples = est.sample(features[:2], 7, np.random.RandomState(0))
    assert samples.shape == (2, 7, est.param_dim)
    assert np.all(np.isfinite(samples))


def test_attn_maf_log_prob_integrates_to_one():
    """Grid quadrature of exp(log_prob) must be ~1: the encoder cannot break it.

    Uses the freshly initialised weights (near-identity flow, ``mu ~ 0``,
    ``alpha ~ 0``) so the density stays inside the quadrature box; the point is
    normalisation, not sharpness.
    """
    est = attn.AttnMAFEstimator(param_dim=2, n_tokens=4, token_dim=3,
                                d_model=8, n_heads=2, n_blocks=1,
                                n_flows=2, hidden_units=6)
    est.weights = est._initialize_weights(np.random.RandomState(0))
    features = np.random.RandomState(2).randn(1, est.feature_dim)
    g = np.linspace(-9, 9, 321)
    xx, yy = np.meshgrid(g, g, indexing='ij')
    grid = np.stack([xx.ravel(), yy.ravel()], axis=1)
    lp = est.log_prob(np.repeat(features, grid.shape[0], axis=0), grid)
    assert np.all(np.isfinite(lp))
    pdf = np.exp(lp).reshape(xx.shape)
    mass = np.trapezoid(np.trapezoid(pdf, g, axis=0), g, axis=0)
    assert abs(mass - 1.0) < 1e-4, f"density integrates to {mass}"


def test_attn_maf_trains_and_improves():
    """A tiny end-to-end fit on synthetic data must reduce the loss."""
    est = attn.AttnMAFEstimator(param_dim=2, n_tokens=4, token_dim=3,
                                d_model=8, n_heads=2, n_blocks=1,
                                n_flows=2, hidden_units=8)
    rng = np.random.RandomState(0)
    n = 256
    params = rng.randn(n, 2) * 0.5
    tokens = np.zeros((n, 4, 3))
    tokens[:, :, 0] = params[:, :1] + 0.1 * rng.randn(n, 4)
    tokens[:, :, 1] = params[:, 1:] + 0.1 * rng.randn(n, 4)
    est.train(params, tokens.reshape(n, -1), n_iter=150, learning_rate=3e-3,
              seed=0, use_tqdm=False)
    losses = np.asarray(est.loss_history)
    assert np.all(np.isfinite(losses))
    assert losses[-10:].mean() < losses[:10].mean() - 0.1


def test_flow_backward_still_returns_bare_grads_by_default():
    """The new ``compute_ctx_grad`` flag must not change existing behaviour."""
    est = cde.MAFEstimator(param_dim=2, feature_dim=3, n_flows=2, hidden_units=6)
    weights = est._initialize_weights(np.random.RandomState(0))
    features = np.random.RandomState(1).randn(9, 3)
    params = np.random.RandomState(2).randn(9, 2)
    _, caches = est._get_log_prob_np(weights, features, params)
    bar = np.full(9, -1.0 / 9)
    out = est._flow_backward_np(weights, caches, bar)
    assert isinstance(out, dict)
    grads, bar_ctx = est._flow_backward_np(weights, caches, bar,
                                           compute_ctx_grad=True)
    assert isinstance(grads, dict) and bar_ctx.shape == features.shape
    for k in out:
        assert np.allclose(out[k], grads[k])


# ---------------------------------------------------------------------------
# Graph (adjacency) attention bias
# ---------------------------------------------------------------------------

def _biased_estimator(n_tokens=5, seed=0):
    rng = np.random.RandomState(seed)
    bias = rng.randn(n_tokens, n_tokens)
    est = attn.AttnMAFEstimator(
        param_dim=2, n_tokens=n_tokens, token_dim=3, d_model=8, n_heads=2,
        n_blocks=1, n_flows=2, hidden_units=6, attn_bias=bias)
    weights = _f64(est._initialize_weights(np.random.RandomState(seed)))
    for k, v in weights.items():
        if 'ln' in k or k == 'e_bias_scale':
            continue
        weights[k] = rng.randn(*v.shape) * 0.4
    return est, weights


def test_attn_bias_scale_gradient_matches_finite_differences():
    est, weights = _biased_estimator()
    features, params = _data(est)

    def loss_fn(w):
        return float(est._loss_and_grad_np(w, features, params)[0])

    _, grads = est._loss_and_grad_np(weights, features, params)
    assert 'e_bias_scale' in grads
    _directional_check(loss_fn, weights, grads, seed=9)


def test_attn_bias_changes_the_encoding():
    est, weights = _biased_estimator()
    tokens = np.random.RandomState(3).randn(4, est.n_tokens, est.token_dim)
    c_bias, _ = est.encoder.forward(weights, tokens)
    weights0 = dict(weights, e_bias_scale=np.zeros((), dtype=np.float64))
    c_flat, _ = est.encoder.forward(weights0, tokens)
    c_scaled, _ = est.encoder.forward(
        dict(weights, e_bias_scale=np.float64(5.0)), tokens)
    assert not np.allclose(c_bias, c_flat)
    assert not np.allclose(c_bias, c_scaled)


def test_attn_bias_rejects_wrong_shape():
    with pytest.raises(ValueError, match="attn_bias must be"):
        attn.AttnMAFEstimator(param_dim=2, n_tokens=5, token_dim=3, d_model=8,
                              n_heads=2, n_blocks=1, n_flows=2, hidden_units=6,
                              attn_bias=np.zeros((4, 4)))


def test_no_bias_keeps_weight_dict_unchanged():
    """Without a bias, no ``e_bias_scale`` key is created (back-compat)."""
    est = attn.AttnMAFEstimator(param_dim=2, n_tokens=5, token_dim=3, d_model=8,
                                n_heads=2, n_blocks=1, n_flows=2, hidden_units=6)
    assert 'e_bias_scale' not in est._initialize_weights(np.random.RandomState(0))


# ---------------------------------------------------------------------------
# Linear attention
# ---------------------------------------------------------------------------

def _linear_estimator(n_tokens=6, n_blocks=1, seed=0):
    est = attn.AttnMAFEstimator(
        param_dim=2, n_tokens=n_tokens, token_dim=3, d_model=8, n_heads=2,
        n_blocks=n_blocks, n_flows=2, hidden_units=6, attn_type='linear')
    weights = _f64(est._initialize_weights(np.random.RandomState(seed)))
    rng = np.random.RandomState(seed + 1)
    for k, v in weights.items():
        if 'ln' in k:
            continue
        weights[k] = rng.randn(*v.shape) * 0.4
    return est, weights


@pytest.mark.parametrize("n_blocks", [1, 2])
def test_linear_attention_gradients_match_finite_differences(n_blocks):
    est, weights = _linear_estimator(n_blocks=n_blocks)
    encoder_keys = {k: v for k, v in weights.items() if k.startswith('e_')}
    tokens = np.random.RandomState(3).randn(7, est.n_tokens, est.token_dim)
    bar_c = np.random.RandomState(4).randn(7, est.d_model)

    _, cache = est.encoder.forward(weights, tokens)
    grads = est.encoder.backward(weights, cache, bar_c)

    def loss(w):
        c, _ = est.encoder.forward(w, tokens)
        return float((c * bar_c).sum())

    _directional_check(loss, encoder_keys, grads, seed=5)


def test_linear_attention_full_model_gradients():
    est, weights = _linear_estimator()
    features, params = _data(est)

    def loss_fn(w):
        return float(est._loss_and_grad_np(w, features, params)[0])

    _, grads = est._loss_and_grad_np(weights, features, params)
    _directional_check(loss_fn, weights, grads, seed=11)


def test_linear_attention_has_no_token_squared_memory():
    """Linear attention must not build an (N, H, T, T) score matrix."""
    est, weights = _linear_estimator(n_tokens=64)
    tokens = np.random.RandomState(3).randn(2, est.n_tokens, est.token_dim)
    _, cache = est.encoder.forward(weights, tokens)
    for b in cache['blocks']:
        assert b['probs'] is None
        assert b['lin']['state'].shape == (2, est.encoder.n_heads,
                                           est.encoder.head_dim,
                                           est.encoder.head_dim)
        for key in ('qp', 'kp', 'num', 'state', 'zsum', 'den'):
            assert b['lin'][key].shape[-2:] != (est.n_tokens, est.n_tokens)


def test_linear_attention_rejects_bias():
    with pytest.raises(ValueError, match="requires attn_type='softmax'"):
        attn.AttnMAFEstimator(param_dim=2, n_tokens=4, token_dim=3, d_model=8,
                              n_heads=2, n_blocks=1, n_flows=2, hidden_units=6,
                              attn_bias=np.zeros((4, 4)), attn_type='linear')


def test_linear_attention_large_negative_input_is_finite():
    """elu+1 underflows to 0 in float32 for very negative pre-activations;
    the denominator guard must keep the read-out finite (not 0/0)."""
    est, weights = _linear_estimator()
    tokens = np.full((2, est.n_tokens, est.token_dim), -1e4, dtype=np.float64)
    c, _ = est.encoder.forward(weights, tokens)
    assert np.all(np.isfinite(c))
    tokens32 = tokens.astype(np.float32)
    c32, _ = est.encoder.forward({k: v.astype(np.float32) for k, v in weights.items()},
                                 tokens32)
    assert np.all(np.isfinite(c32))


def test_linear_attention_runs_end_to_end():
    est = attn.AttnMAFEstimator(param_dim=2, n_tokens=4, token_dim=3,
                                d_model=8, n_heads=2, n_blocks=1, n_flows=2,
                                hidden_units=6, attn_type='linear')
    est.weights = est._initialize_weights(np.random.RandomState(0))
    rng = np.random.RandomState(1)
    x = rng.randn(5, 12)
    y = rng.randn(5, 2)
    lp = est.log_prob(x, y)
    assert lp.shape == (5,) and np.all(np.isfinite(lp))
    s = est.sample(x[:2], 7, np.random.RandomState(0))
    assert s.shape == (2, 7, 2) and np.all(np.isfinite(s))


# ---------------------------------------------------------------------------
# Raw time-series ingest
# ---------------------------------------------------------------------------

def test_stft_tokens_shapes_multichannel():
    rng = np.random.RandomState(0)
    traces = rng.randn(5, 200, 3)
    tokens, freqs = attn.stft_tokens(traces, fs=200.0, nperseg=32, hop=8)
    # 200 samples, nperseg=32, hop=8, no padding -> 22 frames, 17 bands
    assert tokens.shape == (5, 3 * 22, 17)
    assert np.all(np.isfinite(tokens))
    one, _ = attn.stft_tokens(rng.randn(4, 200), fs=200.0)
    assert one.shape == (4, 22, 17)
    capped, _ = attn.stft_tokens(traces, fs=200.0, f_max=20.0)
    assert capped.shape[1] == 3 * 22 and capped.shape[2] < 17


def test_raw_ts_estimator_ingests_traces_end_to_end():
    """RawTSMAFEstimator takes (N, T) traces directly: train/log_prob/sample."""
    rng = np.random.RandomState(0)
    T = 64
    traces = rng.randn(24, T) + 3 * np.sin(2 * np.pi * np.arange(T) / 8)
    params = np.stack([traces.std(1), traces.mean(1)], axis=1)
    est = attn.RawTSMAFEstimator(param_dim=2, T=T, fs=50.0, nperseg=16, hop=4,
                                 d_model=8, n_heads=2, n_blocks=1, n_flows=2,
                                 hidden_units=6, attn_type='linear')
    assert est.feature_dim == est.n_tokens * est.token_dim
    est.train(params, traces, n_iter=30, learning_rate=1e-2, seed=0,
              use_tqdm=False, batch_size=8, clip=5.0)
    assert len(est.loss_history) == 30
    assert np.all(np.isfinite(est.loss_history))
    lp = est.log_prob(traces[:4], params[:4])
    assert lp.shape == (4,) and np.all(np.isfinite(lp))
    s = est.sample(traces[:2], 5, np.random.RandomState(0))
    assert s.shape == (2, 5, 2) and np.all(np.isfinite(s))


def test_raw_ts_estimator_matches_manual_tokenisation():
    """Training on raw traces must equal training on the same STFT tokens
    standardised with the training-split statistics."""
    rng = np.random.RandomState(1)
    T = 64
    traces = rng.randn(16, T)
    params = rng.randn(16, 2)

    def build():
        return attn.RawTSMAFEstimator(param_dim=2, T=T, fs=50.0, nperseg=16,
                                      hop=4, d_model=8, n_heads=2, n_blocks=1,
                                      n_flows=2, hidden_units=6,
                                      attn_type='linear')

    est_raw = build()
    est_raw.train(params, traces, n_iter=5, learning_rate=1e-3, seed=3,
                  use_tqdm=False)
    toks, _ = attn.stft_tokens(traces, fs=50.0, nperseg=16, hop=4)
    mu, sd = est_raw._tok_mu, est_raw._tok_sd
    est_tok = attn.AttnMAFEstimator(param_dim=2, n_tokens=est_raw.n_tokens,
                                    token_dim=est_raw.token_dim, d_model=8,
                                    n_heads=2, n_blocks=1, n_flows=2,
                                    hidden_units=6, attn_type='linear')
    est_tok.train(params, ((toks - mu) / sd).reshape(16, -1), n_iter=5,
                  learning_rate=1e-3, seed=3, use_tqdm=False)
    for k in est_raw.weights:
        assert np.allclose(est_raw.weights[k], est_tok.weights[k], rtol=1e-4,
                           atol=1e-6), k


def test_raw_ts_estimator_works_with_sbi_eval_fit():
    """sbi_eval.fit calls _loss_and_grad_np directly with raw traces."""
    from . import sbi_eval
    rng = np.random.RandomState(2)
    T = 64
    traces = rng.randn(24, T)
    params = rng.randn(24, 2)
    est = attn.RawTSMAFEstimator(param_dim=2, T=T, fs=50.0, nperseg=16, hop=4,
                                 d_model=8, n_heads=2, n_blocks=1, n_flows=2,
                                 hidden_units=6, attn_type='linear')
    toks, _ = attn.stft_tokens(traces, fs=50.0, nperseg=16, hop=4)
    est._tok_mu = toks.mean(axis=(0, 1), keepdims=True)
    est._tok_sd = toks.std(axis=(0, 1), keepdims=True) + 1e-6
    history, _ = sbi_eval.fit(est, traces, params, n_iter=10, batch_size=8,
                              lr=1e-2, seed=0)
    assert history.shape == (10,) and np.all(np.isfinite(history))
    m = sbi_eval.evaluate(est, traces[:6], params[:6], n_posterior=8, seed=0)
    assert np.isfinite(m['nll'])


def test_raw_ts_estimator_rejects_wrong_trace_shape():
    est = attn.RawTSMAFEstimator(param_dim=2, T=64, fs=50.0, nperseg=16, hop=4,
                                 d_model=8, n_heads=2, n_blocks=1, n_flows=2,
                                 hidden_units=6, attn_type='linear')
    est.weights = est._initialize_weights(np.random.RandomState(0))
    with pytest.raises(ValueError, match="tokenise to"):
        est.log_prob(np.zeros((2, 100)), np.zeros((2, 2)))
