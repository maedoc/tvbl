"""Reusable SBI evaluation harness.

Training loop and held-out diagnostics for any
:class:`tvbl.cde.ConditionalDensityEstimator`, promoted from the attention
benchmark experiments (see ``cde-attn-handoff.md``): minibatch Adam with
global gradient-norm clipping, test NLL, and amortised-posterior calibration
(z-spread, z-bias, 90% coverage, shrinkage).

Also generic simulator diagnostics: per-node spectral features and the
own/cross max-correlation matrix that tells whether a network's graph
matters for a given coupling strength.
"""

import time

import numpy as np


def fit(est, x, y, n_iter, batch_size=64, lr=3e-3, seed=0, clip=5.0):
    """Minibatch Adam over any estimator exposing ``_loss_and_grad_np``.

    ``ConditionalDensityEstimator.train`` is full-batch, which with ~10^3
    samples and a few thousand iterations overfits; minibatches keep runs
    comparable.  Gradients are clipped by global L2 norm: without it,
    high-dimensional estimators diverge in the saturated ``alpha`` regime.

    Returns ``(history, wall_clock_seconds)``.
    """
    rng = np.random.RandomState(seed)
    est.weights = est._initialize_weights(np.random.RandomState(seed))
    w = est.weights
    m = {k: np.zeros_like(v) for k, v in w.items()}
    v = {k: np.zeros_like(v) for k, v in w.items()}
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    history, n = [], x.shape[0]
    t0 = time.time()
    for i in range(n_iter):
        idx = rng.randint(0, n, batch_size)
        loss, g = est._loss_and_grad_np(w, x[idx], y[idx])
        history.append(loss)
        if not np.isfinite(loss):
            raise RuntimeError(f"non-finite loss at iteration {i} for {est}")
        if clip:
            norm = np.sqrt(sum(float((np.asarray(g[k]) ** 2).sum()) for k in w))
            if norm > clip and np.isfinite(norm):
                scale = clip / norm
                g = {k: g[k] * scale for k in w}
        for k in w:
            m[k] = beta1 * m[k] + (1 - beta1) * g[k]
            v[k] = beta2 * v[k] + (1 - beta2) * g[k] ** 2
            m_hat = m[k] / (1 - beta1 ** (i + 1))
            v_hat = v[k] / (1 - beta2 ** (i + 1))
            w[k] -= lr * m_hat / (np.sqrt(v_hat) + eps)
    return np.asarray(history), time.time() - t0


def evaluate(est, x, y, n_posterior=256, seed=0, prior_std=1.0):
    """Held-out NLL plus amortised-posterior calibration diagnostics.

    Keys: ``nll`` (lower better), ``z_std`` (ideal 1.0), ``z_bias``
    (ideal 0.0), ``coverage`` (ideal 0.90 for the 90% interval), ``shrink``
    (ideal -> 1; posterior much tighter than the prior).  Non-finite
    posteriors yield NaN diagnostics with the NLL still reported.
    """
    nll = float(-np.mean(est.log_prob(x, y)))
    rng = np.random.RandomState(seed)
    posts = est.sample(x, n_posterior, rng)                 # (N, S, D)
    if not np.all(np.isfinite(posts)):
        return dict(nll=nll, z_std=np.nan, coverage=np.nan, shrink=np.nan)

    post_mean, post_std = posts.mean(axis=1), posts.std(axis=1) + 1e-12
    z = (post_mean - y) / post_std
    lo = np.quantile(posts, 0.05, axis=1)
    hi = np.quantile(posts, 0.95, axis=1)
    # Decompose miscalibration: |mean z| is systematic error (bias), std(z) is
    # width error.  With |bias| ~ 0 the posterior is centred; then coverage
    # shortfall can only come from being too narrow.
    return dict(
        nll=nll,
        z_std=float(np.mean(z.std(axis=0))),        # ideal 1.0
        z_bias=float(np.mean(np.abs(z.mean(axis=0)))),  # ideal 0.0
        coverage=float(np.mean((y >= lo) & (y <= hi))),  # ideal 0.90
        shrink=float(np.mean(1.0 - post_std / prior_std)),  # ideal -> 1
    )


def n_params(est):
    """Total number of scalar parameters in ``est.weights``."""
    return int(sum(np.asarray(v).size for v in est.weights.values()))


def trace_features(traces, fs, n_bands=10, f_max=2.0):
    """Per-node ``(n, node, n_features)`` descriptive features of a simulated
    network.  ``traces`` is ``(n, time, node)``.

    Features: mean, std, dominant frequency, and log10 band power in
    ``n_bands`` equal slices of ``(0.05, f_max]`` Hz.  Simulator-agnostic.
    """
    from scipy.signal import periodogram

    f, P = periodogram(traces.transpose(0, 2, 1), fs=fs, axis=-1)
    keep = (f > 0.05) & (f <= f_max)
    idx = np.where(keep)[0]
    bands = np.array_split(idx, n_bands)
    band_pow = np.stack([np.log10(P[:, :, b].mean(-1) + 1e-10) for b in bands], -1)
    dom = f[idx][np.argmax(P[:, :, idx], axis=-1)]
    cols = [traces.mean(1), traces.std(1), dom]
    return np.concatenate([np.stack(cols, -1), band_pow], axis=-1)


def max_corr(feat, params):
    """``(nodes, nodes)`` matrix of max |corr| over features: node j -> param i.

    ``feat`` is ``(n, node, n_feature)``, ``params`` is ``(n, node)``.
    ``diag`` is how well a node's own dynamics reveal its own parameter;
    the off-diagonal mean is cross-talk (graph relevance).
    """
    n, nodes, nf = feat.shape
    out = np.zeros((nodes, nodes))
    for i in range(nodes):
        for j in range(nodes):
            c = [abs(np.corrcoef(feat[:, j, k], params[:, i])[0, 1])
                 for k in range(nf)]
            out[i, j] = np.nanmax(c)
    return out


def cross_talk(feat, params):
    """``(own, cross, ratio)`` from :func:`max_corr`: mean diagonal, mean
    off-diagonal, and cross/own.  Ratio -> 0 means nodes are independent
    (graph irrelevant); ratio -> 1 means a node's dynamics are as informative
    about neighbours' parameters as about its own (graph dominates)."""
    C = max_corr(feat, params)
    own = float(np.mean(np.diag(C)))
    off = C.copy()
    np.fill_diagonal(off, np.nan)
    cross = float(np.nanmean(off))
    return own, cross, (cross / own if own > 0 else np.nan)
