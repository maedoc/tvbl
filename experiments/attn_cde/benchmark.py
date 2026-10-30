"""Benchmark: ``STFT -> self-attention -> MADE`` vs ``STFT -> MADE`` (no encoder).

Both models see exactly the same STFT tokens and use the *same* MADE flow
implementation (:class:`tvbl.cde.MAFEstimator`); the only difference is the
conditioning pathway.  Three arms:

``made-raw``      MAF directly on the flattened STFT (374 dims) -- the
                  "MADE alone" baseline.
``linear-pool``   learned linear summary + mean pool -> 32-d context -> MAF.
                  Isolates *learned pooling* with no attention.
``attn-2blk``     two pre-LN self-attention blocks -> mean pool -> MAF.

Reported: held-out NLL (the number the models are trained on), posterior
calibration (z-score spread should be ~1, 90% interval coverage should be
~0.9), posterior shrinkage, parameter count and wall clock.

Usage (from the repo root)::

    PYTHONPATH=content python3 experiments/attn_cde/benchmark.py
    PYTHONPATH=content python3 experiments/attn_cde/benchmark.py --iters 6000 --plot
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from jr_toy import (sample_dataset, standardize, stft_tokens, zscore_params)  # noqa: E402

from tvbl import attn, cde  # noqa: E402


# ---------------------------------------------------------------------------
# Training / evaluation
# ---------------------------------------------------------------------------

def fit(est, x, y, n_iter, batch_size=64, lr=3e-3, seed=0, clip=5.0):
    """Minibatch Adam over any estimator exposing ``_loss_and_grad_np``.

    ``ConditionalDensityEstimator.train`` is full-batch, which with ~10^3
    samples and a few thousand iterations overfits; minibatches keep the three
    arms comparable.  Gradients are clipped by global L2 norm: without it the
    high-dimensional ``made-raw`` arm diverges in the saturated ``alpha``
    regime, which would make the comparison meaningless rather than
    informative.
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
    """Held-out NLL plus amortised-posterior calibration diagnostics."""
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
    return int(sum(np.asarray(v).size for v in est.weights.values()))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--n-train', type=int, default=1200)
    p.add_argument('--n-test', type=int, default=400)
    p.add_argument('--iters', type=int, default=3000)
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--lr', type=float, default=3e-3)
    p.add_argument('--lrs', default=None,
                   help='comma-separated lr sweep; overrides --lr')
    p.add_argument('--clip', type=float, default=5.0,
                   help='global gradient-norm clip (0 disables)')
    p.add_argument('--d-model', type=int, default=32)
    p.add_argument('--n-heads', type=int, default=2)
    p.add_argument('--n-flows', type=int, default=4)
    p.add_argument('--hidden-units', type=int, default=64)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--n-repeats', type=int, default=1,
                   help='refit each arm with different model seeds; reports mean/std')
    p.add_argument('--plot', action='store_true', help='write posteriors.png')
    p.add_argument('--arms', default='made-raw,linear-pool,attn-2blk')
    args = p.parse_args(argv)

    # --- data -------------------------------------------------------------
    print(f"simulating {args.n_train + args.n_test} Jansen-Rit runs ...")
    t0 = time.time()
    params_tr, traces_tr = sample_dataset(args.n_train, args.seed + 1)
    params_te, traces_te = sample_dataset(args.n_test, args.seed + 2)
    print(f"  {time.time() - t0:.1f}s")

    tok_tr, freqs = stft_tokens(traces_tr)
    tok_te, _ = stft_tokens(traces_te)
    tok_tr, tok_te = standardize(tok_tr, tok_te)
    y_tr, y_te, p_mu, p_sd = zscore_params(params_tr, params_te)

    n_tokens, token_dim = tok_tr.shape[1], tok_tr.shape[2]
    x_tr, x_te = tok_tr.reshape(len(tok_tr), -1), tok_te.reshape(len(tok_te), -1)
    print(f"tokens: {n_tokens} frames x {token_dim} bands "
          f"({freqs[1] - freqs[0]:.2f} Hz spacing), flat dim {x_tr.shape[1]}")

    # --- arms -------------------------------------------------------------
    def make(name):
        common = dict(param_dim=2, n_flows=args.n_flows,
                      hidden_units=args.hidden_units)
        if name == 'made-raw':
            return cde.MAFEstimator(feature_dim=x_tr.shape[1], **common)
        enc = dict(n_tokens=n_tokens, token_dim=token_dim,
                   d_model=args.d_model, n_heads=args.n_heads)
        if name == 'linear-pool':
            return attn.AttnMAFEstimator(n_blocks=0, **enc, **common)
        if name == 'attn-1blk':
            return attn.AttnMAFEstimator(n_blocks=1, **enc, **common)
        if name == 'attn-2blk':
            return attn.AttnMAFEstimator(n_blocks=2, **enc, **common)
        raise ValueError(f'unknown arm {name!r}')

    rows, fitted = [], {}
    lrs = ([float(v) for v in args.lrs.split(',')] if args.lrs
           else [args.lr])
    for name in args.arms.split(','):
        for lr in lrs:
            runs, seconds = [], 0.0
            for r in range(args.n_repeats):
                est = make(name)
                history, sec = fit(est, x_tr, y_tr, args.iters,
                                   args.batch_size, lr, args.seed + 1000 * r,
                                   args.clip)
                m = evaluate(est, x_te, y_te, seed=args.seed + 1000 * r)
                m['train_nll'] = float(-np.mean(est.log_prob(x_tr, y_tr)))
                runs.append(m)
                seconds += sec
            agg = {k: (float(np.mean([m[k] for m in runs])),
                       float(np.std([m[k] for m in runs])))
                   for k in runs[0]}
            rows.append((f'{name}@{lr:g}', n_params(est), agg,
                         seconds / args.n_repeats))
            print(f"\n[{name} lr={lr:g}] {n_params(est)} params, "
                  f"{seconds / args.n_repeats:.1f}s "
                  f"({1000 * seconds / args.n_repeats / args.iters:.2f} ms/iter)")
            fitted[f'{name}@{lr:g}'] = est

    # --- report -----------------------------------------------------------
    print("\n" + "=" * 100)
    print(f"{'arm@lr':<20}{'params':>9}{'train NLL':>13}{'test NLL':>13}"
          f"{'z-spread':>13}{'|z-bias|':>11}{'cov90':>12}{'shrink':>12}"
          f"{'s/iter':>8}")
    print("-" * 100)
    for name, n, agg, sec in rows:
        mean = {k: v[0] for k, v in agg.items()}
        std = {k: v[1] for k, v in agg.items()}

        def fmt(k, digits=3):
            mu, sd = mean[k], std[k]
            if not np.isfinite(mu) or abs(mu) > 1e3:
                return f"{mu:.1e}"
            if args.n_repeats == 1:
                return f"{mu:.{digits}f}"
            return f"{mu:.{digits}f}+-{sd:.2f}"

        print(f"{name:<20}{n:>9,}{fmt('train_nll'):>13}{fmt('nll'):>13}"
              f"{fmt('z_std'):>13}{fmt('z_bias'):>11}{fmt('coverage'):>12}"
              f"{fmt('shrink'):>12}{sec / args.iters:>8.4f}")
    print("=" * 100)
    print("test NLL lower is better.  Calibration: z-spread 1.0, |z-bias| 0.0, "
          "cov90 0.90;\nshrink -> 1 means the posterior is much tighter than "
          "the prior.")

    if args.plot:
        make_plot(fitted, x_te, params_te, y_te, p_mu, p_sd, args)
    return rows


def make_plot(fitted, x_te, params_te, y_te, p_mu, p_sd, args):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    n_cases, n_show = 4, 3
    idx = np.linspace(0, len(x_te) - 1, n_cases).astype(int)
    names = list(fitted)
    rng = np.random.RandomState(0)
    fig, axes = plt.subplots(len(names), n_show * 2,
                            figsize=(3.0 * n_show * 2, 2.1 * len(names)),
                            squeeze=False)
    labels = ['A', 'a']
    for r, name in enumerate(names):
        est = fitted[name]
        posts = est.sample(x_te[idx], 1000, rng)
        for c in range(n_show):
            for d in range(2):
                ax = axes[r][c * 2 + d]
                raw = posts[c, :, d] * p_sd[0, d] + p_mu[0, d]
                ax.hist(raw, bins=40, color='C0', alpha=.7, density=True)
                ax.axvline(params_te[idx[c], d], color='C3', lw=1.5)
                if r == 0:
                    ax.set_title(f"case {idx[c]}  {labels[d]}")
                if d == 0:
                    ax.set_ylabel(name, fontsize=8)
                ax.tick_params(labelsize=7)
    fig.suptitle('Jansen-Rit posteriors (red = truth)', fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = Path(__file__).with_name('posteriors.png')
    fig.savefig(out, dpi=110)
    print(f"\nwrote {out}")


if __name__ == '__main__':
    main()
