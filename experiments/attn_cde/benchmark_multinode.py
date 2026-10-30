"""Multi-node benchmark: does attention earn its cost once tokens differ?

Four arms, all with the same MADE flow and the same 40 tokens
(8 nodes x 5 STFT frames, 65 bands):

``made-raw``     MAF straight on the flattened tokens (no encoder).
``linear-pool``  learned per-token embedding (which includes a learned
                 positional embedding, so node identity *is* linearly
                 available) + mean pool over all 40 tokens -> MAF.
``attn-1blk``    one pre-LN self-attention block, **no** graph bias, mean pool.
``attn-graph``   same, plus the structural-connectivity adjacency bias.

The prediction from the JR toy: pooling was fine there because all tokens said
the same thing.  Here each node's tokens carry *its own* regional parameter
(corr ~ -0.9 with own ``a``, ~0 with the others), so uniform averaging mixes
eight different parameters together and should hurt.  That is the routing
problem attention is supposed to solve.

Usage::

    PYTHONPATH=content python3 experiments/attn_cde/benchmark_multinode.py
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import multinode as mn                                            # noqa: E402
from benchmark import evaluate, fit, n_params                     # noqa: E402
from jr_toy import zscore_params                                  # noqa: E402

from tvbl import attn, cde                                        # noqa: E402


ARMS = ['made-raw', 'linear-pool', 'attn-1blk', 'attn-graph']


def make_arm(name, num_node, n_tokens, token_dim, bias, args):
    """Construct one arm; every arm shares the MADE flow configuration."""
    common = dict(param_dim=num_node, n_flows=args.n_flows,
                  hidden_units=args.hidden_units)
    enc = dict(n_tokens=n_tokens, token_dim=token_dim,
               d_model=args.d_model, n_heads=args.n_heads)
    return {
        'made-raw': lambda: cde.MAFEstimator(
            feature_dim=n_tokens * token_dim, **common),
        'linear-pool': lambda: attn.AttnMAFEstimator(
            n_blocks=0, **enc, **common),
        'attn-1blk': lambda: attn.AttnMAFEstimator(
            n_blocks=1, **enc, **common),
        'attn-2blk': lambda: attn.AttnMAFEstimator(
            n_blocks=2, **enc, **common),
        'attn-graph': lambda: attn.AttnMAFEstimator(
            n_blocks=1, attn_bias=bias, **enc, **common),
        'attn-graph2': lambda: attn.AttnMAFEstimator(
            n_blocks=2, attn_bias=bias, **enc, **common),
        'attn-linear': lambda: attn.AttnMAFEstimator(
            n_blocks=1, attn_type='linear', **enc, **common),
        'attn-linear2': lambda: attn.AttnMAFEstimator(
            n_blocks=2, attn_type='linear', **enc, **common),
    }[name]()


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--num-node', type=int, default=8)
    p.add_argument('--n-train', type=int, default=1500)
    p.add_argument('--n-test', type=int, default=300)
    p.add_argument('--iters', type=int, default=8000)
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--clip', type=float, default=5.0)
    p.add_argument('--d-model', type=int, default=32)
    p.add_argument('--n-heads', type=int, default=2)
    p.add_argument('--n-flows', type=int, default=4)
    p.add_argument('--hidden-units', type=int, default=64)
    p.add_argument('--n-posterior', type=int, default=128)
    p.add_argument('--self-weight', type=float, default=0.0,
                   help='identity added to the graph bias (self-loops)')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--arm', default='all')
    p.add_argument('--tag', default='')
    p.add_argument('--coupling-scale', type=float, default=1.0,
                   help='global multiplier on the connectivity')
    args = p.parse_args(argv)

    num_node = args.num_node
    csr, idelays, W = mn.make_network(num_node, seed=args.seed,
                                     scale=args.coupling_scale)

    print(f"simulating {args.n_train + args.n_test} runs ({num_node} nodes, "
          f"a in {mn.A_BOUNDS}, coupling scale {args.coupling_scale}) ...")
    t0 = time.time()
    params_tr = mn.sample_params(args.n_train, num_node, args.seed + 1)
    params_te = mn.sample_params(args.n_test, num_node, args.seed + 2)
    tr = mn.simulate(params_tr, csr, idelays, num_node)
    te = mn.simulate(params_te, csr, idelays, num_node)
    print(f"  {time.time() - t0:.1f}s   traces {tr.shape}")

    tok_tr, freqs, n_frames = mn.node_tokens(tr)
    tok_te, _, _ = mn.node_tokens(te)
    tok_tr, tok_te = mn.standardize(tok_tr, tok_te)
    y_tr, y_te, p_mu, p_sd = zscore_params(params_tr, params_te)
    n_tokens, token_dim = tok_tr.shape[1], tok_tr.shape[2]
    x_tr = tok_tr.reshape(len(tok_tr), -1)
    x_te = tok_te.reshape(len(tok_te), -1)
    bias = mn.graph_bias(W, n_frames, self_weight=args.self_weight)

    print(f"tokens {n_tokens} ({num_node} nodes x {n_frames} frames), "
          f"{token_dim} bands, flat dim {x_tr.shape[1]}")
    sd = tr.std(axis=1)
    print("own-node |corr(std_i, a_i)|: "
          f"{np.round(np.abs([np.corrcoef(sd[:, i], params_tr[:, i])[0, 1] for i in range(num_node)]), 2)}")

    arms = ARMS if args.arm == 'all' else args.arm.split(',')

    rows = []
    for name in arms:
        est = make_arm(name, num_node, n_tokens, token_dim, bias, args)
        _, sec = fit(est, x_tr, y_tr, args.iters, args.batch_size, args.lr,
                     args.seed, args.clip)
        m = evaluate(est, x_te, y_te, n_posterior=args.n_posterior,
                     seed=args.seed)
        m['train_nll'] = float(-np.mean(est.log_prob(x_tr, y_tr)))
        rows.append((f'{name}{args.tag}', n_params(est), m, sec))
        print(f"\n[{name}{args.tag}] {n_params(est)} params, {sec:.1f}s "
              f"({1000 * sec / args.iters:.2f} ms/iter)")

    print("\n" + "=" * 100)
    print(f"{'arm':<13}{'params':>9}{'train NLL':>12}{'test NLL':>11}"
          f"{'z-spread':>10}{'|z-bias|':>10}{'cov90':>8}{'shrink':>8}{'s/iter':>8}")
    print("-" * 100)
    for name, n, m, sec in rows:
        print(f"{name:<13}{n:>9,}{m['train_nll']:>12.3f}{m['nll']:>11.3f}"
              f"{m['z_std']:>10.3f}{m['z_bias']:>10.3f}{m['coverage']:>8.3f}"
              f"{m['shrink']:>8.3f}{sec / args.iters:>8.4f}")
    print("=" * 100)
    print("params z-scored, so prior std = 1 per node; ideal z-spread 1.0, "
          "|z-bias| 0.0, cov90 0.90.")
    return rows


if __name__ == '__main__':
    main()
