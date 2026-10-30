"""Where does the graph start to matter?

Sweeps a global coupling scale ``g`` (multiplying every weight of a *fixed*
topology) and reports three things per value:

* ``own``  -- mean over nodes of ``max_f |corr(feature_f(node_i), a_i)|``:
  how well a node's own dynamics reveal its own parameter.
* ``cross`` -- the same but for ``a_j``, ``j != i`` averaged: how much a node's
  dynamics reveal *other* nodes' parameters. This is the number that has to
  grow for the graph to matter.
* ``sync`` -- variance of the population mean divided by mean single-node
  variance. ~1 means nodes move together (strong coupling / synchrony),
  ~1/N means effectively independent.

Also reports how many nodes still oscillate, so a regime that "maximises
cross-talk" by killing the dynamics is not mistaken for a good one.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.signal import periodogram

sys.path.insert(0, str(Path(__file__).resolve().parent))

import multinode as mn


def features(traces, fs=1.0 / mn.DT, n_bands=10, f_max=2.0):
    """Per-node (n, node, n_features) descriptive features."""
    f, P = periodogram(traces.transpose(0, 2, 1), fs=fs, axis=-1)
    keep = (f > 0.05) & (f <= f_max)
    idx = np.where(keep)[0]
    bands = np.array_split(idx, n_bands)
    band_pow = np.stack([np.log10(P[:, :, b].mean(-1) + 1e-10) for b in bands], -1)
    dom = f[idx][np.argmax(P[:, :, idx], axis=-1)]
    cols = [traces.mean(1), traces.std(1), dom]
    return np.concatenate([np.stack(cols, -1), band_pow], axis=-1)


def max_corr(feat, params):
    """(nodes, nodes) matrix of max |corr| over features: node j -> param i."""
    n, nodes, nf = feat.shape
    out = np.zeros((nodes, nodes))
    for i in range(nodes):
        for j in range(nodes):
            c = [abs(np.corrcoef(feat[:, j, k], params[:, i])[0, 1])
                 for k in range(nf)]
            out[i, j] = np.nanmax(c)
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--num-node', type=int, default=8)
    p.add_argument('--n', type=int, default=600)
    p.add_argument('--scales', default='0.1,0.3,0.6,1,1.5,2,3,4,6,8,12')
    p.add_argument('--seed', type=int, default=0)
    args = p.parse_args(argv)

    scales = [float(v) for v in args.scales.split(',')]
    print(f"{'g':>6}{'osc':>6}{'own':>8}{'cross':>8}{'ratio':>8}"
          f"{'sync':>8}{'amp':>8}  verdict")
    best = None
    for g in scales:
        csr, idelays, W = mn.make_network(args.num_node, seed=args.seed, scale=g)
        prm = mn.sample_params(args.n, args.num_node, args.seed + 1)
        tr = mn.simulate(prm, csr, idelays, args.num_node)
        if not np.isfinite(tr).all():
            print(f"{g:>6.1f}{'':>6}{'non-finite':>32}")
            continue
        amp = tr.std(axis=1)
        feat = features(tr)
        C = max_corr(feat, prm)
        own = np.mean(np.diag(C))
        off = C.copy()
        np.fill_diagonal(off, np.nan)
        cross = np.nanmean(off)
        ratio = cross / own
        pm = tr.mean(2)
        sync = (pm.var(0).mean() / amp.var(0).mean()) if amp.var(0).mean() > 0 else 0.0
        oscillating = int(np.mean(amp.mean(0) > 0.3) * args.num_node)
        verdict = ''
        if best is None or ratio > best[1]:
            best = (g, ratio); verdict = '  <- max cross-talk so far'
        print(f"{g:>6.1f}{oscillating:>6d}{own:>8.3f}{cross:>8.3f}{ratio:>8.3f}"
              f"{sync:>8.3f}{amp.mean():>8.3f}{verdict}")
    print(f"\nmax off-diagonal/own ratio at g = {best[0]} ({best[1]:.3f})")
    print("own/cross are max|corr| over {mean, std, dom. freq, 10 bands}.")
    print("ratio -> 0: nodes are independent (graph irrelevant).")
    print("ratio -> 1: a node's dynamics are as informative about its")
    print("neighbours' parameters as about its own (graph dominates).")
    return best


if __name__ == '__main__':
    main()
