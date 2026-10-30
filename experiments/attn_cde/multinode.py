"""Multi-node toy: one *regional* parameter per node over a fixed graph.

Where the single-node Jansen-Rit toy was information-poor (a 7-cycle narrowband
rhythm whose 374 STFT tokens had an effective rank of 2), this one is designed
so that tokens are *not* interchangeable:

* ``num_node`` FitzHugh-Nagumo-like nodes (the ``tvbl.core`` oscillator) coupled
  through a fixed sparse structural connectivity with conduction delays;
* one regional parameter ``a_i`` per node, so ``param_dim = num_node``;
* changing ``a_i`` mostly moves node ``i`` but leaks into its neighbours, so
  the posterior is genuinely correlated across parameters.

That is the regime in which content-based attention has something to route.
Everything is deterministic (``z_scale = 0``) and sized so a batch of
simulations fits comfortably in memory.
"""

import numpy as np
import scipy.sparse
from scipy.signal import stft

from tvbl.core import run

# Oscillator defaults: stable, oscillatory, and sensitive to `a` in this range.
TAU = 1.35
K_COUPLING = 0.1
A_BOUNDS = (0.35, 0.95)   # |a| < 1: limit-cycle regime
DT = 0.1
NUM_TIME = 400
HORIZON = 16


class RegionalFHN:
    """``tvbl.core`` 2-D oscillator with a per-node, per-item ``a``.

    ``a`` has shape ``(num_node, num_item)`` so the bias term broadcasts
    against ``cx`` of the same shape -- that is what makes the parameter
    regional inside a single batched simulation.
    """

    def __init__(self, a, tau=TAU, k=K_COUPLING):
        self.a = np.asarray(a, dtype=np.float32).reshape(len(a), -1)
        self.tau = tau
        self.k = k

    def __call__(self, x, cx):
        return np.array([
            self.tau * (x[0] - x[0] ** 3 / 3.0 + x[1]),
            (1.0 / self.tau) * (self.a - self.k * cx - x[0]),
        ])


def make_network(num_node, density=0.35, seed=0, w_lo=0.5, w_hi=1.0,
                 scale=1.0):
    """Fixed sparse connectivity + delays. Returns ``(csr, idelays, W)``.

    ``scale`` multiplies every weight, so a coupling sweep can be done without
    rebuilding the topology (and therefore without changing the graph).
    """
    rng = np.random.RandomState(seed)
    W = ((rng.rand(num_node, num_node) < density)
         * rng.uniform(w_lo, w_hi, (num_node, num_node))).astype('f')
    np.fill_diagonal(W, 0.0)
    W = (W * scale).astype('f')
    csr = scipy.sparse.csr_matrix(W)
    idelays = rng.randint(1, HORIZON - 10, size=csr.nnz).astype('i')
    return csr, idelays, W


def sample_params(n, num_node, seed, bounds=A_BOUNDS):
    rng = np.random.RandomState(seed)
    return rng.uniform(*bounds, size=(n, num_node)).astype('f')


def simulate(a, csr, idelays, num_node, batch=32, num_time=NUM_TIME,
             dt=DT, show_time=False):
    """Simulate ``a.shape[0]`` parameter sets, ``batch`` at a time.

    Returns ``(n, num_time+1, num_node)`` float32 traces of the first state
    variable.  Batching keeps peak memory at ``batch * num_node * num_time``
    rather than ``n * ...``.
    """
    a = np.asarray(a, dtype='f')
    n = a.shape[0]
    out = np.empty((n, num_time + 1, num_node), dtype='f')
    for lo in range(0, n, batch):
        hi = min(lo + batch, n)
        chunk = a[lo:hi].T                              # (node, item)
        trace = run(csr, idelays, RegionalFHN(chunk), 0.0, horizon=HORIZON,
                    num_item=hi - lo, num_node=num_node, num_time=num_time,
                    dt=dt, num_skip=1, show_time=show_time)
        out[lo:hi] = trace[:, 0].transpose(2, 0, 1)     # (item, time, node)
    return out


def node_tokens(traces, fs=1.0 / DT, nperseg=128, hop=64, db_floor=1e-10,
                f_max=None):
    """Log-power STFT tokens per node, shape ``(n, num_node*n_frames, F)``.

    Token index is ``node * n_frames + frame`` so a graph bias can be built as
    a block matrix over nodes.
    """
    n, T, num_node = traces.shape
    flat = traces.transpose(0, 2, 1).reshape(n * num_node, T)
    freqs, frames, Z = stft(flat, fs=fs, nperseg=nperseg,
                            noverlap=nperseg - hop, boundary=None,
                            padded=False, axis=-1)
    power = np.abs(Z) ** 2
    if f_max is not None:
        power = power[:, freqs <= f_max]
        freqs = freqs[freqs <= f_max]
    lp = np.log10(power + db_floor).astype('f')          # (n*node, F, frames)
    n_frames = lp.shape[2]
    lp = lp.reshape(n, num_node, lp.shape[1], n_frames)
    lp = lp.transpose(0, 1, 3, 2)                        # (n, node, frame, F)
    return lp.reshape(n, num_node * n_frames, lp.shape[-1]), freqs, n_frames


def graph_bias(W, n_frames, normalise=True, self_weight=0.0):
    """Block-copy the connectivity into a ``(tokens, tokens)`` attention bias.

    ``bias[node_i*frame_t, node_j*frame_s] = W[i, j]`` for every pair of frames:
    tokens of neighbouring nodes are nudged together regardless of when they
    occur.

    ``self_weight`` adds that much identity, i.e. a self-loop, which is *not*
    in the connectivity matrix (its diagonal is zeroed).  For a regional
    parameter this matters a lot: without a self-loop the bias actively
    down-weights a node's own tokens in favour of its neighbours'.
    """
    Wb = np.asarray(W, dtype=np.float64)
    if normalise and Wb.max() > 0:
        Wb = Wb / Wb.max()
    if self_weight:
        Wb = Wb + self_weight * np.eye(len(Wb))
    bias = np.repeat(np.repeat(Wb, n_frames, axis=0), n_frames, axis=1)
    assert bias.shape == (Wb.shape[0] * n_frames, Wb.shape[1] * n_frames)
    return bias.astype('f')


def standardize(tok_train, tok_test):
    mu = tok_train.mean(axis=(0, 1), keepdims=True)
    sd = tok_train.std(axis=(0, 1), keepdims=True) + 1e-6
    return ((tok_train - mu) / sd).astype('f'), ((tok_test - mu) / sd).astype('f')


# ---------------------------------------------------------------------------
# Diagnostics: are the tokens now genuinely distinct (unlike the JR toy)?
# ---------------------------------------------------------------------------

def diagnostics(params, traces, W, n_frames, token_dim):
    from scipy.signal import periodogram

    n, num_node = params.shape
    tok, _, _ = node_tokens(traces)
    N, Tn, F = tok.shape
    print(f"  traces {traces.shape}, tokens {tok.shape} "
          f"({num_node} nodes x {n_frames} frames x {F} bands)")

    X = tok - tok.mean((0, 1), keepdims=True)
    X = X.reshape(N, -1)
    S = np.linalg.svd(X - X.mean(0), compute_uv=False)
    ev = S ** 2 / (S ** 2).sum()
    k90 = int(np.searchsorted(np.cumsum(ev), 0.90) + 1)
    print(f"  PCA: 90% of token variance needs {k90} of {Tn * F} components "
          f"(top-3 = {ev[:3].sum() * 100:.1f}%)  -- compare 2 for the JR toy")

    # Square spectral features so the linear probe is symmetric across nodes.
    f, P = periodogram(traces.transpose(0, 2, 1), fs=1.0 / DT, axis=-1)
    keep = (f > 0.2) & (f <= 4.5)
    Feat = np.log10(P[:, :, keep] + 1e-10)                # (n, node, Fk)
    ntr = int(0.75 * N)

    def heldout_r2(Fe):
        A = np.c_[Fe, np.ones(len(Fe))]
        Wt = np.linalg.lstsq(A[:ntr], params[:ntr], rcond=None)[0]
        res = ((A[ntr:] @ Wt - params[ntr:]) ** 2).sum(0)
        tot = ((params[ntr:] - params[:ntr].mean(0)) ** 2).sum(0)
        return 1.0 - res / tot

    per_node = Feat.reshape(N, -1)
    pooled = Feat.mean(1)
    print(f"  held-out linear R^2 -> regional a: per-node spectra "
          f"{np.round(heldout_r2(per_node), 2)}")
    print(f"                                     pooled spectra "
          f"{np.round(heldout_r2(pooled), 2)}")
    print("  (per-node >> pooled means the node axis actually carries "
          "information)")
    return tok, n_frames, F
