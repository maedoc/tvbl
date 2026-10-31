# EXPERIMENTAL: see cde-attn-handoff.md. API is not stable.
"""Self-attention context encoders for conditional density estimation.

Prototype module: a small pre-LN transformer encoder that maps a *token
sequence* (typically an STFT of a simulation, shape ``(n_tokens, token_dim)``)
to a fixed-size context vector ``c``, plus :class:`AttnMAFEstimator` which
feeds that vector into the existing :class:`tvbl.cde.MAFEstimator`.

Why the split works
-------------------
Self-attention is a permutation-equivariant *set* encoder, so it can consume a
variable number of tokens; MADE needs a fixed-size, ordered context vector and
supplies the *exactly normalised* density over the (small) parameter space.
Since the encoder is a deterministic function of the data,

    log p(y | x) = log p(y | enc(x))

is still a proper density in ``y``: adding the encoder cannot break the flow's
normalisation.  It only changes the conditioning.

Everything here is plain NumPy with hand-written backward passes, matching the
rest of the package (no autograd at runtime).

Notes on the implementation
---------------------------
* Pre-LN residual blocks (attention, then a tanh FFN), final LayerNorm, mean
  pool over tokens.
* Dtypes follow the input tensor: pass ``float64`` tokens and every weight is
  promoted, which is what the finite-difference / autograd tests rely on.
* Positions are a learned embedding added once at the input, so the encoder is
  not permutation equivariant across time -- that is intentional for
  time-frequency tokens.
"""

import numpy as np

from .cde import ConditionalDensityEstimator, MAFEstimator

LN_EPS = 1e-5


def stft_tokens(traces, fs=1.0, nperseg=32, hop=8, db_floor=1e-10,
                f_max=None):
    """Log-power STFT tokens of raw traces, shape ``(N, n_frames, n_bands)``.

    ``traces`` is ``(N, T)`` or ``(N, T, n_channels)``; channels are flattened
    into the token axis as ``channel * n_frames + frame`` so a graph bias can
    be built as a block matrix over channels.  With the defaults and T=200 at
    fs=200 this gives 22 frames of 17 bands.  Pure NumPy/SciPy, so it also
    runs under wasm.
    """
    from scipy.signal import stft

    traces = np.asarray(traces)
    if traces.ndim == 2:
        traces = traces[:, :, None]
    n, T, n_ch = traces.shape
    flat = traces.transpose(0, 2, 1).reshape(n * n_ch, T)
    freqs, frames, Z = stft(flat, fs=fs, nperseg=nperseg,
                            noverlap=nperseg - hop, boundary=None,
                            padded=False, axis=-1)
    power = np.abs(Z) ** 2
    if f_max is not None:
        keep = freqs <= f_max
        power, freqs = power[:, keep], freqs[keep]
    lp = np.log10(power + db_floor).astype(traces.dtype if traces.dtype.kind == 'f'
                                           else np.float32)
    n_frames = lp.shape[2]
    lp = lp.reshape(n, n_ch, lp.shape[1], n_frames)
    lp = lp.transpose(0, 1, 3, 2)                    # (N, ch, frame, band)
    return lp.reshape(n, n_ch * n_frames, lp.shape[-1]), freqs


def _elu_plus1(x):
    """Positive feature map for linear attention: ``elu(x) + 1``.

    ``x + 1`` for ``x > 0`` and ``exp(x)`` otherwise -- smooth across the
    origin and strictly positive, so ``phi(q) . sum_j phi(k_j) > 0`` and the
    read-out is a genuine convex combination.
    """
    return np.where(x > 0, x + 1.0, np.exp(np.minimum(x, 0.0)))


def _elu_plus1_grad(x):
    """Derivative of :func:`_elu_plus1`: ``1`` above zero, ``exp(x)`` below."""
    return np.where(x > 0, 1.0, np.exp(np.minimum(x, 0.0)))


# =============================================================================
# == LayerNorm helpers
# =============================================================================

def _ln_forward(x, gain, bias, eps=LN_EPS):
    """LayerNorm over the last axis. Returns (y, cache)."""
    mu = x.mean(axis=-1, keepdims=True)
    xc = x - mu
    var = (xc * xc).mean(axis=-1, keepdims=True)
    rstd = 1.0 / np.sqrt(var + eps)
    xhat = xc * rstd
    return xhat * gain + bias, (xhat, rstd, gain)


def _ln_backward(cache, bar_y):
    """Adjoint of :func:`_ln_forward`. Returns (bar_x, d_gain, d_bias)."""
    xhat, rstd, gain = cache
    flat = bar_y.reshape(-1, bar_y.shape[-1])
    d_gain = (flat * xhat.reshape(flat.shape)).sum(axis=0)
    d_bias = flat.sum(axis=0)
    bar_xhat = bar_y * gain
    m1 = bar_xhat.mean(axis=-1, keepdims=True)
    m2 = (bar_xhat * xhat).mean(axis=-1, keepdims=True)
    return rstd * (bar_xhat - m1 - xhat * m2), d_gain, d_bias


# =============================================================================
# == Attention encoder
# =============================================================================

class AttnEncoder:
    """Pre-LN transformer encoder over a token sequence, mean-pooled.

    Parameters
    ----------
    n_tokens : int
        Sequence length ``T`` (e.g. number of STFT frames).
    token_dim : int
        Per-token feature dimension ``F`` (e.g. number of frequency bands).
    d_model : int, optional
        Embedding width.
    n_heads : int, optional
        Attention heads; ``d_model`` must be divisible by it.
    n_blocks : int, optional
        Number of pre-LN blocks. ``0`` degenerates to a learned linear summary
        plus mean pooling -- the ablation that isolates *attention* from
        *learned pooling*.
    ffn_mult : int, optional
        Hidden width of the FFN, as a multiple of ``d_model``.
    attn_bias : numpy.ndarray, shape (n_tokens, n_tokens), optional
        Additive logit bias applied identically in every block, scaled by one
        trainable scalar (``e_bias_scale``).  This is how a structural
        connectivity matrix is injected: tokens belonging to neighbouring
        nodes get higher attention weight.  The bias itself is fixed; only its
        scale is learned.  Softmax attention only -- linear attention has no
        explicit attention matrix to bias.
    attn_type : {'softmax', 'linear'}, optional
        ``'softmax'`` is exact softmax attention, ``O(T^2 * d)`` time and an
        ``O(T^2)`` score matrix per head.  ``'linear'`` uses the elu+1 feature
        map and the associativity trick (Katharopoulos et al.), giving
        ``O(T * d^2)`` time and an ``O(d^2)`` state per head with no score
        matrix at all.  At T=450 the linear form is ~11x faster in NumPy
        because ~70% of softmax attention's cost is the ``T^2``
        transcendentals, not the matmuls.
    """

    def __init__(self, n_tokens: int, token_dim: int, d_model: int = 32,
                 n_heads: int = 2, n_blocks: int = 2, ffn_mult: int = 2,
                 attn_bias=None, attn_type: str = 'softmax'):
        if n_tokens < 1 or token_dim < 1:
            raise ValueError("n_tokens and token_dim must be positive.")
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads.")
        if attn_type not in ('softmax', 'linear'):
            raise ValueError(f"attn_type must be 'softmax' or 'linear', "
                             f"got {attn_type!r}")
        if attn_bias is not None:
            if attn_type == 'linear':
                raise ValueError("attn_bias requires attn_type='softmax'; "
                                 "linear attention has no score matrix.")
            attn_bias = np.asarray(attn_bias)
            if attn_bias.shape != (n_tokens, n_tokens):
                raise ValueError(
                    f"attn_bias must be {(n_tokens, n_tokens)}, "
                    f"got {attn_bias.shape}")
        self.n_tokens = n_tokens
        self.token_dim = token_dim
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_blocks = n_blocks
        self.head_dim = d_model // n_heads
        self.d_ff = ffn_mult * d_model
        self.scale = 1.0 / np.sqrt(self.head_dim)
        self.attn_bias = attn_bias
        self.attn_type = attn_type

    # -- weight lifecycle ----------------------------------------------------
    def _initialize_weights(self, rng: np.random.RandomState, dtype='f') -> dict:
        d, dt = self.d_model, np.dtype(dtype)
        w = {
            'e_Win': (rng.randn(self.token_dim, d)
                      * np.sqrt(2.0 / (self.token_dim + d))).astype(dt),
            'e_bin': np.zeros(d, dt),
            'e_pos': (rng.randn(self.n_tokens, d) * 0.02).astype(dt),
        }
        for l in range(self.n_blocks):
            w[f'e_ln1_g_{l}'] = np.ones(d, dt)
            w[f'e_ln1_b_{l}'] = np.zeros(d, dt)
            w[f'e_Wqkv_{l}'] = (rng.randn(d, 3 * d) * np.sqrt(1.0 / d)).astype(dt)
            w[f'e_bqkv_{l}'] = np.zeros(3 * d, dt)
            # Residual branches start small so each block is near-identity.
            w[f'e_Wo_{l}'] = (rng.randn(d, d) * np.sqrt(0.5 / d)).astype(dt)
            w[f'e_bo_{l}'] = np.zeros(d, dt)
            w[f'e_ln2_g_{l}'] = np.ones(d, dt)
            w[f'e_ln2_b_{l}'] = np.zeros(d, dt)
            w[f'e_Wf1_{l}'] = (rng.randn(d, self.d_ff)
                               * np.sqrt(2.0 / (d + self.d_ff))).astype(dt)
            w[f'e_bf1_{l}'] = np.zeros(self.d_ff, dt)
            w[f'e_Wf2_{l}'] = (rng.randn(self.d_ff, d)
                               * np.sqrt(0.5 / self.d_ff)).astype(dt)
            w[f'e_bf2_{l}'] = np.zeros(d, dt)
        w['e_lnf_g'] = np.ones(d, dt)
        w['e_lnf_b'] = np.zeros(d, dt)
        if self.attn_bias is not None:
            w['e_bias_scale'] = np.ones((), dt)
        return w

    # -- forward -------------------------------------------------------------
    def forward(self, weights: dict, tokens: np.ndarray):
        """Encode ``tokens`` (N, T, F) into ``c`` (N, d_model).

        Returns
        -------
        c : numpy.ndarray, shape (N, d_model)
        cache : dict
            Everything the backward pass needs.
        """
        n, T, F = tokens.shape
        if T != self.n_tokens or F != self.token_dim:
            raise ValueError(
                f"expected tokens of shape (N, {self.n_tokens}, {self.token_dim}), "
                f"got {tokens.shape}")
        d, H, dh = self.d_model, self.n_heads, self.head_dim

        h = tokens @ weights['e_Win'] + weights['e_bin'] + weights['e_pos']
        bias = None
        if self.attn_bias is not None:
            bias = weights['e_bias_scale'] * self.attn_bias
        blocks = []
        for l in range(self.n_blocks):
            a1, ln1 = _ln_forward(h, weights[f'e_ln1_g_{l}'], weights[f'e_ln1_b_{l}'])
            qkv = a1 @ weights[f'e_Wqkv_{l}'] + weights[f'e_bqkv_{l}']
            q, k, v = qkv[..., :d], qkv[..., d:2 * d], qkv[..., 2 * d:]

            def heads(z):
                return z.reshape(n, T, H, dh).transpose(0, 2, 1, 3)

            qh, kh, vh = heads(q), heads(k), heads(v)
            if self.attn_type == 'softmax':
                logits = (qh @ kh.transpose(0, 1, 3, 2)) * self.scale
                if bias is not None:
                    logits = logits + bias
                logits -= logits.max(axis=-1, keepdims=True)
                probs = np.exp(logits)
                probs /= probs.sum(axis=-1, keepdims=True)
                oh = probs @ vh
                lin = None
            else:
                qp, kp = _elu_plus1(qh), _elu_plus1(kh)
                # O(T*d^2) with plain BLAS matmuls: einsum here is ~5.5x slower.
                # Summarise keys/values once, then read out; no T x T matrix.
                state = kp.transpose(0, 1, 3, 2) @ vh
                zsum = kp.sum(axis=2)
                num = qp @ state
                den = (qp * zsum[:, :, None, :]).sum(axis=-1)
                # +1e-6: _elu_plus1 can underflow to 0 in float32 for very
                # negative pre-activations; without the guard den == 0 and
                # the read-out is 0/0.
                oh = num / (den + 1e-6)[..., None]
                probs = None
                lin = {'qp': qp, 'kp': kp, 'state': state, 'zsum': zsum,
                       'num': num, 'den': den, 'qh': qh, 'kh': kh}
            o = oh.transpose(0, 2, 1, 3).reshape(n, T, d)
            attn_out = o @ weights[f'e_Wo_{l}'] + weights[f'e_bo_{l}']
            h = h + attn_out

            a2, ln2 = _ln_forward(h, weights[f'e_ln2_g_{l}'], weights[f'e_ln2_b_{l}'])
            fh = np.tanh(a2 @ weights[f'e_Wf1_{l}'] + weights[f'e_bf1_{l}'])
            ffn_out = fh @ weights[f'e_Wf2_{l}'] + weights[f'e_bf2_{l}']
            h = h + ffn_out

            blocks.append({'ln1': ln1, 'a1': a1, 'qh': qh, 'kh': kh, 'vh': vh,
                           'probs': probs, 'lin': lin, 'o': o,
                           'attn_out': attn_out, 'ln2': ln2, 'a2': a2,
                           'fh': fh, 'ffn_out': ffn_out})

        hn, lnf = _ln_forward(h, weights['e_lnf_g'], weights['e_lnf_b'])
        c = hn.mean(axis=1)
        cache = {'tokens': tokens, 'h0': h, 'blocks': blocks,
                 'hn': hn, 'lnf': lnf, 'c': c}
        return c, cache

    # -- backward ------------------------------------------------------------
    def backward(self, weights: dict, cache: dict, bar_c: np.ndarray) -> dict:
        """Adjoint of :meth:`forward`. Returns gradients keyed like ``weights``."""
        d, H, dh = self.d_model, self.n_heads, self.head_dim
        tokens = cache['tokens']
        n, T = tokens.shape[0], tokens.shape[1]
        grads = {}

        bar_hn = np.broadcast_to(bar_c[:, None, :] / T, cache['hn'].shape)
        bar_h, grads['e_lnf_g'], grads['e_lnf_b'] = _ln_backward(
            cache['lnf'], bar_hn)

        for l in reversed(range(self.n_blocks)):
            b = cache['blocks'][l]
            # h_out = h_mid + ffn_out
            bar_ffn = bar_h
            fhn = b['fh'].reshape(-1, self.d_ff)
            grads[f'e_Wf2_{l}'] = fhn.T @ bar_ffn.reshape(-1, d)
            grads[f'e_bf2_{l}'] = bar_ffn.sum(axis=(0, 1))
            bar_pre = (bar_ffn @ weights[f'e_Wf2_{l}'].T) * (1.0 - b['fh'] ** 2)
            grads[f'e_Wf1_{l}'] = (b['a2'].reshape(-1, d).T
                                   @ bar_pre.reshape(-1, self.d_ff))
            grads[f'e_bf1_{l}'] = bar_pre.sum(axis=(0, 1))
            bar_a2 = bar_pre @ weights[f'e_Wf1_{l}'].T
            bar_ln2, grads[f'e_ln2_g_{l}'], grads[f'e_ln2_b_{l}'] = _ln_backward(
                b['ln2'], bar_a2)
            bar_h = bar_h + bar_ln2

            # h_mid = h_in + attn_out
            bar_attn = bar_h
            grads[f'e_Wo_{l}'] = (b['o'].reshape(-1, d).T
                                  @ bar_attn.reshape(-1, d))
            grads[f'e_bo_{l}'] = bar_attn.sum(axis=(0, 1))
            bar_o = (bar_attn @ weights[f'e_Wo_{l}'].T).reshape(n, T, H, dh)
            bar_o = bar_o.transpose(0, 2, 1, 3)

            if self.attn_type == 'softmax':
                probs, vh, qh, kh = b['probs'], b['vh'], b['qh'], b['kh']
                bar_p = bar_o @ vh.transpose(0, 1, 3, 2)
                bar_v = probs.transpose(0, 1, 3, 2) @ bar_o
                d_logits = probs * (bar_p - (bar_p * probs).sum(axis=-1, keepdims=True))
                if self.attn_bias is not None:
                    grads['e_bias_scale'] = (d_logits * self.attn_bias).sum()
                bar_scaled = d_logits * self.scale
                bar_q = bar_scaled @ kh
                bar_k = bar_scaled.transpose(0, 1, 3, 2) @ qh
            else:
                L = b['lin']
                qp, kp, state, zsum = L['qp'], L['kp'], L['state'], L['zsum']
                num, den = L['num'], L['den']
                inv = 1.0 / (den + 1e-6)  # same guard as in forward
                bar_num = bar_o * inv[..., None]
                bar_den = -(bar_o * num * inv[..., None] ** 2).sum(axis=-1)
                # num = qp @ state ; den = qp . zsum
                bar_qp = (bar_num @ state.transpose(0, 1, 3, 2)
                          + bar_den[..., None] * zsum[:, :, None, :])
                bar_state = qp.transpose(0, 1, 3, 2) @ bar_num
                bar_zsum = (bar_den[..., None] * qp).sum(axis=2)
                # state = kp^T v ; zsum = sum_T kp
                #   dkp[t,i] = sum_j vh[t,j] * bar_state[i,j] = (vh @ bar_state^T)[t,i]
                bar_kp = (b['vh'] @ bar_state.transpose(0, 1, 3, 2)
                          + bar_zsum[:, :, None, :])
                bar_v = kp @ bar_state
                bar_q = bar_qp * _elu_plus1_grad(L['qh'])
                bar_k = bar_kp * _elu_plus1_grad(L['kh'])

            def unheads(z):
                return z.transpose(0, 2, 1, 3).reshape(n, T, d)

            bar_qkv = np.concatenate([unheads(bar_q), unheads(bar_k),
                                      unheads(bar_v)], axis=-1)
            grads[f'e_Wqkv_{l}'] = (b['a1'].reshape(-1, d).T
                                    @ bar_qkv.reshape(-1, 3 * d))
            grads[f'e_bqkv_{l}'] = bar_qkv.sum(axis=(0, 1))
            bar_a1 = bar_qkv @ weights[f'e_Wqkv_{l}'].T
            bar_ln1, grads[f'e_ln1_g_{l}'], grads[f'e_ln1_b_{l}'] = _ln_backward(
                b['ln1'], bar_a1)
            bar_h = bar_h + bar_ln1

        grads['e_Win'] = tokens.reshape(-1, self.token_dim).T @ bar_h.reshape(-1, d)
        grads['e_bin'] = bar_h.sum(axis=(0, 1))
        grads['e_pos'] = bar_h.sum(axis=0)
        return grads


# =============================================================================
# == Attention-conditioned MAF
# =============================================================================

class AttnMAFEstimator(ConditionalDensityEstimator):
    """``attention(strft tokens) -> c -> MADE flows -> p(params | data)``.

    Composition, not inheritance: the MADE machinery (masks, exact
    log-likelihood, invertible sampling) is reused unchanged from
    :class:`tvbl.cde.MAFEstimator`; only the conditioning pathway is new.

    ``features`` is expected as ``(N, n_tokens * token_dim)`` so the base
    ``train()`` validation and dtype handling keep working; it is reshaped to
    ``(N, n_tokens, token_dim)`` internally.

    Parameters
    ----------
    param_dim : int
        Dimensionality of the target parameters.
    n_tokens, token_dim : int
        Token-sequence shape produced by the caller's embedder.
    d_model, n_heads, n_blocks, ffn_mult : optional
        Passed to :class:`AttnEncoder`.
    n_flows, hidden_units : optional
        Passed to the MADE flows.
    """

    def __init__(self, param_dim: int, n_tokens: int, token_dim: int,
                 d_model: int = 32, n_heads: int = 2, n_blocks: int = 2,
                 ffn_mult: int = 2, n_flows: int = 4, hidden_units: int = 64,
                 attn_bias=None, attn_type: str = 'softmax'):
        super().__init__(param_dim, n_tokens * token_dim)
        self.n_tokens = n_tokens
        self.token_dim = token_dim
        self.d_model = d_model
        self.encoder = AttnEncoder(n_tokens, token_dim, d_model, n_heads,
                                   n_blocks, ffn_mult, attn_bias, attn_type)
        self.flow = MAFEstimator(param_dim=param_dim, feature_dim=d_model,
                                 n_flows=n_flows, hidden_units=hidden_units)

    # -- weight lifecycle ----------------------------------------------------
    def _initialize_weights(self, rng: np.random.RandomState) -> dict:
        weights = self.encoder._initialize_weights(rng)
        # Also installs ``self.flow.model_constants`` (MADE masks/permutations).
        weights.update(self.flow._initialize_weights(rng))
        return weights

    def _tokens(self, features):
        features = np.asarray(features)
        if features.ndim == 1:
            features = features.reshape(1, -1)
        return features.reshape(-1, self.n_tokens, self.token_dim)

    def _encode(self, weights, features):
        return self.encoder.forward(weights, self._tokens(features))

    def _sync_flow(self):
        """Share the flat weight dict with the inner flow.

        ``MAFEstimator.sample`` reads ``self.weights`` from the *flow*, while
        the estimator keeps one flat dict for encoder + flow.  Keeping the
        same object on both keeps them in sync without copying.
        """
        self.flow.weights = self.weights

    # -- loss / gradients ----------------------------------------------------
    def _loss_and_grad_np(self, weights, features, params):
        ctx, enc_cache = self._encode(weights, features)
        log_prob, caches = self.flow._get_log_prob_np(weights, ctx, params)
        n = log_prob.shape[0]
        loss = float(-np.mean(log_prob))

        grads, bar_ctx = self.flow._flow_backward_np(
            weights, caches, np.full(n, -1.0 / n), compute_ctx_grad=True)
        grads.update(self.encoder.backward(weights, enc_cache, bar_ctx))
        return loss, grads

    def _loss_and_grad(self, weights, features, params):
        return self._loss_and_grad_np(weights, features, params)

    def _loss_function(self, weights, features, params) -> float:
        ctx, _ = self._encode(weights, features)
        return self.flow._loss_function(weights, ctx, params)

    # -- inference -----------------------------------------------------------
    def log_prob(self, features, params):
        super().log_prob(features, params)
        self._sync_flow()
        ctx, _ = self._encode(self.weights, features)
        return self.flow._get_log_prob(self.weights, ctx, params)

    def sample(self, features, n_samples, rng):
        super().sample(features, n_samples, rng)
        self._sync_flow()
        ctx, _ = self._encode(self.weights, features)
        return self.flow.sample(ctx, n_samples, rng)


class RawTSMAFEstimator(AttnMAFEstimator):
    """Attention CDE that ingests raw time series directly.

    Same model as :class:`AttnMAFEstimator`, but ``features`` is the raw
    simulation output ``(N, T)`` or ``(N, T, n_channels)``; the log-power
    STFT tokenisation (:func:`stft_tokens`) happens inside, so callers never
    pre-tokenise::

        est = RawTSMAFEstimator(param_dim=2, T=200, n_channels=1, fs=200.0)
        est.train(params, traces, n_iter=3000)          # traces: (N, 200)
        post = est.sample(traces[:5], 256, rng)         # posteriors over params

    Token statistics (per-band mean/std of the log-power STFT) are computed
    on the training split inside :meth:`train` and applied to any later
    input, so train and test see identical scaling.

    Parameters
    ----------
    T : int
        Number of time samples per simulation.
    n_channels : int
        Number of channels (e.g. nodes); flattened into the token axis.
    fs, nperseg, hop, db_floor, f_max : optional
        STFT settings, see :func:`stft_tokens`.
    Plus all :class:`AttnMAFEstimator` parameters.
    """

    def __init__(self, param_dim: int, T: int, n_channels: int = 1, fs: float = 1.0,
                 nperseg: int = 32, hop: int = 8, db_floor: float = 1e-10,
                 f_max: float = None, **kwargs):
        self.T = T
        self.n_channels = n_channels
        self.fs, self.nperseg, self.hop = fs, nperseg, hop
        self.db_floor, self.f_max = db_floor, f_max
        # Token shape is fixed by the STFT geometry; compute it once on a
        # dummy signal so feature_dim is known before any data arrives.
        dummy = np.zeros((1, T, n_channels), dtype=np.float64)
        tokens, _ = stft_tokens(dummy, fs=fs, nperseg=nperseg, hop=hop,
                                db_floor=db_floor, f_max=f_max)
        n_tokens, token_dim = tokens.shape[1], tokens.shape[2]
        super().__init__(param_dim=param_dim, n_tokens=n_tokens,
                         token_dim=token_dim, **kwargs)
        self._tok_mu = None
        self._tok_sd = None

    def _raw_to_tokens(self, traces):
        tokens, _ = stft_tokens(traces, fs=self.fs, nperseg=self.nperseg,
                                hop=self.hop, db_floor=self.db_floor,
                                f_max=self.f_max)
        if tokens.shape[1:] != (self.n_tokens, self.token_dim):
            raise ValueError(
                f"traces {np.asarray(traces).shape} tokenise to "
                f"{tokens.shape[1:]}, expected {(self.n_tokens, self.token_dim)}; "
                f"check T/n_channels/nperseg/hop")
        if self._tok_mu is not None:
            tokens = (tokens - self._tok_mu) / self._tok_sd
        return tokens

    def _features(self, traces):
        traces = np.asarray(traces)
        # Idempotent: input already flattened to (N, n_tokens * token_dim)
        # (e.g. from train(), which pre-tokenises for base-class validation)
        # is passed through unchanged.
        if traces.ndim == 2 and traces.shape[1] == self.n_tokens * self.token_dim:
            return traces
        tokens = self._raw_to_tokens(traces)
        return tokens.reshape(tokens.shape[0], -1)

    # sbi_eval.fit and the base train() call these directly with whatever
    # ``features`` they were given, so the raw->token map must happen here too.
    def _loss_and_grad_np(self, weights, features, params):
        return super()._loss_and_grad_np(weights, self._features(features), params)

    def _loss_and_grad(self, weights, features, params):
        return super()._loss_and_grad(weights, self._features(features), params)

    def _loss_function(self, weights, features, params):
        return super()._loss_function(weights, self._features(features), params)

    def train(self, params, features, **kwargs):
        # Compute token statistics on the unstandardised tokens of this
        # training split, then standardise; _raw_to_tokens applies the stored
        # statistics to any later input.
        self._tok_mu = None
        raw_tokens = self._raw_to_tokens(features)
        self._tok_mu = raw_tokens.mean(axis=(0, 1), keepdims=True)
        self._tok_sd = raw_tokens.std(axis=(0, 1), keepdims=True) + 1e-6
        # Base train() validates on the flat token width, so feed it the
        # standardised tokens; its inner _loss_and_grad re-maps them through
        # _features, which is idempotent on already-tokenised input.
        flat = self._raw_to_tokens(features).reshape(features.shape[0], -1)
        super().train(params, flat, **kwargs)

    def log_prob(self, features, params):
        return super().log_prob(self._features(features), params)

    def sample(self, features, n_samples, rng):
        return super().sample(self._features(features), n_samples, rng)
