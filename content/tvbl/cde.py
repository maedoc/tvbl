"""
A module for conditional density estimation using Mixture Density
Networks (MDNs) and Masked Autoregressive Flows (MAFs).

"""

import abc
import math
from dataclasses import dataclass, field

import numpy as np

from scipy.special import logsumexp as np_logsumexp
from scipy.stats import t
from tqdm.auto import trange

MAF_ALPHA_LIMIT = 7.0

# Symmetric bound applied to the MDN precision log-diagonal before
# exponentiation.  Keeping ``exp(ld)`` within ``exp(+-20)`` prevents overflow
# (and catastrophic underflow) for extreme network outputs.
MDN_LOG_DIAG_LIMIT = 20.0

# =============================================================================
# == Base Class for Conditional Density Estimators
# =============================================================================

class ConditionalDensityEstimator(abc.ABC):
    """
    Abstract base class for conditional density estimators.

    This class provides a unified training interface using the Adam optimizer
    and standardizes the API for training, sampling, and log-probability
    evaluation.

    Parameters
    ----------
    param_dim : int
        Dimensionality of the target variable (parameters to be estimated).
    feature_dim : int
        Dimensionality of the conditional variable (features).
    """
    def __init__(self, param_dim: int, feature_dim: int):
        if not (param_dim > 0 and feature_dim >= 0):
            raise ValueError("Parameter and feature dimensions must be positive/non-negative.")
        self.param_dim = param_dim
        self.feature_dim = feature_dim
        self.weights = None
        self.loss_history = []

    @abc.abstractmethod
    def _initialize_weights(self, rng: np.random.RandomState) -> dict:
        """
        Initialize the trainable weights of the model.

        Parameters
        ----------
        rng : numpy.random.RandomState
            A random number generator for reproducible initialization.

        Returns
        -------
        dict
            A dictionary of initialized weight arrays.
        """
        pass

    @abc.abstractmethod
    def _loss_function(self, weights: dict, features: np.ndarray, params: np.ndarray) -> float:
        """
        Compute the negative log-likelihood loss for a batch of data.

        Parameters
        ----------
        weights : dict
            A dictionary of the model's trainable weights.
        features : np.ndarray
            A (N, feature_dim) array of conditional features.
        params : np.ndarray
            A (N, param_dim) array of target parameters.

        Returns
        -------
        float
            The mean negative log-likelihood of the batch.
        """
        pass

    @abc.abstractmethod
    def sample(self, features: np.ndarray, n_samples: int, rng: np.random.RandomState) -> np.ndarray:
        """
        Generate samples from the learned conditional distribution p(params|features).

        Parameters
        ----------
        features : np.ndarray
            A (n_conditions, feature_dim) array of features to condition on.
        n_samples : int
            The number of samples to generate for each condition.
        rng : numpy.random.RandomState
            A random number generator for sampling.

        Returns
        -------
        np.ndarray
            An array of generated samples of shape (n_conditions, n_samples, param_dim).
        """
        if self.weights is None:
            raise RuntimeError("Model has not been trained yet. Call train() first.")

    @abc.abstractmethod
    def log_prob(self, features: np.ndarray, params: np.ndarray) -> np.ndarray:
        """
        Compute the log-probability log p(params|features).

        Parameters
        ----------
        features : np.ndarray
            A (N, feature_dim) array of conditional features.
        params : np.ndarray
            A (N, param_dim) array of target parameters.

        Returns
        -------
        np.ndarray
            A (N,) array of log-probabilities.
        """
        if self.weights is None:
            raise RuntimeError("Model has not been trained yet. Call train() first.")

    def train(self, params: np.ndarray, features: np.ndarray,
              n_iter: int = 2000, learning_rate: float = 1e-3,
              seed: int = 0, use_tqdm: bool = True):
        """
        Trains the model using the Adam optimizer.

        Parameters
        ----------
        params : np.ndarray
            An (N, param_dim) matrix of simulated parameters.
        features : np.ndarray
            An (N, feature_dim) matrix of corresponding data features.
        n_iter : int, optional
            The number of gradient descent iterations.
        learning_rate : float, optional
            The learning rate for the Adam optimizer.
        seed : int, optional
            Seed for reproducible weight initialization and training.
        use_tqdm : bool, optional
            If True, displays a progress bar during training.
        """
        # --- 1. Data Validation ---
        if params.shape[0] != features.shape[0]:
            raise ValueError("Params and features must have the same number of samples.")
        if params.shape[1] != self.param_dim or features.shape[1] != self.feature_dim:
            raise ValueError("Data dimensions do not match model dimensions.")

        # Filter out non-finite values
        finite_idx = np.all(np.isfinite(params), axis=1) & np.all(np.isfinite(features), axis=1)
        params = params[finite_idx]
        features = features[finite_idx]

        if params.shape[0] == 0:
            raise ValueError("All data points contained non-finite values.")

        # --- 2. Initialization ---
        rng = np.random.RandomState(seed)
        self.weights = self._initialize_weights(rng)

        # Match the data dtype to the initialized weights (e.g. float64 subclasses)
        dtype = np.asarray(next(iter(self.weights.values()))).dtype
        params = params.astype(dtype, copy=False)
        features = features.astype(dtype, copy=False)

        self.loss_history = []

        # --- 3. Optimization Loop ---
        # Adam optimizer state
        m = {key: np.zeros_like(val) for key, val in self.weights.items()}
        v = {key: np.zeros_like(val) for key, val in self.weights.items()}
        beta1, beta2, epsilon = 0.9, 0.999, 1e-8

        iterator = trange(n_iter, desc="Training", disable=not use_tqdm)
        for i in iterator:
            loss, g = self._loss_and_grad(self.weights, features, params)
            self.loss_history.append(loss)

            if not np.isfinite(loss):
                print(f"Warning: Loss is non-finite at iteration {i}. Stopping training.")
                break

            if use_tqdm:
                iterator.set_postfix(loss=f"{loss:.4f}")

            # Adam update step: validate every gradient before changing any
            # weight, so a late non-finite gradient cannot leave a partial update.
            for key in self.weights:
                if not np.all(np.isfinite(g[key])):
                    print(f"Warning: Non-finite gradient for '{key}' at iteration {i}. Stopping.")
                    return
            for key in self.weights:
                m[key] = beta1 * m[key] + (1 - beta1) * g[key]
                v[key] = beta2 * v[key] + (1 - beta2) * (g[key]**2)
                m_hat = m[key] / (1 - beta1**(i + 1))
                v_hat = v[key] / (1 - beta2**(i + 1))
                self.weights[key] -= learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)


# =============================================================================
# == MDN Implementation
# =============================================================================

@dataclass
class MDNEstimator(ConditionalDensityEstimator):
    """
    Mixture Density Network for conditional density estimation.

    Parameters
    ----------
    param_dim : int
        Dimensionality of the target variable.
    feature_dim : int
        Dimensionality of the conditional variable.
    n_components : int, optional
        The number of Gaussian mixture components.
    hidden_sizes : tuple[int, ...], optional
        A tuple specifying the number of units in each hidden layer.
    """
    param_dim: int
    feature_dim: int
    n_components: int = 5
    hidden_sizes: tuple[int, ...] = (32, 32)

    def __post_init__(self):
        super().__init__(self.param_dim, self.feature_dim)
        self._offdiag_basis = self._create_offdiag_basis()
        self._offdiag_rows, self._offdiag_cols = np.triu_indices(self.param_dim, k=1)

    def _create_offdiag_basis(self):
        n_off_diag = self.param_dim * (self.param_dim - 1) // 2
        if n_off_diag == 0:
            return None
        basis = np.zeros((n_off_diag, self.param_dim, self.param_dim), dtype='f')
        rows, cols = np.triu_indices(self.param_dim, k=1)
        basis[np.arange(n_off_diag), rows, cols] = 1
        return basis

    def _initialize_weights(self, rng: np.random.RandomState) -> dict:
        """Initializes weights for the MLP and GMM output layers."""
        weights = {}
        in_size = self.feature_dim
        for i, out_size in enumerate(self.hidden_sizes):
            weights[f'W{i}'] = (rng.randn(in_size, out_size) * np.sqrt(2.0 / in_size)).astype('f')
            weights[f'b{i}'] = np.zeros(out_size, dtype='f')
            in_size = out_size

        last_hidden_size = self.hidden_sizes[-1] if self.hidden_sizes else self.feature_dim

        # GMM output layers
        K, D_out = self.n_components, self.param_dim
        weights['W_alpha'] = (rng.randn(last_hidden_size, K) * 0.01).astype('f')
        weights['b_alpha'] = np.zeros(K, dtype='f')
        weights['W_mu'] = (rng.randn(last_hidden_size, K * D_out) * 0.01).astype('f')
        weights['b_mu'] = np.zeros(K * D_out, dtype='f')
        weights['W_L_prec_log_diag'] = (rng.randn(last_hidden_size, K * D_out) * 0.01).astype('f')
        weights['b_L_prec_log_diag'] = np.zeros(K * D_out, dtype='f')

        n_off_diag = D_out * (D_out - 1) // 2
        if n_off_diag > 0:
            weights['W_L_prec_offdiag'] = (rng.randn(last_hidden_size, K * n_off_diag) * 0.01).astype('f')
            weights['b_L_prec_offdiag'] = np.zeros(K * n_off_diag, dtype='f')

        return weights

    def _forward_pass(self, weights: dict, features: np.ndarray):
        """Maps input features to GMM parameters."""
        alpha, mu, L_prec, ld, _ = self._forward_pass_np(weights, features)
        return alpha, mu, L_prec, ld

    def _forward_pass_np(self, weights: dict, features: np.ndarray):
        """Plain-NumPy version of ``_forward_pass``, plus a backward cache."""
        hs = [features]
        for i in range(len(self.hidden_sizes)):
            hs.append(np.tanh(hs[-1] @ weights[f'W{i}'] + weights[f'b{i}']))
        h = hs[-1]

        K, D = self.n_components, self.param_dim
        log_alpha = h @ weights['W_alpha'] + weights['b_alpha']
        alpha = np.exp(log_alpha - np_logsumexp(log_alpha, axis=1, keepdims=True))
        mu = (h @ weights['W_mu'] + weights['b_mu']).reshape(-1, K, D)
        ld_raw = (h @ weights['W_L_prec_log_diag'] + weights['b_L_prec_log_diag']).reshape(-1, K, D)
        # Bound the log-diagonal before exponentiation so the precision
        # diagonal stays finite for extreme network outputs.
        ld = np.clip(ld_raw, -MDN_LOG_DIAG_LIMIT, MDN_LOG_DIAG_LIMIT)
        ld_mask = (ld_raw > -MDN_LOG_DIAG_LIMIT) & (ld_raw < MDN_LOG_DIAG_LIMIT)

        N = h.shape[0]
        L_prec = np.zeros((N, K, D, D), dtype=ld.dtype)
        idx = np.arange(D)
        L_prec[:, :, idx, idx] = np.exp(ld)

        off = None
        if self._offdiag_rows.size > 0:
            off = (h @ weights['W_L_prec_offdiag'] + weights['b_L_prec_offdiag']).reshape(
                -1, K, self._offdiag_rows.size)
            L_prec[:, :, self._offdiag_rows, self._offdiag_cols] = off

        cache = {'hs': hs, 'h': h, 'log_alpha': log_alpha, 'off': off,
                 'ld_raw': ld_raw, 'ld_mask': ld_mask}
        return alpha, mu, L_prec, ld, cache

    def _mdn_log_prob_from_cache(self, log_alpha, mu, L_prec, ld, params):
        """Mixture log-density with per-component responsibilities."""
        delta = params[:, np.newaxis, :] - mu
        z = np.einsum('nkij,nkj->nki', L_prec, delta)
        quad_term = -0.5 * np.sum(z**2, axis=2)
        log_det_term = np.sum(ld, axis=2)
        log_probs_k = quad_term + log_det_term - 0.5 * self.param_dim * np.log(2 * math.pi)

        log_joint = log_alpha + log_probs_k
        log_prob = np_logsumexp(log_joint, axis=1)
        responsibilities = np.exp(log_joint - log_prob[:, np.newaxis])

        return log_prob, responsibilities, delta, z

    def _loss_and_grad_np(self, weights, features, params):
        """Plain-NumPy negative mean log-likelihood and its weight gradients."""
        alpha, mu, L_prec, ld, cache = self._forward_pass_np(weights, features)
        log_prob, r, delta, z = self._mdn_log_prob_from_cache(
            cache['log_alpha'], mu, L_prec, ld, params)
        # numpy/scipy scalar promotion upcasts float32 there; keep adjoints in
        # the forward dtype.
        r = r.astype(alpha.dtype, copy=False)
        N = params.shape[0]
        loss = float(-np.mean(log_prob))

        bar_a = -r / N
        bar_lp = bar_a

        bar_z = -bar_lp[..., None] * z
        # Diagonal of bar_z[:, :, :, None] * delta[:, :, None, :] is bar_z * delta.
        # The clip has zero derivative outside the bound, so the direct
        # log-diagonal gradient is gated by the clip mask.
        bar_ld = (bar_lp[..., None] + bar_z * delta * np.exp(ld)) * cache['ld_mask']
        bar_L = bar_z[..., :, None] * delta[..., None, :]

        bar_delta = np.einsum('nkij,nki->nkj', L_prec, bar_z)
        bar_mu = -bar_delta

        M = self._offdiag_rows.size
        bar_off = bar_L[:, :, self._offdiag_rows, self._offdiag_cols] if M else None

        bar_log_alpha = bar_a

        h = cache['h']
        bar_mu_flat = bar_mu.reshape(N, -1)
        bar_ld_flat = bar_ld.reshape(N, -1)
        grads = {
            'W_alpha': h.T @ bar_log_alpha,
            'b_alpha': bar_log_alpha.sum(axis=0),
            'W_mu': h.T @ bar_mu_flat,
            'b_mu': bar_mu_flat.sum(axis=0),
            'W_L_prec_log_diag': h.T @ bar_ld_flat,
            'b_L_prec_log_diag': bar_ld_flat.sum(axis=0),
        }
        bar_h = (bar_log_alpha @ weights['W_alpha'].T
                 + bar_mu_flat @ weights['W_mu'].T
                 + bar_ld_flat @ weights['W_L_prec_log_diag'].T)
        if M:
            bar_off_flat = bar_off.reshape(N, -1)
            grads['W_L_prec_offdiag'] = h.T @ bar_off_flat
            grads['b_L_prec_offdiag'] = bar_off_flat.sum(axis=0)
            bar_h = bar_h + bar_off_flat @ weights['W_L_prec_offdiag'].T

        hs = cache['hs']
        for i in reversed(range(len(self.hidden_sizes))):
            bar_hidden = bar_h * (1.0 - hs[i + 1] ** 2)
            grads[f'W{i}'] = hs[i].T @ bar_hidden
            grads[f'b{i}'] = bar_hidden.sum(axis=0)
            bar_h = bar_hidden @ weights[f'W{i}'].T

        return loss, grads

    def _loss_function(self, weights: dict, features: np.ndarray, params: np.ndarray) -> float:
        """Negative log-likelihood of the true parameters under the GMM."""
        alpha, mu, L_prec, ld, cache = self._forward_pass_np(weights, features)
        log_prob, _, _, _ = self._mdn_log_prob_from_cache(
            cache['log_alpha'], mu, L_prec, ld, params)
        return float(-np.mean(log_prob))

    def _loss_and_grad(self, weights, features, params):
        """Train on the plain-NumPy loss/gradient path."""
        return self._loss_and_grad_np(weights, features, params)

    def log_prob(self, features: np.ndarray, params: np.ndarray) -> np.ndarray:
        """Computes the log-probability log p(params|features) for each sample."""
        super().log_prob(features, params)
        alpha, mu, L_prec, ld, cache = self._forward_pass_np(self.weights, features)
        log_prob, _, _, _ = self._mdn_log_prob_from_cache(
            cache['log_alpha'], mu, L_prec, ld, params)
        return log_prob

    def sample(self, features: np.ndarray, n_samples: int, rng: np.random.RandomState) -> np.ndarray:
        super().sample(features, n_samples, rng)
        features = np.asarray(features, dtype='f')
        if features.ndim == 1:
            features = features.reshape(1, -1)

        alpha, mu, L_prec, _, _ = self._forward_pass_np(self.weights, features)
        n_cond, K, D_out = mu.shape

        log_alpha = np.log(alpha + 1e-9)
        gumbel_noise = -np.log(-np.log(rng.uniform(size=(n_cond, n_samples, K))))
        component_indices = np.argmax(log_alpha[:, np.newaxis, :] + gumbel_noise, axis=2)

        cond_idx = np.arange(n_cond)[:, np.newaxis]
        chosen_mu = mu[cond_idx, component_indices]
        chosen_L_prec = L_prec[cond_idx, component_indices]

        try:
            L_cov_factor = np.linalg.inv(chosen_L_prec)
        except np.linalg.LinAlgError:
            print("Warning: Singular precision matrix encountered during sampling. Returning NaNs.")
            return np.full((n_cond, n_samples, D_out), np.nan)

        z = rng.randn(n_cond, n_samples, D_out)
        samples = chosen_mu + np.einsum('nsij,nsj->nsi', L_cov_factor, z)

        return samples

# =============================================================================
# == MAF Implementation
# =============================================================================

@dataclass
class MAFEstimator(ConditionalDensityEstimator):
    """
    Masked Autoregressive Flow for conditional density estimation.

    Parameters
    ----------
    param_dim : int
        Dimensionality of the target variable.
    feature_dim : int
        Dimensionality of the conditional variable.
    n_flows : int, optional
        The number of flow layers (MADE blocks).
    hidden_units : int, optional
        The number of hidden units in each MADE block.
    """
    param_dim: int
    feature_dim: int
    n_flows: int = 4
    hidden_units: int = 64

    def __post_init__(self):
        super().__init__(self.param_dim, self.feature_dim)
        self.model_constants = None # For non-trainable parts like masks

    def _initialize_weights(self, rng: np.random.RandomState) -> dict:
        """Initializes weights and model constants (masks, permutations)."""
        weights = {}
        layers = []
        D, C, H = self.param_dim, self.feature_dim, self.hidden_units

        for k in range(self.n_flows):
            # MADE masks and permutation
            m_in = np.arange(1, D + 1)
            m_hidden = rng.randint(1, D, size=H) if D > 1 else np.ones(H, dtype=int)
            M1 = (m_in[None, :] <= m_hidden[:, None]).astype('f')
            m_out = m_in.copy()
            M2 = (m_hidden[None, :] < m_out[:, None]).astype('f')
            M2t = np.concatenate([M2, M2], axis=0)
            perm = rng.permutation(D)
            inv_perm = np.empty(D, dtype=int); inv_perm[perm] = np.arange(D)

            layers.append({'M1': M1, 'M2': M2, 'M2t': M2t,
                           'perm': perm, 'inv_perm': inv_perm})

            # Trainable parameters
            w_std = 0.01
            weights[f'W1y_{k}'] = (rng.randn(H, D) * w_std).astype('f')
            weights[f'W1c_{k}'] = (rng.randn(H, C) * w_std).astype('f') if C > 0 else np.zeros((H, C), dtype='f')
            weights[f'b1_{k}'] = np.zeros(H, dtype='f')
            weights[f'W2_{k}'] = (rng.randn(2 * D, H) * w_std).astype('f')
            weights[f'W2c_{k}'] = (rng.randn(2 * D, C) * w_std).astype('f') if C > 0 else np.zeros((2*D, C), dtype='f')
            weights[f'b2_{k}'] = np.zeros(2 * D, dtype='f')

        self.model_constants = {'layers': layers}
        return weights

    def _made_forward(self, y, ctx, layer_const, k, weights):
        """Single forward pass through a MADE block."""
        mu, alpha, _ = self._made_forward_np(weights, y, ctx, layer_const, k)
        return mu, alpha

    def _made_forward_np(self, weights, y, ctx, layer_const, k):
        """Pure NumPy equivalent of :meth:`_made_forward`, with a backward cache.

        This is a drop-in numerical replacement for ``_made_forward``: it uses
        regular ``numpy`` and additionally returns
        every intermediate quantity needed by an explicit, hand-written
        backward pass.  The forward semantics are identical to
        ``_made_forward``, including the ``feature_dim == 0`` branch where the
        context contribution is the scalar ``0.0``.

        Parameters
        ----------
        weights : dict
            Full weight dictionary of the estimator.
        y : numpy.ndarray, shape (n_samples, D)
            Input to this block (already permuted by ``layer_const['perm']``).
        ctx : numpy.ndarray, shape (n_samples, C)
            Conditioning features. Ignored when ``self.feature_dim == 0``.
        layer_const : dict
            Per-layer constants with ``'M1'`` (H, D) and ``'M2'`` (D, H)
            masks.
        k : int
            Layer index used to look up ``W1y_{k}``, ``W1c_{k}``, ``b1_{k}``,
            ``W2_{k}``, ``W2c_{k}`` and ``b2_{k}``.

        Returns
        -------
        mu : numpy.ndarray, shape (n_samples, D)
            Location output of the block.
        alpha : numpy.ndarray, shape (n_samples, D)
            Log-scale output of the block, clipped to ``[-7, 7]``.
        cache : dict
            Backward cache. Keys:

            ``'p'``
                The input ``y`` of this block (post-permutation).
            ``'ctx'``
                The conditioning features ``ctx``.
            ``'M1'``, ``'M2'``
                The masks used for this block (``'M2'`` untiled, shape (D, H)).
            ``'W1y'``, ``'W1c'``, ``'b1'``, ``'W2'``, ``'W2c'``, ``'b2'``
                The raw (unmasked) parameter arrays for this block.
            ``'W1y_masked'``
                ``W1y * M1``, shape (H, D).
            ``'W2_masked'``
                ``W2 * M2_tiled`` where ``M2_tiled = concatenate([M2, M2])``,
                shape (2D, H).
            ``'y_h'``
                ``p @ W1y_masked.T``, shape (n_samples, H).
            ``'c_h'``
                Context contribution to the hidden pre-activation,
                ``ctx @ W1c.T``, or the scalar ``0.0`` when
                ``feature_dim == 0`` (broadcastable, as in ``_made_forward``).
            ``'hidden_pre'``
                ``y_h + c_h + b1``, the pre-activation of ``tanh``.
            ``'h'``
                ``tanh(hidden_pre)``, shape (n_samples, H).
            ``'c_out'``
                Context contribution to the output, ``ctx @ W2c.T``, or
                ``None`` when ``feature_dim == 0``.
            ``'out'``
                Pre-split, pre-clip output with ``2 * D`` columns.
            ``'out_alpha_preclip'``
                ``out[:, D:]`` before clipping, shape (n_samples, D).
            ``'mu'``
                ``out[:, :D]``, shape (n_samples, D).
            ``'alpha'``
                ``7 * tanh(out_alpha_preclip / 7)``, shape (n_samples, D).
            ``'alpha_preclip'``
                Alias for ``'out_alpha_preclip'`` kept for backward readability.
            ``'alpha_deriv'``
                Derivative ``1 - (alpha / 7)**2`` of the smooth bounded
                log-scale. Same shape as ``alpha``.
            ``'param_dim'``
                ``self.param_dim`` (``D``), for convenience in the backward
                pass.
        """
        D = self.param_dim
        M1, M2 = layer_const['M1'], layer_const['M2']
        W1y, W1c, b1 = weights[f'W1y_{k}'], weights[f'W1c_{k}'], weights[f'b1_{k}']
        W2, W2c, b2 = weights[f'W2_{k}'], weights[f'W2c_{k}'], weights[f'b2_{k}']

        W1y_masked = W1y * M1
        y_h = np.dot(y, W1y_masked.T)
        c_h = np.dot(ctx, W1c.T) if self.feature_dim > 0 else 0.0
        hidden_pre = y_h + c_h + b1
        h = np.tanh(hidden_pre)

        M2_tiled = layer_const['M2t'] if 'M2t' in layer_const else np.concatenate([M2, M2], axis=0)
        W2_masked = W2 * M2_tiled
        out = np.dot(h, W2_masked.T)
        c_out = None
        if self.feature_dim > 0:
            c_out = np.dot(ctx, W2c.T)
            out += c_out
        out += b2

        mu = out[:, :D]
        out_alpha_preclip = out[:, D:]
        alpha = MAF_ALPHA_LIMIT * np.tanh(out_alpha_preclip / MAF_ALPHA_LIMIT)
        alpha_deriv = 1.0 - (alpha / MAF_ALPHA_LIMIT) ** 2

        cache = {
            'p': y,
            'ctx': ctx,
            'M1': M1,
            'M2': M2,
            'M2t': M2_tiled,
            'W1y': W1y,
            'W1c': W1c,
            'b1': b1,
            'W2': W2,
            'W2c': W2c,
            'b2': b2,
            'W1y_masked': W1y_masked,
            'W2_masked': W2_masked,
            'y_h': y_h,
            'c_h': c_h,
            'hidden_pre': hidden_pre,
            'h': h,
            'c_out': c_out,
            'out': out,
            'out_alpha_preclip': out_alpha_preclip,
            'mu': mu,
            'alpha': alpha,
            'alpha_preclip': out_alpha_preclip,
            'alpha_deriv': alpha_deriv,
            'param_dim': D,
        }
        return mu, alpha, cache

    def _made_backward_np(self, weights, cache, k, bar_mu, bar_alpha,
                          compute_ctx_grad=True):
        """Pure NumPy backward for one MADE block.

        This is the explicit, hand-written adjoint of
        :meth:`_made_forward_np`.  It uses regular ``numpy`` only (never
        plain ``numpy``) and reconstructs the parameter gradients from the
        forward cache, so it can be combined with a future manual multi-flow
        loop.

        Parameters
        ----------
        weights : dict
            Full weight dictionary (same one used in the forward).
        cache : dict
            Cache returned by ``_made_forward_np`` for this block.
        k : int
            Layer index, used only to name the returned gradient keys.
        bar_mu : numpy.ndarray, shape (N, D)
            ``dL/dmu`` for this block.
        bar_alpha : numpy.ndarray, shape (N, D)
            ``dL/dalpha`` for this block, where ``alpha`` is the smooth
            bounded log-scale ``7 * tanh(out_alpha_preclip / 7)``.

        Returns
        -------
        grads : dict
            Keys ``'W1y_{k}'``, ``'W1c_{k}'``, ``'b1_{k}'``, ``'W2_{k}'``,
            ``'W2c_{k}'``, ``'b2_{k}'``, each with the same shape as the
            corresponding weight.
        bar_p : numpy.ndarray, shape (N, D)
            ``dL/dp`` through the MADE network only (``p = cache['p']``).
        bar_ctx : numpy.ndarray, shape (N, C)
            ``dL/dctx`` through both the hidden and output context paths.

        Notes
        -----
        The derivative of ``alpha = 7 * tanh(pre / 7)`` is
        ``cache['alpha_deriv'] = 1 - (alpha / 7)**2``, which stays positive
        even for large pre-activations.
        """
        D = cache['param_dim']
        ctx = cache['ctx']
        p = cache['p']
        h = cache['h']
        W1y_masked = cache['W1y_masked']
        W2_masked = cache['W2_masked']
        W1c = cache['W1c']
        W2c = cache['W2c']
        M1 = cache['M1']
        M2_tiled = cache['M2t'] if 'M2t' in cache else np.concatenate([cache['M2'], cache['M2']], axis=0)

        # Smooth bound derivative: alpha = 7*tanh(preclip/7).
        # Write the gated adjoint straight into bar_out; no concatenate or
        # separate masked-alpha temporary.
        bar_out = np.empty((bar_mu.shape[0], 2 * D), dtype=bar_mu.dtype)
        bar_out[:, :D] = bar_mu
        np.multiply(bar_alpha, cache['alpha_deriv'], out=bar_out[:, D:])

        # Output layer: out = h @ W2_masked.T + ctx @ W2c.T + b2.
        dW2 = np.dot(bar_out.T, h)
        dW2 *= M2_tiled
        dW2c = np.dot(bar_out.T, ctx)
        db2 = np.sum(bar_out, axis=0)

        # Hidden layer: h = tanh(p @ W1y_masked.T + ctx @ W1c.T + b1).
        bar_h = np.dot(bar_out, W2_masked)
        bar_a = np.empty_like(bar_h)
        np.multiply(h, h, out=bar_a)
        np.subtract(1.0, bar_a, out=bar_a)
        np.multiply(bar_h, bar_a, out=bar_a)
        db1 = np.sum(bar_a, axis=0)
        dW1y = np.dot(bar_a.T, p)
        dW1y *= M1
        dW1c = np.dot(bar_a.T, ctx)

        # Inputs of the block. Feature gradients are only needed for
        # diagnostics; training uses the weights and bar_p.
        bar_p = np.dot(bar_a, W1y_masked)
        bar_ctx = None
        if compute_ctx_grad:
            bar_ctx = np.dot(bar_a, W1c) + np.dot(bar_out, W2c)

        # ``W1c`` / ``W2c`` have shape (H, 0) / (2D, 0) when
        # ``feature_dim == 0``; the products above then already resolve to
        # correctly shaped zero arrays, so no branch is needed here.
        grads = {
            f'W1y_{k}': np.asarray(dW1y),
            f'W1c_{k}': np.asarray(dW1c),
            f'b1_{k}': np.asarray(db1),
            f'W2_{k}': np.asarray(dW2),
            f'W2c_{k}': np.asarray(dW2c),
            f'b2_{k}': np.asarray(db2),
        }
        return grads, np.asarray(bar_p), bar_ctx

    def _get_log_prob_np(self, weights, features, params):
        """Pure NumPy forward pass over the full MAF stack.

        This mirrors :meth:`_get_log_prob` exactly but uses regular ``numpy``
        only and caches every per-flow quantity needed by
        :meth:`_flow_backward_np`.

        Parameters
        ----------
        weights : dict
            Full weight dictionary of the estimator.
        features : numpy.ndarray, shape (N, C)
            Conditioning features.
        params : numpy.ndarray, shape (N, D)
            Observations whose log density is evaluated. The input array is
            never mutated.

        Returns
        -------
        log_prob : numpy.ndarray, shape (N,)
            ``base_logp + log_det`` for the complete flow stack.
        caches : list of dict
            One entry per flow (in forward order). Each entry contains:

            ``'block'``
                Cache returned by :meth:`_made_forward_np` for this flow.
            ``'perm'``, ``'inv_perm'``
                The layer permutation and its inverse.
            ``'u_in'``
                Flow input BEFORE the permutation, shape (N, D).
            ``'p'``
                Flow input AFTER the permutation (``u_in[:, perm]``).
            ``'exp_neg_alpha'``
                ``exp(-alpha)`` with the clipped ``alpha``.
            ``'u_out'``
                ``(p - mu) * exp_neg_alpha``, the flow output.
        """
        u = np.asarray(params)
        n_samples = u.shape[0]
        log_det = np.zeros(n_samples, dtype=u.dtype)
        caches = []

        for k, layer_const in enumerate(self.model_constants['layers']):
            perm = layer_const['perm']
            inv_perm = layer_const['inv_perm']

            u_in = u
            p = u_in[:, perm]
            mu, alpha, block = self._made_forward_np(
                weights, p, features, layer_const, k)

            exp_neg_alpha = np.empty_like(alpha)
            np.negative(alpha, out=exp_neg_alpha)
            np.exp(exp_neg_alpha, out=exp_neg_alpha)
            u_out = p - mu
            u_out *= exp_neg_alpha
            log_det -= np.sum(alpha, axis=1)

            caches.append({
                'block': block,
                'perm': perm,
                'inv_perm': inv_perm,
                'u_in': u_in,
                'p': p,
                'exp_neg_alpha': exp_neg_alpha,
                'u_out': u_out,
            })
            u = u_out

        base_logp = (-0.5 * np.sum(u ** 2, axis=1)
                     - 0.5 * self.param_dim * np.log(2.0 * np.pi))
        return base_logp + log_det, caches

    def _flow_backward_np(self, weights, caches, bar_log_prob):
        """Pure NumPy reverse pass over the cached MAF flows.

        Parameters
        ----------
        weights : dict
            Full weight dictionary (same one used in the forward).
        caches : list of dict
            Per-flow caches returned by :meth:`_get_log_prob_np`.
        bar_log_prob : numpy.ndarray, shape (N,)
            ``dL/dlog_prob`` for each sample. For the negative mean log
            probability loss this is ``-1/N`` per sample.

        Returns
        -------
        grads : dict
            Gradients for every flow, keyed exactly like ``weights``
            (``'W1y_{k}'``, ``'W1c_{k}'``, ``'b1_{k}'``, ``'W2_{k}'``,
            ``'W2c_{k}'``, ``'b2_{k}'``).

        Notes
        -----
        With ``u_K`` the output of the last flow the base density gives
        ``dbase_logp/du_K = -u_K``, hence ``bar_u = -bar_logp * u_K``. For
        each flow ``u_out = (p - mu) * exp(-alpha)`` so the adjoint entering
        the affine transform is ``g = bar_u * exp(-alpha)``. The log-det term
        is ``log_det -= sum_k sum_d alpha_{k,d}``, which contributes
        ``-bar_logp`` to each ``bar_alpha``. Finally ``p = u_in[:, perm]`` is
        inverted by scattering ``bar_p`` back with ``bar_u_in[:, perm]``.
        """
        # Keep the batch dtype: promoting to float64 here would also promote
        # every gradient returned for float32 weights.
        batch_dtype = np.asarray(caches[0]['u_in']).dtype if caches else np.float64
        bar_logp = np.asarray(bar_log_prob, dtype=batch_dtype)
        u_K = caches[-1]['u_out']
        bar_u = np.empty_like(u_K)
        np.multiply(bar_logp[:, None], u_K, out=bar_u)
        np.negative(bar_u, out=bar_u)
        bar_ld = bar_logp
        grads = {}

        for k in range(len(caches) - 1, -1, -1):
            c = caches[k]
            exp_neg_alpha = c['exp_neg_alpha']
            u_out = c['u_out']

            # Adjoint through the affine transform u_out = (p - mu) * e^-alpha.
            g = np.empty_like(bar_u)
            np.multiply(bar_u, exp_neg_alpha, out=g)
            bar_mu = np.empty_like(g)
            np.negative(g, out=bar_mu)
            bar_alpha = np.empty_like(bar_u)
            np.multiply(bar_u, u_out, out=bar_alpha)
            np.negative(bar_alpha, out=bar_alpha)
            bar_alpha -= bar_ld[:, None]

            grads_k, bar_p_net, _ = self._made_backward_np(
                weights, c['block'], k, bar_mu, bar_alpha,
                compute_ctx_grad=False)
            grads.update(grads_k)

            # Total adjoint wrt p, then scatter through p = u_in[:, perm].
            g += bar_p_net
            perm = c['perm']
            bar_u_next = np.empty_like(c['u_in'])
            bar_u_next[:, perm] = g
            bar_u = bar_u_next

        return grads

    def _loss_and_grad_np(self, weights, features, params):
        """Pure NumPy negative mean log probability and its weight gradient.

        Returns
        -------
        loss : float
            ``-mean(log_prob)``.
        grads : dict
            Gradient of ``loss`` wrt ``weights``, keyed exactly like
            ``weights``.
        """
        log_prob, caches = self._get_log_prob_np(weights, features, params)
        n_samples = log_prob.shape[0]
        loss = float(-np.mean(log_prob))
        grads = self._flow_backward_np(
            weights, caches, np.full(n_samples, -1.0 / n_samples))
        return loss, grads

    def _loss_and_grad(self, weights, features, params):
        """Loss and gradient for one batch."""
        return self._loss_and_grad_np(weights, features, params)

    def _get_log_prob(self, weights: dict, features: np.ndarray, params: np.ndarray):
        """Computes log probability for the MAF."""
        return self._get_log_prob_np(weights, features, params)[0]

    def _loss_function(self, weights: dict, features: np.ndarray, params: np.ndarray) -> float:
        return float(-np.mean(self._get_log_prob_np(weights, features, params)[0]))

    def log_prob(self, features: np.ndarray, params: np.ndarray) -> np.ndarray:
        super().log_prob(features, params)
        return self._get_log_prob(self.weights, features, params)

    def sample(self, features: np.ndarray, n_samples: int, rng: np.random.RandomState) -> np.ndarray:
        super().sample(features, n_samples, rng)
        features = features.astype('f')
        if features.ndim == 1:
            features = features.reshape(1, -1)

        n_cond = features.shape[0]
        features = np.repeat(features, n_samples, axis=0)

        z = rng.randn(n_cond * n_samples, self.param_dim).astype('f')
        x = z

        # Invert the flow stack
        for k, layer_const in reversed(list(enumerate(self.model_constants['layers']))):
            y_perm = x
            u = np.zeros_like(y_perm)
            for i in range(self.param_dim):
                mu, alpha = self._made_forward(u, features, layer_const, k, self.weights)
                u[:, i] = y_perm[:, i] * np.exp(alpha[:, i]) + mu[:, i]
            x = u[:, layer_const['inv_perm']]

        return x.reshape(n_cond, n_samples, self.param_dim)


# =============================================================================
# == Test Datasets and Visualization
# =============================================================================

def generate_test_data(dataset_name: str, n_samples: int, seed: int = 42):
    """
    Generates complex, conditional 2D datasets for testing.

    The first feature dimension is the conditioning variable.
    The two parameter dimensions are the target variables.

    Parameters
    ----------
    dataset_name : {'banana', 'student_t', 'moons'}
        The name of the dataset to generate.
    n_samples : int
        The number of samples to generate.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        A tuple containing (parameters, features).
    """
    rng = np.random.RandomState(seed)
    params = np.zeros((n_samples, 2), dtype='f')
    features = np.zeros((n_samples, 1), dtype='f')

    if dataset_name == 'banana':
        # Feature controls the curvature of the banana shape
        features[:, 0] = rng.uniform(0.5, 2.0, size=n_samples)
        x = rng.randn(n_samples, 2).astype('f')
        params[:, 0] = x[:, 0]
        params[:, 1] = x[:, 1] - features[:, 0] * (x[:, 0]**2 - 2.0)

    elif dataset_name == 'student_t':
        # Feature controls the degrees of freedom (tail heaviness)
        features[:, 0] = rng.uniform(1.0, 10.0, size=n_samples)
        for i in range(n_samples):
            df = features[i, 0]
            params[i, :] = t.rvs(df, size=2, random_state=rng)

    elif dataset_name == 'moons':
        from sklearn.datasets import make_moons
        # Feature controls the noise level
        features[:, 0] = rng.uniform(0.05, 0.2, size=n_samples)
        for i in range(n_samples):
            p, _ = make_moons(n_samples=2, noise=features[i, 0], random_state=rng)
            params[i, :] = p[0]

    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    return params, features


def run_test(estimator: ConditionalDensityEstimator, dataset_name: str, plot: bool = True):
    """
    Runs a standardized test for a given estimator and dataset.

    Parameters
    ----------
    estimator : ConditionalDensityEstimator
        An instance of the estimator to test.
    dataset_name : str
        The name of the dataset to use for the test.
    plot : bool, optional
        If True, generates and displays a plot of the results.
    """
    print(f"\n--- Testing {estimator.__class__.__name__} on '{dataset_name}' dataset ---")

    # 1. Generate data and train
    params, features = generate_test_data(dataset_name, n_samples=5000)
    estimator.train(params, features, n_iter=400, learning_rate=1e-3)

    if not plot:
        return

    # 2. Setup for plotting
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    fig = plt.figure(figsize=(12, 6))
    gs = gridspec.GridSpec(1, 2, figure=fig)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])

    # Define conditions to test (low and high ends of the feature range)
    low_cond = np.array([[features.min()]])
    high_cond = np.array([[features.max()]])

    # 3. Generate samples from the trained model
    rng = np.random.RandomState(0)
    samples_low = estimator.sample(low_cond, 1000, rng)[0]
    samples_high = estimator.sample(high_cond, 1000, rng)[0]

    # 4. Plot results
    # Create a grid to evaluate the learned density
    x_range = (params[:, 0].min() - 1, params[:, 0].max() + 1)
    y_range = (params[:, 1].min() - 1, params[:, 1].max() + 1)
    grid_x, grid_y = np.meshgrid(np.linspace(*x_range, 100), np.linspace(*y_range, 100))
    grid_params = np.vstack([grid_x.ravel(), grid_y.ravel()]).T

    # Plot for the 'low' condition
    grid_feat_low = np.repeat(low_cond, grid_params.shape[0], axis=0)
    log_p_low = estimator.log_prob(grid_feat_low, grid_params).reshape(100, 100)
    ax1.contourf(grid_x, grid_y, np.exp(log_p_low), levels=10, cmap='Blues', alpha=0.8)
    ax1.plot(samples_low[:, 0], samples_low[:, 1], ',', c='navy', alpha=0.5, label='Generated Samples')
    ax1.set_title(f"Condition: feature = {low_cond[0,0]:.2f}")
    ax1.set(xlim=x_range, ylim=y_range, xlabel="Param 1", ylabel="Param 2")
    ax1.legend()

    # Plot for the 'high' condition
    grid_feat_high = np.repeat(high_cond, grid_params.shape[0], axis=0)
    log_p_high = estimator.log_prob(grid_feat_high, grid_params).reshape(100, 100)
    ax2.contourf(grid_x, grid_y, np.exp(log_p_high), levels=10, cmap='Oranges', alpha=0.8)
    ax2.plot(samples_high[:, 0], samples_high[:, 1], ',', c='darkred', alpha=0.5, label='Generated Samples')
    ax2.set_title(f"Condition: feature = {high_cond[0,0]:.2f}")
    ax2.set(xlim=x_range, ylim=y_range, xlabel="Param 1", yticklabels=[])
    ax2.legend()

    fig.suptitle(f"Density Estimation Results for {estimator.__class__.__name__} on '{dataset_name}'", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()




def shrinkage_zscore(x, x_hat, x_prior):
    """
    Computes the posterior shrinkage and z-score for 
    a parameter x.

    Parameters
    ----------
    x : array
        The true parameter.
    x_hat : array
        The posterior samples of the parameter.
    x_prior : array
        The prior standard deviation of the parameter.

    Returns
    -------
    shrinkage : array
        The posterior shrinkage.
    zscore : array
        The posterior z-score.

    """
    # compute posterior mean
    x_mean = x_hat.mean(0)

    # compute posterior standard deviation
    x_std = x_hat.std(0)

    # compute posterior shrinkage
    shrinkage = 1 - x_std / x_prior

    # compute posterior z-score
    zscore = (x_mean - x) / x_std

    return shrinkage, zscore


if __name__ == '__main__':
    # --- MDN Tests ---
    mdn_banana = MDNEstimator(param_dim=2, feature_dim=1, n_components=8, hidden_sizes=(64, 64))
    run_test(mdn_banana, 'banana')

    mdn_moons = MDNEstimator(param_dim=2, feature_dim=1, n_components=10, hidden_sizes=(64, 64))
    run_test(mdn_moons, 'moons')

    # --- MAF Tests ---
    maf_banana = MAFEstimator(param_dim=2, feature_dim=1, n_flows=5, hidden_units=128)
    run_test(maf_banana, 'banana')

    maf_moons = MAFEstimator(param_dim=2, feature_dim=1, n_flows=5, hidden_units=128)
    run_test(maf_moons, 'moons')
