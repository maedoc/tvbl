"""
A module for conditional density estimation using Mixture Density
Networks (MDNs), Masked Autoregressive Flows (MAFs), and Neural Spline Flows (NSFs).

"""

import abc
import math
from dataclasses import dataclass, field

import autograd.numpy as anp
from autograd import grad
from autograd.scipy.special import logsumexp
from scipy.stats import t
from tqdm.auto import trange

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
    def _initialize_weights(self, rng: anp.random.RandomState) -> dict:
        """
        Initialize the trainable weights of the model.

        Parameters
        ----------
        rng : autograd.numpy.random.RandomState
            A random number generator for reproducible initialization.

        Returns
        -------
        dict
            A dictionary of initialized weight arrays.
        """
        pass

    @abc.abstractmethod
    def _loss_function(self, weights: dict, features: anp.ndarray, params: anp.ndarray) -> float:
        """
        Compute the negative log-likelihood loss for a batch of data.

        Parameters
        ----------
        weights : dict
            A dictionary of the model's trainable weights.
        features : anp.ndarray
            A (N, feature_dim) array of conditional features.
        params : anp.ndarray
            A (N, param_dim) array of target parameters.

        Returns
        -------
        float
            The mean negative log-likelihood of the batch.
        """
        pass

    @abc.abstractmethod
    def sample(self, features: anp.ndarray, n_samples: int, rng: anp.random.RandomState) -> anp.ndarray:
        """
        Generate samples from the learned conditional distribution p(params|features).

        Parameters
        ----------
        features : anp.ndarray
            A (n_conditions, feature_dim) array of features to condition on.
        n_samples : int
            The number of samples to generate for each condition.
        rng : autograd.numpy.random.RandomState
            A random number generator for sampling.

        Returns
        -------
        anp.ndarray
            An array of generated samples of shape (n_conditions, n_samples, param_dim).
        """
        if self.weights is None:
            raise RuntimeError("Model has not been trained yet. Call train() first.")

    @abc.abstractmethod
    def log_prob(self, features: anp.ndarray, params: anp.ndarray) -> anp.ndarray:
        """
        Compute the log-probability log p(params|features).

        Parameters
        ----------
        features : anp.ndarray
            A (N, feature_dim) array of conditional features.
        params : anp.ndarray
            A (N, param_dim) array of target parameters.

        Returns
        -------
        anp.ndarray
            A (N,) array of log-probabilities.
        """
        if self.weights is None:
            raise RuntimeError("Model has not been trained yet. Call train() first.")

    def train(self, params: anp.ndarray, features: anp.ndarray,
              n_iter: int = 2000, learning_rate: float = 1e-3,
              seed: int = 0, use_tqdm: bool = True):
        """
        Trains the model using the Adam optimizer.

        Parameters
        ----------
        params : anp.ndarray
            An (N, param_dim) matrix of simulated parameters.
        features : anp.ndarray
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
        finite_idx = anp.all(anp.isfinite(params), axis=1) & anp.all(anp.isfinite(features), axis=1)
        params = params[finite_idx].astype(anp.float64)
        features = features[finite_idx].astype(anp.float64)

        if params.shape[0] == 0:
            raise ValueError("All data points contained non-finite values.")

        # --- 2. Initialization ---
        rng = anp.random.RandomState(seed)
        self.weights = self._initialize_weights(rng)
        self.loss_history = []

        # Adam optimizer state
        m = {key: anp.zeros_like(val) for key, val in self.weights.items()}
        v = {key: anp.zeros_like(val) for key, val in self.weights.items()}
        beta1, beta2, epsilon = 0.9, 0.999, 1e-8

        # --- 3. Optimization Loop ---
        gradient_func = grad(self._loss_function)

        iterator = trange(n_iter, desc="Training", disable=not use_tqdm)
        for i in iterator:
            g = gradient_func(self.weights, features, params)
            loss = self._loss_function(self.weights, features, params)
            self.loss_history.append(loss)

            if not anp.isfinite(loss):
                print(f"Warning: Loss is non-finite at iteration {i}. Stopping training.")
                break

            if use_tqdm:
                iterator.set_postfix(loss=f"{loss:.4f}")

            # Adam update step
            for key in self.weights:
                if not anp.all(anp.isfinite(g[key])):
                    print(f"Warning: Non-finite gradient for '{key}' at iteration {i}. Stopping.")
                    return
                m[key] = beta1 * m[key] + (1 - beta1) * g[key]
                v[key] = beta2 * v[key] + (1 - beta2) * (g[key]**2)
                m_hat = m[key] / (1 - beta1**(i + 1))
                v_hat = v[key] / (1 - beta2**(i + 1))
                self.weights[key] -= learning_rate * m_hat / (anp.sqrt(v_hat) + epsilon)


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

    def _create_offdiag_basis(self):
        n_off_diag = self.param_dim * (self.param_dim - 1) // 2
        if n_off_diag == 0:
            return None
        basis = anp.zeros((n_off_diag, self.param_dim, self.param_dim), dtype='f')
        rows, cols = anp.triu_indices(self.param_dim, k=1)
        basis[anp.arange(n_off_diag), rows, cols] = 1
        return basis

    def _initialize_weights(self, rng: anp.random.RandomState) -> dict:
        """Initializes weights for the MLP and GMM output layers."""
        weights = {}
        in_size = self.feature_dim
        for i, out_size in enumerate(self.hidden_sizes):
            weights[f'W{i}'] = (rng.randn(in_size, out_size) * anp.sqrt(2.0 / in_size)).astype('f')
            weights[f'b{i}'] = anp.zeros(out_size, dtype='f')
            in_size = out_size

        last_hidden_size = self.hidden_sizes[-1] if self.hidden_sizes else self.feature_dim

        # GMM output layers
        K, D_out = self.n_components, self.param_dim
        weights['W_alpha'] = (rng.randn(last_hidden_size, K) * 0.01).astype('f')
        weights['b_alpha'] = anp.zeros(K, dtype='f')
        weights['W_mu'] = (rng.randn(last_hidden_size, K * D_out) * 0.01).astype('f')
        weights['b_mu'] = anp.zeros(K * D_out, dtype='f')
        weights['W_L_prec_log_diag'] = (rng.randn(last_hidden_size, K * D_out) * 0.01).astype('f')
        weights['b_L_prec_log_diag'] = anp.zeros(K * D_out, dtype='f')

        n_off_diag = D_out * (D_out - 1) // 2
        if n_off_diag > 0:
            weights['W_L_prec_offdiag'] = (rng.randn(last_hidden_size, K * n_off_diag) * 0.01).astype('f')
            weights['b_L_prec_offdiag'] = anp.zeros(K * n_off_diag, dtype='f')

        return weights

    def _forward_pass(self, weights: dict, features: anp.ndarray):
        """Maps input features to GMM parameters."""
        h = features
        for i in range(len(self.hidden_sizes)):
            h = anp.tanh(h @ weights[f'W{i}'] + weights[f'b{i}'])

        K, D_out = self.n_components, self.param_dim
        log_alpha = h @ weights['W_alpha'] + weights['b_alpha']
        alpha = anp.exp(log_alpha - logsumexp(log_alpha, axis=1, keepdims=True))
        mu = (h @ weights['W_mu'] + weights['b_mu']).reshape(-1, K, D_out)

        L_prec_log_diag = (h @ weights['W_L_prec_log_diag'] + weights['b_L_prec_log_diag']).reshape(-1, K, D_out)
        L_prec_diag_mat = anp.einsum('nki,ij->nkij', anp.exp(L_prec_log_diag), anp.eye(D_out, dtype='f'))

        n_off_diag = D_out * (D_out - 1) // 2
        if n_off_diag > 0:
            L_prec_offdiag_vals = (h @ weights['W_L_prec_offdiag'] + weights['b_L_prec_offdiag']).reshape(-1, K, n_off_diag)
            L_prec_offdiag_mat = anp.einsum('nkl,lij->nkij', L_prec_offdiag_vals, self._offdiag_basis)
            L_prec = L_prec_diag_mat + L_prec_offdiag_mat
        else:
            L_prec = L_prec_diag_mat

        return alpha, mu, L_prec, L_prec_log_diag

    def _loss_function(self, weights: dict, features: anp.ndarray, params: anp.ndarray) -> float:
        """Computes the negative log-likelihood of the true parameters under the GMM."""
        alpha, mu, L_prec, L_prec_log_diag = self._forward_pass(weights, features)

        y_true_reshaped = params[:, anp.newaxis, :]
        delta = y_true_reshaped - mu

        z = anp.einsum('nkij,nkj->nki', L_prec, delta)
        quad_term = -0.5 * anp.sum(z**2, axis=2)
        log_det_term = anp.sum(L_prec_log_diag, axis=2)

        log_probs_k = quad_term + log_det_term - 0.5 * self.param_dim * anp.log(2 * math.pi)
        total_log_prob = logsumexp(anp.log(alpha + 1e-9) + log_probs_k, axis=1)

        return -anp.mean(total_log_prob)

    def log_prob(self, features: anp.ndarray, params: anp.ndarray) -> anp.ndarray:
        """
        Computes the log-probability log p(params|features) for each sample.
        """
        super().log_prob(features, params)
    
        # Perform a forward pass to get GMM parameters
        alpha, mu, L_prec, L_prec_log_diag = self._forward_pass(self.weights, features)
    
        # Reshape parameters for broadcasting against mixture components
        y_true_reshaped = params[:, anp.newaxis, :]
        delta = y_true_reshaped - mu
    
        # Compute the log-probability for each component (k) for each sample (n)
        z = anp.einsum('nkij,nkj->nki', L_prec, delta)
        quad_term = -0.5 * anp.sum(z**2, axis=2)
        log_det_term = anp.sum(L_prec_log_diag, axis=2)
    
        log_probs_k = quad_term + log_det_term - 0.5 * self.param_dim * anp.log(2 * math.pi)
    
        # Combine component log-probabilities using the mixture weights (alpha)
        # This returns a vector of shape (N,)
        total_log_prob = logsumexp(anp.log(alpha + 1e-9) + log_probs_k, axis=1)
    
        return total_log_prob


    def sample(self, features: anp.ndarray, n_samples: int, rng: anp.random.RandomState) -> anp.ndarray:
        super().sample(features, n_samples, rng)
        features = features.astype('f')
        if features.ndim == 1:
            features = features.reshape(1, -1)

        alpha, mu, L_prec, _ = self._forward_pass(self.weights, features)
        n_cond, K, D_out = mu.shape

        log_alpha = anp.log(alpha + 1e-9)
        gumbel_noise = -anp.log(-anp.log(rng.uniform(size=(n_cond, n_samples, K))))
        component_indices = anp.argmax(log_alpha[:, anp.newaxis, :] + gumbel_noise, axis=2)

        cond_idx = anp.arange(n_cond)[:, anp.newaxis]
        chosen_mu = mu[cond_idx, component_indices]
        chosen_L_prec = L_prec[cond_idx, component_indices]

        try:
            L_cov_factor = anp.linalg.inv(chosen_L_prec)
        except anp.linalg.LinAlgError:
            print("Warning: Singular precision matrix encountered during sampling. Returning NaNs.")
            return anp.full((n_cond, n_samples, D_out), anp.nan)

        z = rng.randn(n_cond, n_samples, D_out)
        samples = chosen_mu + anp.einsum('nsij,nsj->nsi', L_cov_factor, z)

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

    def _initialize_weights(self, rng: anp.random.RandomState) -> dict:
        """Initializes weights and model constants (masks, permutations)."""
        weights = {}
        layers = []
        D, C, H = self.param_dim, self.feature_dim, self.hidden_units

        for k in range(self.n_flows):
            # MADE masks and permutation
            m_in = anp.arange(1, D + 1)
            m_hidden = rng.randint(1, D, size=H)
            M1 = (m_in[None, :] <= m_hidden[:, None]).astype('f')
            m_out = m_in.copy()
            M2 = (m_hidden[None, :] < m_out[:, None]).astype('f')
            perm = rng.permutation(D)
            inv_perm = anp.empty(D, dtype=int); inv_perm[perm] = anp.arange(D)

            layers.append({'M1': M1, 'M2': M2, 'perm': perm, 'inv_perm': inv_perm})

            # Trainable parameters
            w_std = 0.01
            weights[f'W1y_{k}'] = (rng.randn(H, D) * w_std).astype('f')
            weights[f'W1c_{k}'] = (rng.randn(H, C) * w_std).astype('f') if C > 0 else anp.zeros((H, C), dtype='f')
            weights[f'b1_{k}'] = anp.zeros(H, dtype='f')
            weights[f'W2_{k}'] = anp.zeros((2 * D, H), dtype='f')
            weights[f'W2c_{k}'] = anp.zeros((2 * D, C), dtype='f') if C > 0 else anp.zeros((2*D, C), dtype='f')
            weights[f'b2_{k}'] = anp.zeros(2 * D, dtype='f')

        self.model_constants = {'layers': layers}
        return weights

    def _made_forward(self, y, ctx, layer_const, k, weights):
        """Single forward pass through a MADE block."""
        M1, M2 = layer_const['M1'], layer_const['M2']
        W1y, W1c, b1 = weights[f'W1y_{k}'], weights[f'W1c_{k}'], weights[f'b1_{k}']
        W2, W2c, b2 = weights[f'W2_{k}'], weights[f'W2c_{k}'], weights[f'b2_{k}']

        y_h = anp.dot(y, (W1y * M1).T)
        c_h = anp.dot(ctx, W1c.T) if self.feature_dim > 0 else 0.0
        h = anp.tanh(y_h + c_h + b1)

        M2_tiled = anp.concatenate([M2, M2], axis=0)
        out = anp.dot(h, (W2 * M2_tiled).T)
        if self.feature_dim > 0:
            out = out + anp.dot(ctx, W2c.T)
        out = out + b2

        mu, alpha = out[:, :self.param_dim], anp.clip(out[:, self.param_dim:], -7.0, 7.0)
        return mu, alpha

    def _get_log_prob(self, weights: dict, features: anp.ndarray, params: anp.ndarray):
        """Computes log probability for the MAF."""
        u = params
        log_det = anp.zeros(params.shape[0])

        for k, layer_const in enumerate(self.model_constants['layers']):
            u = u[:, layer_const['perm']]
            mu, alpha = self._made_forward(u, features, layer_const, k, weights)
            u = (u - mu) * anp.exp(-alpha)
            log_det -= anp.sum(alpha, axis=1)

        base_logp = -0.5 * anp.sum(u**2, axis=1) - 0.5 * self.param_dim * anp.log(2.0 * anp.pi)
        return base_logp + log_det

    def _loss_function(self, weights: dict, features: anp.ndarray, params: anp.ndarray) -> float:
        return -anp.mean(self._get_log_prob(weights, features, params))

    def log_prob(self, features: anp.ndarray, params: anp.ndarray) -> anp.ndarray:
        super().log_prob(features, params)
        return self._get_log_prob(self.weights, features, params)

    def sample(self, features: anp.ndarray, n_samples: int, rng: anp.random.RandomState) -> anp.ndarray:
        super().sample(features, n_samples, rng)
        features = features.astype('f')
        if features.ndim == 1:
            features = features.reshape(1, -1)

        n_cond = features.shape[0]
        # Broadcast features to match number of samples
        if n_cond != n_samples:
            features = anp.repeat(features, n_samples, axis=0)

        z = rng.randn(n_samples, self.param_dim).astype('f')
        x = z

        # Invert the flow stack
        for k, layer_const in reversed(list(enumerate(self.model_constants['layers']))):
            y_perm = x
            u = anp.zeros_like(y_perm)
            for i in range(self.param_dim):
                mu, alpha = self._made_forward(u, features, layer_const, k, self.weights)
                u[:, i] = y_perm[:, i] * anp.exp(alpha[:, i]) + mu[:, i]
            x = u[:, layer_const['inv_perm']]

        # Reshape to (n_conditions, n_samples, param_dim)
        return x.reshape(features.shape[0] // n_samples, n_samples, self.param_dim)


# =============================================================================
# == NSF Implementation (Neural Spline Flow)
# =============================================================================

DEFAULT_MIN_BIN_WIDTH = 1e-3
DEFAULT_MIN_BIN_HEIGHT = 1e-3
DEFAULT_MIN_DERIVATIVE = 1e-3

def _searchsorted_autograd(bin_locations, inputs, eps=1e-6):
    """
    Computes searchsorted (right) on the last dimension of bin_locations.
    bin_locations: (..., K+1)
    inputs: (..., ) or (..., 1)
    Returns: indices (..., ) in range [0, K-1]
    """
    # Add eps to the last bin boundary to ensure inputs exactly at the boundary are included
    # We create a new array to avoid in-place modification
    # bin_locations shape: (N, D, K+1)
    last = bin_locations[..., -1:] + eps
    locs = anp.concatenate([bin_locations[..., :-1], last], axis=-1)
    
    # inputs: (N, D). Broadcasting needs inputs to be (N, D, 1)
    # Cast to int to ensure Autograd treats it as an index/constant w.r.t differentiation
    return (anp.sum(inputs[..., anp.newaxis] >= locs, axis=-1) - 1).astype(int)

def _gather_elementwise(params, indices):
    """
    Gather values from params at indices.
    params: (N, D, K)
    indices: (N, D)
    Returns: (N, D)
    """
    N, D = indices.shape
    # Advanced indexing
    # We want result[i, j] = params[i, j, indices[i, j]]
    # Flatten everything
    params_flat = params.reshape(-1, params.shape[-1]) # (N*D, K)
    indices_flat = indices.flatten().astype(int) # (N*D,)
    
    # Use integer array indexing on flattened array
    gathered = params_flat[anp.arange(len(indices_flat)), indices_flat]
    return gathered.reshape(N, D)

def _rational_quadratic_spline(inputs, unnormalized_widths, unnormalized_heights,
                               unnormalized_derivatives, inverse=False,
                               left=-2.5, right=2.5, bottom=-2.5, top=2.5,
                               min_bin_width=DEFAULT_MIN_BIN_WIDTH,
                               min_bin_height=DEFAULT_MIN_BIN_HEIGHT,
                               min_derivative=DEFAULT_MIN_DERIVATIVE):
    
    num_bins = unnormalized_widths.shape[-1]
    
    # --- 1. Define widths and heights (softmax) ---
    widths = anp.exp(unnormalized_widths - logsumexp(unnormalized_widths, axis=-1, keepdims=True))
    widths = min_bin_width + (1 - min_bin_width * num_bins) * widths
    
    cumwidths = anp.cumsum(widths, axis=-1)
    # Pad with 0 at start. cumwidths shape becomes (..., K+1)
    pad_shape = list(cumwidths.shape); pad_shape[-1] = 1
    cumwidths = anp.concatenate([anp.zeros(pad_shape), cumwidths], axis=-1)
    
    # Scale to interval
    cumwidths = (right - left) * cumwidths + left
    
    # Hard-enforce boundaries (for stability and correctness) using concat instead of assignment
    cumwidths_inner = cumwidths[..., 1:-1]
    left_edge = anp.full(pad_shape, left)
    right_edge = anp.full(pad_shape, right)
    cumwidths = anp.concatenate([left_edge, cumwidths_inner, right_edge], axis=-1)
    
    widths = cumwidths[..., 1:] - cumwidths[..., :-1]
    
    # --- 2. Define derivatives (softplus) ---
    derivatives = min_derivative + anp.logaddexp(0., unnormalized_derivatives)
    
    # --- 3. Define heights ---
    heights = anp.exp(unnormalized_heights - logsumexp(unnormalized_heights, axis=-1, keepdims=True))
    heights = min_bin_height + (1 - min_bin_height * num_bins) * heights
    
    cumheights = anp.cumsum(heights, axis=-1)
    cumheights = anp.concatenate([anp.zeros(pad_shape), cumheights], axis=-1)
    cumheights = (top - bottom) * cumheights + bottom
    
    cumheights_inner = cumheights[..., 1:-1]
    bottom_edge = anp.full(pad_shape, bottom)
    top_edge = anp.full(pad_shape, top)
    cumheights = anp.concatenate([bottom_edge, cumheights_inner, top_edge], axis=-1)
    
    heights = cumheights[..., 1:] - cumheights[..., :-1]
    
    # --- 4. Spline Calculation ---
    if inverse:
        bin_idx = _searchsorted_autograd(cumheights, inputs)
    else:
        bin_idx = _searchsorted_autograd(cumwidths, inputs)
        
    # Clamp bin_idx to valid range [0, K-1] just in case of float errors
    bin_idx = anp.clip(bin_idx, 0, num_bins - 1)

    input_cumwidths = _gather_elementwise(cumwidths, bin_idx)
    input_bin_widths = _gather_elementwise(widths, bin_idx)
    input_cumheights = _gather_elementwise(cumheights, bin_idx)
    input_bin_heights = _gather_elementwise(heights, bin_idx) # Used as heights later
    
    delta = input_bin_heights / input_bin_widths
    
    input_derivatives = _gather_elementwise(derivatives, bin_idx)
    input_derivatives_plus_one = _gather_elementwise(derivatives, bin_idx + 1)
    
    if inverse:
        a = (((inputs - input_cumheights) * (input_derivatives + input_derivatives_plus_one - 2 * delta)
              + input_bin_heights * (delta - input_derivatives)))
        b = (input_bin_heights * input_derivatives
             - (inputs - input_cumheights) * (input_derivatives + input_derivatives_plus_one - 2 * delta))
        c = - delta * (inputs - input_cumheights)

        discriminant = b**2 - 4 * a * c
        # discriminant = anp.maximum(discriminant, 0) # Ensure non-negative? autograd safe?

        root = (2 * c) / (-b - anp.sqrt(discriminant))
        outputs = root * input_bin_widths + input_cumwidths

        theta = root
        theta_one_minus_theta = theta * (1 - theta)
        
        denominator = delta + ((input_derivatives + input_derivatives_plus_one - 2 * delta) * theta_one_minus_theta)
        derivative_numerator = delta**2 * (input_derivatives_plus_one * theta**2
                                            + 2 * delta * theta_one_minus_theta
                                            + input_derivatives * (1 - theta)**2)
        logabsdet = anp.log(derivative_numerator) - 2 * anp.log(denominator)
        return outputs, -logabsdet

    else:
        theta = (inputs - input_cumwidths) / input_bin_widths
        theta_one_minus_theta = theta * (1 - theta)

        numerator = input_bin_heights * (delta * theta**2 + input_derivatives * theta_one_minus_theta)
        denominator = delta + ((input_derivatives + input_derivatives_plus_one - 2 * delta) * theta_one_minus_theta)
        
        outputs = input_cumheights + numerator / denominator

        derivative_numerator = delta**2 * (input_derivatives_plus_one * theta**2
                                            + 2 * delta * theta_one_minus_theta
                                            + input_derivatives * (1 - theta)**2)
        logabsdet = anp.log(derivative_numerator) - 2 * anp.log(denominator)
        
        return outputs, logabsdet


@dataclass
class NSFEstimator(ConditionalDensityEstimator):
    """
    Neural Spline Flow for conditional density estimation.
    Uses Rational Quadratic Splines with coupling layers.

    Parameters
    ----------
    param_dim : int
        Dimensionality of the target variable.
    feature_dim : int
        Dimensionality of the conditional variable.
    n_flow_steps : int
        Number of coupling layers.
    n_bins : int
        Number of bins for the spline.
    hidden_features : int
        Number of hidden units in the transform net (MLP).
    tail_bound : float
        The bound of the spline interval (i.e. [-B, B]).
    """
    param_dim: int
    feature_dim: int
    n_flow_steps: int = 4
    n_bins: int = 8
    hidden_features: int = 64
    tail_bound: float = 8.0

    def __post_init__(self):
        super().__init__(self.param_dim, self.feature_dim)
        self.model_constants = None

    def _initialize_weights(self, rng: anp.random.RandomState) -> dict:
        weights = {}
        layers = []
        
        # Dimensions
        D = self.param_dim
        C = self.feature_dim
        
        # Output dim of MLP: for each transformed dim, we need (3 * K - 1) params if linear tails.
        # We assume linear tails outside [-B, B].
        # For K bins: K widths, K heights, K-1 derivatives (since ends are 1).
        # Total: 3*K - 1.
        param_per_dim = 3 * self.n_bins - 1
        
        for i in range(self.n_flow_steps):
            # Alternating mask
            # If D=1, we cannot really split. But typically flows are D>=2.
            # If D=1, NSF usually implies unconditional on x, conditional on c.
            # For CDE, if D=1, we can't do coupling on x itself unless we treat it as 1D flow cond on C.
            # In that case, identity set is empty?
            # If D > 1:
            mask = anp.zeros(D)
            if D > 1:
                mask[::2] = 1 if i % 2 == 0 else 0 # 1 means transformed, 0 means identity
                if i % 2 != 0:
                    mask = 1 - mask
            else:
                # If D=1, everything is transformed. Identity is empty.
                mask[:] = 1

            n_identity = int(anp.sum(1 - mask))
            n_transform = int(anp.sum(mask))
            
            # MLP inputs: identity part of x + features
            in_dim = n_identity + C
            out_dim = n_transform * param_per_dim
            
            # Weights for MLP (2 hidden layers)
            H = self.hidden_features
            w_std = 0.01
            
            # W1: in -> H
            weights[f'step{i}_W1'] = (rng.randn(in_dim, H) * anp.sqrt(2/in_dim)).astype('f')
            weights[f'step{i}_b1'] = anp.zeros(H, dtype='f')
            
            # W2: H -> H
            weights[f'step{i}_W2'] = (rng.randn(H, H) * anp.sqrt(2/H)).astype('f')
            weights[f'step{i}_b2'] = anp.zeros(H, dtype='f')
            
            # W3: H -> out
            weights[f'step{i}_W3'] = (rng.randn(H, out_dim) * 0.01).astype('f') # Initialize slightly larger to break symmetry
            weights[f'step{i}_b3'] = anp.zeros(out_dim, dtype='f')
            
            # Constants to reconstruction
            # We need to know which indices are identity and which are transform
            id_indices = anp.where(mask == 0)[0]
            tr_indices = anp.where(mask == 1)[0]
            
            layers.append({
                'mask': mask,
                'id_indices': id_indices,
                'tr_indices': tr_indices,
                'n_transform': n_transform
            })
            
        self.model_constants = {'layers': layers}
        return weights

    def _mlp_forward(self, weights, step_idx, inputs):
        """Standard MLP forward pass."""
        W1 = weights[f'step{step_idx}_W1']
        b1 = weights[f'step{step_idx}_b1']
        W2 = weights[f'step{step_idx}_W2']
        b2 = weights[f'step{step_idx}_b2']
        W3 = weights[f'step{step_idx}_W3']
        b3 = weights[f'step{step_idx}_b3']
        
        h = anp.tanh(anp.dot(inputs, W1) + b1)
        h = anp.tanh(anp.dot(h, W2) + b2)
        out = anp.dot(h, W3) + b3
        return out

    def _transform_step(self, x, features, layer_const, step_idx, weights, inverse=False):
        # x: (N, D)
        # Split
        id_idx = layer_const['id_indices']
        tr_idx = layer_const['tr_indices']
        
        x_id = x[:, id_idx]
        x_tr = x[:, tr_idx]
        
        # MLP input
        if features is not None and self.feature_dim > 0:
            if x_id.shape[1] > 0:
                mlp_in = anp.concatenate([x_id, features], axis=1)
            else:
                mlp_in = features
        else:
            mlp_in = x_id
            
        params = self._mlp_forward(weights, step_idx, mlp_in)
        
        # Reshape params: (N, n_transform, 3*K - 1)
        N = x.shape[0]
        K = self.n_bins
        n_tr = layer_const['n_transform']
        params = params.reshape(N, n_tr, -1)
        
        # Split params
        # Widths: K, Heights: K, Derivatives: K-1
        unnorm_widths = params[..., :K]
        unnorm_heights = params[..., K:2*K]
        unnorm_derivatives = params[..., 2*K:]
        
        # Pad derivatives (linear tails -> deriv=1 at ends => unnorm=constant s.t. softplus(c)=1 => c = log(e-1))
        # But we pass unnormalized.
        # Constant for derivative=1: min_deriv + softplus(x) = 1. 
        # For default min=1e-3, 1e-3 + log(1+exp(x)) = 1 => log(1+exp(x)) = 0.999 => 1+exp(x) = exp(0.999) => x = log(exp(0.999)-1)
        # We can just pad with a value that results in 1.
        # Let's compute that constant.
        min_deriv = DEFAULT_MIN_DERIVATIVE
        c_val = anp.log(anp.exp(1 - min_deriv) - 1)
        
        pad_shape = list(unnorm_derivatives.shape); pad_shape[-1] = 1
        c_tensor = anp.full(pad_shape, c_val)
        unnorm_derivatives = anp.concatenate([c_tensor, unnorm_derivatives, c_tensor], axis=-1)
        
        # Rational Quadratic Spline
        # Identify inputs inside the bound
        # We apply spline only within [-B, B]. Outside is identity.
        B = self.tail_bound
        
        inside_mask = (x_tr >= -B) & (x_tr <= B)
        # Note: masks in autograd are fine for indexing/selection but careful with in-place.
        # We process everything with spline, then mix based on mask?
        # Or, we can use the `rational_quadratic_spline` logic which assumes it receives valid inputs.
        # But `rational_quadratic_spline` uses searchsorted which requires inputs in range.
        # We should clamp inputs passed to spline, and then mask the output.
        
        x_tr_clamped = anp.clip(x_tr, -B, B)
        
        y_tr, log_det_tr = _rational_quadratic_spline(
            x_tr_clamped, unnorm_widths, unnorm_heights, unnorm_derivatives,
            inverse=inverse,
            left=-B, right=B, bottom=-B, top=B
        )
        
        # Apply mask: if outside, identity transform (y=x), log_det=0
        
        # Debug print (only print once or sparsely to avoid spam)
        # We can't use static variable in method easily, but we can check if it's the first step of first iteration?
        # Or just print.
        if step_idx == 0 and anp.random.rand() < 0.001:
             print(f"DEBUG: Step {step_idx} Mask coverage: {anp.mean(inside_mask):.2f}")

        final_y_tr = anp.where(inside_mask, y_tr, x_tr)
        final_log_det = anp.where(inside_mask, log_det_tr, 0.)
        
        # Reconstruct full vector
        # We need to put x_id and final_y_tr back into their places.
        # Construct empty (N, D) and fill.
        # autograd doesn't like item assignment.
        # We can use a permutation matrix or just sorting indices.
        
        # Indices to place back
        # We have id_idx and tr_idx.
        # We can concat [x_id, final_y_tr] then apply inverse permutation.
        
        concat_res = anp.concatenate([x_id, final_y_tr], axis=1) # (N, n_id + n_tr)
        
        # We need to reorder columns from [id..., tr...] to [0, 1, 2...]
        # Current order of columns is id_idx followed by tr_idx.
        current_order = anp.concatenate([id_idx, tr_idx])
        # We want to map current_order[k] -> k.
        # Actually we want to permute columns such that they align with 0..D-1.
        # We need argsort of current_order.
        inv_perm = anp.argsort(current_order)
        
        final_out = concat_res[:, inv_perm]
        
        return final_out, anp.sum(final_log_det, axis=1)


    def _flow_forward(self, params, features):
        z = params
        log_det_sum = 0.
        
        for k in range(self.n_flow_steps):
            layer_const = self.model_constants['layers'][k]
            z, log_det = self._transform_step(z, features, layer_const, k, self.weights, inverse=False)
            log_det_sum = log_det_sum + log_det
            
        return z, log_det_sum

    def _flow_inverse(self, z, features):
        x = z
        # Inverse: reverse order of steps
        for k in reversed(range(self.n_flow_steps)):
            layer_const = self.model_constants['layers'][k]
            x, _ = self._transform_step(x, features, layer_const, k, self.weights, inverse=True)
            # We don't need log_det for sampling
            
        return x

    def _loss_function(self, weights, features, params):
        z, log_det_jac = self._flow_forward(params, features)
        
        # Base distribution: Standard Normal N(0, I)
        # log p(z) = -0.5 * z^2 - 0.5 * log(2pi)
        log_p_z = -0.5 * anp.sum(z**2, axis=1) - 0.5 * self.param_dim * anp.log(2 * anp.pi)
        
        log_prob = log_p_z + log_det_jac
        return -anp.mean(log_prob)

    def log_prob(self, features: anp.ndarray, params: anp.ndarray) -> anp.ndarray:
        super().log_prob(features, params)
        z, log_det_jac = self._flow_forward(params, features)
        log_p_z = -0.5 * anp.sum(z**2, axis=1) - 0.5 * self.param_dim * anp.log(2 * anp.pi)
        return log_p_z + log_det_jac

    def sample(self, features: anp.ndarray, n_samples: int, rng: anp.random.RandomState) -> anp.ndarray:
        super().sample(features, n_samples, rng)
        features = features.astype('f')
        if features.ndim == 1:
            features = features.reshape(1, -1)

        n_cond = features.shape[0]
        # Broadcast features
        if n_cond != n_samples:
             features = anp.repeat(features, n_samples, axis=0)

        # Sample z ~ N(0, I)
        z = rng.randn(n_samples, self.param_dim).astype('f')
        
        # Transform z -> x
        x = self._flow_inverse(z, features)
        
        return x.reshape(n_cond, -1, self.param_dim)


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
    tuple[anp.ndarray, anp.ndarray]
        A tuple containing (parameters, features).
    """
    rng = anp.random.RandomState(seed)
    params = anp.zeros((n_samples, 2), dtype='f')
    features = anp.zeros((n_samples, 1), dtype='f')

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
    low_cond = anp.array([[features.min()]])
    high_cond = anp.array([[features.max()]])

    # 3. Generate samples from the trained model
    rng = anp.random.RandomState(0)
    samples_low = estimator.sample(low_cond, 1000, rng)[0]
    samples_high = estimator.sample(high_cond, 1000, rng)[0]

    # 4. Plot results
    # Create a grid to evaluate the learned density
    x_range = (params[:, 0].min() - 1, params[:, 0].max() + 1)
    y_range = (params[:, 1].min() - 1, params[:, 1].max() + 1)
    grid_x, grid_y = anp.meshgrid(anp.linspace(*x_range, 100), anp.linspace(*y_range, 100))
    grid_params = anp.vstack([grid_x.ravel(), grid_y.ravel()]).T

    # Plot for the 'low' condition
    grid_feat_low = anp.repeat(low_cond, grid_params.shape[0], axis=0)
    log_p_low = estimator.log_prob(grid_feat_low, grid_params).reshape(100, 100)
    ax1.contourf(grid_x, grid_y, anp.exp(log_p_low), levels=10, cmap='Blues', alpha=0.8)
    ax1.plot(samples_low[:, 0], samples_low[:, 1], ',', c='navy', alpha=0.5, label='Generated Samples')
    ax1.set_title(f"Condition: feature = {low_cond[0,0]:.2f}")
    ax1.set(xlim=x_range, ylim=y_range, xlabel="Param 1", ylabel="Param 2")
    ax1.legend()

    # Plot for the 'high' condition
    grid_feat_high = anp.repeat(high_cond, grid_params.shape[0], axis=0)
    log_p_high = estimator.log_prob(grid_feat_high, grid_params).reshape(100, 100)
    ax2.contourf(grid_x, grid_y, anp.exp(log_p_high), levels=10, cmap='Oranges', alpha=0.8)
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
    run_test(mdn_banana, 'banana', plot=False)

    # --- MAF Tests ---
    maf_banana = MAFEstimator(param_dim=2, feature_dim=1, n_flows=5, hidden_units=128)
    run_test(maf_banana, 'banana', plot=False)
    
    # --- NSF Tests ---
    print("\nStarting NSF Test...")
    nsf_banana = NSFEstimator(param_dim=2, feature_dim=1, n_flow_steps=4, n_bins=8, hidden_features=64)
    run_test(nsf_banana, 'banana', plot=False)
    print("NSF Test Complete.")
