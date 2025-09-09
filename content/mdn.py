import autograd.numpy as np
from autograd import grad
from autograd.scipy.special import logsumexp
import math

def train_mdn(theta, feats, K=5, H=(32, 32), niter=2000, learning_rate=1e-3):
    """
    Trains a Mixture Density Network to approximate a posterior distribution p(theta|feats).

    This function implements the method from "Fast e-free Inference of Simulation
    Models with Bayesian Conditional Density Estimation" (Papamakarios & Murray, 2018).

    Args:
        theta (np.ndarray): An N x D_out matrix of simulated parameters.
        feats (np.ndarray): An N x D_in matrix of corresponding data features.
        K (int): The number of Gaussian mixture components to use.
        H (tuple): A tuple specifying the number of units in each hidden layer.
        niter (int): The number of gradient descent iterations.
        learning_rate (float): The learning rate for the Adam optimizer.

    Returns:
        tuple: A tuple containing:
            - dict: The dictionary of trained network weights.
            - list: A list of the loss values recorded during training.
            - function: The forward_pass function for making predictions.
    """
    N, D_out = theta.shape
    N_feats, D_in = feats.shape
    assert N == N_feats, "Theta and feats must have the same number of rows (simulations)."

    # --- Use float32 for all operations ---
    theta = theta.astype('f')
    feats = feats.astype('f')

    # --- 1. Initialize Network Weights ---
    layer_sizes = (D_in,) + H
    weights = {}

    # Initialize MLP weights
    in_size = D_in
    for i, out_size in enumerate(H):
        weights[f'W{i}'] = (np.random.randn(in_size, out_size) * np.sqrt(2.0 / in_size)).astype('f')
        weights[f'b{i}'] = np.zeros(out_size, dtype='f')
        in_size = out_size
    
    last_hidden_size = H[-1] if H else D_in

    # Initialize weights for the GMM output layers
    # Mixing coefficients (alphas)
    weights['W_alpha'] = (np.random.randn(last_hidden_size, K) * 0.01).astype('f')
    weights['b_alpha'] = np.zeros(K, dtype='f')
    # Means (mus)
    weights['W_mu'] = (np.random.randn(last_hidden_size, K * D_out) * 0.01).astype('f')
    weights['b_mu'] = np.zeros(K * D_out, dtype='f')

    # Cholesky factor of the precision matrix (inverse covariance)
    # This ensures the covariance matrix is always positive semi-definite.
    n_off_diag = D_out * (D_out - 1) // 2
    weights['W_L_prec_log_diag'] = (np.random.randn(last_hidden_size, K * D_out) * 0.01).astype('f')
    weights['b_L_prec_log_diag'] = np.zeros(K * D_out, dtype='f')
    weights['W_L_prec_offdiag'] = (np.random.randn(last_hidden_size, K * n_off_diag) * 0.01).astype('f')
    weights['b_L_prec_offdiag'] = np.zeros(K * n_off_diag, dtype='f')
    
    # Pre-compute a basis tensor to build the off-diagonal part of L_prec.
    # This is a clean, vectorized way to construct the matrix that autograd can handle.
    M_offdiag_basis = np.zeros((n_off_diag, D_out, D_out), dtype='f')
    rows, cols = np.triu_indices(D_out, k=1)
    M_offdiag_basis[np.arange(n_off_diag), rows, cols] = 1

    def forward_pass(W, x_input):
        """Maps input features to the parameters of a Gaussian Mixture Model."""
        h = x_input
        for i in range(len(H)):
            h = np.tanh(h @ W[f'W{i}'] + W[f'b{i}'])

        # Calculate GMM parameters from the last hidden layer
        log_alpha = h @ W['W_alpha'] + W['b_alpha']
        # Use logsumexp for stable softmax
        alpha = np.exp(log_alpha - logsumexp(log_alpha, axis=1, keepdims=True))
        mu = (h @ W['W_mu'] + W['b_mu']).reshape(-1, K, D_out)
        
        # Build the Cholesky factor of the precision matrix, L_prec
        # L_prec is upper triangular. Precision matrix = L_prec.T @ L_prec
        L_prec_log_diag = (h @ W['W_L_prec_log_diag'] + W['b_L_prec_log_diag']).reshape(-1, K, D_out)
        L_prec_offdiag = (h @ W['W_L_prec_offdiag'] + W['b_L_prec_offdiag']).reshape(-1, K, n_off_diag)
        
        # Diagonal part
        L_prec_diag_mat = np.einsum('nki,ij->nkij', np.exp(L_prec_log_diag), np.eye(D_out, dtype='f'))
        # Off-diagonal part
        L_prec_offdiag_mat = np.einsum('nkl,lij->nkij', L_prec_offdiag, M_offdiag_basis)

        L_prec = L_prec_diag_mat + L_prec_offdiag_mat
        
        return alpha, mu, L_prec, L_prec_log_diag

    def neg_log_likelihood(W, x_input, y_true):
        """Computes the negative log-likelihood of the true parameters under the GMM."""
        alpha, mu, L_prec, L_prec_log_diag = forward_pass(W, x_input)
        
        # Reshape for broadcasting: y_true from (N, D_out) to (N, 1, D_out)
        y_true_reshaped = y_true[:, np.newaxis, :]
        
        # Calculate log PDF of multivariate normal for each component k
        # log p(y|k) = -0.5 * ||L_prec_k @ (y-mu_k)||^2 + log|det(L_prec_k)| - const
        delta = y_true_reshaped - mu  # Shape (N, K, D_out)
        
        # L_prec @ delta, result shape (N, K, D_out)
        z = np.einsum('nkij,nkj->nki', L_prec, delta)
        
        # Quadratic term: -0.5 * z^T @ z
        quad_term = -0.5 * np.sum(z**2, axis=2)
        
        # Log determinant term: sum of log-diagonals of L_prec
        log_det_term = np.sum(L_prec_log_diag, axis=2)
        
        # Log probability for each sample n, for each component k
        log_probs_k = quad_term + log_det_term - 0.5 * D_out * np.log(2 * math.pi)
        
        # Total log probability, weighted by mixing coefficients (alpha)
        # Use log-sum-exp trick for numerical stability
        total_log_prob = logsumexp(np.log(alpha) + log_probs_k, axis=1)
        
        # Return the mean of the negative log-likelihood
        return -np.mean(total_log_prob)

    # --- 3. Optimization Loop (Adam) ---
    gradient_func = grad(neg_log_likelihood)
    m = {key: np.zeros_like(val) for key, val in weights.items()}
    v = {key: np.zeros_like(val) for key, val in weights.items()}
    beta1, beta2, epsilon = 0.9, 0.999, 1e-8
    loss_history = []

    print("Starting training...")
    for i in range(niter):
        g = gradient_func(weights, feats, theta)
        loss = neg_log_likelihood(weights, feats, theta)
        loss_history.append(loss)
        
        if (i + 1) % 200 == 0 or i == 0:
            print(f"Iteration {i+1:5d}/{niter}, Loss: {loss:.4f}")

        # Adam update step
        for key in weights:
            if not np.all(np.isfinite(g[key])):
                print(f"Warning: Non-finite gradient for {key} at iteration {i}. Stopping.")
                return weights, loss_history
            m[key] = beta1 * m[key] + (1 - beta1) * g[key]
            v[key] = beta2 * v[key] + (1 - beta2) * (g[key]**2)
            m_hat = m[key] / (1 - beta1**(i + 1))
            v_hat = v[key] / (1 - beta2**(i + 1))
            weights[key] -= learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)
            
    print("Training finished.")
    # Return the forward_pass function as a closure for easy prediction
    return weights, loss_history, forward_pass

def sample(trained_weights, predictor_func, x_input, n_samples=1):
    """
    Generates samples from the posterior distribution p(y|x) learned by the MDN.

    Args:
        trained_weights (dict): The dictionary of trained network weights.
        predictor_func (function): The forward_pass function from the trained MDN.
        x_input (np.ndarray): An array of input features (conditions) of shape (N_cond, D_in).
                              If a single condition is given, it can be shape (D_in,).
        n_samples (int): The number of samples to generate for each input condition.

    Returns:
        np.ndarray: An array of generated samples of shape (N_cond, n_samples, D_out).
                    If the input was 1D, the output shape is squeezed to (n_samples, D_out).
    """
    import autograd.numpy as np

    # --- 1. Input validation and reshaping ---
    was_1d = False
    if x_input.ndim == 1:
        x_input = x_input.reshape(1, -1)
        was_1d = True

    # Ensure input is float32, consistent with training
    x_input = x_input.astype('f')

    # --- 2. Get GMM parameters from the network ---
    alpha, mu, L_prec, _ = predictor_func(trained_weights, x_input)
    # alpha: (N_cond, K)
    # mu: (N_cond, K, D_out)
    # L_prec: (N_cond, K, D_out, D_out)
    N_cond, K, D_out = mu.shape

    # --- 3. Select mixture components for each sample ---
    # We use the Gumbel-Max trick for efficient, vectorized sampling from the
    # categorical distribution defined by the mixing coefficients (alpha).
    log_alpha = np.log(alpha + 1e-9) # Add epsilon for numerical stability

    # Gumbel noise, broadcastable to (N_cond, n_samples, K)
    gumbel_noise = -np.log(-np.log(np.random.uniform(size=(N_cond, n_samples, K))))

    # Select component indices by finding the argmax
    component_indices = np.argmax(log_alpha[:, np.newaxis, :] + gumbel_noise, axis=2)
    # component_indices has shape (N_cond, n_samples)

    # --- 4. Gather the parameters (mu, L_prec) for the chosen components ---
    # Use advanced indexing for an efficient, vectorized lookup.
    cond_idx = np.arange(N_cond)[:, np.newaxis] # Shape (N_cond, 1) for broadcasting

    chosen_mu = mu[cond_idx, component_indices]
    # -> shape (N_cond, n_samples, D_out)

    chosen_L_prec = L_prec[cond_idx, component_indices]
    # -> shape (N_cond, n_samples, D_out, D_out)

    # --- 5. Sample from the selected multivariate Gaussian components ---
    # As derived above, we need the inverse of L_prec.
    try:
        L_cov_factor = np.linalg.inv(chosen_L_prec)
    except np.linalg.LinAlgError as e:
        print("Error: Failed to invert the Cholesky factor of the precision matrix.")
        print("This can happen if a component's precision matrix is singular or ill-conditioned.")
        raise e

    # Generate standard normal random vectors
    z = np.random.randn(N_cond, n_samples, D_out)

    # Transform the standard normal samples to the target distribution: y = mu + L_cov_factor @ z
    samples = chosen_mu + np.einsum('nsij,nsj->nsi', L_cov_factor, z)
    # samples has shape (N_cond, n_samples, D_out)

    # --- 6. Squeeze output if input was 1D for convenience ---
    if was_1d:
        return samples.squeeze(axis=0)

    return samples


def plot_ellipse(mean, cov, color, label, std_devs=1.0):
    """
    Plots a covariance ellipse on a given matplotlib axes instance.
    """
    import matplotlib.pyplot as pl
    from matplotlib.patches import Ellipse
    # Eigenvalue decomposition to find the axes of the ellipse
    eigenvalues, eigenvectors = np.linalg.eigh(cov)

    # Get the angle of the major axis
    angle = np.degrees(np.arctan2(*eigenvectors[:, 0][::-1]))

    # Get the width and height of the ellipse (major and minor axes)
    # The eigenvalues are the variances, so the std dev is the sqrt
    width, height = 2 * std_devs * np.sqrt(eigenvalues)

    ellipse = Ellipse(xy=mean, width=width, height=height, angle=angle,
                      edgecolor=color, facecolor='none', linewidth=2,
                      label=label)
    pl.gca().add_patch(ellipse)


def test_mdn(plot=False):
    # Generate some synthetic data for a clear demonstration.
    # The goal is to learn a bimodal distribution where the features determine the mode.
    N_samples = 4000
    D_features = 1
    D_params = 2
    
    # Create features that switch between two states
    sim_feats = np.random.randn(N_samples, D_features).astype('f')
    # Based on the feature's sign, draw theta from one of two Gaussians
    mode_selector = sim_feats[:, 0] > 0
    sim_theta = np.zeros((N_samples, D_params), dtype='f')
    
    mean1, cov1 = np.array([2, 3], dtype='f'), np.array([[0.5, 0.3], [0.3, 0.5]], dtype='f')
    mean2, cov2 = np.array([-2, -3], dtype='f'), np.array([[0.5, -0.3], [-0.3, 0.5]], dtype='f')
    
    sim_theta[mode_selector] = np.random.multivariate_normal(mean1, cov1, size=np.sum(mode_selector)).astype('f')
    sim_theta[~mode_selector] = np.random.multivariate_normal(mean2, cov2, size=np.sum(~mode_selector)).astype('f')

    print(f"Training MDN on {N_samples} samples...")
    print(f"Input features dimension: {D_features}, Output parameters dimension: {D_params}\n")

    # Train the MDN
    trained_weights, losses, predictor_func = train_mdn(
        sim_theta, 
        sim_feats, 
        K=2,            # We know there are 2 modes in this toy problem
        H=(24, 24),     # A small MLP
        niter=400,
        learning_rate=5e-3
    )

    print(f"\nFinal loss: {losses[-1]:.4f}")

    # --- Validation Step ---
    # Use the returned predictor function to check if the network learned the modes.
    print("\n--- Validation ---")

    # Test with a feature that should select the first mode (positive feature)
    test_feat1 = np.array([[1.5]], dtype='f')
    alpha1, mu1, _, _ = predictor_func(trained_weights, test_feat1)
    dominant_mode_idx1 = np.argmax(alpha1[0])
    found_mode1 = mu1[0, dominant_mode_idx1, :]

    # Test with a feature that should select the second mode (negative feature)
    test_feat2 = np.array([[-1.5]], dtype='f')
    alpha2, mu2, _, _ = predictor_func(trained_weights, test_feat2)
    dominant_mode_idx2 = np.argmax(alpha2[0])
    found_mode2 = mu2[0, dominant_mode_idx2, :]

    # The assignment of found modes to true modes might be swapped, so we check both ways
    is_correct1 = np.allclose(found_mode1, mean1, atol=0.2) and np.allclose(found_mode2, mean2, atol=0.2)
    is_correct2 = np.allclose(found_mode1, mean2, atol=0.2) and np.allclose(found_mode2, mean1, atol=0.2)

    print(f"True Mode 1 (for positive feats): {np.round(mean1, 2)}")
    print(f"True Mode 2 (for negative feats): {np.round(mean2, 2)}")
    print(f"Predicted mode for positive feature: {np.round(found_mode1, 2)}")
    print(f"Predicted mode for negative feature: {np.round(found_mode2, 2)}")

    if is_correct1 or is_correct2:
        print("\nSuccess: The network correctly identified the true modes of the distribution.")
    else:
        print("\nFailure: The network did not correctly identify the modes.")

    if plot:    
        # Define two test features to elicit the two different posterior modes
        test_feat_pos = np.array([[1.5]], dtype='f')  # Should trigger mode 1
        test_feat_neg = np.array([[-1.5]], dtype='f') # Should trigger mode 2
        n_plot_samples = 1000
        
        print("Generating samples for positive feature...")
        samples_pos = sample(trained_weights, predictor_func, test_feat_pos, n_samples=n_plot_samples)
        
        print("Generating samples for negative feature...")
        samples_neg = sample(trained_weights, predictor_func, test_feat_neg, n_samples=n_plot_samples)

        import matplotlib.pyplot as pl
        
        pl.plot(samples_neg[0,:,0], samples_neg[0,:,1], ',b')
        pl.plot(samples_pos[0,:,0], samples_pos[0,:,1], ',g')
        
        pl.plot(mean1[0], mean1[1], 'x', color='black', markersize=12, mew=3, label='True Mean 1')
        pl.plot(mean2[0], mean2[1], '+', color='black', markersize=12, mew=3, label='True Mean 2')
        
        # Plot the 1-standard deviation ellipses for the true distributions
        plot_ellipse(mean1, cov1, color='royalblue', label='1-std dev (True Dist. 1)')
        plot_ellipse(mean2, cov2, color='firebrick', label='1-std dev (True Dist. 2)')
        
        pl.show()

if __name__ == "__main__":
    test_mdn()