"""Toy Jansen-Rit dataset for the attention-vs-MADE benchmark.

Deliberately small: two parameters ``(A, a)``, one second of pyramidal output
at 200 Hz, and an STFT tokenisation instead of a hand-built summary vector.

The point of using an STFT rather than mean/std/PSD is that the STFT is a
fixed, linear, *overcomplete* re-representation -- nothing is thrown away, the
learned pooling happens afterwards inside the encoder.  A hand-built summary
vector is a lossy projection, and information destroyed there is unrecoverable
by any downstream model.
"""

import numpy as np
from scipy.integrate import odeint
from scipy.signal import stft

V_MAX, R = 5.0, 0.56
A_BOUNDS = (3.0, 3.5)          # Jansen-Rit A: excitatory gain
AA_BOUNDS = (90.0, 110.0)      # Jansen-Rit a: excitatory rate constant


def _sigmoid(x, v0):
    return V_MAX / (1.0 + np.exp(R * (v0 - x)))


def _jr_ode(y, t, A, a, B=22.0, b=50.0, v0=6.0, C=135.0):
    """Standard Jansen-Rit system (pyramidal, excitatory, inhibitory)."""
    y0, y1, y2, y3, y4, y5 = y
    C1 = C2 = C
    C3, C4 = 0.8 * C, 0.25 * C
    p_mu = 220.0
    return [
        y3, y4, y5,
        A * a * _sigmoid(y1 - y2, v0) - 2 * a * y3 - a * a * y0,
        A * a * (p_mu + C2 * _sigmoid(C1 * y0, v0)) - 2 * a * y4 - a * a * y1,
        B * b * C4 * _sigmoid(C3 * y0, v0) - 2 * b * y5 - b * b * y2,
    ]


def simulate(A, a, fs=200, duration=1.2, discard=0.2):
    """Pyramidal-minus-excitatory trace, initial transient discarded."""
    t = np.arange(0.0, duration, 1.0 / fs)
    sol = odeint(_jr_ode, np.zeros(6), t, args=(A, a), rtol=1e-5, atol=1e-7)
    return (sol[:, 1] - sol[:, 2])[int(round(discard * fs)):]


def sample_dataset(n, seed, **sim_kwargs):
    """Draw ``n`` parameter sets uniformly and simulate each one."""
    rng = np.random.RandomState(seed)
    A = rng.uniform(*A_BOUNDS, size=n)
    a = rng.uniform(*AA_BOUNDS, size=n)
    traces = np.array([simulate(A[i], a[i], **sim_kwargs) for i in range(n)],
                      dtype=np.float64)
    return np.c_[A, a], traces


def stft_tokens(traces, fs=200, nperseg=32, hop=8, db_floor=1e-10):
    """Log-power STFT tokens of shape ``(N, n_frames, n_bands)``.

    With ``fs=200, nperseg=32, hop=8`` and 200 samples this gives 22 frames of
    17 bands (6.25 Hz spacing): small enough that a global self-attention layer
    stays cheap in pure NumPy.
    """
    freqs, frames, Z = stft(traces, fs=fs, nperseg=nperseg,
                            noverlap=nperseg - hop, boundary=None,
                            padded=False, axis=-1)
    power = np.abs(Z) ** 2
    log_power = np.log10(power + db_floor)
    return log_power.transpose(0, 2, 1).astype('f'), freqs


def standardize(tokens_train, tokens_test):
    """Z-score per band using pooled train statistics (a fixed transform)."""
    mu = tokens_train.mean(axis=(0, 1), keepdims=True)
    sd = tokens_train.std(axis=(0, 1), keepdims=True) + 1e-6
    return ((tokens_train - mu) / sd).astype('f'), ((tokens_test - mu) / sd).astype('f')


def zscore_params(params_train, params_test):
    """Standardise targets so the flow's N(0, I) base is a sane starting point."""
    mu = params_train.mean(axis=0, keepdims=True)
    sd = params_train.std(axis=0, keepdims=True) + 1e-12
    return ((params_train - mu) / sd).astype('f'), ((params_test - mu) / sd).astype('f'), mu, sd
