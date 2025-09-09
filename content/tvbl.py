import time
import numpy as np
import scipy.sparse
import tqdm

def _cfun(t, buffer, csr_weights, idelays2, horizon):
    cx = buffer[csr_weights.indices, (t-idelays2) % horizon]
    cx *= csr_weights.data.reshape(-1, 1)
    cx = np.add.reduceat(cx, csr_weights.indptr[:-1], axis=1)
    return cx  # (2, num_node, num_item)

def _dfun(x, cx, sim_params):
    a, tau, k = sim_params
    return np.array([
        tau*(x[0] - x[0]**3/3 + x[1]),
        (1/tau)*(a + k*cx - x[0])
    ])  # (2, num_node, num_item)

def _heun(x, cx, dt, num_node, num_item, dfun, sim_params):
    z = np.random.randn(2, num_node, num_item)
    z *= 0  # z_scale.reshape((2, 1, 1))
    dx1 = dfun(x, cx[0], sim_params)
    dx2 = dfun(x + dt*dx1 + z, cx[1], sim_params)
    return x + dt/2*(dx1 + dx2) + z


def run(
    csr_weights: scipy.sparse.csr_matrix,
    idelays: np.ndarray,
    sim_params: np.ndarray,
    z_scale: np.ndarray,
    horizon: int,
    rng_seed=43, num_item=8, num_node=90, num_svar=2, num_time=1000, dt=0.1,
    num_skip=5
):
    trace_shape = num_time // num_skip + 1, num_svar, num_node, num_item
    trace = np.zeros(trace_shape, 'f')
    assert idelays.max() < horizon-2
    idelays2 = -horizon + np.c_[idelays, idelays-1].T
    assert idelays2.shape == (2, csr_weights.nnz)
    buffer = np.zeros((num_node, horizon, num_item))

    x = np.zeros((2, num_node, num_item), 'f')
    
    tic = time.time()

    for t in tqdm.trange(trace.shape[0], ncols=79):
        for tt in range(num_skip):
            ttt = t*num_skip + tt
            cx = _cfun(ttt, buffer, csr_weights, idelays2, horizon)
            x = _heun(x, cx, dt, num_node, num_item, _dfun, sim_params)
            buffer[:, ttt % horizon] = x[0]
        trace[t] = x
        
    tok = time.time()
    print(tok - tic, 's', num_time*num_item/(tok-tic), 'iter/s')

    return trace
