import time
import numpy as np
import scipy.sparse
import tqdm
from dataclasses import dataclass

def cfun(t, buffer, csr_weights, idelays2, horizon):
    cx = buffer[csr_weights.indices, (t-idelays2) % horizon]
    cx *= csr_weights.data.reshape(-1, 1)
    cx = np.add.reduceat(cx, csr_weights.indptr[:-1], axis=1)
    return cx  # (2, num_node, num_item)


try:
    import pyjs
    _js_cfun = pyjs.js.Function("cx", "t", "buffer", "indices", "weights", "indptr", "idelays2", "horizon", """
    """)
    _np_cfun = cfun
    def cfun(t, buffer, csr_weights, idelays2, horizon, validate=True):
        cx_np = _np_cfun(t, buffer, csr_weights, idelays2, horizon)
        cx = np.zeros(cx_np.shape, 'f')
        j_cx = pyjs.buffer_to_js_typed_array(cx, view=True)
        j_buffer = pyjs.buffer_to_js_typed_array(buffer, view=True)
        j_indices = pyjs.buffer_to_js_typed_array(csr_weights.indices, view=True)
        j_weights = pyjs.buffer_to_js_typed_array(csr_weights.data, view=True)
        j_indptr = pyjs.buffer_to_js_typed_array(csr_weights.indptr, view=True)
        j_idelays2 = pyjs.buffer_to_js_typed_array(idelays2, view=True)
        _js_cfun(j_cx, t, j_buffer, j_indices, j_weights, j_indptr, j_idelays2, horizon)
        assert np.allclose(cx, cx_np, atol=1e-5)
        return cx
except ImportError:
    print("pyjs not found, using pure python implementation")


@dataclass
class DFun:
    sim_params: np.ndarray

    def __call__(self, x, cx):
        a, tau, k = self.sim_params
        return np.array([
            tau*(x[0] - x[0]**3/3 + x[1]),
            (1/tau)*(a - k*cx - x[0])
        ])  # (2, num_node, num_item)


def heun(x, cx, dt, num_node, num_item, dfun, z_scale):
    if isinstance(z_scale, (int, float)) and z_scale == 0:
        z = 0
    else:
        z = np.random.randn(2, num_node, num_item)
        z *= z_scale.reshape((2, 1, -1))
    dx1 = dfun(x, cx[0])
    dx2 = dfun(x + dt*dx1 + z, cx[1])
    return x + dt/2*(dx1 + dx2) + z


def run(
    csr_weights: scipy.sparse.csr_matrix,
    idelays: np.ndarray,
    dfun: DFun,
    z_scale: np.ndarray,
    horizon: int,
    rng_seed=43, num_item=8, num_node=90, num_svar=2, num_time=1000, dt=0.1,
    num_skip=5, show_time=True,
):
    trace_shape = num_time // num_skip + 1, num_svar, num_node, num_item
    trace = np.zeros(trace_shape, 'f')
    assert idelays.max() < horizon-2
    idelays2 = -horizon + np.c_[idelays, idelays-1].T
    assert idelays2.shape == (2, csr_weights.nnz)
    buffer = np.zeros((num_node, horizon, num_item))

    x = np.zeros((2, num_node, num_item), 'f')
    
    tic = time.time()

    steps = tqdm.trange(trace.shape[0], ncols=79) if show_time else range(trace.shape[0])
    for t in steps:
        for tt in range(num_skip):
            ttt = t*num_skip + tt
            cx = cfun(ttt, buffer, csr_weights, idelays2, horizon)
            x = heun(x, cx, dt, num_node, num_item, dfun, z_scale)
            buffer[:, ttt % horizon] = x[0]
        trace[t] = x
        
    tok = time.time()
    if show_time:
        print(tok - tic, 's', num_time*num_item/(tok-tic), 'iter/s')

    return trace
