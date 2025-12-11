import autograd.numpy as anp
from autograd import grad

def gather_exact_match(params):
    # params: (N, D, K) = (2, 2, 3)
    N, D, K = params.shape
    # indices derived from params? No, from "input".
    # But let's assume indices are fixed or derived from constant x.
    indices = anp.array([[0, 1], [2, 0]]) # (2, 2)
    
    # EXACT COPY of _gather_elementwise body
    N_idx, D_idx = indices.shape
    params_flat = params.reshape(-1, params.shape[-1]) # (N*D, K)
    indices_flat = indices.flatten().astype(int) # (N*D,)
    
    # Use integer array indexing on flattened array
    gathered = params_flat[anp.arange(len(indices_flat)), indices_flat]
    result = gathered.reshape(N, D)
    
    return anp.sum(result**2)

if __name__ == "__main__":
    params = anp.random.randn(2, 2, 3)
    print("\nTesting exact match gather...")
    try:
        g = grad(gather_exact_match)(params)
        print("Gradient sum:", anp.sum(anp.abs(g)))
        print("Gradient shape:", g.shape)
    except Exception as e:
        print("Error:", e)

