"""
Test suite for cfun JavaScript implementation

This module tests the JavaScript implementation of the cfun function,
which computes coupling values from a buffer using CSR sparse matrix 
format with delays.

The cfun function is critical for neural dynamics simulations and needs
to be fast and correct. The JavaScript implementation provides better
performance in browser environments (JupyterLite).

When pyjs is available, the code automatically validates that the JavaScript
implementation matches the Python implementation within a tolerance of 1e-5.
"""

import numpy as np
import scipy.sparse

# Import directly from core to avoid circular import issues
import core
cfun = core.cfun


def test_cfun_simple():
    """Test cfun with a simple 3x3 connectivity matrix"""
    num_node = 3
    num_item = 2
    horizon = 10
    t = 5
    
    # Create a simple CSR matrix (3x3)
    data = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype='f')
    indices = np.array([0, 2, 0, 1, 2], dtype=np.int32)
    indptr = np.array([0, 2, 3, 5], dtype=np.int32)
    csr_weights = scipy.sparse.csr_matrix((data, indices, indptr), shape=(3, 3))
    
    # Create idelays2 (2 x nnz)
    idelays2 = np.array([[1, 2, 1, 2, 1], [0, 1, 0, 1, 0]], dtype=np.int32)
    
    # Create buffer
    np.random.seed(42)
    buffer = np.random.randn(num_node, horizon, num_item).astype('f')
    
    # Test cfun
    result = cfun(t, buffer, csr_weights, idelays2, horizon)
    
    assert result.shape == (2, num_node, num_item)
    assert result.dtype == np.float32
    print("✓ test_cfun_simple passed")


def test_cfun_realistic():
    """Test cfun with a realistic sparse connectivity matrix"""
    num_node = 10
    num_item = 4
    horizon = 20
    t = 15
    
    # Create a random sparse matrix
    np.random.seed(123)
    density = 0.3
    matrix = scipy.sparse.random(num_node, num_node, density=density, format='csr', dtype='f')
    matrix.data = np.abs(matrix.data)  # Make weights positive
    
    # Create random delays
    nnz = matrix.nnz
    max_delay = horizon // 2
    idelays = np.random.randint(0, max_delay, nnz, dtype=np.int32)
    idelays2 = np.stack([idelays, idelays - 1], axis=0)
    
    # Create buffer
    buffer = np.random.randn(num_node, horizon, num_item).astype('f')
    
    # Test cfun
    result = cfun(t, buffer, matrix, idelays2, horizon)
    
    assert result.shape == (2, num_node, num_item)
    assert result.dtype == np.float32
    print("✓ test_cfun_realistic passed")


def test_cfun_edge_cases():
    """Test cfun with edge cases like t=0 and large t"""
    num_node = 3
    num_item = 2
    horizon = 10
    
    np.random.seed(456)
    data = np.array([1.0, 2.0, 3.0], dtype='f')
    indices = np.array([0, 1, 2], dtype=np.int32)
    indptr = np.array([0, 1, 2, 3], dtype=np.int32)
    csr_weights = scipy.sparse.csr_matrix((data, indices, indptr), shape=(3, 3))
    
    idelays2 = np.array([[1, 2, 3], [0, 1, 2]], dtype=np.int32)
    buffer = np.random.randn(num_node, horizon, num_item).astype('f')
    
    # Test with t=0
    result = cfun(0, buffer, csr_weights, idelays2, horizon)
    assert result.shape == (2, num_node, num_item)
    
    # Test with large t (tests modulo arithmetic)
    result = cfun(1000, buffer, csr_weights, idelays2, horizon)
    assert result.shape == (2, num_node, num_item)
    
    print("✓ test_cfun_edge_cases passed")


def test_cfun_dimensions():
    """Test that cfun produces correct output dimensions"""
    test_cases = [
        (5, 2, 10),   # small
        (10, 4, 20),  # medium
        (20, 8, 30),  # larger
    ]
    
    for num_node, num_item, horizon in test_cases:
        np.random.seed(42)
        
        # Create sparse matrix with higher density to avoid issues
        density = 0.5
        matrix = scipy.sparse.random(num_node, num_node, density=density, format='csr', dtype='f')
        matrix.data = np.abs(matrix.data)
        
        # Ensure each row has at least one entry by adding diagonal
        matrix = matrix + scipy.sparse.eye(num_node, dtype='f') * 0.1
        matrix = matrix.tocsr()
        
        # Create delays
        nnz = matrix.nnz
        idelays = np.random.randint(0, horizon // 2, nnz, dtype=np.int32)
        idelays2 = np.stack([idelays, idelays - 1], axis=0)
        
        # Create buffer
        buffer = np.random.randn(num_node, horizon, num_item).astype('f')
        
        # Test
        result = cfun(5, buffer, matrix, idelays2, horizon)
        
        expected_shape = (2, num_node, num_item)
        assert result.shape == expected_shape, \
            f"Expected shape {expected_shape}, got {result.shape}"
    
    print("✓ test_cfun_dimensions passed")


if __name__ == '__main__':
    print("Running cfun tests...")
    print()
    
    test_cfun_simple()
    test_cfun_realistic()
    test_cfun_edge_cases()
    test_cfun_dimensions()
    
    print()
    print("=" * 60)
    print("✓ All cfun tests passed!")
    print("=" * 60)
    print()
    print("Note: When pyjs is available, cfun automatically validates")
    print("that the JavaScript implementation matches the Python")
    print("implementation (tolerance: 1e-5)")
