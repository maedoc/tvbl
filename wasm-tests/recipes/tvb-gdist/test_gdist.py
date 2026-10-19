import numpy as np
import gdist

vertices = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64)
triangles = np.array([[0, 1, 2]], dtype=np.int32)
d = gdist.compute_gdist(vertices, triangles, np.array([0], dtype=np.int32), np.array([1, 2], dtype=np.int32))
assert np.allclose(d, [1.0, 1.0]), d
print("gdist ok", d)
