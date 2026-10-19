"""Acceptance test: the CDE module must work with autograd unavailable.

Runs in a fresh subprocess with an import hook that rejects ``autograd``.
It trains MDN and MAF, evaluates log-probabilities, and samples from both.
"""

import subprocess
import sys
from pathlib import Path


def test_cde_without_autograd():
    root = Path(__file__).resolve().parents[2]
    script = r"""
import sys

class BlockAutograd:
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "autograd" or fullname.startswith("autograd."):
            raise ImportError("autograd is blocked for this test")
        return None

sys.meta_path.insert(0, BlockAutograd())

import numpy as np
from tvbl import cde

rng = np.random.RandomState(0)
params = rng.randn(120, 2).astype("f")
features = rng.randn(120, 3).astype("f")

mdn = cde.MDNEstimator(2, 3, n_components=3, hidden_sizes=(8,))
mdn.train(params, features, n_iter=20, learning_rate=1e-2, seed=1, use_tqdm=False)
assert np.all(np.isfinite(mdn.log_prob(features, params)))
assert mdn.sample(features[:2], 4, np.random.RandomState(2)).shape == (2, 4, 2)

maf = cde.MAFEstimator(2, 3, n_flows=2, hidden_units=8)
maf.train(params, features, n_iter=20, learning_rate=1e-2, seed=1, use_tqdm=False)
assert np.all(np.isfinite(maf.log_prob(features, params)))
assert maf.sample(features[:1], 4, np.random.RandomState(2)).shape == (1, 4, 2)
assert maf.sample(features[:2], 4, np.random.RandomState(2)).shape == (2, 4, 2)

maf1 = cde.MAFEstimator(1, 3, n_flows=1, hidden_units=8)
maf1.train(params[:, :1], features, n_iter=10, learning_rate=1e-2, seed=1, use_tqdm=False)
assert maf1.sample(features[:2], 3, np.random.RandomState(2)).shape == (2, 3, 1)

assert not hasattr(cde, "anp")
assert not hasattr(cde, "grad")
print("NO_AUTOGRAD_OK")
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=root,
        env={"PYTHONPATH": str(root / "content")},
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "NO_AUTOGRAD_OK" in result.stdout
