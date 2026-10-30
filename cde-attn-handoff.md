# CDE + attention: handoff

**Status: research done, code is a prototype. ~1 hour of cleanup before parking.**

Written after building self-attention context encoders for the conditional
density estimators in `content/tvbl/cde.py`, and benchmarking them on
Jansen-Rit (single node, global parameters) and a coupled FitzHugh-Nagumo
network (regional parameters).

The one-sentence answer to the original question: **attention-conditioned CDE
is clearly worth it when each parameter is read off a different part of the
data, and a tie when they are not.** Numbers below. The reason to park rather
than ship is that the current TVB demo infers *global* parameters, which is
the tie case.

---

## 1. What is in the tree

| path | lines | state |
| --- | --- | --- |
| `content/tvbl/attn.py` | 435 | prototype. `AttnEncoder` (softmax + linear attention, hand-written backward), `AttnMAFEstimator`. 23 tests. **Not public-API quality.** |
| `content/tvbl/test_attn.py` | 366 | gradient checks (central finite differences along random directions), normalisation, contract tests |
| `content/tvbl/cde.py` | +22 | `_flow_backward_np(..., compute_ctx_grad=True)` returns `dL/dctx`. Backwards compatible. **Keep.** |
| `content/tvbl/__init__.py` | +1 | exports `attn` — should be reverted, see §3.1 |
| `experiments/attn_cde/README.md` | 390 | **the main deliverable.** Full results, negative results, caveats |
| `experiments/attn_cde/benchmark.py` | 257 | minibatch+clip training loop, NLL/calibration eval |
| `experiments/attn_cde/multinode.py` + `benchmark_multinode.py` | 338 | coupled-network toy, regional parameters |
| `experiments/attn_cde/coupling_scan.py` | 99 | finds the regime where the graph matters, and where the sim blows up |
| `experiments/attn_cde/jr_toy.py` | 83 | single-node Jansen-Rit + STFT tokenisation |

All 356 tests under `content/tvbl/` pass.

```sh
PYTHONPATH=content python3 -m pytest content/tvbl/ -q
PYTHONPATH=content python3 experiments/attn_cde/coupling_scan.py     # ~2 s
PYTHONPATH=content python3 experiments/attn_cde/benchmark.py --help
```

Note `experiments/` is not importable as a package (scripts use
`sys.path.insert`); run them as scripts from the repo root.

---

## 2. The finding, in numbers

Everything below is test NLL on held-out simulations (lower better), same MADE
flow in every arm. `cov90` should be 0.90.

**Single node, 2 global parameters (Jansen-Rit, 22x17 STFT tokens):**

| arm | params | test NLL |
| --- | --- | --- |
| `made-raw` (MAF on 374 flat dims) | 103,536 | −4.01 ± 0.76 |
| `linear-pool` (d=32) | 11,856 | −5.33 ± 0.39 |
| `attn-2blk` (d=32) | 28,944 | −5.42 ± 0.61 |

Encoder >> no encoder (~1.3 nats, 1/3.6–1/8.7 params). Attention ≈ pooling:
the 90% of token variance lives in **2 of 374** principal components, so there
is no routing problem to solve.

**8 nodes, 8 regional parameters, coupling g=1:**

| arm | test NLL |
| --- | --- |
| `linear-pool` | −5.81 |
| `attn-1blk` | **−10.35** |

Now attention wins by 4.5 nats, because tokens are rank-93 of 2600 and each
node's tokens carry its own parameter (`corr(std_i, a_i) ≈ −0.9`, cross-node
≈ 0).

**At g=6 (high cross-talk: cross/own = 0.34) with 6000 sims:**

| arm | test NLL | cov90 |
| --- | --- | --- |
| `linear-pool` | −4.51 | 0.85 |
| `attn-1blk` | −12.81 | 0.84 |
| `attn-graph` (SC bias, self-loop=2) | −12.82 | 0.86 |

Attention beats pooling by **8.3 nats**. The graph bias is NLL-neutral but
halves systematic error (`|z-bias|` 0.21 → 0.10) for one scalar.

Data budget dominates: at 1500 sims attention was at **+0.30** with an 8-nat
train/test gap; 4x the sims moved it to −12.81.

---

## 3. The ~1 hour of work before parking

### 3.1 Stop `attn` looking like public API

`content/tvbl/__init__.py` currently does `from . import ... attn`. Revert
that line. The module stays importable as `from tvbl import attn`, but is no
longer advertised. Optionally add a one-line warning at the top of
`attn.py`:

```python
# EXPERIMENTAL: see cde-attn-handoff.md. API is not stable.
```

### 3.2 Guard the linear-attention denominator

`content/tvbl/attn.py:231` is `oh = num / den[..., None]` with no epsilon.
`_elu_plus1` can underflow to 0 in float32 for very negative pre-activations,
making `den == 0` and the read-out `0/0`. Change to `den[..., None] + 1e-6`
and add a test that a large negative input produces finite output:

```python
est, weights = _linear_estimator()
tokens = np.full((2, est.n_tokens, est.token_dim), -1e4, dtype=np.float64)
c, _ = est.encoder.forward(weights, tokens)
assert np.all(np.isfinite(c))
```

### 3.3 Promote the evaluation harness

`experiments/attn_cde/benchmark.py::fit` and `::evaluate` are the most
reusable artefact here — the repo otherwise has no SBI evaluation harness,
only `cde.run_test` with eyeball plots. Move them to something like
`content/tvbl/sbi_eval.py` (or `testutil`) with tests, and have the
experiment scripts import from there. Keep `experiments/` importing them.

Do the same for `coupling_scan.py`'s feature/`max_corr` diagnostics: they are
useful for *any* simulator, independent of attention.

### 3.4 Note the base-class gaps somewhere

`ConditionalDensityEstimator.train` (`content/tvbl/cde.py:131`) is full-batch,
fixed learning rate, no clipping, no weight decay. We have direct evidence
this matters: at `lr=3e-3` the MAF context path does not merely overfit, it
fails to train (`-0.55` vs `-5.03` at `1e-3`); without clipping it diverges to
~1e17. Every experiment used a hand-rolled loop in `benchmark.py` instead.

Either fix `train()` (batch_size, cosine schedule, grad clip) or add a
comment/TODO pointing at `benchmark.py::fit` as the workaround. Fixing it
would benefit `MDNEstimator` and `MAFEstimator` too, so it is the
highest-value thing in this list.

### 3.5 Commit

Single commit covering the above. Do **not** claim the attention feature is
supported in the message.

---

## 4. Parked, with unpark gates

Do not ship `AttnMAFEstimator` as a supported feature until **all three**:

1. **wasm timing.** This project ships to JupyterLite. Nothing here has been
   timed under emscripten. Cheap to check now, because the linear-attention
   path is pure NumPy (no numba). Use `wasm-tests/run.sh`.
2. **Calibration.** `cov90` is 0.82–0.86 against an ideal 0.90, and has never
   been through SBC. For an inference library the posterior *is* the product.
3. **A real regional-parameter problem.** See §5.

Also unresolved and known to matter:

- **The feature map question changed the answer by 0.4 nats at n=1 seed.** The
  original `_elu_plus1` had a bug (added `+1` to both branches, discontinuous
  at the origin). Fixing it made the *result worse*: −12.14 → −11.71 test NLL,
  `|z-bias|` 0.20 → 0.46. The bug's extra constant acted as a floor keeping
  the denominator well-conditioned. Re-run with ≥3 seeds before trusting
  either number, and consider an explicit floor (`elu(x)+1+eps`, or softplus)
  as a tunable.
- Accuracy at 90 nodes is **unmeasured**. The 30-node probe (params 30,
  tokens 150) confirmed timing but every arm was underfit at 2000 sims
  (`shrink` 0.1–0.2), so its accuracy numbers are meaningless.
  `param_dim` in the tens needs 1e4–1e5 sims. Generation is cheap
  (0.3–3 ms/sim) — at 0.3 ms/sim, 1e5 sims is ~30 s.

---

## 5. Do not re-litigate these

Recorded so the next person does not spend a week rediscovering them.

- **numba / flash attention does not help.** A fused numba kernel (no score
  matrix, `prange`, `fastmath`) was measured *slower* than NumPy: 0.4x at
  T=450. Roughly 70% of softmax attention's cost is the `T^2` transcendentals,
  not the matmuls; NumPy's `exp` is SIMD and its matmuls hit BLAS. Flash
  attention saves memory (which was never binding — 104 MB) and I/O, not
  FLOPs. Separately, this repo's root README documents that numba array
  kernels need an `RTLD_GLOBAL` patch of the pyjs runtime to work under wasm
  at all.
- **Use linear attention for long token sequences.** At 450 tokens, fwd+bwd
  is 35.6 ms vs 271.3 ms (7.6x) and peak memory 142 MB vs 1055 MB.
  **Write every contraction as reshape + `@`, not `np.einsum`** — the einsum
  version was only 2.6x faster than softmax; the BLAS rewrite added a further
  5.5x at identical results.
- **The graph prior belongs in `K`/`V`, not in the logits**, for linear
  attention (there is no score matrix). `K, V <- M @ K, M @ V` over the node
  axis with `M = row_norm(A + I)`. Also: include the self-loop. Zeroing the SC
  diagonal (which `make_network` does) makes the bias actively push attention
  away from a node's own tokens, which cost 1.6 nats before it was spotted.
- **Widening the conditioning path makes things worse, not better.** Pool at
  d=32 → −5.33; d=64 → −3.70; d=128 → −4.84. This is an optimisation
  conditioning problem (the `alpha` head saturates), not a capacity problem.
  Do not chase it with more width.
- **The single-node Jansen-Rit toy cannot test attention.** 7 Hz rhythm, 1.12
  cycles per STFT window, 90% of token variance in 2 of 374 components. If you
  want to test routing behaviour, use the coupled network or make tokens
  genuinely distinct first — and check with
  `coupling_scan.py` that cross-talk is actually non-trivial (cross/own ≥ 0.3).
- **Global parameters are the tie case.** `content/demo.ipynb` infers
  `a, tau, log_k, log_z`, all global. Pooling nodes is fine there. Attention
  becomes interesting only with per-region parameters.
- **Data budget beats architecture.** 1500 → 6000 sims moved attention by 13
  nats. Any arm compared at 1500 sims on this problem is measuring
  memorisation.

---

## 6. Two things noticed in passing

- `content/demo.ipynb` builds its features with
  `np.abs(np.fft.fft(x[-256:], axis=0).mean(axis=1))` — averaging **complex**
  coefficients across nodes *then* taking the magnitude, so random per-node
  phase cancels. On a synthetic 90-node 10 Hz signal: coherent-average
  peak/median 16x, correct `abs`-then-mean 54x, mean power 2290x. The
  "average fft" features are largely degenerate.
- In the same notebook the connectivity and delays are re-randomised per batch
  and never enter `features`, so the CDE learns a marginal over SC. Conditioning
  on that known would sharpen posteriors more than any encoder change.
