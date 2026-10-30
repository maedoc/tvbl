# STFT → self-attention → MADE, vs MADE alone

Toy Jansen-Rit parameter recovery, to test the hypothesis: *replace hand-built
summary vectors (mean/std/PSD) with a learned encoder over a time-frequency
representation, and let self-attention do the pooling.*

**Status: implemented, gradient-verified, benchmarked. Attention is not yet
earning its cost at this data scale — the learned low-dimensional bottleneck
is, and that is the robust finding.**

## What was built

| file | contents |
| --- | --- |
| `content/tvbl/attn.py` | `AttnEncoder` (pre-LN transformer, hand-written backward) and `AttnMAFEstimator` (composes encoder + existing `MAFEstimator`) |
| `content/tvbl/test_attn.py` | 13 tests: backward vs central finite differences, normalisation, end-to-end fit |
| `experiments/attn_cde/jr_toy.py` | Jansen-Rit simulation + log-power STFT tokenisation |
| `experiments/attn_cde/benchmark.py` | three-arm comparison with calibration diagnostics |

The only change to existing code is one optional keyword on
`MAFEstimator._flow_backward_np(..., compute_ctx_grad=True)`, which
additionally returns `dL/dctx` (per-flow adjoints sum, since every flow sees
the same context). The default path is unchanged; all 316 pre-existing tests
still pass.

### Why the composition is safe

The encoder is a deterministic function of the data, so
`log p(y|x) = log p(y|enc(x))` remains a proper density in `y`. Attention can
be made as deep as you like without touching the flow's normalisation
guarantee. This is tested by grid quadrature
(`test_attn_maf_log_prob_integrates_to_one`).

## Setup

- Jansen-Rit, 2 parameters: `A ~ U[3.0, 3.5]`, `a ~ U[90, 110]`, pyramidal
  `y1 - y2`, 200 Hz, 1 s (100 ms transient discarded).
- Tokens: log-power STFT, `nperseg=32, hop=8` → **22 frames x 17 bands**
  (6.25 Hz spacing). 374 dims when flattened.
- Same MADE flows in every arm (`n_flows=4, hidden_units=64`); only the
  conditioning pathway differs.
- Minibatch Adam (`bs=64`, `lr=1e-3`, global grad-norm clip 5.0), 12000
  iterations, 4000 train / 600 test, 2 model seeds.

```sh
PYTHONPATH=content python3 -m pytest content/tvbl/test_attn.py -q
PYTHONPATH=content python3 experiments/attn_cde/benchmark.py \
    --n-train 4000 --n-test 600 --iters 12000 --lrs 1e-3 --n-repeats 2
```

## Results

`cov90` should be 0.90, `z-spread` 1.0, `|z-bias|` 0.0.

| arm | params | test NLL | z-spread | \|z-bias\| | cov90 | ms/iter |
| --- | --- | --- | --- | --- | --- | --- |
| `made-raw` (MAF on 374 flat dims) | 103,536 | −4.01 ± 0.76 | 1.09 | 0.83 | 0.752 | 1.0 |
| `linear-pool` (d=32) | 11,856 | **−5.33 ± 0.39** | 0.88 | 0.33 | 0.915 | 1.0 |
| `attn-2blk` (d=32) | 28,944 | **−5.42 ± 0.61** | 0.77 | 0.20 | 0.948 | 7.5 |

Bottleneck-width ablation on the cheap arm (`linear-pool`, 2 seeds):

| d_model | params | test NLL |
| --- | --- | --- |
| 32 | 11,856 | −5.33 ± 0.39 |
| 64 | 21,904 | −3.70 ± 0.06 |
| 128 | 42,000 | −4.84 ± 0.22 |

## Findings

1. **The learned bottleneck is the win, and it is large and reproducible.**
   Feeding MADE a 32-d learned summary instead of 374 raw dims buys ~1.3 nats
   of held-out NLL with 3.6-8.7x fewer parameters. This confirms the premise:
   `W1c`/`W2c` as a single affine+tanh projection of a wide feature vector is
   the real weakness of `MAFEstimator`, not the flow head.

2. **Attention does not beat a linear projection at this scale.** ΔNLL = 0.09
   against seed noise of ±0.4-0.6, at 7.5x the compute. Attention is better
   centred (`|z-bias|` 0.20 vs 0.33) and better covered (0.948 vs 0.915), but
   its marginals are too wide (`z-spread` 0.77).

3. **It is an optimisation problem, not a capacity problem.** Widening the
   context degrades results (d=32 > d=128 > d=64; 374 raw dims is worst of
   all). More conditioning width makes the `alpha` head saturate and the
   gradient signal worse. Attention at d=32 is not "more capacity" than a
   linear map at d=32 — it is a better-conditioned path to the same width.

4. **`MAFEstimator`'s context pathway is numerically fragile.** At `lr=3e-3`
   the raw arm does not merely overfit, it fails to train (−0.55 vs −5.03 at
   `lr=1e-3`); with clipping disabled it diverges to ~1e17. Any future work
   should treat `lr` and clipping as first-class parts of the method.

## Caveats — read before quoting numbers

- **Test NLL is not monotone in iterations** for any arm: `attn-2blk` reaches
  −6.02 at 12k but −5.83 at 30k in a different run. Absolute values depend on
  where you stop; the encoders show no train/test gap, so this is optimiser
  noise near saturation, not overfitting.
- Single-dataset, single-toy, 2 seeds. The *ordering* of the two encoder arms
  is not established; the ordering of "encoder > raw high-dim" is.
- Calibration is imperfect everywhere, worst for the raw arm. Posteriors should
  be validated with SBC before use, not just NLL.
- `z-spread`/`cov90` are marginal-only; they cannot see whether the joint
  (correlated) posterior shape is right, which is where a flow should shine.

## Suggested next steps

1. **Fix the optimiser before adding capacity**: cosine or step LR decay, and
   longer warmup. Given finding 4, this is likely worth more than architecture.
2. **Widen the toy data**: 4k sims is too few to separate a 29k-param encoder
   from a 12k-param one. The attention result should be re-run at 10^4-10^5
   sims with matched epochs, which is where its extra capacity should pay.
3. **Then** try the architecturally interesting step the flow head makes
   possible and a pooled vector does not: cross-attention from `param_dim`
   queries into *context tokens*, replacing `c_h = ctx @ W1c.T`. Still exact
   likelihood, still invertible, and it lets each parameter dim read a
   different part of the data.
4. **Condition on the known nuisance variables.** In `content/demo.ipynb` the
   connectivity and delays are resampled per batch and never enter the
   features, so the CDE learns a marginal over SC. An adjacency bias on the
   attn logits is the natural fix and should sharpen posteriors more than the
   encoder does.
5. Regional (per-node) parameters would justify the spatial attention stage;
   for global params, pooling nodes is defensible and much cheaper.

---

# Part 2 — multi-node: tokens that are *not* interchangeable

Part 1's negative result had a specific cause: the STFT tokens of a 7 Hz
narrowband rhythm were **rank-2 redundant** (90% of variance in 2 of 374
components), so there was no routing problem for attention to solve. Part 2
builds the opposite regime.

## Setup (`multinode.py`, `benchmark_multinode.py`)

* `num_node=8` oscillators (`tvbl.core` FitzHugh-Nagumo) coupled through a
  fixed sparse SC with conduction delays.
* One **regional** parameter `a_i` per node, `param_dim = 8`, `a in [0.35, 0.95]`
  (the `|a| < 1` limit-cycle regime; outside it the system just relaxes to a
  fixed point and the parameter is only in the DC level).
* 401 samples (~5 cycles), STFT `nperseg=128, hop=64` → 5 frames x 65 bands
  per node → **40 tokens** (8 nodes x 5 frames), 2600 dims flattened.
* Same MADE flow in every arm (`n_flows=4, hidden_units=64`), 1500/300
  train/test, 8000 iters, `bs=64`, `lr=1e-3`, clip 5.

Tokens are now informative and distinct:

| diagnostic | value |
| --- | --- |
| PCA: components for 90% of token variance | **93 of 2600** (vs **2 of 374** for JR) |
| `corr(std_i, a_i)` per node | **−0.85 … −0.92** |
| `corr(std_i, a_j)`, `i != j` | **≈ 0** (−0.04 … +0.09) |

Each node's tokens carry *its own* parameter and essentially nothing about the
others. Uniform mean-pooling therefore mixes eight different parameters
together — the routing problem is real.

## Results (test NLL, 2 seeds)

| arm | params | test NLL | z-spread | cov90 | ms/iter |
| --- | --- | --- | --- | --- | --- |
| `made-raw` | 838,464 | +0.18 | 1.05 | 0.85 | 3.8 |
| `linear-pool` | 20,160 | −5.81 | 1.05 | 0.86 | 1.1 |
| `attn-1blk` | 28,704 | **−10.35** | 1.20 | 0.80 | 6.4 |
| `attn-graph` (self=2) | 28,705 | **−10.52** | 1.20 | 0.79 | 6.9 |

**Attention now beats pooling by ~4.5 nats**, consistently across both seeds
(−10.01/−10.69 vs −5.68/−5.94), at 6x the compute and 1/29 the parameters of
`made-raw`. In the JR toy the same comparison was a tie. The only thing that
changed is whether the tokens were interchangeable.

## The graph bias is a small effect, and was initially harmful

Ablating the self-loop weight in `graph_bias` (seed 0):

| bias | test NLL |
| --- | --- |
| `self_weight=0` (SC only, diagonal zeroed) | −9.14 |
| `self_weight=1` | −9.66 |
| `self_weight=2` | −10.78 |
| none (free attention) | −10.01 |

Two lessons:

1. Zeroing the SC diagonal was a bug in the *prior*, not the idea. Without a
   self-loop the bias down-weights a node's own tokens in favour of its
   neighbours' — backwards for a regional parameter.
2. Even fixed, the SC bias buys ~0.2–0.8 nats, i.e. within seed noise
   (−10.78/−10.26 for `attn-graph` vs −10.01/−10.69 for free attention). That
   is consistent with the diagnostic above: since node effects are nearly
   independent, *which* neighbours you attend to carries little information.
   The graph bias should only pay when parameters are entangled through the
   graph.

## Caveats

* Attention is **overconfident** here: `cov90` 0.79–0.82 (ideal 0.90) and
  `z-spread` ~1.2. Better NLL, worse calibration — don't ship without SBC.
* `param_dim=8` with 1500 sims: `attn` has a ~2.5 nat train/test gap. More
  sims is the obvious next step.
* Only one network realisation (`seed=0` for `make_network`); the SC is not
  resampled, so there is no SC nuisance here.

## What this implies for the real problem

Attention is worth its cost exactly when each parameter is read off a
different part of the data. That maps onto regional parameters (per-node
biophysical gains) much better than onto global ones (`a, tau, k, z` in
`content/demo.ipynb`), where pooling nodes is fine and the encoder bottleneck
is the only thing that matters.

---

# Part 3 — coupling sweep and data scaling

## Where does the graph start to matter? (`coupling_scan.py`)

Global coupling scale `g` multiplying a **fixed** topology; 600 sims per point.
`own` = mean over nodes of `max_f |corr(feature_f(node_i), a_i)|`, `cross` = the
same for `a_j, j != i`.

| `g` | own | cross | cross/own | comment |
| --- | --- | --- | --- | --- |
| 0.1 | 0.980 | 0.066 | 0.067 | effectively independent nodes |
| 1.0 | 0.982 | 0.137 | 0.140 | |
| 2.0 | 0.963 | 0.176 | 0.182 | |
| 4.0 | 0.911 | 0.230 | 0.252 | |
| **6.0** | **0.810** | **0.278** | **0.343** | chosen operating point |
| 8.0 | 0.662 | 0.313 | 0.473 | near the edge |
| 12.0 | — | — | — | **non-finite** (system blows up) |

Cross-talk rises monotonically as the system approaches its stability boundary,
so there is a genuine critical regime rather than a tuned coincidence. `g=6`
was chosen as the highest value that still simulates reliably.

## g = 6, 1500 sims / 8000 iters — the task is much harder

| arm | train NLL | test NLL | train−test gap |
| --- | --- | --- | --- |
| `made-raw` | −0.07 | +18.72 | 18.8 |
| `linear-pool` | −1.15 | +2.40 | 3.6 |
| `attn-1blk` | −7.76 | +0.30 | **8.1** |
| `attn-graph` (self=0) | −6.81 | +1.08 | 7.9 |

Everything degrades relative to `g=0.1`, because entangling the parameters
through the graph makes the posterior substantially harder. Attention in
particular is memorising: an 8-nat train/test gap.

## g = 6, 6000 sims / 32000 iters (epoch-matched) — data fixes it

| arm | params | test NLL | z-spread | \|z-bias\| | cov90 | gap |
| --- | --- | --- | --- | --- | --- | --- |
| `made-raw` | 838,464 | +7.00 | 0.90 | 0.74 | 0.81 | 1.1 |
| `linear-pool` | 20,160 | −4.51 | 1.04 | 0.33 | 0.85 | 0.6 |
| `attn-1blk` | 28,704 | **−12.81** | 1.12 | 0.21 | 0.84 | 1.25 |
| `attn-graph` (self=2) | 28,705 | **−12.82** | 1.11 | **0.10** | **0.86** | 1.18 |

Two things happen:

1. **4x the simulations moves attention from +0.30 to −12.81 test NLL** and
   collapses its overfitting gap from 8.1 to 1.25 nats. The 1500-sim result was
   measuring memorisation, not representation quality — the earlier conclusion
   would have been wrong.
2. **Attention beats pooling by 8.3 nats** at `g=6`, versus 4.5 nats at
   `g=0.1`. The more the graph entangles the parameters, the more the routing
   matters.

## The graph bias helps calibration, not likelihood

At `g=6` / 6000 sims, `attn-graph` and `attn-1blk` are indistinguishable on
NLL (−12.815 vs −12.814) *and* on parameter count (28,705 vs 28,704 — the bias
adds exactly one scalar). But the bias halves the systematic error:
`|z-bias|` 0.21 -> 0.10 and `cov90` 0.84 -> 0.86.

That is a coherent story: the SC prior does not tell the model anything new
about *how likely* a parameter is, but it does tell it *where to look*, which
centres the posterior. It costs one scalar and a matrix you already have.

## Practical takeaways

* **Data budget dominates everything.** At `g=6`, 1500 sims overfits by 8 nats
  and 6000 sims is nearly fine. Any architecture comparison at 1500 sims here
  is meaningless.
* Validate the regime first: `coupling_scan.py` costs seconds and found both
  the informative operating point and the blow-up edge.
* `cov90` is now 0.84-0.86 across the encoder arms (ideal 0.90) — usable but
  still worth an SBC pass before trusting posteriors.
* Attention's advantage is bought with regional structure. For the *global*
  parameters in `content/demo.ipynb`, pooling nodes remains the right call.

---

# Part 4 — making 90 nodes affordable: numba/flash vs linear attention

At the real problem size (90 nodes x 5 frames = 450 tokens) the softmax encoder
cost 271 ms/iter forward+backward (B=64, 1 block, d=32) and 1.05 GB peak. That
is the blocker for shipping. Two candidate fixes were measured.

## Where the time actually goes

| T | full softmax attn | `QK^T` | `+exp` | exp share |
| --- | --- | --- | --- | --- |
| 40 | 0.5 ms | 0.0 | 0.4 | ~65% |
| 90 | 2.2 ms | 0.2 | 1.8 | ~75% |
| 450 | 57.4 ms | 7.7 | 46.9 | ~68% |

**~70% of softmax attention's cost is the `T^2` transcendentals, not the
matmuls.** That is why the naive fix (make it faster) and the right fix
(remove the `T^2`) diverge.

## numba / flash attention: measured *slower*

A fused numba kernel (scores + max + exp + `PV` in one loop, no `T x T`
materialisation, `prange`, `fastmath`) versus plain numpy:

| T | numpy softmax | numba fused | gain |
| --- | --- | --- | --- |
| 40 | 0.6 ms | 0.8 ms | 0.7x |
| 90 | 2.2 ms | 4.5 ms | 0.5x |
| 450 | 59.2 ms | 137.9 ms | **0.4x** |

Do not pursue this.

* Flash attention saves **memory and I/O**, not FLOPs. Memory was never the
  binding constraint (104 MB of score matrix in 8 GB).
* numpy's `exp` is SIMD-vectorised and its matmuls hit BLAS; a hand-written
  scalar-loop kernel competes badly. A highly tuned, tiled kernel could
  plausibly recover 1.5-2x, still short of removing the `T^2` term.
* Separately, this repo already documents (root `README.md`) that numba array
  kernels need an `RTLD_GLOBAL` patch of the pyjs runtime to work at all under
  wasm — a fragile dependency for the in-browser path.

## Linear attention: 7.6x and 7x less memory

`attn_type='linear'` uses the `elu(x)+1` feature map with the associativity
trick, so there is no score matrix at all — `O(T*d^2)` time, `O(d^2)` state.

| T | softmax fwd+bwd | linear fwd+bwd | speedup | linear peak mem | 32k iters |
| --- | --- | --- | --- | --- | --- |
| 40 | 4.7 ms | 3.0 ms | 1.6x | — | 1.6 min |
| 90 | 14.5 ms | 6.8 ms | 2.2x | — | 3.6 min |
| **450** | **271.3 ms** | **35.6 ms** | **7.6x** | **142 MB** (vs 1055 MB) | **19 min** |

The implementation detail matters as much as the algebra: the first version
used `np.einsum` and was only 2.6x faster. Rewriting every contraction as an
explicit reshape + `@` (BLAS) gave a further **5.5x** with identical results
(`max|diff| = 4e-8`). Gradients are verified against central finite
differences in `test_attn.py`.

So 90 nodes x 5 frames goes from ~2.4 h to ~20-30 min per 32k-iteration run.

## What linear attention costs in accuracy

At `g=6`, 6000 sims, 32000 iters, 8 nodes (40 tokens):

| arm | test NLL | z-spread | \|z-bias\| | cov90 | ms/iter |
| --- | --- | --- | --- | --- | --- |
| `linear-pool` | −4.51 | 1.04 | 0.33 | 0.85 | 1.1 |
| `attn-1blk` (softmax) | **−12.81** | 1.12 | 0.21 | 0.84 | 7.07 |
| `attn-linear` | −12.14 | 1.07 | 0.20 | **0.86** | **4.49** |

Linear attention gives up **0.67 nats** but is 1.6x faster, better calibrated,
and still **+7.6 nats over pooling**. Reasonable trade; at 450 tokens it is not
a trade at all.

## Scale probe at 30 nodes (150 tokens)

`g=4` (the 30-node cross-talk 0.370 matches the 8-node `g=6` value; `g=6`
blows up). Timing scales as expected — `attn-linear` 16.3 ms/iter,
`attn-1blk` 43.5 ms/iter (2.7x), `linear-pool` 2.6 ms/iter.

Accuracy is **not** established at this size: with 30 parameters and only 2000
sims / 6000 iters every arm is badly underfit (test NLL +44 to +60, `shrink`
0.1-0.2, i.e. posteriors barely tighter than the prior) and attention is worse
than pooling. That is a data-budget failure, not an architecture result — the
same lesson as Part 3, where 1500 -> 6000 sims moved attention by 13 nats.
`param_dim` in the tens needs 10^4-10^5 simulations.

## Updated verdict

Blocker 1 is **downgraded from decisive to tractable**: the cost at 90 nodes is
now ~20-30 min per run rather than hours, in pure NumPy with no numba, so the
wasm path is untouched. Remaining before writing the library class:

1. **wasm timing** — still unmeasured, and now cheap to measure since the wasm
   path stays pure-numpy.
2. **calibration** — `cov90` 0.86 at 40 tokens; needs the lr-decay + SBC pass.
3. **accuracy at 450 tokens** — needs the 10^4-10^5 simulation budget, which is
   a data-generation question (0.3-3 ms/sim, so 10^5 sims is minutes).
