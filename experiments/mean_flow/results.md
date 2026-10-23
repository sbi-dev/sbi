# Mean flow in sbi: results (2026-10-22 and 2026-10-23)

Question: is a conditional mean flow (see `paper_summary.md`) a useful density estimator in sbi, above all as an NLE likelihood compared with sbi's NSF?

Short answer: at the paper's data scale it **ties NSF in NLE** when both get the same training recipe, is **1.8x slower per new observation** in MCMC, trains 9x longer, and is **worse in NPE**. At 2,000 simulations it beats NSF in NLE on 3 of 4 mini-sbibm tasks. The paper's g-and-k result (mean flow 0.577 vs coupling flow 0.679) is the one claim left to reproduce with a fair baseline; see `gandk_reproduction_plan.md`.

## Implementation

`sbi/neural_nets/estimators/mean_flow_estimator.py` and `sbi/neural_nets/net_builders/mean_flow.py`, selected with `MeanFlowConfig` (registered as `"mean_flow"` for the mini-sbibm harness). It implements the paper's selected configuration: JVP target, auxiliary velocity head with its own last 2 blocks, adaptive weight, power-law times with 70% $t=s$, cycle loss with condition swap, EDM schedule with $\rho=7$, $\sigma_{\min}=10^{-3}$, exact S-step density, backward sampling. `log_prob` evaluates rows in chunks of 1,024.

Differences to the paper's code (`mean_flow/mean_flow.py` in their repository):

| Item | Paper code | sbi branch |
|---|---|---|
| Cycle-loss times | own pair, no collapse to $t=s$ | same since commit "fix: draw separate times for the mean-flow cycle loss"; **all results below were produced before that fix**, with the cycle reusing the mean-flow pairs (active on 30% of each batch) |
| Time input | $(s, s-t)$, Fourier features plus the raw times | $(s, t-s)$, Fourier features only |
| Backbone | BayesFlow `TimeMLP` with FiLM blocks, 5 × 256 | sbi `AdaMLPBlock`s (adaptive layer norm), 5 × 256 (3 shared, 2 per head) |
| Output layers | default initialization | zero-initialized (an untrained flow is the identity) |
| Condition swap in the cycle | shift by one row | random permutation |

## Training regime: the deciding factor

With sbi's trainer defaults (constant learning rate, early stopping after 20 epochs without improvement) the mean flow stops after about 41 to 77 epochs and is far behind. The paper's recipe (AdamW with weight decay 5e-3, learning rate warmed up over 5% of the steps to 5e-4, then cosine decay to zero, gradient clipping at 1.5) fixes that. Two moons NPE, 2,000 simulations, C2ST on 5 observations (`results/training_variant_*.log`):

| Variant | S=1 | S=2 | S=4 | velocity head, 200 Euler steps |
|---|---|---|---|---|
| sbi trainer, defaults (early stop at 41 epochs; 3 observations, first quick run) | 0.914 | 0.920 | 0.916 | n/a |
| sbi trainer, lr 1e-4, 600 epochs | 0.863 | 0.716 | 0.693 | 0.648 |
| paper recipe, 500 epochs | 0.798 | 0.716 | 0.637 | 0.549 |
| paper recipe, 500 epochs, 10,000 simulations | 0.681 | 0.710 | 0.600 | 0.522 |

References at 2,000 simulations: NSF 0.57 to 0.63, sbi FMPE with defaults 0.781.

At 2,000 simulations the paper's fixed epoch count overfits without model selection: on gaussian_linear the held-out loss goes 7.46 (epoch 25), 7.36 (100), 10.78 (200), 36.64 (400) while the training loss falls to 2.6, and the NPE C2ST ends at 0.995. All runs below therefore keep the best epoch on a 10% held-out split (fixed noise per evaluation), for NSF and the mean flow alike (`scripts/mean_flow_paper_trainer.py`).

## mini-sbibm at 2,000 simulations

3 seeds, 10 observations per task, default MCMC (slice sampler, 20 chains, 200 warmup), C2ST against the reference posteriors. Each estimator trained with sbi defaults and with the paper recipe; the table shows the better one per estimator. Full table: `results/summary.txt`; raw rows: `results/grid1.jsonl`, `results/grid2.jsonl`, merged in `results/merged.json`.

**NLE** (C2ST, mean ± sd over seeds; MCMC time for 10 observations at S=1 relative to NSF):

| Task | NSF | Mean flow S=1 | Mean flow S=2 | MCMC time |
|---|---|---|---|---|
| two_moons | **0.584 ± 0.003** | 0.653 ± 0.052 | 0.640 ± 0.046 | 1.9x |
| linear_mvg_2d | 0.519 ± 0.005 | **0.506 ± 0.006** | 0.509 ± 0.002 | 1.7x |
| gaussian_linear | 0.643 ± 0.014 | **0.599 ± 0.009** | 0.599 ± 0.011 | 2.2x |
| slcp | 0.905 ± 0.004 | **0.851 ± 0.006** | 0.848 ± 0.005 | 2.2x |

The clearest gain is the posterior width: std error 0.060 vs 0.109 on gaussian_linear, 0.444 vs 0.615 on slcp (mean error 0.455 vs 0.756). Training takes 5 to 9x longer than NSF. With sbi's trainer defaults the mean flow is still ahead on gaussian_linear (0.601) and slcp (0.861) but fails on two moons (0.80).

**NPE:** no benefit. Sampling is sub-second for both; accuracy is equal or worse and needs S=4 on two moons (0.647 vs NSF 0.622; S=1 gives 0.844).

## Density and sampler consistency (two moons, 2,000 simulations)

`scripts/diagnostics.py`, paper recipe with checkpointing (`results/diagnostics2.json`; `diagnostics1.json` is the same without checkpointing):

| | Round trip, median / 99th pct. | Mass over 20 conditions, mean ± sd (range) | Direct samples vs MCMC on `log_prob`, C2ST |
|---|---|---|---|
| NLE S=1 | 0.049 / 0.151 | 1.083 ± 0.023 (1.04 to 1.11) | n/a |
| NLE S=2 | 0.022 / 0.079 | 1.017 ± 0.006 (1.01 to 1.03) | n/a |
| NLE S=4 | 0.015 / 0.070 | 1.003 ± 0.002 (1.00 to 1.01) | n/a |
| NPE S=1 | 0.049 / 0.470 | 1.010 ± 0.149 (0.49 to 1.22) | 0.65 to 0.73 |
| NPE S=4 | 0.018 / 0.604 | 0.967 ± 0.198 (0.25 to 1.13) | 0.60 to 0.63 |

The NLE density is close to normalized and nearly the same for every condition, which is what MCMC needs. The NPE density is not consistent with the samples. At S=1 the ±3.5% θ-dependence of the NLE mass adds up over i.i.d. trials (about 3.5 nats at 100 trials), so S=2 is the safer default.

## Paper scale: two moons, 50,176 simulations, 1,000 epochs

3 seeds, 10 observations, paper recipe with best-epoch selection for both estimators, plus NSF with sbi defaults (`results/scale.jsonl`, `results/scale_summary.txt`):

| | NSF (sbi defaults) | NSF (paper recipe) | Mean flow S=1 | S=2 | S=4 |
|---|---|---|---|---|---|
| NLE C2ST | 0.543 ± 0.012 | **0.527 ± 0.010** | **0.528 ± 0.005** | 0.535 ± 0.003 | 0.527 ± 0.006 |
| NLE MCMC, 10 observations | 87 s | 87 s | 157 s | 283 s | 542 s |
| NPE C2ST | 0.530 ± 0.003 | **0.518 ± 0.001** | 0.578 ± 0.009 | 0.573 ± 0.023 | 0.552 ± 0.007 |
| Training (CPU, 2 threads, 6 jobs in parallel) | 3 min | 30 min | 268 min | | |

The paper reports NPE S=1 C2ST 0.523 against its coupling flow 0.543. Here sbi's NSF, also a spline coupling flow, reaches 0.518 to 0.530 and the mean flow 0.578. The C2ST classifiers differ (sbi's `c2st` here, the paper's own MLP classifier there), so absolute values compare only roughly; the cycle-loss deviation above may explain part of the NPE gap.

## Effect of the cycle-loss fix (two moons, 2,000 simulations, one seed)

After matching the paper's code (separate cycle times, see the differences table), one rerun with the paper recipe and checkpointing (`results/cyclefix.jsonl`, `results/diagnostics_cyclefix.json`):

| | Before the fix | With the fix |
|---|---|---|
| NPE C2ST, seed 0, S = 1 / 2 / 4 | 0.879 / 0.716 / 0.653 | **0.753 / 0.695 / 0.638** |
| NLE mass at S=1 over 20 conditions | 1.083 ± 0.023 | **0.994 ± 0.006** |
| NLE round trip at S=1, 99th pct. | 0.151 | **0.097** |
| NPE round trip at S=1, 99th pct. | 0.470 | **0.213** |
| NPE mass at S=1 (range) | 0.49 to 1.22 | 0.37 to 1.03 |
| NPE direct samples vs MCMC on `log_prob`, C2ST | 0.65 to 0.73 | 0.63 to 0.70 |

The fix helps few-step sampling (most at S=1) and makes the NLE normalization almost independent of θ. The NPE density stays inconsistent with the samples. The grids above ran without the fix, so they understate the mean flow somewhat, most at S=1; the g-and-k reproduction uses the fixed estimator.

## Bugs in existing sbi found on the way

Both are fixed on branch `fix/sample-shapes-and-mcmc-init-grad` (PR pending) and in the first two commits of this branch.

- `ConditionalDensityEstimator.sample_and_log_prob` passed samples with several sample dimensions to `log_prob`, which takes one. With `sample_shape=(2, 3)` and more than one condition the MDN raised a broadcasting error. Affects the MDN, the MNLE estimator and the categorical estimator.
- The `resample` and `sir` MCMC init strategies evaluate 10,000 candidates per chain inside `torch.no_grad()`, but sbi potentials default to `track_gradients=True` and turn gradients back on. Every MCMC init built an unused autograd graph: NSF (128 hidden features) on SLCP needed 335 MB extra peak memory instead of 46 MB; the mean flow 5.4 GB instead of 0.6 GB. Results are bit-identical with the fix.

## Mistakes in our own benchmark (fixed, recorded so they are not repeated)

- `build_posterior` deep-copies the estimator (`sbi/inference/trainers/base.py:623`). Setting `num_steps` on the trained estimator afterwards does not reach the posterior. The first grid ran every mean flow at S=2; set `num_steps` before `build_posterior`.
- The paper recipe without model selection overfits at 2,000 simulations (above).

## Verdict

Park the mean flow in sbi unless the g-and-k reproduction shows a clear NLE gain over a fairly trained flow. With equal training it gives no accuracy gain over NSF on two moons at the paper's scale, costs 1.8x more MCMC time per new observation, and needs a cosine learning-rate schedule that sbi's `train()` does not offer.
