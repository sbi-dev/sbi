# Mean flow in sbi: results (2026-10-22 to 2026-10-30)

Question: is a conditional mean flow (see `paper_summary.md`) a useful density estimator in sbi, above all as an NLE likelihood compared with sbi's NSF?

Short answer: at the paper's data scale it **ties NSF in NLE** when both get the same training recipe, is **1.8x slower per new observation** in MCMC, trains 9x longer, and is **worse in NPE**. At 2,000 simulations it beats NSF in NLE on 3 of 4 mini-sbibm tasks. The paper's g-and-k result (mean flow 0.577 vs coupling flow 0.679) reproduces with the paper's code; with the coupling flow trained as long as the mean flow the gap halves (0.584 to 0.600 against 0.630) but holds, and sbi's port of the paper's network matches the paper's mean flow while sbi's own network does not (`results_gandk.md`). That does not carry over to other NLE tasks: at 50,176 simulations both networks tie NSF on three mini-sbibm tasks and lose on slcp, and at 2,000 sbi's network beats the paper's on every task (NLE benchmark below).

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

With sbi's trainer defaults (constant learning rate, early stopping after 20 epochs without improvement) the mean flow stops after about 41 to 77 epochs and is far behind. The paper's recipe (AdamW with weight decay 5e-3, learning rate warmed up over 5% of the steps to 5e-4, then cosine decay to zero, gradient clipping at 1.5) fixes that. Two moons NPE, 2,000 simulations, C2ST on 5 observations (`results/training_variant_*.txt`):

| Variant | S=1 | S=2 | S=4 | velocity head, 200 Euler steps |
|---|---|---|---|---|
| sbi trainer, defaults (early stop at 41 epochs; 3 observations, first quick run) | 0.914 | 0.920 | 0.916 | n/a |
| sbi trainer, lr 1e-4, 600 epochs | 0.863 | 0.716 | 0.693 | 0.648 |
| paper recipe, 500 epochs | 0.798 | 0.716 | 0.637 | 0.549 |
| paper recipe, 500 epochs, 10,000 simulations | 0.681 | 0.710 | 0.600 | 0.522 |

References at 2,000 simulations: NSF 0.57 to 0.63, sbi FMPE with defaults 0.781.

At 2,000 simulations the paper's fixed epoch count overfits without model selection: on gaussian_linear the held-out loss goes 7.46 (epoch 25), 7.36 (100), 10.78 (200), 36.64 (400) while the training loss falls to 2.6, and the NPE C2ST ends at 0.995. All runs below therefore keep the best epoch on a 10% held-out split (fixed noise per evaluation), for NSF and the mean flow alike (`scripts/mean_flow_paper_trainer.py`).

This rule does not carry over to the data-rich regime. On g-and-k (100,000 simulations, `results_gandk.md`) the mean flow's held-out loss bottoms out at epochs 233 to 373 and then rises slightly (3.29 to 3.30 to 3.32) while the training loss keeps falling, yet the last epoch gives clearly better posteriors: C2ST 0.573 against 0.649 at S=2 for the paper network, with best-epoch failures at 0.76 to 0.84 on 3 of 10 datasets. The mean-flow objective on held-out data is a poor proxy for density quality. The NLE benchmark below shows that the last epoch is not a safe alternative either: sbi's mean-flow network fails at the last epoch on gaussian_linear even with 50,176 simulations (C2ST 0.979 against 0.513 at the best epoch), while the paper network's last epoch is fine there. Best-epoch selection is the safe default; it costs the paper network 0.04 to 0.08 C2ST on slcp and g-and-k. For the flows, where the loss is the negative log-likelihood, best-epoch selection stays right: the wide NSF trained 1,000 epochs on g-and-k overfits and its last epoch does not sample (C2ST 0.958, no chain converged). A better criterion for the mean flow would be the held-out exact S-step log-likelihood (not tried).

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

## NLE benchmark: paper network, sbi network, single differences, NSF (2026-10-28)

Question: does the g-and-k result (the paper's network ahead of every flow, sbi's own network 0.06 behind it) hold on other NLE tasks, and which difference between the two networks matters? Script `scripts/nle_bench.py`; raw rows `results/nle_bench.jsonl`; every table, including S=1 and all paired differences, in `results/nle_bench_summary.txt` (made by `scripts/nle_bench_summary.py`).

Setup: mini-sbibm, 10 observations per task, sbi's default MCMC, C2ST against the reference posteriors, as in the 2,000-simulation grid above. 2,000 simulations with 400 epochs (CPU) and 50,176 simulations with 1,000 epochs (mean flows on GPU 1), 3 seeds each; the single-difference arms at 50,176 with one seed. Mean flows and `nsf_paper` train with the paper's recipe and are scored with both the best held-out and the last-epoch weights; `nsf_default` trains with `NLE.train()` defaults. All mean flows include the cycle-loss fix.

- `mf_paper`: `MeanFlowConfig(variant="paper")`, the port of the paper's FiLM network (1.76M parameters), which matched the paper's code on g-and-k.
- `mf_sbi`: `MeanFlowConfig()`, sbi's adaptive-layer-norm network (2.85M parameters).
- `abl_time`, `abl_actinit`, `abl_zeroout`: the paper network with one difference switched to sbi's choice: time input $(s, t-s)$ through fixed Fourier features; GELU with PyTorch's default initialization; zero-initialized output layers.
- `nsf_default`, `nsf_paper`: `NSFConfig()` (61k to 98k parameters).

C2ST at S=2, mean ± sd over 3 seeds (0.5 is perfect; paired differences below about 0.02 are mostly noise):

| 2,000 simulations | two_moons | linear_mvg_2d | gaussian_linear | slcp |
|---|---|---|---|---|
| `mf_paper` best | 0.685 ± 0.017 | 0.524 ± 0.005 | 0.750 ± 0.009 | 0.943 ± 0.005 |
| `mf_paper` last | 0.666 ± 0.005 | 0.519 ± 0.006 | 0.884 ± 0.010 | 0.954 ± 0.003 |
| `mf_sbi` best | 0.610 ± 0.038 | 0.516 ± 0.013 | **0.616 ± 0.016** | **0.856 ± 0.005** |
| `mf_sbi` last | **0.580 ± 0.008** | **0.503 ± 0.003** | 0.968 ± 0.006 | 0.863 ± 0.004 |
| `nsf_default` | 0.612 ± 0.008 | 0.517 ± 0.003 | 0.647 ± 0.017 | 0.903 ± 0.006 |
| `nsf_paper` best | 0.597 ± 0.013 | 0.518 ± 0.012 | 0.655 ± 0.015 | 0.913 ± 0.007 |

| 50,176 simulations | two_moons | linear_mvg_2d | gaussian_linear | slcp |
|---|---|---|---|---|
| `mf_paper` best | 0.547 ± 0.008 | 0.505 ± 0.003 | 0.531 ± 0.001 | 0.721 ± 0.015 |
| `mf_paper` last | 0.552 ± 0.005 | 0.502 ± 0.002 | 0.542 ± 0.003 | 0.685 ± 0.013 |
| `mf_sbi` best | 0.533 ± 0.007 | 0.500 ± 0.003 | **0.513 ± 0.003** | 0.703 ± 0.006 |
| `mf_sbi` last | 0.534 ± 0.011 | 0.500 ± 0.003 | 0.979 ± 0.008 | 0.761 ± 0.012 |
| `nsf_default` | 0.535 ± 0.004 | 0.503 ± 0.002 | 0.529 ± 0.002 | **0.651 ± 0.005** |
| `nsf_paper` best | 0.528 ± 0.003 | 0.504 ± 0.004 | 0.528 ± 0.004 | 0.657 ± 0.017 |

Findings (paired differences over seeds and observations, mean with standard error; negative means the first estimator is better):

1. **The mean flow does not beat NSF with plenty of data.** At 50,176, `mf_sbi` (best epoch) ties `nsf_default` on the two 2-D tasks, is 0.016 (0.003) better on gaussian_linear and 0.052 (0.016) worse on slcp. `mf_paper` is behind NSF on two moons (+0.012 to +0.017) and slcp (+0.034 at the last epoch, +0.070 at the best). At 2,000, `mf_sbi` beats NSF on gaussian_linear (−0.031, 0.008) and slcp (−0.047, 0.015) and ties on the 2-D tasks, which repeats the earlier grid; `mf_paper` is behind NSF on every task.
2. **The paper's network is not better in general.** sbi's network beats it at the best epoch on every task at 2,000 (−0.075, −0.009, −0.134, −0.086) and by 0.014 to 0.018 at 50,176 (two moons, gaussian_linear, slcp). On g-and-k the order was the other way round (`results_gandk.md`), so that result is task-specific.
3. **Single differences (2,000, best epoch, against `mf_paper`):** sbi's activation and initialization give the largest single gain: −0.057 (0.008) of the −0.134 gap on gaussian_linear, −0.044 (0.005) of −0.086 on slcp, −0.013 (0.009) of −0.075 on two moons. Zero-initialized outputs help on gaussian_linear (−0.057, 0.004) and slcp (−0.014, 0.004) but hurt on two moons (+0.030, 0.005). The time input changes nothing beyond noise. No single switch closes the gap; the rest is FiLM against adaptive layer norm, the larger network, or interactions. At 50,176 (one seed) all single differences are within ±0.02 except zero-initialized outputs on slcp (+0.039, 0.017 at the best epoch).
4. **Neither epoch rule is safe.** Where the held-out loss bottoms out early (gaussian_linear: best epoch 10 to 124 of 400 at 2,000, 163 to 377 of 1,000 at 50,176), the last epoch fails: `mf_sbi` 0.968 and 0.979, `nsf_paper` 0.752 and 0.702, `mf_paper` 0.884 at 2,000 (but 0.542 at 50,176). Where it bottoms out late, the last epoch is equal or better: `mf_paper` on slcp at 50,176 (−0.036, 0.008), as on g-and-k (−0.076). So the g-and-k rule "last epoch for the mean flow when data is plentiful" holds for the paper network but fails for sbi's network on gaussian_linear even at 50,176 simulations. Best-epoch selection is the safe default; it costs the paper network 0.04 to 0.08 on slcp and g-and-k.
5. **S=1 is enough in NLE.** For `mf_sbi` at the best epoch, S=1 against S=2 is within ±0.004 at 2,000 and within ±0.003 at 50,176 except slcp (S=1 0.017 worse, 0.009). S=1 halves the MCMC cost.
6. **Cost** (medians over tasks and seeds; the machine was fully loaded, so ratios are rough): at 2,000 on CPU, mean-flow training took 640 to 910 s against 46 s for `nsf_default`; MCMC for 10 observations took 1,180 to 1,600 s at S=1 and 2,220 to 2,730 s at S=2, against 864 s for NSF (1.4 to 1.9x at S=1, 2.6 to 3.2x at S=2). At 50,176, the mean flows trained for 3.7 to 5 h on the GPU (12 jobs sharing it) against 12 min for `nsf_default` on 2 CPU threads.

## What this means for the paper

Our results do not contradict the paper; they narrow it. The paper makes two different claims.

**Claim A: few-step mean flows give flow-matching-quality densities at a cost that MCMC can afford.** This is the paper's main story, and its speed-ups (182 to 216x on the drift-diffusion model, about 50x on g-and-k) are against flow matching, which needs hundreds of network evaluations per density. The g-and-k reproduction confirms it: the mean flow (C2ST 0.58 to 0.60) is at least as accurate as flow matching with 200 Euler steps (0.614), which ran about 50x slower.

**Claim B: mean flows are more accurate than normalizing flows.** The evidence in the paper is two moons NPE (0.523 against 0.543) and g-and-k (0.58 against 0.68). Here we disagree on size and generality:

- On g-and-k, half of the gap comes from the baseline's training: the coupling flow trained 200 epochs, the mean flow 1,000 with a tuned recipe. With equal training the gap is 0.03 to 0.05 (`results_gandk.md`).
- On the mini-sbibm NLE tasks at the paper's data scale, both mean-flow networks tie sbi's NSF on three tasks and lose on slcp (benchmark above). At two moons NPE scale, sbi's NSF reaches 0.518 to 0.530 against 0.578 for the mean flow (before the cycle-loss fix).
- The drift-diffusion experiment, where the speed story is strongest, has no normalizing-flow baseline.

**Why the speed claim does not reach sbi users.** sbi's NLE runs on normalizing flows (MAF by default, or NSF), which already evaluate a density in one network pass. Against those the mean flow is not faster: 1.4 to 1.9x slower per MCMC run at S=1 on mini-sbibm and about the same on g-and-k, because every step needs a full Jacobian log-determinant, which costs $O(d^3)$ in the data dimension.

**In one sentence:** mean flows bring flow-matching-class density estimators into likelihood-based inference at normalizing-flow cost, but not at better-than-normalizing-flow accuracy in general.

**Where the method could still matter: architectures that flows cannot use.** A normalizing flow needs an invertible architecture with a cheap Jacobian determinant, so its network is constrained to coupling or autoregressive layers of fixed width. A mean flow only needs a network that predicts an average velocity; the density comes from autodiff of the flow map. Any architecture works, including a transformer over tokens. The paper uses this in its third experiment: a stochastic renewal epidemic model (128 whitened Gaussian-process coefficients for $\log R_t$, 123 observed log-incidence values, 30% missing completely at random, 8,192 simulations). Its mean flow runs on a transformer that treats each observation as a token and takes a mask of the missing ones, so it learns the density of the observed coordinates with the missing ones integrated out, which is the correct likelihood under random missingness. The coupling flow can only be shown a fill-in value (−1) where data are missing. The paper reports lower parameter error (0.187 against 0.199), lower predictive error (0.060 against 0.125) and calibration for the mean flow, with the flow finishing 22x sooner (1,624 s against 36,146 s). If there is a case for the mean flow in sbi, it is this one: likelihoods over observations of varying size and missingness, which sbi's NLE cannot express today. The obvious competitor in sbi is NPE with the missingness mask appended to the observation, which needs no new estimator. The next section tests both on a tractable stand-in.

## Missing data: masked-transformer mean flow against NPE with a mask (2026-10-30)

**Question.** Does the paper's third experiment hold up when the baseline is what sbi offers today, NPE with the missingness mask appended? Plan and decision rule, set in advance: `.claude/plans/weekend-checks-2026-10-31.md`, check 2 (local note).

**Task.** The paper's renewal model has no tractable likelihood, so its scores are errors against the true parameters. We use a tractable stand-in with an exact likelihood for every missingness pattern (`scripts/missing/wave_task.py`): 4 parameters of an epidemic-like wave (growth rate, peak time, peak height, decay rate), log-incidence on 32 time points, Gaussian AR(1) noise with correlation 0.8 and sd 0.15, 30% missing completely at random with the paper's rule. 8,192 simulations, 10 evaluation datasets with one pattern each (19 to 29 of 32 points present). The noise level puts the posterior sd at 23 to 30% of the prior sd (the paper's task: about 15%). Reference posteriors: 20,000 draws per dataset by adaptive importance sampling on the exact likelihood (ESS about 220,000 of 400,000).

**Arms.** 3 training seeds each.

1. `mf_dit`: the paper's mean flow on their diffusion transformer with the missing-target mask, their code (`scripts/missing/paper_arms.py`). The attention is not causal: their causal transformer needs the parameters to span the time grid. 1,000 epochs, last epoch, NUTS at S=1, 4,000 draws, as in the paper.
2. `nf`: the paper's spline coupling flow, shown −1 where a value is missing, depth 5, up to 1,000 epochs with their early stopping (stopped after 87 to 116), NUTS.
3. `sbi_npe`: sbi NPE with the NSF; the observation standardized, missing entries set to 0 and the 0/1 mask appended; a new mask per training simulation every epoch; best epoch (262 to 478 epochs, 8 to 14 min on CPU) (`scripts/missing/sbi_npe.py`).

**Results** (`results/missing/`, from `scripts/missing/score.py` and `stats.py`). C2ST against the reference (4,000 draws per side), mean over 10 datasets; parameter RMSE of the posterior mean against the true θ (prior units; the reference posterior itself gives 0.269).

| Arm | C2ST, mean of 3 seeds | C2ST per seed | RMSE per seed |
|---|---|---|---|
| mean flow, transformer (`mf_dit`) | **0.614** | 0.602, 0.607, 0.632 | 0.268, 0.315, 0.277 |
| sbi NPE, mask appended | 0.645 | 0.669, 0.630, 0.636 | 0.278, 0.276, 0.263 |
| coupling flow, −1 fill (`nf`) | 0.745 | 0.736, 0.762, 0.737 | 0.286, 0.300, 0.343 |

Paired differences in C2ST (mean over seeds per dataset, standard error over the 10 datasets):

| Comparison | Difference |
|---|---|
| mean flow − sbi NPE | −0.031 (0.014) |
| mean flow − sbi NPE, without the 2 non-converged NUTS runs | −0.026 (0.015) |
| coupling flow − sbi NPE | +0.100 (0.012) |
| mean flow − coupling flow | −0.131 (0.013) |

**Cost.** NUTS per dataset (V100, the paper's sampler): median 1,924 s for the mean flow against 300 s for the coupling flow, 46 s against 4.5 s per 1,000 network evaluations. 2 of the 30 mean-flow runs failed the convergence gate (bulk ESS 29 and 7, R-hat 1.11 and 1.53); all 30 coupling-flow runs passed. NPE is amortized and samples a new observation in seconds.

**Findings.**

1. The paper's claim against its own baseline reproduces clearly: the masked transformer beats the coupling flow with a fill-in value by 0.13 C2ST, and by 0.02 in parameter RMSE (0.287 against 0.310, mean of 3 seeds).
2. That baseline is weak. NPE with the mask appended, available in sbi without new code, is 0.10 C2ST better than it.
3. Against NPE with a mask, the mean flow gains 0.03 C2ST. This meets the rule set in advance (more than 0.02 and more than 2 standard errors, 2 × 0.014 = 0.028) only narrowly, and fails it without the two non-converged runs (2 × 0.015 = 0.030). Parameter RMSE shows no gain: all arms sit near the reference's 0.27, with NPE the most stable over seeds.
4. The gain costs a lot: about 30 min of NUTS per observation, against seconds for amortized NPE, and 2 of 30 chains did not converge.

**Caveats.** Arm 1 against arm 3 compares NLE with a transformer against NPE with an MLP, so the difference cannot be pinned on the transformer alone. The mean flow ran at its last epoch only (the plan also asked for the best epoch; not run). The task is small: 4 parameters and 32 tokens favour NPE with a mask, and the paper's regime (128 parameters, 123 tokens) could differ.

**What it means for sbi.** No case for adding the mean flow for missing data on this evidence. The useful outcome is the recipe that matched it at a fraction of the cost: standardize, set missing entries to 0, append the mask, and redraw the mask per epoch. sbi's documentation does not describe missing data at all; a short how-to guide with this recipe would serve users.

## Bugs in existing sbi found on the way

Both are fixed in the first two commits of this branch, and for sbi on their own branches from main: `fix/sample-and-log-prob-sample-shape` and `fix/importance-weights-no-grad` (PRs pending). The second fix grew into a root-cause fix: potentials now follow the caller's grad mode by default, which also removes the unused graph in rejection sampling, importance sampling, the VI quality check and the MAP search.

- `ConditionalDensityEstimator.sample_and_log_prob` passed samples with several sample dimensions to `log_prob`, which takes one. With `sample_shape=(2, 3)` and more than one condition the MDN raised a broadcasting error. Affects the MDN, the MNLE estimator and the categorical estimator.
- The `resample` and `sir` MCMC init strategies evaluate 10,000 candidates per chain inside `torch.no_grad()`, but sbi potentials default to `track_gradients=True` and turn gradients back on. Every MCMC init built an unused autograd graph: NSF (128 hidden features) on SLCP needed 335 MB extra peak memory instead of 46 MB; the mean flow 5.4 GB instead of 0.6 GB. Results are bit-identical with the fix.

## Mistakes in our own benchmark (fixed, recorded so they are not repeated)

- `build_posterior` deep-copies the estimator (`sbi/inference/trainers/base.py:623`). Setting `num_steps` on the trained estimator afterwards does not reach the posterior. The first grid ran every mean flow at S=2; set `num_steps` before `build_posterior`.
- The paper recipe without model selection overfits at 2,000 simulations (above); with plenty of data, best-epoch selection on the mean-flow loss can pick worse weights than the last epoch (g-and-k, slcp), but the last epoch can also fail (gaussian_linear at 50,176 with sbi's network). Report both.

## Verdict

After the missing-data check (2026-10-30): unchanged. The masked-transformer mean flow beats the paper's fill-in coupling flow by 0.13 C2ST, but sbi NPE with the mask appended comes within 0.03 at a fraction of the cost.

After the NLE benchmark (2026-10-28): park the mean flow; do not add it to sbi. With plenty of data it ties sbi's default NSF on three of four mini-sbibm tasks and loses on slcp, with either network. Its gains are narrow: g-and-k with the paper's network only, and 0.03 to 0.05 C2ST on gaussian_linear and slcp at 2,000 simulations with sbi's network only. Against that stand 14 to 20x the training time (2,000 simulations, CPU), 1.4 to 1.9x the MCMC time at S=1, a training recipe that sbi's `train()` does not offer, and no reliable rule for which epoch to keep. The branch stays as a documented experiment; the fixes for sbi that came out of it are listed separately.

Before g-and-k: park the mean flow in sbi unless the g-and-k reproduction shows a clear NLE gain over a fairly trained flow. The g-and-k outcome is in `results_gandk.md`: a gain of 0.03 to 0.05 C2ST over the paper's coupling flow trained equally long and over sbi's best flow, with the paper's network only. The rest of this paragraph is the pre-g-and-k assessment. With equal training it gives no accuracy gain over NSF on two moons at the paper's scale, costs 1.8x more MCMC time per new observation, and needs a cosine learning-rate schedule that sbi's `train()` does not offer.
