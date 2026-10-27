# g-and-k reproduction on the DGX (2026-10-23 to 2026-10-26)

Plan: `gandk_reproduction_plan.md`. Paper: Arruda, Hasenauer, Radev, arXiv:2610.21657, Section 4.4, Table S6.

Status: complete. All 36 arms on all 10 datasets, scored together by the paper's metric code (`results/gandk/cells.jsonl`, `results/gandk/arms.json`).

## Verdict

1. **The paper's code reproduces Table S6** on our hardware: mean flow C2ST 0.584 to 0.600 (paper 0.577 to 0.604), coupling flow 0.678 (paper 0.679), every other arm within 0.01.
2. **Part of the paper's gap is the coupling flow's training, not the method.** Their coupling flow trained like their mean flow (1,000 epochs, same schedule, their code) improves from 0.678 to 0.630. The mean flow still beats it: by 0.030 to 0.046 in C2ST (paired over datasets, standard error 0.014 to 0.017), on 8 or 9 of 10 datasets at every step count. The claim holds in direction; its size is about half of what Table S6 shows.
3. **sbi's port with the paper's network reproduces their mean flow** (C2ST 0.573 to 0.586 at the last epoch) and beats every sbi flow; the best of these is the default NSF at 0.619 (paired difference at S=2: −0.045 ± 0.013, 9 of 10 datasets).
4. **sbi's own mean-flow network is clearly worse:** 0.631 to 0.649 at the last epoch over 3 seeds, 0.059 behind the paper network at S=2 (wins 1 of 10), and level with the default NSF. Same loss, data, recipe and sampler, so the difference is the network (time input, backbone, output initialization; see `results.md`, "Differences").
5. **Keep the last epoch for mean flows.** Best-epoch weights (lowest fixed-noise validation loss, epochs 233 to 373) are worse for both networks and fail on some datasets (paper network at S=2: 0.649 against 0.573, with C2ST 0.84, 0.76 and 0.78 on datasets 2, 4 and 6). The fixed-noise validation loss apparently stops tracking posterior quality late in training, when the cosine schedule refines the flow map. For the coupling flows the opposite holds: the wide sbi NSF trained 1,000 epochs overfits, and its last epoch gives C2ST 0.958 with no converged dataset.
6. **Sampling cost** (ESS per 1,000 network evaluations, comparable across frameworks) is the same for the paper's mean flow and sbi's port at each step count (S=1: 1.12 and 1.05; S=8: 0.108 and 0.104). Mean flows at S=1 and the sbi flows are at about 1 per 1,000; the cost per draw grows linearly in S, while C2ST is flat from S=1 to S=8, so S=1 or 2 is enough on g-and-k.

## Setup

### Hardware

DGX with 5 Tesla V100-SXM2-32GB (compute capability 7.0), driver 550.127.08 (CUDA 12.4), 2 × Xeon E5-2698 v4 (40 cores), 128 GB RAM, no Slurm. Only GPU 1 was used; the others ran other jobs throughout, as did 4 CPU cores. The paper ran on A100s, so wall times are not comparable to its Figure S2B.

### Paper environment (`results/gandk/env_paper.txt`)

Their `pyproject.toml` cannot be installed as published: its BayesFlow source is the anonymized `git = "XXXX"`. The environment is their dependency list (`paper_requirements.txt`) with `bayesflow==2.0.14` from PyPI, installed with `uv pip install --override paper_overrides.txt`. Two fixes, both to the environment only:

- **numba 0.68.0 instead of 0.63.0b1.** Their pins conflict: `arviz<1` forces pymc 5.28.5, whose pytensor 2.38.3 caps numba at 0.65.1, while numpy must be at least 2.5 (their pin and BayesFlow 2.0.14's). Every numba up to 0.65.1 requires numpy below 2.5 except the beta 0.63.0b1, which the resolver picks and which fails on `import nutpie` (`np.row_stack`). numba 0.68.0 supports numpy 2.5.
- **Python 3.14.7 managed by uv** instead of the system 3.14.4, which has no `Python.h` for PyTensor's C compilation.

Versions: Python 3.14.7, jax and jaxlib 0.12.0.1 (cuda12 plugin, CUDA 12.9 libraries from pip), keras 3.15.1 on JAX, bayesflow 2.0.14, pymc 5.28.5, pytensor 2.38.3, nutpie 0.16.11, numpy 2.5.4, arviz 0.23.4.

BayesFlow `dev` (last commit before 2026-10-15: `3777d497`, 18 commits after 2.0.14) was not tried, because 2.0.14 runs their code. Its changes in modules their code imports: a refactor of `integrate_scipy`, `film_use_gamma` added to `ConditionalDenseBlock.get_config` (their code works around its absence), the default adapter no longer casting to float32. None is on the g-and-k path as far as the diff shows.

The paper's code is kept unchanged in `~/meanflow-paper` (snapshot in a local git repository). Our scripts that import it live in `scripts/gandk/` and run with its interpreter.

### sbi environment (`results/gandk/env_sbi.txt`)

Python 3.12.14, sbi at this branch, torch 2.14.1+cu126 (the default 2.14.1 wheel is built for CUDA 13 and refuses the CUDA 12.4 driver; the cu126 build has sm_70 kernels), nutpie 0.16.11 (same as the paper's), pymc 6.3.2, arviz 1.3.0.

### Checks (Phase 0)

| Check | Result |
|---|---|
| JAX on the V100 | runs: matrix product, jitted gradient, their smoke run on `cuda:0` |
| PyTorch on the V100 | runs with the cu126 build |
| nutpie on a Python density | `nutpie.compiled_pyfunc.from_pyfunc` with a torch log-density and autograd gradient; on a 3-parameter Gaussian toy in the g-and-k box, posterior means match nutpie on the equivalent PyMC model within 1 standard error, no divergences |
| Their smoke run (`exp2_gnk.py --smoke`) | completes in 287 s on GPU 1, all 10 arms, TARP and figures; no C2ST by design (smoke arms keep fewer than 500 draws) |
| Their training speed | mean flow 9 ms per step alone on GPU 1, 12 ms while sharing it (391 steps per epoch) |

## Protocol

Their experiment 2 as published (`experiments/exp2_gnk.py`, `tasks/config_gnk.py`, `config_run.py`), with these facts read from the code rather than from the paper:

- **Optimizer** (BayesFlow 2.0.14 `BasicWorkflow.build_optimizer`, offline): AdamW, weight decay 5e-3, epsilon 1e-7; learning rate linear from 5e-5 to 5e-4 over the first 5% of steps, then cosine decay to zero; `clipnorm=1.5`, which in Keras clips **each** gradient tensor to norm 1.5, not the global norm.
- **Coupling flow:** BayesFlow `CouplingFlow(depth=5, transform="spline", subnet_kwargs={"widths": (256, 256)})`, 801,810 parameters. Each depth level is ActNorm, a random permutation and a dual coupling (both halves transformed), so 10 single couplings. Each coupling's subnet takes BayesFlow's `COUPLING_MLP_DEFAULTS`: two dense layers of 256 on the concatenated half and condition, hard_silu, orthogonal initialization, no residual connections, dropout 0.05, then a zero-initialized output projector. 16-bin rational-quadratic splines whose domain starts at [-3, 3] and is learned, with linear tails matching the boundary slope and minimum bin width and height 0.1. 200 epochs, early stopping on validation loss with patience 10, best weights restored (Keras restores them also when patience never runs out). On g-and-k it stops at epoch 38 and keeps epoch 28 (172 s of training, against 4,303 s for the mean flow).
- **Data:** 100,000 training and 1,024 validation simulations from one stream (`offline_training_data`, seed `TrainConfig.seed + 7` = 49); dataset `i` from `default_rng(42 + 1000 * (20000 + i))`.
- **C2ST size:** every cell (arm, dataset) is scored at one common draw count per side, the smallest number of draws any scored cell kept (or the reference's 32,000), not at a fixed 1,024. Cells with fewer than 500 draws are not scored.
- **Checkpoint names** hash the whole `TrainConfig`, including `verbose`, which is `--verbose` minus 1; the dataset stage must be run with the same `--verbose` as the training stage to find the networks.

## Phase 1: the paper's code

Their experiment 2 on 10 datasets, one network per arm, aggregated by their `--aggregate` on CPU. Mean (sd) over datasets; C2ST at 2,744 draws per side (set by the delayed-acceptance arm at S_c = 8 on dataset 0, as in their run).

| Arm | Draws (4 chains) | Min bulk ESS | C2ST | Paper C2ST (Table S6) |
|---|---|---|---|---|
| Mean flow S=1 | 32,000 | 10,808 (1,449) | 0.600 (0.053) | 0.604 (0.049) |
| Mean flow S=2 | 16,000 | 5,230 (952) | 0.588 (0.039) | 0.587 (0.046) |
| Mean flow S=4 | 8,000 | 2,408 (884) | 0.584 (0.042) | 0.577 (0.039) |
| Mean flow S=8 | 4,000 | 1,196 (420) | 0.594 (0.044) | 0.599 (0.030) |
| Coupling flow (200 epochs, early stopping) | 32,000 | 9,331 (2,766) | 0.678 (0.088) | 0.679 (0.078) |
| Flow matching, Euler 200 (off budget) | 4,000 | 1,229 (309) | 0.614 (0.049) | 0.614 (0.048) |
| Delayed acceptance S_c = 1 / 2 / 4 / 8 | 3,656 / 3,492 / 3,200 / 2,744 | 624 / 663 / 585 / 582 | 0.593 / 0.593 / 0.601 / 0.599 | 0.592 / 0.594 / 0.595 / 0.599 |

Every arm passed their convergence gate on every dataset. Draw counts match Table S6 exactly. TARP (no MCMC) reproduces Table S1: mean flow S=1 ATC −0.0043 (0.0118), S=8 −0.0091 (0.0106), coupling flow +0.0001 (0.0111), flow matching +0.0035 (0.0079).

Wall times (MCMC per dataset, median): mean flow 500 to 1,070 s, coupling flow 1,894 s, flow matching 25,573 s, delayed acceptance 350 to 450 s. They were measured on a shared machine (load up to 2x the cores) and are not comparable to the paper's Figure S2B.

## Phase 2: sbi arms on the same data

All sbi arms train on the exported training set (`scripts/gandk/export_paper_data.py`) and are sampled with the paper's kernel and budget (`scripts/gandk/sample_sbi.py`): nutpie NUTS, 4 chains, diagonal mass matrix, 8,000 / S draws per chain, warmup 25%, the paper's chain-seed integers, PyMC's logit transform for the box prior. Everything is scored together with the paper's arms by their code (`scripts/gandk/score.py`).

| Arm | Estimator | Training | Seeds |
|---|---|---|---|
| `mf_sbi` | `MeanFlowConfig()` (sbi network, 2,849,028 parameters) | paper recipe, 1,000 epochs; last and best epoch | 0, 1, 2 |
| `mf_paper` | `MeanFlowConfig(variant="paper")`, port of their network (1,755,940 parameters, as theirs) | paper recipe, 1,000 epochs; last and best epoch | 0 |
| `paper_nf_long` | **their** BayesFlow coupling flow, their code (`scripts/gandk/paper_nf_long.py`) | their mean-flow recipe: 1,000 epochs, no early stopping, schedule over 1,000 epochs; last and best epoch | their seed |
| `nsf_nfprotocol` | wide sbi NSF: `NSFConfig(hidden_features=256, num_transforms=10, num_bins=16, num_blocks=1, dropout_probability=0.05)`, 1,459,730 parameters | paper's coupling-flow protocol: 200 epochs, patience 10, best weights | 0, 1, 2 |
| `nsf_long` | same wide NSF | paper's mean-flow recipe, 1,000 epochs; last and best epoch | 0 |
| `nsf_default` | `NSFConfig()` (61,675 parameters) | `NLE.train()` defaults on all 101,024 simulations (it holds out 10%) | 0 |
| `maf_default` | `MAFConfig()`, NLE's default (28,270 parameters) | as `nsf_default` | 0 |

The paper recipe in `scripts/gandk/train_sbi.py` reproduces BayesFlow's optimizer including per-tensor clipping; the held-out loss uses their 1,024 validation simulations (for the mean flow with fixed noise, so epochs compare).

The fair test of the paper's claim is `paper_nf_long` against their mean flow: same framework, same code, same training length. The wide sbi NSF is **not** the paper's coupling flow. It shares the coupling count, spline bins and initial domain, width and dropout, but each conditioner is an nflows `ResidualNet` (input layer, one residual block of two 256-wide layers with a context gating layer, output layer: one hidden layer more than BayesFlow's, relu, default initialization, so the flow is not the identity at the start), the spline domain is fixed at [-3, 3] with identity tails and minimum bin size 1e-3 (3.4% of the standardized training observations lie outside ±3 in some coordinate), LU-linear layers sit between couplings, and z-scoring uses fixed training-set statistics. On the 1,024 validation simulations its best loss is 3.11 nats against about 3.10 for the paper's coupling flow (2.2383 in BayesFlow's standardized space plus Σ log σ_x = 0.860).

The 1,000-epoch NSF peaks early: the best validation losses fell at epochs 38, 33 and 29 for three seeds, still inside the 50-epoch warmup, and by epoch 220 the validation loss had risen from 3.11 to 3.28 while the training loss fell to 2.83. Its best epoch is therefore close to `nsf_nfprotocol`, and its last epoch is an overfitted flow. Seeds 1 and 2 were stopped at epoch 220 for that reason (logs in `sbi-train/nsf_long_s{1,2}_stopped`); the coupling-flow seed spread comes from `nsf_nfprotocol` instead. The mean flows show no such overfitting in the validation loss: best epochs 233 (paper network) and 285, 355, 373 (sbi network, seeds 0 to 2), with the last-epoch loss close to the best. Best-epoch arms are compared with best-epoch arms, last with last.

### Results

Mean (sd) over the 10 datasets, C2ST at 2,744 draws per side together with the Phase 1 arms. ESS per 1k NFE is the minimum bulk ESS per 1,000 network evaluations over the whole run (warmup included), as their code counts it.

| Arm | Epoch | S | C2ST | Min bulk ESS | ESS per 1k NFE | Divergences (10 datasets) |
|---|---|---|---|---|---|---|
| Paper mean flow (Phase 1) | last | 1 / 2 / 4 / 8 | 0.600 / 0.588 / 0.584 / 0.594 | 10,808 / 5,230 / 2,408 / 1,196 | 1.12 / 0.52 / 0.24 / 0.11 | 1 / 0 / 2 / 1 |
| `mf_paper` | last | 1 / 2 / 4 / 8 | 0.586 / 0.573 / 0.574 / 0.582 | 10,236 / 5,024 / 2,437 / 1,130 | 1.05 / 0.49 / 0.24 / 0.10 | 1 / 0 / 0 / 2 |
| `mf_paper` | best | 2 | 0.649 (0.109) | 6,228 | 0.69 | 1 |
| `mf_sbi` seed 0 | last | 1 / 2 / 4 / 8 | 0.634 / 0.633 / 0.638 / 0.649 | 9,489 / 4,397 / 2,300 / 1,083 | 0.96 / 0.42 / 0.22 / 0.10 | 8 / 0 / 0 / 0 |
| `mf_sbi` seed 0 | best | 1 / 2 / 4 / 8 | 0.672 / 0.679 / 0.655 / 0.651 | 13,081 / 6,344 / 3,063 / 1,476 | 1.47 / 0.70 / 0.32 / 0.15 | 0 |
| `mf_sbi` seeds 0, 1, 2 | last | 2 | 0.633 / 0.645 / 0.631 | 4,397 / 4,944 / 4,705 | 0.42 / 0.49 / 0.45 | 0 / 3 / 0 |
| `mf_sbi` seeds 0, 1, 2 | best | 2 | 0.679 / 0.638 / 0.654 | 6,344 / 5,666 / 5,539 | 0.70 / 0.59 / 0.59 | 0 |
| Paper coupling flow (Phase 1) | best of 38 | - | 0.678 (0.088) | 9,331 | 0.76 | 0 |
| `paper_nf_long` | best (232) / last | - | 0.630 (0.046) / 0.633 (0.039) | 7,898 / 5,978 | 0.61 / 0.40 | 0 / 1 |
| `nsf_nfprotocol` seeds 0, 1, 2 | best | - | 0.668 / 0.695 / 0.720 | 10,046 / 10,137 / 9,568 | 0.95 / 0.97 / 0.91 | 0 |
| `nsf_long` | best (38) | - | 0.643 (0.042) | 10,306 | 1.01 | 0 |
| `nsf_long` | last | - | 0.958 (0.034), not converged | 6 | 0.0003 | 92 |
| `nsf_default` | best (sbi) | - | 0.619 (0.042) | 9,911 | 0.99 | 1 |
| `maf_default` | best (sbi) | - | 0.639 (0.081) | 11,162 | 1.18 | 9 |

Paired differences in C2ST over the 10 datasets (mean ± standard error; "wins" counts datasets where the first arm is lower):

| Comparison | Difference | Wins |
|---|---|---|
| Paper mean flow S=1 / 2 / 4 / 8 − `paper_nf_long` best | −0.030 ± 0.017 / −0.042 ± 0.016 / −0.046 ± 0.014 / −0.036 ± 0.016 | 8 / 8 / 9 / 8 |
| `paper_nf_long` best − paper coupling flow (200 epochs) | −0.048 ± 0.030 | 7 |
| `mf_paper` last S=2 − paper mean flow S=2 | −0.014 ± 0.009 | 8 |
| `mf_paper` last S=2 − `nsf_default` | −0.045 ± 0.013 | 9 |
| `mf_sbi` s0 last S=2 − `mf_paper` last S=2 | +0.059 ± 0.016 | 1 |
| `mf_sbi` s0 last S=2 − `nsf_default` | +0.014 ± 0.015 | 5 |
| `mf_paper` best S=2 − `mf_paper` last S=2 | +0.076 ± 0.029 | 2 |

Readings:

- **Network, not method port.** The paper-network port matches their mean flow (slightly better at S=2, within two standard errors), so sbi's loss, schedule, density and sampler path are right. The 0.06 gap of sbi's default network is architectural. `MeanFlowConfig()` should not be the default network if this holds beyond g-and-k; worth a mini-sbibm check before changing it.
- **sbi's default NSF is the strongest flow**, with 61,675 parameters and sbi's own training loop, ahead of the paper's coupling flow trained 1,000 epochs (0.619 against 0.630) and far ahead of the wide NSF (0.668 to 0.720, a 0.05 seed spread). On this task a small flow with early stopping is better than a wide one.
- **Best-epoch selection hurts the mean flows** (finding 5 of the verdict). This reverses the lesson in `results.md` ("keep the best held-out epoch"), which came from flows and from short CPU runs; for mean flows trained with a schedule that decays to zero, the last epoch is the right default, as in the paper.
- **MCMC wall times** (per dataset, median, CPU, about 2x oversubscribed): sbi mean flows 3,000 to 3,900 s at every S, `nsf_default` 3,716 s, wide NSF 7,000 to 7,500 s, `maf_default` 789 s. The paper's JAX path ran the same chains in 500 to 1,070 s (GPU 1 for the network). Only the within-sbi ratios mean anything.

## Deviations from the paper's protocol

- Environment fixes above (numba, Python build); BayesFlow 2.0.14 rather than their unpublished `dev` revision.
- Hardware: one V100 shared with other jobs instead of A100s.
- sbi arms: PyTorch eager instead of jitted JAX. Training wall times are not comparable even among sbi arms: the mean flows trained on GPU 1 (four at once), the NSF and default arms on CPU threads. MCMC wall times compare only among sbi arms run on the same CPU setup.
- Chain starts: same distribution as the paper's (box centre jittered by U(-1, 1) in logit space, one seed per chain from nutpie), but not the same realization: PyMC draws the jitter from reseeded PyTensor generators and orders the free variables (rho, k, g). The seed integers are the paper's; the chains are not paired with its chains.
- Precision: their PyMC model runs in float32 (`pymc_bridge` sets PyTensor's `floatX` to Keras' float32); the sbi runner applies the logit transform and its Jacobian in float64 and evaluates the network in float32. The difference in log-density is about 1e-5 at |log p| ≈ 200.
- Seeds: the paper's spread is over datasets for one trained network per arm; arms with several training seeds are reported per seed, and the seed spread is given separately.
- Phase 1 datasets 1 to 4 crashed on GPU memory after 7 to 8 hours (four sbi trainings shared GPU 1) and were rerun on Monday; their code reused the arms it had journaled before the crash (flow matching and some mean-flow arms), so those cells come from the first run.
- `mf_paper` was sampled after commit 280e60bc, which writes mish out (`x * tanh(softplus(x))`): PyTorch's `F.mish` has a NaN second derivative under `torch.func` for large inputs, which gave NaN NUTS gradients. The network and weights are unchanged. One cell sampled before the fix (best epoch, S=2, dataset 0) was resampled.
- The sbi sampler loads estimators with `map_location="cpu"`: sbi's `Standardize` keeps plain CUDA `mean`/`std` attributes beside its buffers, so a GPU-trained estimator does not load on a CPU-only process otherwise.
- The sbi defaults (`nsf_default`, `maf_default`) train on all 101,024 simulations and hold out 10% themselves; the other arms train on the paper's 100,000 and validate on its 1,024.
