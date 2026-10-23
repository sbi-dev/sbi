# g-and-k reproduction on the DGX (2026-10-23 to 2026-10-26)

Plan: `gandk_reproduction_plan.md`. Paper: Arruda, Hasenauer, Radev, arXiv:2610.21657, Section 4.4, Table S6.

Status: in progress. Sections marked *pending* are filled when the runs finish.

## Verdict

*pending*

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
- **Coupling flow:** BayesFlow `CouplingFlow(depth=5, transform="spline", subnet_kwargs={"widths": (256, 256)})`. Each depth level is ActNorm, a random permutation and a dual coupling (both halves transformed), so 10 single couplings; 16-bin rational-quadratic splines on [-3, 3]; residual mish MLP subnets with dropout 0.05 and zero-initialized output. 200 epochs, early stopping on validation loss with patience 10, best weights restored (Keras restores them also when patience never runs out).
- **Data:** 100,000 training and 1,024 validation simulations from one stream (`offline_training_data`, seed `TrainConfig.seed + 7` = 49); dataset `i` from `default_rng(42 + 1000 * (20000 + i))`.
- **C2ST size:** every cell (arm, dataset) is scored at one common draw count per side, the smallest number of draws any scored cell kept (or the reference's 32,000), not at a fixed 1,024. Cells with fewer than 500 draws are not scored.
- **Checkpoint names** hash the whole `TrainConfig`, including `verbose`, which is `--verbose` minus 1; the dataset stage must be run with the same `--verbose` as the training stage to find the networks.

## Phase 1: the paper's code

*pending*

## Phase 2: sbi arms on the same data

All sbi arms train on the exported training set (`scripts/gandk/export_paper_data.py`) and are sampled with the paper's kernel and budget (`scripts/gandk/sample_sbi.py`): nutpie NUTS, 4 chains, diagonal mass matrix, 8,000 / S draws per chain, warmup 25%, the paper's chain seeds, PyMC's logit transform for the box prior. Everything is scored together with the paper's arms by their code (`scripts/gandk/score.py`).

| Arm | Estimator | Training | Seeds |
|---|---|---|---|
| `mf_sbi` | `MeanFlowConfig()` (sbi network) | paper recipe, 1,000 epochs; last and best epoch | 0, 1, 2 |
| `mf_paper` | `MeanFlowConfig(variant="paper")`, port of their network (1,755,940 parameters, as theirs) | paper recipe, 1,000 epochs; last and best epoch | 0 |
| `nsf_nfprotocol` | `NSFConfig(hidden_features=256, num_transforms=10, num_bins=16, num_blocks=1, dropout_probability=0.05)` | paper's coupling-flow protocol: 200 epochs, patience 10, best weights | 0 |
| `nsf_long` | same NSF | paper's mean-flow recipe, 1,000 epochs, best epoch | 0, 1, 2 |
| `nsf_default` | `NSFConfig()` | `NLE.train()` defaults | 0 |
| `maf_default` | `MAFConfig()` (NLE's default) | `NLE.train()` defaults | 0 |

The paper recipe in `scripts/gandk/train_sbi.py` reproduces BayesFlow's optimizer including per-tensor clipping; the held-out loss uses their 1,024 validation simulations (for the mean flow with fixed noise, so epochs compare).

The NSF is matched to the paper's coupling flow in coupling count, spline bins and domain, subnet width and dropout. It differs in activation (relu, not mish), in the layers between couplings (LU-linear, not ActNorm plus a random permutation), and in z-scoring (fixed training-set statistics, not BayesFlow's running estimates).

*results pending*

## Deviations from the paper's protocol

- Environment fixes above (numba, Python build); BayesFlow 2.0.14 rather than their unpublished `dev` revision.
- Hardware: one V100 shared with other jobs instead of A100s.
- sbi arms: PyTorch eager instead of jitted JAX, so their wall times are compared only with each other.
