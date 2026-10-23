# Plan: reproduce the paper's g-and-k comparison on the DGX (V100 32GB)

Self-contained handoff for a new session on the DGX machine. Read `paper_summary.md` and `results.md` in this folder first.

## Goal

The paper's strongest NLE claim against a normalizing flow is the g-and-k experiment (paper Section 4.4, Table S6, Figure 4): under one evaluation budget the mean flow reaches C2ST 0.577 to 0.604 and the coupling flow 0.679. We want to know whether that holds with a fairly trained flow, and whether sbi's implementation matches the paper's.

Two questions, in this order:

1. **Reproduction:** does the paper's own code give its Table S6 numbers on our hardware?
2. **Fair comparison:** on the same datasets, reference posteriors and metric, how do sbi's mean flow, a coupling flow trained as long as the mean flow, and sbi's default NLE flows compare?

Stop after question 1 if the reproduction fails by more than seed noise; then the reason is the finding.

## Paper numbers to reproduce (Table S6, 10 datasets, mean (sd))

| Arm | Steps | Samples (all 4 chains) | Min bulk ESS | C2ST |
|---|---|---|---|---|
| Mean flow S=1 | 1 | 32,000 | 10,498 (1,552) | 0.604 (0.049) |
| Mean flow S=2 | 2 | 16,000 | 4,856 (939) | 0.587 (0.046) |
| Mean flow S=4 | 4 | 8,000 | 2,436 (708) | **0.577 (0.039)** |
| Mean flow S=8 | 8 | 4,000 | 1,143 (476) | 0.599 (0.030) |
| Normalizing flow | 1 | 32,000 | 8,023 (3,169) | 0.679 (0.078) |
| Flow matching, Euler 200 (off budget) | 200 | 4,000 | 1,270 (331) | 0.614 (0.048) |
| Delayed acceptance, S_c = 1 / 2 / 4 / 8 | 100 fine | 3,656 / 3,492 / 3,200 / 2,744 | 642 / 649 / 581 / 544 | 0.592 / 0.594 / 0.595 / 0.599 |

TARP (Table S1, g-and-k likelihood): mean flow S=1 ATC −0.004 (0.012), S=8 −0.009 (0.011); NF +0.001 (0.011); FM +0.004 (0.007). Wall times are in Figure S2B (mean flow about 70 to 100 s per dataset, NF about 300 s, FM about 4,000 s, on an A100).

## The paper's protocol (from their code, `experiments/tasks/config_gnk.py`, `experiments/config_run.py`, `experiments/common.py`)

- **Model:** bivariate g-and-k, $W(z;g,k)=(1+0.8\tanh(gz/2))(1+z^2)^k z$ on both coordinates, Gaussian copula with correlation ρ. Priors g ~ U(−1, 1), k ~ U(0, 0.5), ρ ~ U(−0.9, 0.9). The network learns the likelihood of **one** 2-D observation; a dataset has 50 i.i.d. observations. Exact density by bisection inversion of W.
- **Data:** 100,000 training simulations (`TRAIN_SIZE` in `config_gnk.py`), 10 datasets with θ from the prior, seed 42 (`config_run.SEED`).
- **Reference posterior:** adaptive random-walk Metropolis ensemble on the exact likelihood, 8 chains, 5,000 adaptation steps, 40,000 draws thinned by 10, R̂ ≤ 1.05, ESS ≥ 400 (`sampling/reference.py`, `ReferenceConfig()`).
- **Networks:** mean flow MLP 5 × 256 (BayesFlow `TimeMLP` with FiLM), velocity head forks the last 2 blocks; coupling flow depth 5, subnet widths (256, 256) on g-and-k; trained flow matching uses the mean flow's widths without the head.
- **Training:** batch 256, AdamW (weight decay 5e-3), learning rate warmed up over 5% of steps from 5e-5 to 5e-4, cosine decay to zero, gradient clipping 1.5. Mean flow and flow matching 1,000 epochs; coupling flow 200 epochs with early stopping, patience 10, best weights kept.
- **Sampling:** nutpie NUTS on PyMC with the JAX backend and JAX gradients, 4 chains in threads, diagonal mass matrix, max tree depth 10, nutpie's default target acceptance. An arm whose density costs K network evaluations requests 8,000 / K draws per chain (`draws_at_cost`); warmup is 25% of the request (`TUNE_SHARE`). Chain seeds: `SEED + steps + 10_000 * dataset`.
- **Delayed acceptance:** subchain J = 5 random-walk Metropolis steps on the coarse level (mean flow at S_c), fine level = the same network's velocity head with 100 fixed Tsit5 steps and exact divergence; target coarse acceptance 0.3 (Robbins-Monro), adaptation 25% of iterations.
- **Metrics:** C2ST with their MLP classifier at one common draw count per experiment, the smallest any scored cell kept (2,744 in their run, set by a delayed-acceptance arm; corrected 2026-10-23, the code does not use 1,024) (`helpers/metrics.py`, `c2st`), min bulk ESS, ESS per 1,000 network evaluations, draws kept, wall time; convergence gate max R̂ ≤ 1.05, min bulk ESS ≥ 100, ≥ 500 kept draws.

## Phase 0: environments and hardware checks (about 1 hour)

1. **Code.** Pull this sbi branch (`mean-flow-estimator`). Fetch the paper code:
   `python experiments/mean_flow/scripts/fetch_paper_code.py --out ~/meanflow-paper`
2. **Environment A (paper code).** In `~/meanflow-paper`: Python 3.14, `uv sync`. Their `pyproject.toml` pins BayesFlow to an anonymized git source at `rev = "dev"`. Try in order and record which one works:
   - `bayesflow==2.0.14` from PyPI (the paper says it built on 2.0.14),
   - `git+https://github.com/bayesflow-org/bayesflow@dev` at the last commit on or before 2026-10-15.
   Check: `uv run python -c "import bayesflow, nutpie, pymc; from bayesflow.wrappers.pymc import NeuralDistribution"`.
3. **JAX on V100.** V100 is compute capability 7.0. Check that the installed `jax[cuda12]` runs on it: `uv run python -c "import jax, jax.numpy as jnp; print(jax.devices()); print(jnp.ones((1000,1000)) @ jnp.ones((1000,1000)))"`. Check `nvidia-smi` for the driver's CUDA version. If JAX cannot use the GPU, run environment A on CPU (`JAX_PLATFORMS=cpu`): the g-and-k density is 2-D, so MCMC is fine on CPU; training the mean flow (about 390,000 steps) then takes several hours per network. Record the choice; wall times are then not comparable to the paper's.
4. **Environment B (sbi).** In the sbi checkout: `uv venv --python 3.12 && uv pip install -e ".[dev,pymc]" nutpie arviz`. Check that the PyTorch wheel supports sm_70: `python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_arch_list())"`. If `sm_70` is missing, install a PyTorch build that has it (for example the CUDA 12.6 wheel).
5. **Sampler for the sbi arms.** Check whether nutpie can sample a Python log-density with its gradient (look for `nutpie.compiled_pyfunc.from_pyfunc` or the documented equivalent in the installed version). If yes, use it for the sbi arms with a torch `log_prob` plus `torch.autograd.grad` wrapper: then every arm runs the same NUTS. If not, use sbi's `MCMCPosterior(method="nuts_pymc")` with 4 chains and the same draws and warmup; then report that the sampler differs.
6. **Smoke run of their pipeline:** `KERAS_BACKEND=jax uv run experiments/exp2_gnk.py --smoke --results-dir results_smoke --checkpoint-dir checkpoints_smoke`. It must finish in minutes.

## Phase 1: reproduce the paper with its own code

1. Train: `KERAS_BACKEND=jax uv run experiments/exp2_gnk.py --train-only --results-dir results --checkpoint-dir checkpoints --verbose 1`.
2. Datasets: one process per dataset, spread over the GPUs, for example
   `for i in 0 ... 9: CUDA_VISIBLE_DEVICES=$((i % NGPU)) uv run experiments/exp2_gnk.py --dataset-index $i --results-dir results --checkpoint-dir checkpoints &`.
   The flow-matching arm (200 steps) dominates the time, about an hour per dataset on an A100.
3. Aggregate: `uv run experiments/exp2_gnk.py --aggregate --results-dir results --checkpoint-dir checkpoints`.
4. Compare with the table above. Reproduced if every arm's mean C2ST lies within 2 standard deviations of the paper's (sd over 10 datasets) and the ordering mean flow < flow matching < coupling flow holds. Record differences in wall time with the hardware.

## Phase 2: fair comparison on the same data

1. **Export from environment A** (write a small script there; do not change their code): for each of the 10 datasets the true θ, the 50 observations, the 32,000 reference draws, and every arm's posterior draws; plus the 100,000 training simulations and the validation set exactly as their training stage drew them. First check what `save_dataset_shard` (JSON shards in `results/`) already stores; regenerate what is missing with their task code (`GnK`, `sampling/reference.py`) and the same seeds. Save as `.npz`.
2. **sbi arms (environment B)**, all trained on the exported 100,000 simulations, NLE with the 50 observations as i.i.d. trials and the box prior:
   - **sbi mean flow** (`MeanFlowConfig()`, includes the cycle-loss fix), paper recipe for 1,000 epochs (`scripts/mean_flow_paper_trainer.py`; report both the last and the best epoch), evaluated at S = 1, 2, 4, 8.
   - **Coupling flow matched to the paper's NF:** `NSFConfig(hidden_features=256, num_transforms=5, num_blocks=1)` (check that this gives 5 coupling layers with two 256-wide hidden layers each; adjust `num_blocks` otherwise). Train it twice: (a) the paper's NF protocol (200 epochs, early stopping, patience 10, same optimizer), (b) the mean-flow recipe for 1,000 epochs with best-epoch selection. (b) is the comparison the paper did not make.
   - **sbi defaults:** `NLE()` (MAF) and `NSFConfig()` with sbi's `train()`.
3. **Same budget and sampler:** per arm 8,000 / K draws per chain, 4 chains, warmup 25% of the request, diagonal mass matrix, max tree depth 10. Count network evaluations by wrapping `log_prob` (rows × S).
4. **Same metric:** score every arm's draws in environment A with their `c2st` against the exported reference draws, and compute R̂, bulk ESS and ESS per 1,000 network evaluations with ArviZ. Apply their convergence gate.
5. **Optional:** TARP on the sbi arms with 2,000 validation pairs and 200 draws each (`sbi.diagnostics`), to compare with Table S1.

## Deliverables

- `experiments/mean_flow/results_gandk.md` on this branch: the reproduced Table S6, the fair-comparison table (C2ST, min bulk ESS, ESS per 1,000 evaluations, draws kept, training time, MCMC wall time per dataset), hardware and software versions, and every deviation from the paper's protocol.
- Raw results (`.npz`, `.jsonl`) under `experiments/mean_flow/results/gandk/`; no checkpoints or large draw files in git (keep those on the DGX).
- A short verdict: does the mean flow beat a coupling flow trained as long, and does sbi's implementation match the paper's?

## Risks

- **JAX or PyTorch without sm_70 kernels** (Phase 0, steps 3 and 4). Fallbacks are listed there.
- **BayesFlow version.** The paper's code expects BayesFlow features from its `dev` branch (`bayesflow.wrappers.pymc`, `TimeMLP`, `ConditionalDenseBlock`). A mismatch shows up in the smoke run.
- **Sampler mismatch for the sbi arms** if nutpie cannot take a Python density. Accuracy then stays comparable, ESS and wall time do not.
- **Remaining implementation differences** of the sbi mean flow (time input, backbone, output initialization; `results.md`). If the sbi mean flow trails the paper's mean flow on identical data, these are the first suspects.
