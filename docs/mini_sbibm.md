# Benchmarking changes with mini-sbibm

*mini-sbibm* is a small benchmark that runs through `pytest`. It is a minimal version of
[`sbibm`](https://github.com/sbi-benchmark/sbibm). Use it to check that a change to
`sbi` does not make inference worse, for example a new loss, network, or sampler.

It trains sbi methods on four tasks with known reference posteriors (`two_moons`,
`linear_mvg_2d`, `gaussian_linear`, `slcp`) and prints how close the posteriors are.
The tests have no thresholds: they fail only when something crashes. To see the
quality, read the results table. mini-sbibm runs locally, not in CI.

## Quick start

```bash
pytest --bm -n auto                  # default run: each method with default settings
pytest --bm --bm-mode npe -n auto    # one mode: NPE with several density estimators
```

The available modes are `npe`, `npe_pfn`, `nle`, `nre`, `fmpe`, `npse`, `vfpe` (FMPE
and NPSE), `snpe`, `snle`, and `snre`. The sequential modes train for two rounds and
show their methods with an `S` prefix, e.g. `SNPE_C`.

## Reading the results

At the end of a run, mini-sbibm prints one table per metric. Rows are method
configurations with their run label, columns are tasks:

```text
C2ST (0.5 is best)
                   gaussian_linear  two_moons
---------------------------------------------
NPE_C-maf [main]        0.704         0.850
NPE_C-mdn [main]        0.893         0.807
```

- **C2ST**: how well a classifier can tell posterior samples from reference samples.
  0.5 is best, 1.0 means the two sets are fully separated.
- **Posterior mean error** and **posterior std error**: the error of the marginal
  means and standard deviations, divided by the reference standard deviation and
  averaged over parameter dimensions. 0 is best. With 1000 samples, the noise floor
  is about 0.03. A large std error means that the posterior is too wide or too narrow.

Amortized methods are averaged over 10 observations. Sequential methods use one
observation. Within each task, the best value is green and the worst is red.

## Common tasks

### Compare your branch with main

Each run is labeled with the current git branch. Run the same benchmark on both
branches:

```bash
git switch main && pytest --bm --bm-mode npe -n auto
git switch my-branch && pytest --bm --bm-mode npe -n auto
```

The table then shows `NPE_C-nsf [main]` next to `NPE_C-nsf [my-branch]`. A new run
replaces the rows with the same label, method, and task, also when the number of
simulations or seeds is different. With a detached HEAD or outside git, the label is
`default`, so set one with `--bm-label`.

The results are stored in `.bm_results/results_all.csv`, in the folder that you run
`pytest` from. If main and your branch are in different checkouts (e.g., a git
worktree), give both runs the same results folder:

```bash
pytest --bm --bm-mode npe -n auto --bm-results-dir ~/sbi-bm-results
```

To start from scratch, delete the results folder.

### Check whether a difference is real

A single training run can be noisy. Repeat each case with several training seeds:

```bash
pytest --bm --bm-mode npe --bm-seeds 3 -n auto
```

The seeds run in parallel, and the table shows the mean and the standard deviation,
e.g. `0.874 ±0.012`. If two runs differ by less than that spread, the difference is
probably noise.

### Compare estimators

Replace the estimators of a mode with a comma-separated list:

```bash
pytest --bm --bm-mode npe --bm-estimators nsf,zuko_nsf --bm-seeds 3 -n auto
```

### Rerun one case quickly

The test ids have the form `<method>-<settings>-<task>`, with a `seed<N>` part before
the task when you use `--bm-seeds`. Select cases with `-k`, and use a separate label
for quick runs with a small budget:

```bash
pytest --bm --bm-mode npe -k "NPE_C-mdn and slcp" --bm-num-simulations 500 --bm-label quick
```

To keep results from before and after a change on the same branch, use two labels,
e.g. `--bm-label before` and `--bm-label after`.

## Options

| Option | Default | Description |
|---|---|---|
| `--bm` | off | Run the mini-sbibm tests (and only those). |
| `--bm-mode` | default run | Method group to run, see the list above. |
| `--bm-estimators` | mode default | Comma-separated estimators for the selected mode. |
| `--bm-num-simulations` | 2000 | Training simulations per case. Sequential methods split them over two rounds. |
| `--bm-seeds` | 1 | Training seeds per case. |
| `--bm-label` | git branch | Name of the run in the results table. |
| `--bm-results-dir` | `.bm_results` | Folder of the results file. |

## Adding a mode or a task

Modes are defined in `tests/bm_test.py`: add the methods to `METHOD_GROUPS`, their
settings to `METHOD_PARAMS`, and, for `--bm-estimators`, the estimator argument to
`ESTIMATOR_ARGUMENTS`. Add sequential modes to `SEQUENTIAL_MODES`.

Tasks live in `tests/mini_sbibm/`. A task subclasses `Task` and provides a prior, a
simulator, and 10 observations with at least 1000 reference posterior samples each,
either from files or computed (see `gaussian_linear.py`). Register it in
`tests/mini_sbibm/__init__.py` and add it to `TASKS` in `tests/bm_test.py`.
