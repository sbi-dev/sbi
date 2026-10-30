"""The paper's experiment-3 arms on the tractable missing-data task (`wave_task.py`).

Runs with the paper's code and interpreter, which stay unchanged. Their experiment 3
(`experiments/exp3_renewal.py`) is the template: a mean flow on BayesFlow's diffusion
transformer that takes a missing-target mask (`mf_dit`), against a spline coupling flow
shown -1 where a value is missing (`nf`), both sampled by NUTS at S=1. Two changes:

- the transformer is their `ForkableDiffusionTransformer` without the causal attention
  mask (`causal_dit` requires the parameters to span the observation's time grid, which
  4 global parameters do not); it is swapped in for `CausalDiffusionTransformer` where
  their trainer builds the subnet, so the checkpoint still records `causal_dit`;
- the coupling flow has the default depth 5 (their renewal task uses 2 to avoid
  divergence at 123 dimensions) and up to 1,000 epochs with their early stopping
  (patience 10, best weights), as long as the mean flow's 1,000.

The evaluation datasets and patterns are those of `make_reference.py`. Each dataset's
posterior draws go to `<out>/paper_<arm>_d<index>.npz`. `--seed k` trains a replicate
with the paper's seed 42 + k; give each seed its own `--out`, `--results-dir` and
`--checkpoint-dir`.

    cd ~/meanflow-paper && CUDA_VISIBLE_DEVICES=1 \\
        XLA_PYTHON_CLIENT_PREALLOCATE=false .venv-pypi/bin/python -I \\
        <sbi>/experiments/mean_flow/scripts/missing/paper_arms.py \\
        --paper-dir ~/meanflow-paper --out ~/meanflow-runs/missing/paper \\
        --results-dir ~/meanflow-runs/missing/results \\
        --checkpoint-dir ~/meanflow-runs/missing/ckpt --train-only --verbose 2
    # then the same with --dataset-index <i> instead of --train-only
"""

import json
import sys
from pathlib import Path

argv = sys.argv[1:]
paper = Path(argv[argv.index("--paper-dir") + 1]).expanduser()
out = Path(argv[argv.index("--out") + 1]).expanduser()
seed = int(argv[argv.index("--seed") + 1]) if "--seed" in argv else 0
mf_best_epoch = "--mf-best-epoch" in argv
if mf_best_epoch:
    argv.remove("--mf-best-epoch")
for flag in ("--paper-dir", "--out", "--seed"):
    if flag in argv:
        position = argv.index(flag)
        del argv[position : position + 2]
sys.path[:0] = [str(paper), str(paper / "experiments"), str(Path(__file__).parent)]

# A training seed other than 0 replaces the paper's global SEED (42) before any of
# their modules read it: it changes the simulations, the initialization and the chains.
import experiments.config_run as paper_config  # noqa: E402

paper_config.SEED += seed

import numpy as np  # noqa: E402
import wave_task  # noqa: E402
from common import (  # noqa: E402  (sets KERAS_BACKEND)
    DATASET,
    SettingProgress,
    log_parameter_counts,
    n_datasets,
    prior_truth_provenance,
    run_training_stage,
    start_run,
)
from exp3_renewal import pymc_model_gaussian  # noqa: E402
from inductive_bias import (  # noqa: E402
    arm_networks,
    encode_for,
    fixed_step_distributions,
    run_dataset,
)
from mean_flow.diffusion_transformer import ForkableDiffusionTransformer  # noqa: E402

import experiments.training as paper_training  # noqa: E402
from experiments.config_run import (  # noqa: E402
    DRAWS,
    DRAWS_SMOKE,
    STEPS,
    TUNE_SMOKE,
    NetworkConfig,
    smoke_choice,
)
from experiments.tasks.base import Task  # noqa: E402

paper_training.CausalDiffusionTransformer = ForkableDiffusionTransformer

ARMS = {"mf_dit": ("mf", "causal_dit"), "nf": ("nf", "time_mlp")}


class KeepBestWeights(paper_training.keras.callbacks.Callback):
    """Restores the weights of the epoch with the lowest validation loss at the end."""

    stopped_epoch = 0  # read by their trainer's summary line

    def on_train_begin(self, logs=None):
        self.best, self.weights, self.best_epoch = float("inf"), None, None

    def on_epoch_end(self, epoch, logs=None):
        loss = (logs or {}).get("val_loss")
        if loss is not None and loss < self.best:
            self.best, self.best_epoch = loss, epoch + 1
            self.weights = self.model.get_weights()

    def on_train_end(self, logs=None):
        if self.weights is not None:
            self.model.set_weights(self.weights)
            print(f"[mf] kept epoch {self.best_epoch}, validation loss {self.best:.4f}")


if mf_best_epoch:
    # The mean flow alone, at its best epoch; the checkpoint name does not record this,
    # so give such a run its own `--checkpoint-dir`.
    paper_stopping = paper_training.early_stopping
    paper_training.early_stopping = lambda kind, config, networks: (
        KeepBestWeights() if kind == "mf" else paper_stopping(kind, config, networks)
    )
    ARMS = {"mf_dit": ARMS["mf_dit"]}
KINDS = {arm: kind for arm, (kind, _) in ARMS.items()}
TRAIN_SIZE = 8192


class MissingWave(Task):
    """`wave_task` in the paper's task interface, with a standard-normal prior."""

    name = "missing_wave"
    has_exact_likelihood = True
    standardize_observations = True
    networks = NetworkConfig(
        mf_subnet="causal_dit",
        mf_widths=(128, 128, 128, 128, 128),
        dit_heads=4,
        missing_rate=wave_task.MISSING_RATE,
        nf_epochs=1000,
    )

    def __init__(self, encode_missing: bool = False):
        self.obs_dim = wave_task.T
        self.param_names = tuple(f"theta_{i}" for i in range(wave_task.NUM_PARAMS))
        self.prior_bounds = {name: (-3.0, 3.0) for name in self.param_names}
        self.true_params = dict.fromkeys(self.param_names, 0.0)
        self.missing_rate = wave_task.MISSING_RATE
        self.missing_fill = wave_task.MISSING_FILL
        self.encode_missing = bool(encode_missing)

    def sample_prior(self, n, rng):
        return rng.standard_normal((int(n), wave_task.NUM_PARAMS))

    def prior_dict(self, rng):
        draw = rng.standard_normal(wave_task.NUM_PARAMS)
        return {name: float(draw[i]) for i, name in enumerate(self.param_names)}

    def log_prior(self, thetas):
        thetas = np.atleast_2d(np.asarray(thetas, dtype=float))
        return -0.5 * np.sum(thetas**2, -1) - 0.5 * thetas.shape[-1] * np.log(2 * np.pi)

    def simulate_batch(self, thetas, rng, **kwargs):
        observations = wave_task.simulate(np.atleast_2d(thetas), rng)
        if not self.encode_missing:
            return observations
        present = wave_task.present_mask(len(observations), rng)
        return self.with_sentinel(observations, present, self.missing_fill)

    def log_likelihood_pointwise(self, thetas, data):
        thetas = np.atleast_2d(np.asarray(thetas, dtype=float))
        everything = np.ones(wave_task.T, dtype=bool)
        return np.stack(
            [
                wave_task.log_likelihood(thetas, y, everything)
                for y in np.atleast_2d(data)
            ],
            axis=-1,
        )

    def present_mask(self, n, rng):
        return wave_task.present_mask(n, rng)

    @staticmethod
    def with_sentinel(y, present, fill):
        present = np.asarray(present, dtype=bool)
        return np.where(present, np.asarray(y, float), float(fill))


def main():
    ctx, mode, indices = start_run("exp3_missing_wave", __doc__, argv)
    task, encoded = MissingWave(), MissingWave(encode_missing=True)
    tasks = {arm: encoded if kind == "nf" else task for arm, kind in KINDS.items()}
    ctx.config.offline_size = smoke_choice(ctx.smoke, TRAIN_SIZE, 256)
    draws = smoke_choice(ctx.smoke, DRAWS, DRAWS_SMOKE)
    tune = smoke_choice(ctx.smoke, None, TUNE_SMOKE)
    sets = wave_task.datasets(n_datasets(ctx.args))

    def provenance(index):
        return prior_truth_provenance(sets[index]["theta"]) | {
            "present_mask": [int(v) for v in sets[index]["present"]]
        }

    models, skipped, _, early = run_training_stage(
        ctx,
        task,
        mode,
        indices,
        KINDS,
        provenance_of=provenance,
        networks_of=lambda arm: arm_networks(ctx, task, ARMS, arm),
        task_of=lambda arm: tasks[arm],
        require_all=False,
    )
    if early is not None:
        return early
    ctx.log(f"skipped arms: {skipped}")
    log_parameter_counts(
        ctx, task, KINDS, networks_of=lambda arm: arm_networks(ctx, task, ARMS, arm)
    )
    arms = tuple(models)
    out.mkdir(parents=True, exist_ok=True)
    for index in indices if mode == DATASET else range(len(sets)):
        present, data = sets[index]["present"], sets[index]["x"][None, :]
        progress = SettingProgress(ctx, index, provenance=provenance(index))
        distributions = fixed_step_distributions(models, task, KINDS, STEPS, present)
        results = run_dataset(
            ctx,
            task,
            arms,
            KINDS,
            index=index,
            data_of=lambda arm, data=data, present=present: encode_for(
                task, KINDS[arm], data, present
            ),
            truth=sets[index]["theta"],
            distributions=distributions,
            progress=progress,
            draws=draws,
            tune=tune,
            steps=STEPS,
            model_of=pymc_model_gaussian,
        )
        for arm, result in results.items():
            row = {k: v for k, v in result["row"].items() if np.isscalar(v)}
            np.savez(
                out / f"paper_{arm}_d{index}.npz",
                samples=np.asarray(result["samples"]),
                row=json.dumps(row),
            )
            ctx.log(
                f"dataset {index} {arm}: {len(result['samples'])} draws, "
                f"wall {row.get('wall_seconds')}"
            )


if __name__ == "__main__":
    main()
