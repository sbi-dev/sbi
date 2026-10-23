"""The paper's coupling flow trained like its mean flow, with the paper's own code.

Their experiment 2 trains the coupling flow for at most 200 epochs with early
stopping (on g-and-k it stops at epoch 38 and keeps epoch 28) against 1,000 epochs
for the mean flow. This trains the same `CouplingFlow` through their
`build_workflow` and BayesFlow's `fit_offline` for 1,000 epochs without early
stopping, so the learning-rate schedule spans 1,000 epochs as the mean flow's does,
and keeps the last and the best (lowest validation loss) weights. It then samples
it with their NUTS path (`pymc_model`, `FixedStepNeuralDistribution`, `sample_nuts`)
at the coupling-flow arm's settings of `exp2_gnk.run_arm`, writing the format of
`sample_sbi.py` so that `score.py` scores it with every other arm.

Runs in the paper's environment and imports their code without changing it:

    cd ~/meanflow-paper && CUDA_VISIBLE_DEVICES=1 XLA_PYTHON_CLIENT_PREALLOCATE=false \
        .venv-pypi/bin/python -I <script> train \
        --paper-dir ~/meanflow-paper --out ~/meanflow-runs/paper-nf-long
    ... <script> sample --weights last --dataset 0 \
        --paper-dir ~/meanflow-paper --out ~/meanflow-runs/paper-nf-long \
        --data ~/meanflow-runs/gandk-data --mcmc ~/meanflow-runs/sbi-mcmc

where <script> is <sbi>/experiments/mean_flow/scripts/gandk/paper_nf_long.py.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("mode", choices=("train", "sample"))
parser.add_argument("--paper-dir", type=Path, required=True)
parser.add_argument("--out", type=Path, required=True)
parser.add_argument("--weights", choices=("last", "best"))
parser.add_argument("--dataset", type=int)
parser.add_argument("--data", type=Path)
parser.add_argument("--mcmc", type=Path)
parser.add_argument("--epochs", type=int, default=1000)
parser.add_argument("--draws", type=int, help="override, for smoke tests")
args = parser.parse_args()

paper = args.paper_dir.resolve()
sys.path[:0] = [str(paper), str(paper / "experiments")]
os.environ.setdefault("KERAS_BACKEND", "jax")

import keras  # noqa: E402
from common import pymc_model, sample_nuts  # noqa: E402  (sets KERAS_BACKEND)
from sampling.pymc_bridge import FixedStepNeuralDistribution  # noqa: E402

from experiments.config_run import CHAINS, SEED  # noqa: E402
from experiments.tasks.config_gnk import (  # noqa: E402
    BASE_DRAWS,
    TRAIN_SIZE,
    TUNE_SHARE,
    GnK,
)
from experiments.training import (  # noqa: E402
    StopOnNonFiniteLoss,
    TrainConfig,
    build_workflow,
    count_params,
    offline_training_data,
)

task = GnK()
names = list(task.param_names)


class KeepBest(keras.callbacks.Callback):
    """The weights of the epoch with the lowest validation loss."""

    def __init__(self):
        super().__init__()
        self.best, self.best_epoch, self.weights = np.inf, 0, None

    def on_epoch_end(self, epoch, logs=None):
        if logs["val_loss"] < self.best:
            self.best, self.best_epoch = float(logs["val_loss"]), epoch + 1
            self.weights = self.model.get_weights()


def train() -> None:
    # resolve_train_config(--verbose 2) followed by exp2's offline_size, as Phase 1
    config = TrainConfig(seed=SEED, verbose=1)
    config.offline_size = TRAIN_SIZE
    networks = task.networks.variant(nf_epochs=args.epochs)
    workflow = build_workflow(task, "nf", config, networks)
    training, validation = offline_training_data(task, config)
    watch, best = StopOnNonFiniteLoss(), KeepBest()
    started = time.perf_counter()
    history = workflow.fit_offline(
        training,
        epochs=args.epochs,
        batch_size=config.batch_size,
        validation_data=validation,
        verbose=2,
        callbacks=[watch, best],
    )
    seconds = time.perf_counter() - started
    if watch.failed_at is not None:
        raise RuntimeError(f"non-finite loss at (epoch, batch) {watch.failed_at}")
    approximator = workflow.approximator
    args.out.mkdir(parents=True, exist_ok=True)
    approximator.save(args.out / "last.keras")
    approximator.set_weights(best.weights)
    approximator.save(args.out / "best.keras")
    record = {
        "train_loss": [float(v) for v in history.history["loss"]],
        "val_loss": [float(v) for v in history.history["val_loss"]],
        "best_epoch": best.best_epoch,
        "best_val_loss": best.best,
        "epochs_run": len(history.history["loss"]),
        "train_seconds": seconds,
        "parameters": count_params(approximator.inference_network),
    }
    (args.out / "history.json").write_text(json.dumps(record))
    print(f"done: best epoch {best.best_epoch}, {seconds / 3600:.2f} h")


def sample() -> None:
    label = f"paper_nf_long_{args.weights}"
    target = args.mcmc / label / f"d{args.dataset}.npz"
    if target.exists():
        print(f"{target} exists, skipping")
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    model_path = args.out / f"{args.weights}.keras"
    approximator = keras.saving.load_model(model_path)
    datasets = np.load(args.data / "datasets.npz")
    data, truth = (
        datasets["observations"][args.dataset],
        datasets["truths"][args.dataset],
    )

    distribution = FixedStepNeuralDistribution(approximator, names, kind="nf")
    model = pymc_model(task, distribution, data)
    distribution.warm_up(data, truth)
    draws = args.draws or BASE_DRAWS
    tune = max(1, round(TUNE_SHARE * draws))
    seed = SEED + 0 + 10_000 * args.dataset
    started = time.perf_counter()
    idata = sample_nuts(model, draws=draws, tune=tune, chains=CHAINS, seed=seed)
    wall = time.perf_counter() - started

    posterior = idata.posterior
    np.savez(
        target,
        theta=np.stack([np.asarray(posterior[p]) for p in names], axis=-1),
        n_steps=np.asarray(idata.sample_stats["n_steps"]).astype(np.int64),
        warmup_n_steps=np.asarray(idata.warmup_sample_stats["n_steps"]).astype(
            np.int64
        ),
        diverging=np.asarray(idata.sample_stats["diverging"]),
        warmup_diverging=np.asarray(idata.warmup_sample_stats["diverging"]),
        step_size=np.asarray(idata.sample_stats["step_size"]),
        wall_seconds=wall,
        logp_calls=-1,
        draws=draws,
        tune=tune,
        seed=seed,
        nfe_per_density=1,
        kind="nf",
        source="paper-retrained",
        model=str(model_path),
    )
    print(f"{label} d{args.dataset}: {draws} draws x {CHAINS} chains in {wall:.0f}s")


train() if args.mode == "train" else sample()
