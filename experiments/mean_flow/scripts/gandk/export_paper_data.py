"""Export the g-and-k data of Arruda et al. (2026) as their experiment 2 draws it.

Runs in the paper's environment and imports their code without changing it:

    cd ~/meanflow-paper && .venv-pypi/bin/python -I \
        <sbi>/experiments/mean_flow/scripts/gandk/export_paper_data.py \
        --paper-dir ~/meanflow-paper --out ~/meanflow-runs/gandk-data \
        --checkpoint-dir ~/meanflow-runs/phase1/checkpoints --verbose 2

Writes `train.npz` (the 100,000 training and 1,024 validation simulations of their
training stage), `datasets.npz` (the 10 true parameters and their 50 observations) and
`tarp.npz` (their 2,000 TARP validation pairs). `--verbose` must be the value the
training stage ran with: it enters their `TrainConfig`, whose hash names the
checkpoints, and finding them confirms that the exported training set is theirs.
"""

import argparse
import sys
from pathlib import Path

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--paper-dir", type=Path, required=True)
parser.add_argument("--out", type=Path, required=True)
parser.add_argument("--checkpoint-dir", type=Path, required=True)
parser.add_argument("--verbose", type=int, required=True)
args = parser.parse_args()

paper = args.paper_dir.resolve()
sys.path[:0] = [str(paper), str(paper / "experiments")]

from common import validation_pairs  # noqa: E402  (sets KERAS_BACKEND)

from experiments.config_run import N_DATASETS, N_VALIDATION, SEED  # noqa: E402
from experiments.tasks.config_gnk import N_OBS, TRAIN_SIZE, GnK  # noqa: E402
from experiments.training import (  # noqa: E402
    TrainConfig,
    checkpoint_path,
    offline_training_data,
)


def child_rng(offset: int) -> np.random.Generator:
    """`RunContext.child_rng`, which needs a whole run context to call."""
    return np.random.default_rng(SEED + 1000 * offset)


task = GnK()
# resolve_train_config(args) followed by exp2's override of offline_size
config = TrainConfig(seed=SEED, verbose=max(args.verbose - 1, 0))
config.offline_size = TRAIN_SIZE
for kind in ("mf", "nf", "fm_trained"):
    path = checkpoint_path(task, kind, args.checkpoint_dir, config=config)
    print(
        f"{kind}: checkpoint {'found' if path.exists() else 'MISSING'} at {path.name}"
    )


def split(pairs: dict) -> tuple[np.ndarray, np.ndarray]:
    theta = np.concatenate(
        [np.asarray(pairs[n], float).reshape(-1, 1) for n in task.param_names], 1
    )
    return theta, np.asarray(pairs["y"], float)


training, validation = offline_training_data(task, config)
theta_train, x_train = split(training)
theta_val, x_val = split(validation)

truths, observations = [], []
for index in range(N_DATASETS):
    truth, data = task.sample_observed_dataset(N_OBS, child_rng(20_000 + index))
    truths.append(truth)
    observations.append(data)

tarp_theta, tarp_x = validation_pairs(task, N_VALIDATION, child_rng(7))

args.out.mkdir(parents=True, exist_ok=True)
np.savez(
    args.out / "train.npz",
    theta_train=theta_train,
    x_train=x_train,
    theta_val=theta_val,
    x_val=x_val,
    param_names=np.array(task.param_names),
    lower=task.lower,
    upper=task.upper,
)
np.savez(
    args.out / "datasets.npz",
    truths=np.stack(truths),
    observations=np.stack(observations),
)
np.savez(args.out / "tarp.npz", theta=tarp_theta, x=tarp_x)
print(
    f"train {x_train.shape}, validation {x_val.shape}, "
    f"datasets {np.stack(observations).shape}, "
    f"tarp {tarp_x.shape} -> {args.out}"
)
