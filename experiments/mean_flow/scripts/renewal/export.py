"""Evaluation datasets of the paper's experiment 3 and a training set for sbi NPE.

Run with the paper's interpreter. The 10 datasets (truth, observation, missingness
pattern) are rebuilt exactly as `experiments/exp3_renewal.py` draws them: truths and
observations from `child_rng(30_000 + k)`, patterns from `child_rng(40_000 + k)`, where
`child_rng(o)` is `default_rng(SEED + 1000 * o)`. The training set has the paper's size
(8,192 complete simulations, `TRAIN_SIZE` in `config_renewal.py`) from a stream the
paper does not use; NPE draws the missingness patterns itself.

    cd ~/meanflow-paper && .venv-pypi/bin/python \\
        <sbi>/experiments/mean_flow/scripts/renewal/export.py --out data.npz
"""

import argparse
import sys
from pathlib import Path

sys.path[:0] = [str(Path.cwd()), str(Path.cwd() / "experiments")]

import numpy as np  # noqa: E402

from experiments.config_run import N_DATASETS, SEED  # noqa: E402
from experiments.tasks.config_renewal import N_OBS, TRAIN_SIZE, Renewal  # noqa: E402


def child_rng(offset: int) -> np.random.Generator:
    return np.random.default_rng(SEED + 1000 * offset)


parser = argparse.ArgumentParser()
parser.add_argument("--out", type=Path, required=True)
args = parser.parse_args()

task = Renewal()
truths, observations, patterns = [], [], []
for index in range(N_DATASETS):
    truth = task.sample_prior(1, child_rng(30_000 + index))[0]
    sampled_truth, data = task.sample_observed_dataset(N_OBS, child_rng(30_000 + index))
    np.testing.assert_array_equal(sampled_truth, truth)
    truths.append(truth)
    observations.append(data[0])
    patterns.append(task.present_mask(1, child_rng(40_000 + index))[0])

train = task.simulate_dataset(TRAIN_SIZE, child_rng(70_000))
theta_train = np.concatenate([train[name] for name in task.param_names], axis=1)
np.savez(
    args.out,
    theta=np.stack(truths),
    x=np.stack(observations),
    present=np.stack(patterns),
    theta_train=theta_train,
    x_train=train["y"],
    missing_rate=task.missing_rate,
)
print(
    f"{N_DATASETS} datasets, {len(theta_train)} training simulations, "
    f"theta {theta_train.shape[1]}-d, x {train['y'].shape[1]}-d, "
    f"{np.stack(patterns).mean():.2f} of coordinates present"
)
