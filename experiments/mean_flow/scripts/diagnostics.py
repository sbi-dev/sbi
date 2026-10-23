"""Density/sampler consistency of a trained mean flow (plan Step 4).

On mini-sbibm two_moons, in NPE (q(theta|x)) and NLE (q(x|theta)) direction:

1. Round trip: held-out z -> forward maps -> backward maps, error in standardized
   units.
2. Mass across conditions: integral of exp(log q) on a fine grid, per condition.
   Spread across conditions biases NLE posteriors; a constant offset does not.
3. Sampler vs density (NPE): C2ST between direct samples and MCMC samples that
   target the estimator's log_prob. 0.5 means both describe the same distribution.

    python experiments/mean_flow/scripts/diagnostics.py --out diagnostics.json \
        --train-kwargs '{"learning_rate": 1e-4, "max_num_epochs": 600, ...}'
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path.cwd()))
sys.path.insert(0, str(Path(__file__).parent))

from mean_flow_paper_trainer import train_paper_recipe  # noqa: E402

from sbi.inference import NLE, NPE  # noqa: E402
from sbi.inference.posteriors.posterior_parameters import (  # noqa: E402
    MCMCPosteriorParameters,
)
from sbi.neural_nets import MeanFlowConfig  # noqa: E402
from sbi.utils.metrics import c2st  # noqa: E402
from tests.mini_sbibm import get_task  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--num-simulations", type=int, default=2000)
parser.add_argument("--steps", default="1,2,4")
parser.add_argument("--train-kwargs", default="{}")
parser.add_argument("--trainer", choices=["sbi", "paper"], default="paper")
parser.add_argument("--epochs", type=int, default=400)
parser.add_argument("--threads", type=int, default=4)
parser.add_argument("--out", required=True)
args = parser.parse_args()
torch.set_num_threads(args.threads)
steps = [int(s) for s in args.steps.split(",")]

task = get_task("two_moons")
torch.manual_seed(0)
theta, x = task.get_data(args.num_simulations)
theta_test, x_test = task.get_data(200)
prior = task.get_prior()
results: dict = {
    "trainer": args.trainer,
    "train_kwargs": args.train_kwargs,
    "steps": steps,
}


def round_trip(estimator, input, condition, num_steps):
    with torch.no_grad():
        z = estimator._standardize(input)
        emb = estimator.embedding_net(condition)
        times = estimator._time_schedule(num_steps, z)
        z_noise = z
        for s, t in zip(times[:-1], times[1:], strict=True):
            z_noise = estimator._flow_map(z_noise, emb, s, t)
        z_back = z_noise
        for s, t in zip(times.flip(0)[:-1], times.flip(0)[1:], strict=True):
            z_back = estimator._flow_map(z_back, emb, s, t)
        error = (z_back - z).norm(dim=-1)
    return {
        "median": round(float(error.median()), 4),
        "q99": round(float(error.quantile(0.99)), 4),
    }


def grid_mass(estimator, condition, low, high, num_steps, resolution=256):
    """Integral of exp(log q(. | condition)) over the box [low, high]."""
    axes = [
        torch.linspace(lo, hi, resolution) for lo, hi in zip(low, high, strict=False)
    ]
    cell = torch.prod((high - low) / (resolution - 1))
    grid = torch.cartesian_prod(*axes)
    log_probs = []
    with torch.no_grad():
        for chunk in grid.split(8192):
            log_probs.append(
                estimator.log_prob(chunk, condition.expand(len(chunk), -1), num_steps)[
                    0
                ]
            )
    return float(torch.cat(log_probs).exp().sum() * cell)


for mode, method in (("npe", NPE), ("nle", NLE)):
    torch.manual_seed(1)
    inference = method(
        prior, density_estimator=MeanFlowConfig(), show_progress_bars=False
    )
    start = time.perf_counter()
    if args.trainer == "sbi":
        estimator = inference.append_simulations(theta, x).train(
            **json.loads(args.train_kwargs)
        )
        results[f"{mode}_epochs"] = inference.summary["epochs_trained"][-1]
    else:
        inference.append_simulations(theta, x)
        train_input, train_condition = (theta, x) if mode == "npe" else (x, theta)
        estimator, best_epoch = train_paper_recipe(
            MeanFlowConfig().build(train_input, train_condition),
            train_input,
            train_condition,
            args.epochs,
        )
        results[f"{mode}_epochs"] = best_epoch
    results[f"{mode}_train_time"] = round(time.perf_counter() - start, 1)
    estimator.eval()
    input, condition = (theta_test, x_test) if mode == "npe" else (x_test, theta_test)

    for num_steps in steps:
        key = f"{mode}_S{num_steps}"
        results[key] = {
            "round_trip": round_trip(estimator, input, condition, num_steps)
        }

        masses = []
        for i in range(20):
            if mode == "npe":
                low, high = torch.tensor([-1.0, -1.0]), torch.tensor([1.0, 1.0])
            else:
                # The two-moons likelihood is a thin half ring around its mean.
                simulated = task.get_simulator()(theta_test[i].expand(2000, -1))
                low, high = simulated.min(0).values - 0.1, simulated.max(0).values + 0.1
            masses.append(
                grid_mass(estimator, condition[i : i + 1], low, high, num_steps)
            )
        masses_t = torch.tensor(masses)
        results[key]["mass"] = {
            "mean": round(float(masses_t.mean()), 4),
            "std": round(float(masses_t.std()), 4),
            "min": round(float(masses_t.min()), 4),
            "max": round(float(masses_t.max()), 4),
        }

        if mode == "npe":
            estimator.num_steps = num_steps
            direct = inference.build_posterior(estimator)
            mcmc = inference.build_posterior(
                estimator,
                sample_with="mcmc",
                posterior_parameters=MCMCPosteriorParameters(
                    method="slice_np_vectorized", num_chains=20, warmup_steps=200
                ),
            )
            scores = []
            for idx in range(1, 6):
                x_o = task.get_observation(idx)
                a = direct.sample((1000,), x=x_o, show_progress_bars=False)
                b = mcmc.sample((1000,), x=x_o, show_progress_bars=False)
                scores.append(round(float(c2st(a, b)), 3))
            results[key]["direct_vs_mcmc_c2st"] = scores
        print(key, json.dumps(results[key]), flush=True)

Path(args.out).write_text(json.dumps(results, indent=2))
