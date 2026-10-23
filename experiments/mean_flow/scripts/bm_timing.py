"""Timed mini-sbibm runs for the mean-flow estimator (plan Step 5b).

Same tasks, observations, metrics and default posterior settings as
`tests/bm_test.py`, but training time and posterior sampling time are measured
separately, and a mean flow is trained once and evaluated at several step counts.

Run from an sbi checkout that has `MeanFlowConfig`:

    python experiments/mean_flow/scripts/bm_timing.py --mode nle --task two_moons \
        --estimator mean_flow --seed 0 --out results.jsonl
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
from sbi.neural_nets import MeanFlowConfig, NSFConfig  # noqa: E402
from sbi.utils.metrics import c2st  # noqa: E402
from tests.mini_sbibm import get_task  # noqa: E402

NUM_EVALUATION_OBS = 10  # as tests/bm_test.py
NUM_SAMPLES = 1000

parser = argparse.ArgumentParser()
parser.add_argument("--mode", choices=["nle", "npe"], required=True)
parser.add_argument("--task", required=True)
parser.add_argument("--estimator", choices=["nsf", "mean_flow"], required=True)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--num-simulations", type=int, default=2000)
parser.add_argument("--steps", default="1,2,4", help="mean-flow step counts")
parser.add_argument("--train-kwargs", default="{}", help="JSON for .train()")
parser.add_argument(
    "--trainer",
    choices=["sbi", "paper"],
    default="sbi",
    help="sbi: .train() defaults; paper: AdamW, warmup + cosine decay, clip 1.5",
)
parser.add_argument("--epochs", type=int, default=400, help="paper trainer only")
parser.add_argument("--label", default="")
parser.add_argument("--threads", type=int, default=2)
parser.add_argument("--out", required=True)
args = parser.parse_args()
torch.set_num_threads(args.threads)

task = get_task(args.task)
torch.manual_seed(args.seed)
theta, x = task.get_data(args.num_simulations)
prior = task.get_prior()
config = MeanFlowConfig() if args.estimator == "mean_flow" else NSFConfig()
method = NLE if args.mode == "nle" else NPE
inference = method(prior, density_estimator=config, show_progress_bars=False)


start = time.perf_counter()
if args.trainer == "sbi":
    estimator = inference.append_simulations(theta, x).train(
        **json.loads(args.train_kwargs)
    )
    epochs = inference.summary["epochs_trained"][-1]
else:
    inference.append_simulations(theta, x)
    input, condition = (x, theta) if args.mode == "nle" else (theta, x)
    estimator, epochs = train_paper_recipe(
        config.build(input, condition), input, condition, args.epochs
    )
train_time = time.perf_counter() - start

steps = (
    [int(s) for s in args.steps.split(",")] if args.estimator == "mean_flow" else [None]
)
for num_steps in steps:
    if num_steps is not None:
        estimator.num_steps = num_steps
    # build_posterior works on a deep copy, so set num_steps before building it.
    posterior = inference.build_posterior(estimator)
    sample_time, metrics = 0.0, []
    for idx in range(1, NUM_EVALUATION_OBS + 1):
        x_o = task.get_observation(idx)
        torch.manual_seed(args.seed * 1000 + idx)
        start = time.perf_counter()
        samples = posterior.sample((NUM_SAMPLES,), x=x_o, show_progress_bars=False)
        sample_time += time.perf_counter() - start
        reference = task.get_reference_posterior_samples(idx)
        reference_std = reference.std(0)
        metrics.append({
            "c2st": float(c2st(reference[:NUM_SAMPLES], samples)),
            "mean_err": float(
                ((samples.mean(0) - reference.mean(0)).abs() / reference_std).mean()
            ),
            "std_err": float(
                ((samples.std(0) - reference_std).abs() / reference_std).mean()
            ),
        })
    row = {
        "label": args.label,
        "mode": args.mode,
        "task": args.task,
        "estimator": args.estimator,
        "num_steps": num_steps,
        "seed": args.seed,
        "num_simulations": args.num_simulations,
        "trainer": args.trainer,
        "train_kwargs": args.train_kwargs,
        "epochs": epochs,
        "train_time": round(train_time, 1),
        "sample_time": round(sample_time, 1),
        **{
            key: round(sum(m[key] for m in metrics) / len(metrics), 4)
            for key in metrics[0]
        },
        "c2st_per_obs": [round(m["c2st"], 3) for m in metrics],
    }
    with open(args.out, "a") as f:
        f.write(json.dumps(row) + "\n")
    print(json.dumps(row), flush=True)
