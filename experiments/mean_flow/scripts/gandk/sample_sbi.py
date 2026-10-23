"""NUTS on one sbi likelihood estimator for one g-and-k dataset, as the paper runs it.

    python experiments/mean_flow/scripts/gandk/sample_sbi.py \
        --model ~/meanflow-runs/sbi-train/mf_sbi_s0/last.pt --steps 2 \
        --label mf_sbi_s0_last-2 --dataset 0 \
        --data ~/meanflow-runs/gandk-data --out ~/meanflow-runs/sbi-mcmc

Same kernel and budget as the paper's experiment 2 (`exp2_gnk.run_arm`): nutpie
0.16.11 NUTS, 4 chains in threads, diagonal mass matrix, nutpie's defaults otherwise;
8,000 / S draws per chain for a density that costs S network evaluations, warmup 25%
of the request, chain seed `42 + S + 10_000 * dataset` (S = 0 for a one-pass flow,
as `arm.steps or 0` there). The box prior enters as PyMC's `Uniform` does: NUTS runs
on the logit-transformed parameters with the log-Jacobian added, and each chain
starts at the box centre jittered by U(-1, 1) in that space.

The density is a Python function (`nutpie.compiled_pyfunc.from_pyfunc`) returning
the sum of the estimator's log-likelihood over the 50 observations and its gradient
from `torch.autograd`. Writes `<out>/<label>/d<dataset>.npz` with the draws and the
sampler statistics the paper's `mcmc_summary` reads; the metrics are computed with
their code (`score.py`).
"""

import argparse
import time
from pathlib import Path

import numpy as np
import nutpie
import torch

from sbi.neural_nets.estimators import MeanFlowEstimator

SEED = 42
BASE_DRAWS = 8_000
TUNE_SHARE = 0.25
CHAINS = 4


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=None, help="mean flow only")
    parser.add_argument("--label", required=True)
    parser.add_argument("--dataset", type=int, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--draws", type=int, help="override, for smoke tests")
    args = parser.parse_args()

    target = args.out / args.label / f"d{args.dataset}.npz"
    if target.exists():
        print(f"{target} exists, skipping")
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.threads)

    estimator = torch.load(args.model, weights_only=False).eval()
    if isinstance(estimator, MeanFlowEstimator):
        if args.steps is None:
            raise SystemExit("a mean flow needs --steps (its cost per density is S)")
        estimator.num_steps = args.steps
        kind, nfe_per_density = "mf", args.steps
    else:
        if args.steps is not None:
            raise SystemExit(f"--steps given for a one-pass {type(estimator).__name__}")
        kind, nfe_per_density = "nf", 1

    train = np.load(args.data / "train.npz")
    lower = torch.as_tensor(train["lower"], dtype=torch.float64)
    upper = torch.as_tensor(train["upper"], dtype=torch.float64)
    observations = np.load(args.data / "datasets.npz")["observations"][args.dataset]
    x = torch.as_tensor(observations, dtype=torch.float32).unsqueeze(1)
    calls = [0]

    def make_logp_fn():
        def logp(z_np, **_):
            calls[0] += 1
            z = torch.tensor(z_np, dtype=torch.float64, requires_grad=True)
            theta = lower + (upper - lower) * torch.sigmoid(z)
            log_jacobian = (
                torch.log(upper - lower)
                + torch.nn.functional.logsigmoid(z)
                + torch.nn.functional.logsigmoid(-z)
            ).sum()
            log_likelihood = estimator.log_prob(x, theta.float()[None]).sum()
            value = log_likelihood.double() + log_jacobian
            (grad,) = torch.autograd.grad(value, z)
            return float(value), grad.numpy()

        return logp

    def make_expand_fn(seed1, seed2, chain):
        lower_np, upper_np = lower.numpy(), upper.numpy()

        def expand(z_np, **_):
            return {"theta": lower_np + (upper_np - lower_np) / (1 + np.exp(-z_np))}

        return expand

    def initial_point(seed):
        return np.random.default_rng(seed).uniform(-1.0, 1.0, size=3)

    model = nutpie.compiled_pyfunc.from_pyfunc(
        3,
        make_logp_fn,
        make_expand_fn,
        [np.dtype("float64")],
        [(3,)],
        ["theta"],
        make_initial_point_fn=initial_point,
    )
    # exp2's draws_at_cost
    draws = args.draws or max(1, round(BASE_DRAWS / nfe_per_density))
    tune = max(1, round(TUNE_SHARE * draws))
    seed = SEED + (args.steps or 0) + 10_000 * args.dataset

    started = time.perf_counter()
    trace = nutpie.sample(
        model,
        draws=draws,
        tune=tune,
        chains=CHAINS,
        cores=CHAINS,
        seed=seed,
        save_warmup=True,
        progress_bar=False,
        adaptation="diag",
    )
    wall = time.perf_counter() - started

    def stat(group: str, name: str) -> np.ndarray:
        return np.asarray(trace[group][name].values)

    np.savez(
        target,
        theta=np.asarray(trace["posterior"]["theta"].values),
        n_steps=stat("sample_stats", "n_steps").astype(np.int64),
        warmup_n_steps=stat("warmup_sample_stats", "n_steps").astype(np.int64),
        diverging=stat("sample_stats", "diverging"),
        warmup_diverging=stat("warmup_sample_stats", "diverging"),
        step_size=stat("sample_stats", "step_size"),
        wall_seconds=wall,
        logp_calls=calls[0],
        draws=draws,
        tune=tune,
        seed=seed,
        nfe_per_density=nfe_per_density,
        kind=kind,
        source="sbi",
        model=str(args.model),
    )
    print(
        f"{args.label} d{args.dataset}: {draws} draws x {CHAINS} chains "
        f"in {wall:.0f}s, {calls[0]} density calls"
    )


if __name__ == "__main__":
    main()
