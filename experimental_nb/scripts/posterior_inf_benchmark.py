import sbi
from sbi.utils.get_nn_models import likelihood_nn
from sbi.analysis import pairplot
from sbi import inference

import sbibm
from sbibm.metrics import c2st, ksd, mmd, ppc, mvn_kl

import torch
from torch import nn

import numpy as np
import pandas as pd
import time

import multiprocessing

import argparse

parser = argparse.ArgumentParser(description="Benchmark VI and MCMC methods.")
parser.add_argument(
    "task",
    type=str,
    nargs=1,
    help="One of: two_moons, linear_gaussian, slcp, gaussian_mixture, bernoulli_glm",
)
parser.add_argument(
    "method", type=str, help="One of vi or mcmc", nargs=1,
)
parser.add_argument(
    "out", type=str, help="Output file", nargs=1,
)
parser.add_argument(
    "--out_samples",
    type=str,
    help="Output directory of samples",
    default="samples/",
)
parser.add_argument(
    "--num_repeat",
    type=int,
    help="Number of repeats",
    default=1,
)
parser.add_argument(
    "--num_chains",
    type=int,
    help="Number of mcmc chains - only used if method is mcmc",
    default=1,
)
parser.add_argument("--num_observation", help="Number of observation", default=1)
parser.add_argument("--num_flows", type=int, help="Number of flows", default=5)
parser.add_argument(
    "--num_comps", type=int, help="If > 1 then a mixture of flows is used", default=1
)
parser.add_argument(
    "--type_flow", type=str, help="Type of the flow", default="spline_autoregressive"
)
parser.add_argument("--loss", type=str, help="type of loss", default="elbo")
parser.add_argument(
    "--n_particles", type=int, help="Number of elbo particles", default=128
)
parser.add_argument(
    "--max_num_iters", type=int, help="Number of elbo gradient stes", default=1000
)
parser.add_argument(
    "--alpha", type=float, help="Alpha when renjey divergence is used", default=0.5
)
parser.add_argument(
    "--beta", type=float, help="Alpha when tail adaptive loss is used", default=-0.5
)


def run_mcmc_bm(args):
    task = args.task[0]
    num_chains = args.num_chains
    num_observation = args.num_observation
    inf, task = get_model(task, "mcmc")
    x_obs = task.get_observation(num_observation)
    post = inf.build_posterior(mcmc_method="slice_np_vectorized")
    post.set_default_x(x_obs)

    samples = post.sample((10000,), x_obs, mcmc_parameters={"num_chains": num_chains})
    metrics = evaluate_metric(task, samples, 10000, num_observation, "SNL_mcmc")
    return samples, metrics


def run_vi_bm(args):
    task = args.task[0]
    loss = args.loss
    num_observation = args.num_observation
    num_flows = args.num_flows
    num_comps = args.num_comps
    type_flow = args.type_flow

    inf, task = get_model(task, "vi")
    x_obs = task.get_observation(num_observation)
    post = inf.build_posterior(num_flows=num_flows, num_comps=num_comps, flow=type_flow)
    post.set_default_x(x_obs)
  
    if loss == "elbo":
        print("Elbo is optimized")
        post.train(
            loss=loss, n_particles=args.n_particles, max_num_iters=args.max_num_iters
        )
    elif loss == "renjey_divergence":
        print("Renjey is optimized")
        post.train(
            loss=loss,
            n_particles=args.n_particles,
            alpha=args.alpha,
            max_num_iters=args.max_num_iters,
        )
    elif loss == "tail_adaptive_fdivergence":
        print("TailAptivefdiv is optimized")
        post.train(
            loss=loss,
            n_particles=args.n_particles,
            beta=args.beta,
            max_num_iters=args.max_num_iters,
        )
    else:
        raise NotImplementedError("Unknown loss")

    samples = post.sample((10000,))
    metrics = evaluate_metric(task, samples, 10000, num_observation, "SNL_vi")
    return samples, metrics


def get_model(task_name, method):
    task = sbibm.get_task(task_name)
    prior = task.prior_dist
    simulator = task.get_simulator()
    observation = task.get_observation(num_observation=1)

    thetas = prior.sample((10000,))
    xs = simulator(thetas)

    # VI inference
    if method == "vi":
        inf_vi = inference.SNLVI(prior)
        density_estimator = inf_vi.append_simulations(thetas, xs).train(
            max_num_epochs=0
        )
        state_dict = torch.load(f"likelihood_models/{task_name}.net")
        density_estimator = inf_vi._neural_net
        density_estimator.load_state_dict(state_dict())
        return inf_vi, task
    else:
        # MCMC inference
        inf_mcmc = inference.SNL(prior)
        density_estimator = inf_mcmc.append_simulations(thetas, xs).train(
            max_num_epochs=0
        )
        state_dict = torch.load(f"likelihood_models/{task_name}.net")
        density_estimator = inf_mcmc._neural_net
        density_estimator.load_state_dict(state_dict())
        return inf_mcmc, task


def evaluate_metric(
    task, samples: torch.Tensor, num_simulations: int, num_observation: int, algo: str,
):
    r""" Will evaluate the metrics c2st, mmd and mean_dist for a given set of samples """
    reference_samples = task.get_reference_posterior_samples(num_observation)
    c2st_accuracy = float(c2st(reference_samples, samples))
    mmd_metric = float(mmd(reference_samples, samples))
    median_dist = float(ppc.median_distance(reference_samples, samples))

    df = pd.DataFrame(
        {
            "algorithm": [algo],
            "num_simulations": [num_simulations],
            "c2st": [c2st_accuracy],
            "mmd": [mmd_metric],
            "median_dist": [median_dist],
        }
    )
    return df


if __name__ == "__main__":
    args = parser.parse_args()
    iters = args.num_repeat
    method = args.method[0]
    out = args.out[0]
    for _ in range(iters):
        if method == "mcmc":
            print("Running MCMC bm -----------------------------------------")
            samples, metrics = run_mcmc_bm(args)
        else:
            print("Running VI bm -----------------------------------------")
            samples, metrics = run_vi_bm(args)
        print(metrics)
        print(samples)
        with open(out, "a") as f:
            metrics.to_csv(f, mode="a", header=f.tell() == 0)
        samples = np.array(samples)
        np.save(f"{str(args.out_samples)}samples_{method}_{str(args.task[0])}_{str(time.time())}", samples)
