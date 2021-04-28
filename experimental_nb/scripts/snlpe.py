import torch
from sbi import inference as inference
from sbi.utils.get_nn_models import likelihood_nn
from sbibm.algorithms.sbi.utils import (
    wrap_posterior,
    wrap_prior_dist,
    wrap_simulator_fn,
)

from sbibm.metrics import c2st, ksd, mmd, ppc, mvn_kl
import sbibm
import pandas as pd

import logging
import math
from typing import Any, Dict, Optional, Tuple
from sbi.inference.snle.snle_c import SNLE_C
from sbi.inference.posteriors.variational_posterior import build_flow


from sbibm.tasks.task import Task


def run(
    task: Task,
    num_samples: int,
    num_simulations: int,
    num_observation: Optional[int] = None,
    observation: Optional[torch.Tensor] = None,
    num_rounds: int = 10,
    neural_net: str = "maf",
    hidden_features: int = 50,
    simulation_batch_size: int = 1000,
    training_batch_size: int = 10000,
    automatic_transforms_enabled: bool = True,
    z_score_x: bool = True,
    z_score_theta: bool = True,
    link_support: bool = False,
    num_flows: int = 5,
    num_steps: int = 1001,
    type: str = "iaf",
) -> Tuple[torch.Tensor, int, Optional[torch.Tensor]]:
    """Runs (S)NLE from `sbi`
Args:
    task: Task instance
    num_observation: Observation number to load, alternative to `observation`
    observation: Observation, alternative to `num_observation`
    num_samples: Number of samples to generate from posterior
    num_simulations: Simulation budget
    num_rounds: Number of rounds
    neural_net: Neural network to use, one of maf / mdn / made / nsf
    hidden_features: Number of hidden features in network
    simulation_batch_size: Batch size for simulator
    training_batch_size: Batch size for training network
    automatic_transforms_enabled: Whether to enable automatic transforms
    mcmc_method: MCMC method
    mcmc_parameters: MCMC parameters
    z_score_x: Whether to z-score x
    z_score_theta: Whether to z-score theta
Returns:
    Samples from posterior, number of simulator calls, log probability of true params if computable
"""
    assert not (num_observation is None and observation is None)
    assert not (num_observation is not None and observation is not None)

    log = logging.getLogger(__name__)

    prior = task.prior_dist
    simulator = task.get_simulator()
    obs = task.get_observation(num_observation=num_observation)

    thetas = prior.sample((100,)).detach()
    xs = simulator(thetas)

    infer = SNLE_C(prior, simulator)
    _ = infer.append_simulations(thetas, xs)

    infer.train(
        obs,
        max_num_simulation=num_simulations - 100,
        stop_after_epochs=int(num_simulations / 10) + 100,
        posterior_kwargs={
            "flow": "affine_autoregressive",
            "num_flows": 10,
            "hidden_dims": [50],
        },
    )

    posterior = infer.build_posterior()

    samples = posterior.sample((num_samples,)).detach()

    return samples, num_simulations, "SNLRP"


def evaluate_metric(
    task: Task,
    samples: torch.Tensor,
    num_simulations: int,
    num_observation: int,
    algo: str,
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


def benchmark_single(
    task: str,
    num_observation: int,
    num_simulations=[1000, 10000, 100000],
    num_rounds=[1],
    link_support=True,
    num_flows=10,
    num_steps=1001,
    type="iaf",
):
    # init thinks
    task = sbibm.get_task(task)
    metric_df = pd.DataFrame()
    # Run and evaluate
    for num_round in num_rounds:
        for num_sim in num_simulations:
            samples, num_sim, _ = run(
                task,
                10000,
                num_sim,
                num_observation=num_observation,
                type=type,
                num_rounds=num_round,
                link_support=link_support,
                num_flows=num_flows,
                num_steps=num_steps,
            )
            df = evaluate_metric(task, samples, num_sim, num_observation, "SNLPE")
            metric_df = metric_df.append(df)
    return metric_df
