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
    num_flows : int = 5,
    num_steps : int = 1001,
    type : str = "iaf",
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

    if num_rounds == 1:
        log.info(f"Running NLE")
        num_simulations_per_round = num_simulations
    else:
        log.info(f"Running SNLE")
        num_simulations_per_round = math.floor(num_simulations / num_rounds)

    if simulation_batch_size > num_simulations_per_round:
        simulation_batch_size = num_simulations_per_round
        log.warn("Reduced simulation_batch_size to num_simulation_per_round")

    if training_batch_size > num_simulations_per_round:
        training_batch_size = num_simulations_per_round
        log.warn("Reduced training_batch_size to num_simulation_per_round")

    prior = task.get_prior_dist()
    if observation is None:
        observation = task.get_observation(num_observation)

    simulator = task.get_simulator(max_calls=num_simulations)

    transforms = task._get_transforms(automatic_transforms_enabled)["parameters"]
    prior = wrap_prior_dist(prior, transforms)
    simulator = wrap_simulator_fn(simulator, transforms)

    density_estimator_fun = likelihood_nn(
        model=neural_net.lower(),
        hidden_features=hidden_features,
        z_score_x=z_score_x,
        z_score_theta=z_score_theta,
    )
    inference_method = inference.SNLE_B(
        density_estimator=density_estimator_fun, prior=prior,
    )

    posteriors = []
    proposal = prior

    for r in range(num_rounds):
        theta, x = inference.simulate_for_sbi(
            simulator,
            proposal,
            num_simulations=num_simulations_per_round,
            simulation_batch_size=simulation_batch_size,
        )

        density_estimator = inference_method.append_simulations(
            theta, x, from_round=r
        ).train(
            training_batch_size=training_batch_size,
            retrain_from_scratch_each_round=False,
            discard_prior_samples=False,
            show_train_summary=True,
        )
        posterior = inference_method.build_posterior(
            density_estimator)
        # Current bug
        posterior._x_shape = observation.shape
        posterior.q = build_flow(posterior._prior, permute=True, type=type, num_flows=num_flows, link_support=link_support)  # TODO Fix this 
        proposal = posterior.set_default_x(observation)
        posterior.train(observation, steps=num_steps, clip_value=5., lr=1e-2)
        # Copy hyperparameters, e.g., mcmc_init_samples for "latest_sample" strategy.
        posteriors.append(posterior)

    posterior = wrap_posterior(posteriors[-1], transforms)

    assert simulator.num_simulations == num_simulations

    samples = posterior.sample((num_samples,)).detach()

    return samples, simulator.num_simulations, inference_method


def evaluate_metric(task: Task, samples: torch.Tensor, num_simulations: int, num_observation: int, algo: str):
    r""" Will evaluate the metrics c2st, mmd and mean_dist for a given set of samples """
    reference_samples = task.get_reference_posterior_samples(num_observation)
    c2st_accuracy = float(c2st(reference_samples, samples))
    mmd_metric = float(mmd(reference_samples, samples))
    median_dist = float(ppc.median_distance(reference_samples, samples))

    df = pd.DataFrame({"algorithm": [algo], "num_simulations": [num_simulations], "c2st": [c2st_accuracy], "mmd":[mmd_metric], "median_dist":[median_dist]})
    return df 

def benchmark_single(task:str, num_observation:int, num_simulations = [1000, 10000, 100000],
num_rounds = [1,10], link_support=True, num_flows=10, num_steps=1001, type="iaf"):
    # init thinks
    task = sbibm.get_task(task) 
    metric_df = pd.DataFrame()
    # Run and evaluate
    for num_round in num_rounds:
        for num_sim in num_simulations:
            samples, num_sim, _ = run(task, 10000, num_sim, num_observation=num_observation, type=type, num_rounds=num_round, link_support=link_support, num_flows=num_flows, num_steps=num_steps)
            if num_round == 1:
                df = evaluate_metric(task, samples, num_sim, num_observation, "NLVI")
            else:
                df = evaluate_metric(task, samples, num_sim, num_observation, "SNLVI")
            metric_df = metric_df.append(df)
    return metric_df