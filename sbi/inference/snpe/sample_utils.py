import numpy as np
import torch
from torch import distributions
from torch import multiprocessing as mp
from pyro.infer.mcmc import HMC, NUTS
from pyro.infer.mcmc.api import MCMC
from sbi.mcmc import Slice
from sbi.mcmc import SliceSampler
import sbi.utils as utils


def sample_posterior(
        neural_posterior,
        prior,
        num_samples,
        true_observation):
    """
    Sample a posterior.

    Args:
        neural_posterior:
        prior: distribution
        num_samples:  int. Number of samples to generate.
        true_observation: torch.Tensor [observation_dim] or [1, observation_dim]
            Pass true observation for inference.

    Returns:
        torch.Tensor [num_samples, parameter_dim]
        Posterior parameter samples.

    """

    # Always sample in eval mode.
    neural_posterior.eval()

    # Rejection sampling is potentially needed for the posterior.
    # This is because the prior may not have support everywhere.
    # The posterior may also be constrained to the same support,
    # but we don't know this a priori.
    samples = []
    num_remaining_samples = num_samples
    total_num_accepted, _total_num_generated_examples = 0, 0
    while num_remaining_samples > 0:

        # Generate samples from posterior.
        candidate_samples = neural_posterior.sample(
            max(10000, num_samples), context=true_observation.reshape(1, -1)
        ).squeeze(0)

        # Evaluate posterior samples under the prior.
        prior_log_prob = prior.log_prob(candidate_samples)
        if isinstance(prior, distributions.Uniform):
            prior_log_prob = prior_log_prob.sum(-1)

        # Keep those samples which have non-zero probability under the prior.
        accepted_samples = candidate_samples[~torch.isinf(prior_log_prob)]
        samples.append(accepted_samples.detach())

        # Update remaining number of samples needed.
        num_accepted = (~torch.isinf(prior_log_prob)).sum().item()
        num_remaining_samples -= num_accepted
        total_num_accepted += num_accepted

        # Keep track of acceptance rate
        _total_num_generated_examples += candidate_samples.shape[0]

    # Back to training mode.
    neural_posterior.train()

    # Aggregate collected samples.
    samples = torch.cat(samples)

    # Make sure we have the right amount.
    samples = samples[:num_samples, ...]
    assert samples.shape[0] == num_samples

    return samples


def sample_posterior_mcmc(
        neural_posterior,
        prior,
        true_observation,
        num_samples,
        mcmc_method='slice_np',
        thin=10):
    """
    Sample the posterior using MCMC

    Args:
        neural_posterior: nn
        prior: distribution
        true_observation: torch.tensor, observation/context/conditioning
        num_samples: int, Number of samples to generate.
        potential_function: NeuralPotentialFunction
        mcmc_method: Which MCMC method to use ['metropolis-hastings', 'slice', 'hmc', 'nuts']
        thin: thin: Generate (num_samples * thin) samples in total, then select every
            'thin' sample.

    Returns:
        torch.Tensor of shape [num_samples, parameter_dim]

    """

    # HMC and NUTS from Pyro.
    # Defining the potential function as an object means Pyro's MCMC scheme
    # can pickle it to be used across multiple chains in parallel, even if
    # the potential function requires evaluating a neural likelihood as is the
    # case here.
    potential_function = NeuralPotentialFunction(
        neural_posterior, prior, true_observation
    )

    # Axis-aligned slice sampling implementation in NumPy
    target_log_prob = (
        lambda parameters: neural_posterior.log_prob(
            inputs=torch.Tensor(parameters).reshape(1, -1),
            context=true_observation.reshape(1, -1),
        ).item()
        if not np.isinf(prior.log_prob(torch.Tensor(parameters)).sum().item())
        else -np.inf
    )
    neural_posterior.eval()
    posterior_sampler = SliceSampler(
        utils.tensor2numpy(prior.sample((1,))).reshape(-1),
        lp_f=target_log_prob,
        thin=10,
    )

    # Always sample in eval mode.
    neural_posterior.eval()

    if mcmc_method == "slice-np":
        posterior_sampler.gen(20)  # Burn-in for 200 samples
        samples = torch.Tensor(posterior_sampler.gen(num_samples))

    else:
        if mcmc_method == "slice":
            kernel = Slice(potential_function=potential_function)
        elif mcmc_method == "hmc":
            kernel = HMC(potential_fn=potential_function)
        elif mcmc_method == "nuts":
            kernel = NUTS(potential_fn=potential_function)
        else:
            raise ValueError(
                "'mcmc_method' must be one of ['slice', 'hmc', 'nuts']."
            )
        num_chains = mp.cpu_count() - 1

        initial_params = prior.sample((num_chains,))
        sampler = MCMC(
            kernel=kernel,
            num_samples=(thin * num_samples) // num_chains + num_chains,
            warmup_steps=200,
            initial_params={"": initial_params},
            num_chains=num_chains,
            mp_context="spawn",
        )
        sampler.run()
        samples = next(iter(sampler.get_samples().values())).reshape(
            -1, len(prior.mean)  # len(prior.mean) = dim of theta
        )

        samples = samples[::thin][:num_samples]
        assert samples.shape[0] == num_samples

    # Back to training mode.
    neural_posterior.train()

    return samples


def _estimate_acceptance_rate(
        neural_posterior,
        prior,
        true_observation,
        num_samples=int(1e4)
    ):
    """
    Estimates rejection sampling acceptance rates.

    Args:
        neural_posterior: nn
        prior: torch.distribution
        true_observation: Observation on which to condition.
            If None, use true observation given at initialization.
        num_samples: int, Number of samples to use.

    Returns: float in [0, 1]
        Fraction of accepted samples.

    """

    # Always sample in eval mode.
    neural_posterior.eval()

    total_num_accepted_samples, total_num_generated_samples = 0, 0
    while total_num_generated_samples < num_samples:

        # Generate samples from posterior.
        candidate_samples = neural_posterior.sample(
            10000, context=true_observation.reshape(1, -1)
        ).squeeze(0)

        # Evaluate posterior samples under the prior.
        prior_log_prob = prior.log_prob(candidate_samples)
        if isinstance(prior, distributions.Uniform):
            prior_log_prob = prior_log_prob.sum(-1)

        # Update remaining number of samples needed.
        num_accepted_samples = (~torch.isinf(prior_log_prob)).sum().item()
        total_num_accepted_samples += num_accepted_samples

        # Keep track of acceptance rate
        total_num_generated_samples += candidate_samples.shape[0]

    # Back to training mode.
    neural_posterior.train()

    return total_num_accepted_samples / total_num_generated_samples

@property
def summary(self):
    return self._summary


class NeuralPotentialFunction:
    """
    Implementation of a potential function for Pyro MCMC which uses a classifier
    to evaluate a quantity proportional to the likelihood.
    """

    def __init__(self, posterior, prior, true_observation):
        """
        Args:
            posterior: nn
            prior: torch.distribution, Distribution object with 'log_prob' method.
            true_observation:torch.Tensor containing true observation x0.
        """

        self.prior = prior
        self.posterior = posterior
        self.true_observation = true_observation

    def __call__(self, parameters_dict):
        """
        Call method allows the object to be used as a function.
        Evaluates the given parameters using a given neural likelhood, prior,
        and true observation.

        Args:
            parameters_dict: dict of parameter values which need evaluation for MCMC.

        Returns:
            torch.Tensor potential ~ -[log r(x0, theta) + log p(theta)]

        """

        parameters = next(iter(parameters_dict.values()))
        potential = -self.posterior.log_prob(
            inputs=parameters, context=self.true_observation
        )
        if isinstance(self.prior, distributions.Uniform):
            log_prob_prior = self.prior.log_prob(parameters).sum(-1)
        else:
            log_prob_prior = self.prior.log_prob(parameters)
        log_prob_prior[~torch.isinf(log_prob_prior)] = 1
        potential *= log_prob_prior

        return potential
