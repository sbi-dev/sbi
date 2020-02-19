import numpy as np
import torch
from torch import distributions
import sbi.utils as utils
from pyro.infer.mcmc import HMC, NUTS
from pyro.infer.mcmc.api import MCMC
from sbi.mcmc import Slice
from torch import multiprocessing as mp

from nflows import flows
from sbi.mcmc import SliceSampler


class FlowPosterior(flows.Flow):
    def __init__(
        self,
        transform,
        distribution,
        prior,
        context,
        embedding=None,
        train_with_mcmc=False,
        mcmc_method="slice-np",
    ):

        super().__init__(transform, distribution, embedding)
        self.prior = prior
        self.context = context
        self._train_with_mcmc = train_with_mcmc
        self._mcmc_method = mcmc_method

    def log_prob(
        self, inputs: torch.tensor, context: torch.tensor = None, normalize: bool = True
    ):
        """Calculate log probability under the distribution.

        Args:
            inputs: Tensor, input variables.
            context: Tensor or None, conditioning variables. If a Tensor, it must have the same
                number or rows as the inputs. If None, the context is ignored.
            normalize:
                If True, we normalize the output density
                by drawing samples, estimating the acceptance
                ratio, and then scaling the probability with it

        Returns:
            A Tensor of shape [input_size], the log probability of the inputs given the context.
        """

        # format inputs and context into the correct shape
        inputs, context = utils.build_inputs_and_contexts(
            inputs, context, self.context, normalize
        )

        # go into evaluation mode
        self.eval()

        # compute the unnormalized log probability by evaluating the MDN
        unnormalized_log_prob = self._log_prob(inputs, context)

        # find the acceptance rate
        if normalize:
            acceptance_prob = self.estimate_acceptance_rate(context=context[0])
        else:
            acceptance_prob = 1.0

        return np.log(acceptance_prob) + unnormalized_log_prob

    def _log_prob(self, inputs: torch.tensor, context: torch.tensor = None):
        """
        Evaluate posterior distribution at datapoint

        Args:
            inputs: where to evaluate posterior
            context: context/conditioning/observation

        Returns: log-probability
        """
        if context is None:
            context = self.context[
                None,
            ]

        if len(inputs.shape) == 1:
            inputs = inputs[
                None,
            ]  # append a dimension

        self.eval()

        return super()._log_prob(inputs, context=context)

    def estimate_acceptance_rate(
        self, num_samples: int = int(1e4), context: torch.tensor = None
    ):
        """
        Estimates rejection sampling acceptance rates.

        Args:
            context: Observation on which to condition.
                If None, use true observation given at initialization.
            num_samples: Number of samples to use.

        Returns: float in [0, 1]
            Fraction of accepted samples.
        """

        if context is None:
            context = self.context

        # Always sample in eval mode.
        self.eval()

        total_num_accepted_samples, total_num_generated_samples = 0, 0
        while total_num_generated_samples < num_samples:

            # Generate samples from posterior.
            candidate_samples = (
                super()
                ._sample(  # sample from unbounded flow
                    10000, context=context.reshape(1, -1)
                )
                .squeeze(0)
            )

            # Evaluate posterior samples under the prior.
            prior_log_prob = self.prior.log_prob(candidate_samples)
            if isinstance(self.prior, distributions.Uniform):
                prior_log_prob = prior_log_prob.sum(-1)

            # Update remaining number of samples needed.
            num_accepted_samples = (~torch.isinf(prior_log_prob)).sum().item()
            total_num_accepted_samples += num_accepted_samples

            # Keep track of acceptance rate
            total_num_generated_samples += candidate_samples.shape[0]

        # Back to training mode.
        self.train()

        return total_num_accepted_samples / total_num_generated_samples

    def _sample(self, num_samples: int, context: torch.tensor = None):
        """
        Sample posterior distribution.

        Args:
            num_samples: number of samples
            context:
                Provide observation/context/condition.
                Will be _true_observation if None

        Returns: torch.tensor, samples from posterior
        """

        if context is None:
            context = self.context

        if self._train_with_mcmc:
            return self._sample_posterior_mcmc(
                context=context, num_samples=num_samples, mcmc_method=self._mcmc_method,
            )
        else:
            return self._sample_posterior_rejection(
                context=context, num_samples=num_samples
            )

    def _sample_posterior_rejection(
        self, num_samples: int, context: torch.tensor = None
    ):
        """
        Rejection sample a posterior.

        Args:
            num_samples: Number of samples to generate.
            context: [observation_dim] or [1, observation_dim]
                Pass true observation for inference.

        Returns:
            torch.Tensor [num_samples, parameter_dim]
            Posterior parameter samples.
        """

        # Always sample in eval mode.
        self.eval()

        # Rejection sampling is potentially needed for the posterior.
        # This is because the prior may not have support everywhere.
        # The posterior may also be constrained to the same support,
        # but we don't know this a priori.
        samples = []
        num_remaining_samples = num_samples
        total_num_accepted, _total_num_generated_examples = 0, 0
        while num_remaining_samples > 0:

            # Generate samples from unbounded posterior.
            candidate_samples = (
                super()
                ._sample(max(10000, num_samples), context=context.reshape(1, -1))
                .squeeze(0)
            )

            # Evaluate posterior samples under the prior.
            prior_log_prob = self.prior.log_prob(candidate_samples)
            if isinstance(self.prior, distributions.Uniform):
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
        self.train()

        # Aggregate collected samples.
        samples = torch.cat(samples)

        # Make sure we have the right amount.
        samples = samples[:num_samples, ...]
        assert samples.shape[0] == num_samples

        return samples

    def _sample_posterior_mcmc(
        self,
        num_samples: int,
        context: torch.tensor = None,
        mcmc_method: str = "slice_np",
        thin: int = 10,
    ):
        """
        Sample the posterior using MCMC

        Args:
            context: observation/context/conditioning
            num_samples: Number of samples to generate.
            mcmc_method: Which MCMC method to use ['metropolis-hastings', 'slice', 'hmc', 'nuts']
            thin: Generate (num_samples * thin) samples in total, then select every
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
            self, self.prior, context  # TODO: passing self not nice
        )

        # Axis-aligned slice sampling implementation in NumPy
        target_log_prob = (
            lambda parameters: self.log_prob(
                inputs=torch.Tensor(parameters).reshape(1, -1),
                context=context.reshape(1, -1),
            ).item()
            if not np.isinf(self.prior.log_prob(torch.Tensor(parameters)).sum().item())
            else -np.inf
        )
        self.eval()
        posterior_sampler = SliceSampler(
            utils.tensor2numpy(self.prior.sample((1,))).reshape(-1),
            lp_f=target_log_prob,
            thin=10,
        )

        # Always sample in eval mode.
        self.eval()

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

            initial_params = self.prior.sample((num_chains,))
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
                -1, len(self.prior.mean)  # len(prior.mean) = dim of theta
            )

            samples = samples[::thin][:num_samples]
            assert samples.shape[0] == num_samples

        # Back to training mode.
        self.train()

        return samples

    def set_embedding_net(self, embedding_net):
        """
        Set the embedding net to encode the context

        Args:
            embedding_net: nn.Module
                neural net to encode the context
        """
        assert isinstance(embedding_net, torch.nn.Module), 'embedding_net is not a nn.Module. ' \
                                                           'If you want to use hard-coded summary features, ' \
                                                           'please simply pass the encoded features and pass ' \
                                                           'embedding_net=None'
        self.embedding_net = embedding_net


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

    def evaluate(self, point):
        raise NotImplementedError
