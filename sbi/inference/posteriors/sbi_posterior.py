from typing import Union
import numpy as np
import torch
from torch import distributions
import sbi.utils as utils
from pyro.infer.mcmc import HMC, NUTS
from pyro.infer.mcmc.api import MCMC
from sbi.mcmc import Slice
from torch import multiprocessing as mp
from sbi.mcmc import SliceSampler
import sbi.inference


class Posterior:
    """Posterior with evaluation and sampling methods.
    
    This class is used by inference algorithms as follows:
    
    - SNPE-family algorithms put density outside of the prior. This class uses the prior to adjust evaluation and sampling and correct for that.
    - SNL and and SRE methods don't return posteriors directly. This class provides MCMC methods that given the prior, allow to sample from the posterior.
    
    """

    def __init__(
        self,
        algorithm_family: str,
        neural_net: torch.nn.Module,
        prior: torch.distributions.Distribution,
        context: torch.Tensor,
        train_with_mcmc: bool = True,  # XXX make sure it IS True
        mcmc_method: str = "slice-np",
    ):
        """
        Args:
            algorithm_family: one of 'snpe', 'snl', 'sre'
            neural_net: depends on algorithm: classifier for sre, density estimator for snpe and snl
            prior: prior dist
            context: Tensor or None, conditioning variables. If a Tensor, it must have the same
                number or rows as the inputs. If None, the context is ignored.
            #XXX sample with MCMC ????
            train_with_mcmc: bool. Sample rejection or MCMC?
        """

        self.neural_net = neural_net
        self._prior = prior
        self._context = context
        self._train_with_mcmc = train_with_mcmc
        self._mcmc_method = mcmc_method
        assert algorithm_family in ["snpe", "snl", "sre"], "Not supported."
        self._alg_family = algorithm_family

    def log_prob(
        self,
        inputs: torch.Tensor,
        context: torch.Tensor = None,
        normalize_snpe: bool = True,
    ):
        """Calculate log probability under the distribution.

        Args:
            inputs: Tensor, input variables.
            context: Tensor or None, conditioning variables. If a Tensor, it must have the same
                number or rows as the inputs. If None, the context is ignored.
            normalize_snpe:
                whether to normalize the output density when using snpe (by drawing samples, estimating the acceptance
                ratio, and then scaling the probability with it)

        Returns:
            A Tensor of shape [input_size], the log probability of the inputs given the context.
        """

        # we care about the normalized density only when we do snpe.
        normalize = normalize_snpe and self._alg_family == "snpe"

        # format inputs and context into the correct shape
        inputs, context = utils.build_inputs_and_contexts(
            inputs, context, self._context, normalize
        )

        # go into evaluation mode
        self.neural_net.eval()

        # compute the unnormalized log probability by evaluating the network
        unnormalized_log_prob = self.neural_net.log_prob(inputs, context)

        # find the acceptance rate
        if normalize:
            acceptance_prob = self.estimate_acceptance_rate(context=context[0])
        else:
            acceptance_prob = 1.0

        # XXX torch.log?
        return -np.log(acceptance_prob) + unnormalized_log_prob

    def estimate_acceptance_rate(
        self, num_samples: int = int(1e4), context: torch.Tensor = None
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
            context = self._context

        # Always sample in eval mode.
        self.neural_net.eval()

        total_num_accepted_samples, total_num_generated_samples = 0, 0
        while total_num_generated_samples < num_samples:

            # Generate samples from posterior.
            candidate_samples = self.neural_net.sample(  # sample from unbounded flow
                10000, context=context.reshape(1, -1)
            ).squeeze(0)

            # Evaluate posterior samples under the prior.
            prior_log_prob = self._prior.log_prob(candidate_samples)
            if isinstance(self._prior, distributions.Uniform):
                prior_log_prob = prior_log_prob.sum(-1)

            # Update remaining number of samples needed.
            num_accepted_samples = (~torch.isinf(prior_log_prob)).sum().item()
            total_num_accepted_samples += num_accepted_samples

            # Keep track of acceptance rate
            total_num_generated_samples += candidate_samples.shape[0]

        # Back to training mode.
        self.neural_net.train()

        return total_num_accepted_samples / total_num_generated_samples

    def sample(self, num_samples: int, context: torch.Tensor = None, **kwargs):
        """
        Sample posterior distribution.

        Args:
            num_samples: number of samples
            context:
                Provide observation/context/condition.
                Will be _true_observation if None
            **kwargs:
                Additional parameters for MCMC. thin and warmup

        Returns: torch.tensor, samples from posterior
        """

        if context is None:
            context = self._context

        if self._train_with_mcmc:
            return self._sample_posterior_mcmc(
                context=context,
                num_samples=num_samples,
                mcmc_method=self._mcmc_method,
                **kwargs,
            )
        else:
            return self._sample_posterior_rejection(
                context=context, num_samples=num_samples
            )

    def _sample_posterior_rejection(
        self, num_samples: int, context: torch.Tensor = None
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
        self.neural_net.eval()

        # Rejection sampling is potentially needed for the posterior.
        # This is because the prior may not have support everywhere.
        # The posterior may also be constrained to the same support,
        # but we don't know this a priori.
        samples = []
        num_remaining_samples = num_samples
        total_num_accepted, _total_num_generated_examples = 0, 0
        while num_remaining_samples > 0:

            # Generate samples from unbounded posterior.
            candidate_samples = self.neural_net.sample(
                max(10000, num_samples), context=context.reshape(1, -1)
            ).squeeze(0)

            # Evaluate posterior samples under the prior.
            prior_log_prob = self._prior.log_prob(candidate_samples)
            if isinstance(self._prior, distributions.Uniform):
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
        self.neural_net.train()

        # Aggregate collected samples.
        samples = torch.cat(samples)

        # Make sure we have the right amount.
        samples = samples[:num_samples, ...]
        assert samples.shape[0] == num_samples

        return samples

    def _sample_posterior_mcmc(
        self,
        num_samples: int,
        context: torch.Tensor = None,
        mcmc_method: str = "slice_np",
        thin: int = 10,
        warmup: int = 20,
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

        if mcmc_method == "slice-np":
            samples = self.slice_np_mcmc(num_samples, context, thin, warmup)
        else:
            samples = self.pyro_mcmc(num_samples, context, mcmc_method, thin, warmup)

        # Back to training mode.
        self.neural_net.train()

        return samples

    def slice_np_mcmc(
        self,
        num_samples: int,
        context: torch.Tensor,
        thin: int = 10,
        warmup_steps: int = 20,
    ):

        if self._alg_family == "snpe":
            potential_function = sbi.inference.snpe.base_snpe.SliceNpNeuralPotentialFunction(
                self, self._prior, context
            )
        elif self._alg_family == "snl":
            potential_function = sbi.inference.snl.SliceNpNeuralPotentialFunction(
                self, self._prior, context
            )
        elif self._alg_family == "sre":
            potential_function = sbi.inference.sre.SliceNpNeuralPotentialFunction(
                self, self._prior, context
            )
        else:
            raise NameError
        self.neural_net.eval()

        posterior_sampler = SliceSampler(
            utils.tensor2numpy(self._prior.sample((1,))).reshape(-1),
            lp_f=potential_function,
            thin=thin,
        )

        self.neural_net.train()

        posterior_sampler.gen(warmup_steps)

        samples = torch.Tensor(posterior_sampler.gen(num_samples))

        return samples

    def pyro_mcmc(
        self,
        num_samples: int,
        context: torch.Tensor,
        mcmc_method: str = "slice",
        thin: int = 10,
        warmup_steps: int = 200,
    ):
        # HMC and NUTS from Pyro.
        # Defining the potential function as an object means Pyro's MCMC scheme
        # can pickle it to be used across multiple chains in parallel, even if
        # the potential function requires evaluating a neural likelihood as is the
        # case here.
        # build potential function depending on what algorithm is used
        if self._alg_family == "snpe":
            potential_function = sbi.inference.snpe.base_snpe.NeuralPotentialFunction(
                self.neural_net, self._prior, context
            )
        elif self._alg_family == "snl":
            potential_function = sbi.inference.snl.NeuralPotentialFunction(
                self.neural_net, self._prior, context
            )
        elif self._alg_family == "sre":
            potential_function = sbi.inference.sre.NeuralPotentialFunction(
                self.neural_net, self._prior, context
            )
        else:
            raise NameError

        # Always sample in eval mode.
        self.neural_net.eval()

        if mcmc_method == "slice":
            kernel = Slice(potential_function=potential_function)
        elif mcmc_method == "hmc":
            kernel = HMC(potential_fn=potential_function)
        elif mcmc_method == "nuts":
            kernel = NUTS(potential_fn=potential_function)
        else:
            raise ValueError("'mcmc_method' must be one of ['slice', 'hmc', 'nuts'].")
        num_chains = mp.cpu_count() - 1

        initial_params = self._prior.sample((num_chains,))
        sampler = MCMC(
            kernel=kernel,
            num_samples=(thin * num_samples) // num_chains + num_chains,
            warmup_steps=warmup_steps,
            initial_params={"": initial_params},
            num_chains=num_chains,
            # mp_context="spawn",
        )
        sampler.run()
        samples = next(iter(sampler.get_samples().values())).reshape(
            -1, len(self._prior.mean)  # len(prior.mean) = dim of theta
        )

        samples = samples[::thin][:num_samples]
        assert samples.shape[0] == num_samples

        return samples

    def set_embedding_net(self, embedding_net):
        """
        Set the embedding net to encode the context

        Args:
            embedding_net: nn.Module
                neural net to encode the context
        """
        assert isinstance(embedding_net, torch.nn.Module), (
            "embedding_net is not a nn.Module. "
            "If you want to use hard-coded summary features, "
            "please simply pass the encoded features and pass "
            "embedding_net=None"
        )
        self.neural_net.embedding_net = embedding_net
