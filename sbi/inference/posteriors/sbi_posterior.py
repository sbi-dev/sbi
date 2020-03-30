from typing import Callable, Optional
from warnings import warn

import numpy as np
import torch
from pyro.infer.mcmc import HMC, NUTS
from pyro.infer.mcmc.api import MCMC
from torch import distributions
from torch import multiprocessing as mp

import sbi.inference
import sbi.utils as utils
from sbi.mcmc import Slice, SliceSampler


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
        get_potential_function: Optional[Callable] = None,
    ):
        """
        Args:
            algorithm_family: one of 'snpe', 'snl', 'sre'
            neural_net: depends on algorithm: classifier for sre, density estimator for snpe and snl
            prior: prior dist
            context: Tensor or None, conditioning variables, i.e., observed data. If a Tensor, it must have the same
                number or rows as the inputs. If None, the context is ignored.
            #XXX sample with MCMC ????
            train_with_mcmc: bool. Sample rejection or MCMC?
        """

        self.neural_net = neural_net
        self._prior = prior
        self._context = context
        self._train_with_mcmc = train_with_mcmc
        self._mcmc_method = mcmc_method
        assert algorithm_family in ["snpe", "snl", "sre", "aalr"], "Not supported."
        self._alg_family = algorithm_family
        self._get_potential_function = get_potential_function
        # correction factor for snpe leakage
        self._leakage_density_correction_factor = None
        self._num_trained_rounds = 0

    def log_prob(
        self,
        inputs: torch.Tensor,
        context: torch.Tensor = None,
        normalize_snpe: bool = True,  # TODO: new variable name? This sounds as if we were 'normalizing snpe'
    ) -> torch.Tensor:
        """Calculate log probability under the distribution.

        Args:
            inputs: Tensor, input variables.
            context: Tensor or None, conditioning variables. If a Tensor, it must have the same
                number or rows as the inputs. If None, self._context is used.
            normalize_snpe:
                If True, normalize the output density when using snpe (by drawing
                 samples, estimating the acceptance ratio, and then scaling the
                 probability with it) and return -infinity if inputs are outside of
                 the prior support. If False, return the output from the density
                 estimator.

        Returns:
            A Tensor of shape [input_size], the log probability of the inputs given the context.
        """

        # we care about the normalized density only when we do snpe.
        correct_for_leakage = normalize_snpe and self._alg_family == "snpe"

        # format inputs and context into the correct shape
        inputs, context = utils.match_shapes_of_inputs_and_contexts(
            inputs, context, self._context, correct_for_leakage
        )

        # go into evaluation mode
        self.neural_net.eval()

        # compute the unnormalized log probability by evaluating the network
        if self._alg_family == "snpe":
            unnormalized_log_prob = self.neural_net.log_prob(inputs, context)
        elif self._alg_family == "sre" or self._alg_family == "aalr":
            if self._num_trained_rounds > 1 or self._alg_family == "sre":
                # if we train single round with aalr loss (Hermans et al. 2019), the
                # density is normalized and we do not raise the warning.
                warn(
                    "The log-probability returned by SRE is only correct up to a "
                    "normalizing constant."
                )
            log_ratio = self.neural_net(torch.cat((inputs, context)).reshape(1, -1))
            unnormalized_log_prob = log_ratio + self._prior.log_prob(inputs)
        else:
            raise NameError(
                "Evaluating the log-probability can only be done for SNPE and SRE."
            )

        if correct_for_leakage:
            # set the log-likelihood to -infinity if parameter set (inputs) is outside
            # of prior bounds.
            prior_log_prob = self._prior.log_prob(inputs)
            within_prior = torch.isfinite(prior_log_prob)
            unnormalized_log_prob = torch.where(
                within_prior, unnormalized_log_prob, prior_log_prob
            )

        # find the acceptance rate
        leakage_correction = (
            self.get_leakage_correction(context=context) if correct_for_leakage else 1.0
        )

        # return the normalized (leakage corrected) log prob: divide by acceptance prob
        # of rejection sampling
        return -torch.log(torch.tensor([leakage_correction])) + unnormalized_log_prob

    def get_leakage_correction(
        self, context: torch.Tensor, num_rejection_samples: int = 10000
    ) -> float:
        """Return factor for correcting the posterior density for leakage. 
        
        The factor is estimated from the acceptance probability during rejection
         sampling from the posterior.
        
        NOTE: This is to avoid re-estimating the acceptance probability from scratch
         whenever log_prob is called and normalize_snpe is True. Here, it is estimated
          only once for the default context, i.e., self._context, and saved for later,
           and whenever a new context is passed.
        
        Arguments:
            context {torch.Tensor} -- Context to condition the posterior. If None, uses
             the "default" context the posterior was trained for in multi-round mode.
        
        Keyword Arguments:
            num_rejection_samples {int} -- Number of samples used to estimate the factor
             (default: {10000})
        
        Returns:
            float -- Saved or newly estimated correction factor.
        """

        # check whether context is new
        new_context = (context != self._context).all()

        if new_context:
            _, acceptance_rate = utils.sample_posterior_within_prior(
                self.neural_net, self._prior, context, num_samples=num_rejection_samples
            )
            return acceptance_rate
        # if factor for default context wasnt estimated yet, estimate and set it
        elif self._leakage_density_correction_factor is None:
            _, acceptance_rate = utils.sample_posterior_within_prior(
                self.neural_net,
                self._prior,
                self._context,
                num_samples=num_rejection_samples,
            )
            self._leakage_density_correction_factor = acceptance_rate
            return self._leakage_density_correction_factor
        # otherwise just return the saved correction factor
        else:
            return self._leakage_density_correction_factor

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
        else:
            context = utils.torchutils.atleast_2d(context)

        if self._train_with_mcmc:
            return self._sample_posterior_mcmc(
                context=context,
                num_samples=num_samples,
                mcmc_method=self._mcmc_method,
                **kwargs,
            )
        else:
            samples, _ = utils.sample_posterior_within_prior(
                self.neural_net, self._prior, context, num_samples=num_samples
            )
            return samples

    def _sample_posterior_mcmc(
        self,
        num_samples: int,
        context: torch.Tensor,
        mcmc_method: str = "slice_np",
        thin: int = 10,
        warmup: int = 20,
        num_chains: Optional[int] = 1,
    ):
        """
        Sample the posterior using MCMC

        Args:
            context: observation/context/conditioning
            num_samples: Number of samples to generate.
            mcmc_method: Which MCMC method to use ['metropolis-hastings', 'slice', 'hmc', 'nuts']
            thin: Generate (num_samples * thin) samples in total, then select every
                'thin' sample.

        Returns:None
            torch.Tensor of shape [num_samples, parameter_dim]
        """

        # when using slice_np as mcmc sampler, we can have only a single chain.
        if mcmc_method == "slice_np" and num_chains > 1:
            warn(
                "slice_np does not support multiple mcmc chains. Using just a single chain."
            )

        # XXX: maybe get whole sampler instead of just potential function?
        potential_function = self._get_potential_function(
            self._prior, self.neural_net, context, mcmc_method
        )
        if mcmc_method == "slice-np":
            samples = self.slice_np_mcmc(
                num_samples, potential_function, context, thin, warmup
            )
        else:
            samples = self.pyro_mcmc(
                num_samples,
                potential_function,
                context,
                mcmc_method,
                thin,
                warmup,
                num_chains,
            )

        # Back to training mode.
        self.neural_net.train()

        return samples

    def slice_np_mcmc(
        self,
        num_samples: int,
        potential_function: Callable,
        context: torch.Tensor,
        thin: int = 10,
        warmup_steps: int = 20,
    ) -> torch.Tensor:

        # go into eval mode for evaluating during sampling
        self.neural_net.eval()

        posterior_sampler = SliceSampler(
            utils.tensor2numpy(self._prior.sample((1,))).reshape(-1),
            lp_f=potential_function,
            thin=thin,
        )

        posterior_sampler.gen(warmup_steps)

        samples = posterior_sampler.gen(num_samples)

        # back to training mode
        self.neural_net.train()

        return torch.tensor(samples, dtype=torch.float32)

    def pyro_mcmc(
        self,
        num_samples: int,
        potential_function: Callable,
        context: torch.Tensor,
        mcmc_method: str = "slice",
        thin: int = 10,
        warmup_steps: int = 200,
        num_chains: Optional[int] = 1,
    ):
        """Return samples obtained using Pyro's HMC, NUTS or slice kernels.

        Args:
            num_samples: desired number of samples
            potential_fn: Defining the potential function as a callable **class** makes it picklable for Pyro's MCMC to use it across chains in parallel, even if the potential function requires evaluating a neural network.

            context (torch.Tensor): conditioning observation
            mcmc_method (str, optional): One of "hmc", "nuts" or "slice" (default).
            thin (int, optional): thinning (subsampling) factor. Defaults to 10.
            warmup_steps (int, optional): initial samples to discard. Defaults to 200.
            num_chains (int, optional): whether to sample in parallel. Defaults to 1, will use all CPUs minus one if None.

        Raises:
            NameError: [description]
            ValueError: [description]

        Returns:
            [type]: [description]
        """
        if num_chains is None:
            num_chains = mp.cpu_count - 1

        # XXX move outside function, and assert inside; remember return to train
        # Always sample in eval mode.
        self.neural_net.eval()

        kernels = dict(slice=Slice, hmc=HMC, nuts=NUTS)
        try:
            kernel = kernels[mcmc_method](potential_fn=potential_function)
        except KeyError:
            raise ValueError("`mcmc_method` not one of 'slice', 'hmc', 'nuts'.")

        initial_params = self._prior.sample((num_chains,))

        sampler = MCMC(
            kernel=kernel,
            num_samples=(thin * num_samples) // num_chains + num_chains,
            warmup_steps=warmup_steps,
            initial_params={"": initial_params},
            num_chains=num_chains,
            mp_context="fork",
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
