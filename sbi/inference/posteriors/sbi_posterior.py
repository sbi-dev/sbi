from typing import Callable, Optional
from warnings import warn

from pyro.infer.mcmc import HMC, NUTS
from pyro.infer.mcmc.api import MCMC
from torch import Tensor
import torch
from torch import nn
from torch import multiprocessing as mp

from sbi.mcmc import Slice, SliceSampler
import sbi.utils as utils
from sbi.utils.torchutils import atleast_2d


NEG_INF = torch.tensor(float("-inf"), dtype=torch.float32)


class Posterior:
    """Posterior with evaluation and sampling methods.
    
    This class is used by inference algorithms as follows:
    
    - SNPE-family algorithms put density outside of the prior. This class uses the  
      prior to adjust evaluation and sampling and correct for that.
    - SNL and and SRE methods don't return posteriors directly. This class provides
      MCMC methods that given the prior, allow to sample from the posterior.
    """

    def __init__(
        self,
        algorithm_family: str,
        neural_net: nn.Module,
        prior,
        context: Optional[Tensor],
        sample_with_mcmc: bool = True,
        mcmc_method: str = "slice-np",
        get_potential_function: Optional[Callable] = None,
    ):
        """
        Args:
            algorithm_family: one of 'snpe', 'snl', 'sre' or 'aalr'
            neural_net: a classifier for sre/aalr, a density estimator for snpe/snl   
            prior: prior distribution with methods `log_prob` and `sample`
            context: observations acting as conditioning variables. Absent if 
                None. If provided, it must have same leading dimension as the inputs.  
            sample_with_mcmc: sample with MCMC for leakage
        """

        self.neural_net = neural_net
        self._prior = prior
        self._context = context  # atleast_2d(context)

        self._sample_with_mcmc = sample_with_mcmc
        self._mcmc_method = mcmc_method
        self._get_potential_function = get_potential_function

        if algorithm_family in ("snpe", "snl", "sre", "aalr"):
            self._alg_family = algorithm_family
        else:
            raise ValueError("Algorithm family unsupported.")

        self._num_trained_rounds = 0

        # correction factor for snpe leakage
        self._leakage_density_correction_factor = None

    def log_prob(
        self,
        inputs: Tensor,
        context: Optional[Tensor] = None,
        normalize_snpe_density: bool = True,
    ) -> Tensor:
        """Return posterior log probability.

        Args: 
            inputs: parameters.
            context: observations acting as conditioning variables. 
                If None, uses the "default" context the posterior was trained for in multi-round mode. If provided, it must have same leading dimension as the inputs.
            normalize_snpe_density:
                If True, normalize the output density when using snpe (by drawing samples, estimating the acceptance ratio, and then scaling the probability with it) and return -infinity where inputs are outside of the prior support. If False, directly return the output from the density estimator.

        Returns: 
            Tensor shaped like the input, containing the log probability of
            the inputs given the context.
        """

        # XXX I would like to remove this and deal with everything leakage-
        # XXX related down locally in the _log_prob_snpe function
        # XXX See draft code commented down below.
        correct_leakage = normalize_snpe_density and self._alg_family == "snpe"

        inputs, context = utils.match_shapes_of_inputs_and_contexts(
            inputs, context, self._context, correct_leakage
        )

        # XXX train exited here, entered after sampling?
        self.neural_net.eval()

        try:
            log_prob_fn = getattr(self, f"_log_prob_{self._alg_family}")
        except AttributeError:
            raise ValueError(f"{self.alg_family} cannot evaluate probabilities.")

        return log_prob_fn(
            inputs, context, normalize_snpe_density=normalize_snpe_density
        )

    def _log_prob_snpe(self, inputs: Tensor, context: Tensor, **kwargs) -> Tensor:
        unnormalized_log_prob = self.neural_net.log_prob(inputs, context)

        if not kwargs.get("normalize_snpe_density", True):
            # XXX Should we test if a leakage correction is due, warn only then?
            # XXX or is it too expensive?
            warn("No leakage correction was requested.")
            return unnormalized_log_prob
        else:
            # XXX See above note on defining `correct_leakage`
            # if context.shape[0] > 1:
            #     raise ValueError("etc")
            #
            # Set log-likelihood to -infinity if parameters outside prior support.
            is_prior_finite = torch.isfinite(self._prior.log_prob(inputs))
            masked_log_prob = torch.where(
                is_prior_finite, unnormalized_log_prob, NEG_INF
            )
            leakage_correction = self.get_leakage_correction(context=context)
            return masked_log_prob - torch.log(leakage_correction)

    def _log_prob_classifier(self, inputs: Tensor, context: Tensor, **kwargs) -> Tensor:
        log_ratio = self.neural_net(torch.cat((inputs, context)).reshape(1, -1))
        return log_ratio + self._prior.log_prob(inputs)

    def _log_prob_sre(self, inputs: Tensor, context: Tensor, **kwargs) -> Tensor:
        warn(
            "The log-probability returned by SRE is only correct up to a normalizing constant."
        )
        return self._log_prob_classifier(inputs, context)

    def _log_prob_aalr(self, inputs: Tensor, context: Tensor, **kwargs) -> Tensor:
        if self._num_trained_rounds > 1:
            warn(
                "The log-probability returned by AALR beyond round 1 is only correct up to a normalizing constant."
            )
        return self._log_prob_classifier(inputs, context)

    def get_leakage_correction(
        self, context: Tensor, num_rejection_samples: int = 10000
    ) -> Tensor:
        """Return leakage-correction factor for a leaky posterior density. 
        
        The factor is estimated from the acceptance probability during rejection
         sampling from the posterior.
        
        NOTE: This is to avoid re-estimating the acceptance probability from scratch whenever log_prob is called and normalize_snpe_density is True. Here, it is estimated only once for the default context, i.e., self._context, and saved for later, and whenever a new context is passed.
        
        Arguments:
            context: conditioning observation. If None, uses the "default"
                context the posterior was trained for in multi-round mode.
        
            num_rejection_samples: number of samples used to estimate the factor
             (default: 10000).
        
        Returns:
            Saved or newly estimated correction factor (scalar Tensor).
        """

        is_new_context = (context != self._context).all()

        if is_new_context:
            _, acceptance_rate = utils.sample_posterior_within_prior(
                self.neural_net, self._prior, context, num_samples=num_rejection_samples
            )
            return torch.as_tensor(acceptance_rate)
        # if factor for default context wasn't estimated yet, estimate and set
        elif self._leakage_density_correction_factor is None:
            _, acceptance_rate = utils.sample_posterior_within_prior(
                self.neural_net,
                self._prior,
                self._context,
                num_samples=num_rejection_samples,
            )
            self._leakage_density_correction_factor = acceptance_rate
            # XXX merge with next return (move out)
            return torch.as_tensor(self._leakage_density_correction_factor)
        # otherwise just return the saved correction factor
        else:
            return torch.as_tensor(self._leakage_density_correction_factor)

    def sample(self, num_samples: int, context: Tensor = None, **kwargs) -> Tensor:
        """
        Return samples from posterior distribution.

        Args:
            num_samples: number of samples
            context: conditioning observation. Will be _true_observation if None
            **kwargs:
                Additional parameters passed to MCMC sampler (thin and warmup)

        Returns: samples from posterior.
        """

        context = self._context if context is None else atleast_2d(context)

        if self._sample_with_mcmc:
            return self._sample_posterior_mcmc(
                context=context,
                num_samples=num_samples,
                mcmc_method=self._mcmc_method,
                **kwargs,
            )
        else:
            # rejection sampling
            samples, _ = utils.sample_posterior_within_prior(
                self.neural_net, self._prior, context, num_samples=num_samples
            )
            return samples

    def _sample_posterior_mcmc(
        self,
        num_samples: int,
        context: Tensor,
        mcmc_method: str = "slice_np",
        thin: int = 10,
        warmup: int = 20,
        num_chains: Optional[int] = 1,
    ) -> Tensor:
        """
        Return posterior samples using MCMC.

        Args:
            context: conditioning observation
            
            num_samples: desired output samples
            
            mcmc_method: one of 'metropolis-hastings', 'slice', 'hmc', 'nuts'
            
            thin: thinning factor for the chain, e.g. for thin=3 only every
                third sample will be returned, until a total of num_samples

        Returns:
            tensor of shape num_samples x parameter dimension
        """

        # when using slice_np as mcmc sampler, we can only have a single chain
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

        # XXX train exited in log_prob, entered here?
        # back to training mode
        self.neural_net.train(True)

        return samples

    def slice_np_mcmc(
        self,
        num_samples: int,
        potential_function: Callable,
        context: torch.Tensor,
        thin: int = 10,
        warmup_steps: int = 20,
    ) -> Tensor:

        # go into eval mode for evaluating during sampling
        # XXX set eval mode outside of calls to sample
        self.neural_net.eval()

        posterior_sampler = SliceSampler(
            utils.tensor2numpy(self._prior.sample((1,))).reshape(-1),
            lp_f=potential_function,
            thin=thin,
        )

        posterior_sampler.gen(warmup_steps)

        samples = posterior_sampler.gen(num_samples)

        # back to training mode
        # XXX train exited in log_prob, entered here?
        self.neural_net.train(True)

        return torch.tensor(samples, dtype=torch.float32)

    def pyro_mcmc(
        self,
        num_samples: int,
        potential_function: Callable,
        context: Tensor,
        mcmc_method: str = "slice",
        thin: int = 10,
        warmup_steps: int = 200,
        num_chains: Optional[int] = 1,
    ):
        """Return samples obtained using Pyro's HMC, NUTS or slice kernels.

        Args: 
            num_samples: desired number of samples  
            potential_fn: defining the potential function as a callable **class** makes
                it picklable for Pyro's MCMC to use it across chains in parallel, even
                if the potential function requires evaluating a neural network. context:
                conditioning observation
            mcmc_method: one of "hmc", "nuts" or "slice" (default "slice")
            thin: thinning (subsampling) factor (default 10)
            warmup_steps: initial samples to discard (defaults to 200)
            num_chains: whether to sample in parallel. If None, will use all
                CPUs except one (default 1)

        Returns: tensor of shape num_samples x parameter dimension
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

    def set_embedding_net(self, embedding_net: nn.Module) -> None:
        """
        Set the embedding net that encodes the context as an attribute.

        Args:
            embedding_net: neural net to encode the context
        """
        assert isinstance(embedding_net, nn.Module), (
            "embedding_net is not a nn.Module. "
            "If you want to use hard-coded summary features, "
            "please simply pass the already encoded as input and pass "
            "embedding_net=None"
        )
        self.neural_net._embedding_net = embedding_net
