from typing import Callable, Optional, Union
from warnings import warn

from pyro.infer.mcmc import HMC, NUTS
from pyro.infer.mcmc.api import MCMC
import torch
from torch import nn, Tensor, log, as_tensor
from torch import multiprocessing as mp

from sbi.mcmc import Slice, SliceSampler
import sbi.utils as utils
from sbi.utils.torchutils import atleast_2d


NEG_INF = torch.tensor(float("-inf"), dtype=torch.float32)

Array = Union[torch.Tensor, np.ndarray]


class Posterior:
    r"""Posterior $p(\theta|x_o)$ with evaluation and sampling methods.
    
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
        x_o: Optional[Tensor],
        sample_with_mcmc: bool = True,
        mcmc_method: str = "slice-np",
        get_potential_function: Optional[Callable] = None,
    ):
        """
        Args:
            algorithm_family: one of 'snpe', 'snl', 'sre' or 'aalr'
            neural_net: a classifier for sre/aalr, a density estimator for snpe/snl   
            prior: prior distribution with methods `log_prob` and `sample`
            x_o: observations acting as conditioning context. Absent if None.
            TODO: Why is x_o optional?
            sample_with_mcmc: sample with MCMC for leakage
        """

        self.neural_net = neural_net
        self._prior = prior
        self.x_o = x_o

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
        theta: Tensor,
        x: Optional[Tensor] = None,
        normalize_snpe_density: bool = True,
    ) -> Tensor:
        r"""Return posterior $p(\theta|x)$ log probability.

        Args: 
            theta: parameters $\theta$.
            x: conditioning context for posterior $p(\theta|x)$.
                If None, use the observation self.x_o.
            normalize_snpe_density:
                If True, normalize the output density when using snpe (by drawing
                samples, estimating the acceptance ratio, and then scaling the
                probability with it) and return -infinity where theta is outside of
                the prior support. If False, directly return the output from the density
                estimator.

        Returns: 
            Tensor of shape theta.shape[0], containing the log probability of
            the posterior $p(\theta|x)$
        """

        # XXX I would like to remove this and deal with everything leakage-
        # XXX related down locally in the _log_prob_snpe function
        # XXX See draft code commented down below.
        correct_leakage = normalize_snpe_density and self._alg_family == "snpe"

        theta, x = utils.match_shapes_of_theta_and_x(
            theta, x, self.x_o, correct_leakage
        )

        # XXX train exited here, entered after sampling?
        self.neural_net.eval()

        try:
            log_prob_fn = getattr(self, f"_log_prob_{self._alg_family}")
        except AttributeError:
            raise ValueError(f"{self._alg_family} cannot evaluate probabilities.")

        return log_prob_fn(theta, x, normalize_snpe_density=normalize_snpe_density)

    def _log_prob_snpe(self, theta: Tensor, x: Tensor, **kwargs) -> Tensor:
        r"""
        Return posterior log probability $p(\theta|x)$ for SNPE.

        TODO: Does it make sense to put this into snpe/snpe_base.py?
        """
        unnormalized_log_prob = self.neural_net.log_prob(theta, x)

        if not kwargs.get("normalize_snpe_density", True):
            # XXX Should we test if a leakage correction is due, warn only then?
            # XXX or is it too expensive?
            warn("No leakage correction was requested.")
            return unnormalized_log_prob
        else:
            # Set log-likelihood to -infinity if theta outside prior support.
            is_prior_finite = torch.isfinite(self._prior.log_prob(theta))
            masked_log_prob = torch.where(
                is_prior_finite, unnormalized_log_prob, NEG_INF
            )
            leakage_correction = self.get_leakage_correction(x=x)
            return masked_log_prob - torch.log(leakage_correction)

    def _log_prob_classifier(self, theta: Tensor, x: Tensor, **kwargs) -> Tensor:
        log_ratio = self.neural_net(torch.cat((theta, x)).reshape(1, -1))
        return log_ratio + self._prior.log_prob(theta)

    def _log_prob_sre(self, theta: Tensor, x: Tensor, **kwargs) -> Tensor:
        warn(
            "The log-probability returned by SRE is only correct up to a normalizing constant."
        )
        return self._log_prob_classifier(theta, x)

    def _log_prob_aalr(self, theta: Tensor, x: Tensor, **kwargs) -> Tensor:
        if self._num_trained_rounds > 1:
            warn(
                "The log-probability returned by AALR beyond round 1 is only correct up to a normalizing constant."
            )
        return self._log_prob_classifier(theta, x)

    def get_leakage_correction(
        self,
        x: Tensor,
        num_rejection_samples: int = 10_000,
        force_update: bool = False,
        show_progressbar: bool = False,
    ) -> Tensor:
        r"""Return leakage correction factor for a leaky posterior density.

        The factor is estimated from the acceptance probability during rejection
         sampling from the posterior.

        NOTE: This is to avoid re-estimating the acceptance probability from scratch
              whenever log_prob is called and normalize_snpe_density is True. Here, it
              is estimated only once for self.x_o and saved for later. We re-evaluate
              only whenever a new x is passed.
        
        Arguments:
            x: Conditioning context for posterior $p(\theta|x)$. If None, use self.x_o.
            num_rejection_samples: Number of samples used to estimate correction factor.
            force_update: Whether to force a reevaluation of the leakage correction even
                if the context x is the same as self.x_o. This is useful to enforce a 
                new estimate of the leakage after later rounds, i.e. round 2, 3, ...
            show_progressbar: Whether to show a progressbar during sampling.

        Returns:
            Saved or newly estimated correction factor (scalar Tensor).
        """

        def acceptance_at(x: Tensor) -> Tensor:
            return utils.sample_posterior_within_prior(
                self.neural_net, self._prior, x, num_rejection_samples, show_progressbar
            )[1]

        # Short-circuit here for performance: if identical no need to check equality.
        is_new_x = (x is not self.x_o) and (x != self.x_o).all()
        not_saved_at_x_o = self._leakage_density_correction_factor is None

        if is_new_x:  # Calculate at x; don't save.
            return acceptance_at(x)
        elif not_saved_at_x_o or force_update:  # Calculate at x_o; save.
            self._leakage_density_correction_factor = acceptance_at(self.x_o)

        return self._leakage_density_correction_factor  # type:ignore

    def sample(
        self,
        num_samples: int,
        x: Tensor = None,
        show_progressbar: bool = False,
        **kwargs,
    ) -> Tensor:
        r"""
        Return samples from posterior distribution $p(\theta|x)$.

        Args:
            num_samples: number of samples
            x: conditioning context for posterior $p(\theta|x)$. Will be self.x_o if
             None
            show_progressbar: whether to plot a progressbar showing how many samples have already
             been drawn
            **kwargs:
                Additional parameters passed to MCMC sampler (thin and warmup)

        Returns: samples from posterior.
        """

        x = atleast_2d(as_tensor(self._x_else_x_o(x)))

        with torch.no_grad():
            if self._sample_with_mcmc:

                samples = self._sample_posterior_mcmc(
                    x=x,
                    num_samples=num_samples,
                    mcmc_method=self._mcmc_method,
                    show_progressbar=show_progressbar,
                    **kwargs,
                )
            else:
                # rejection sampling
                samples, _ = utils.sample_posterior_within_prior(
                    self.neural_net,
                    self._prior,
                    x,
                    num_samples=num_samples,
                    show_progressbar=show_progressbar,
                )

        return samples

    def _sample_posterior_mcmc(
        self,
        num_samples: int,
        x: Tensor,
        mcmc_method: str = "slice_np",
        thin: int = 10,
        warmup: int = 20,
        num_chains: Optional[int] = 1,
        show_progressbar: Optional[int] = True,
    ) -> Tensor:
        r"""
        Return MCMC samples from posterior $p(\theta|x)$.

        Args:
            x: conditioning context for posterior $p(\theta|x)$
            
            num_samples: desired output samples
            
            mcmc_method: one of 'metropolis-hastings', 'slice', 'hmc', 'nuts'
            
            thin: thinning factor for the chain, e.g. for thin=3 only every
                third sample will be returned, until a total of num_samples

            show_progressbar: whether to show a progressbar during sampling

        Returns:
            tensor of shape (num_samples, shape_of_single_theta)
        """

        # when using slice_np as mcmc sampler, we can only have a single chain
        if mcmc_method == "slice_np" and num_chains > 1:
            warn(
                "slice_np does not support multiple mcmc chains. Using just a single chain."
            )

        # XXX: maybe get whole sampler instead of just potential function?
        potential_function = self._get_potential_function(
            self._prior, self.neural_net, x, mcmc_method
        )
        if mcmc_method == "slice-np":
            samples = self.slice_np_mcmc(
                num_samples, potential_function, x, thin, warmup
            )
        else:
            samples = self.pyro_mcmc(
                num_samples,
                potential_function,
                x,
                mcmc_method,
                thin,
                warmup,
                num_chains,
                show_progressbar=show_progressbar,
            )

        return samples

    def slice_np_mcmc(
        self,
        num_samples: int,
        potential_function: Callable,
        x: torch.Tensor,
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
        x: Tensor,
        mcmc_method: str = "slice",
        thin: int = 10,
        warmup_steps: int = 200,
        num_chains: Optional[int] = 1,
        show_progressbar: Optional[bool] = True,
    ):
        r"""Return samples obtained using Pyro's HMC, NUTS or slice kernels.

        Args: 
            num_samples: desired number of samples  
            potential_function: defining the potential function as a callable **class**
                makes it picklable for Pyro's MCMC to use it across chains in parallel,
                even if the potential function requires evaluating a neural network.
            x: conditioning context for posterior $p(\theta|x)$
            mcmc_method: one of "hmc", "nuts" or "slice" (default "slice")
            thin: thinning (subsampling) factor (default 10)
            warmup_steps: initial samples to discard (defaults to 200)
            num_chains: whether to sample in parallel. If None, will use all
                CPUs except one (default 1)
            show_progressbar: whether to show a progressbar during sampling

        Returns: tensor of shape (num_samples, shape_of_single_theta)
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
            disable_progbar=not show_progressbar,
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
        Set the embedding net that encodes x as an attribute of the neural_net.

        Args:
            embedding_net: Neural net to encode x.
        """
        assert isinstance(embedding_net, nn.Module), (
            "embedding_net is not a nn.Module. "
            "If you want to use hard-coded summary features, "
            "please simply pass the already encoded summary features as input and pass "
            "embedding_net=None"
        )
        self.neural_net._embedding_net = embedding_net

    def _x_else_x_o(self, x: Optional[Array]) -> Array:
        if x is not None:
            return x
        elif self.x_o is None:
            raise ValueError("Context x needed when not preconditioned to x_o.")
        else:
            return self.x_o
