from typing import Callable, Optional, Union
from numpy import ndarray
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
        mcmc_method: str = "slice_np",
        get_potential_function: Optional[Callable] = None,
    ):
        """
        Args:
            algorithm_family: One of 'snpe', 'snl', 'sre' or 'aalr'.
            neural_net: A classifier for sre/aalr, a density estimator for snpe/snl.
            prior: Prior distribution with methods `log_prob` and `sample`.
            x_o: Observation acting as conditioning context. Absent if None.
            sample_with_mcmc: Whether to sample with MCMC to correct leakage.
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

        # Correction factor for snpe leakage.
        self._leakage_density_correction_factor = None

    def log_prob(
        self,
        theta: Tensor,
        x: Optional[Tensor] = None,
        norm_posterior_snpe: bool = True,
    ) -> Tensor:
        r"""Return posterior $p(\theta|x)$ log probability.

        Args:
            theta: Parameters $\theta$.
            x: Conditioning context for posterior $p(\theta|x)$. Per default, use the
               observation self.x_o.
            norm_posterior_snpe: Whether to enforce a normalized posterior density when
                using snpe. Renormalization of the posterior is useful when some
                probability falls out or 'leaks' out of the prescribed prior support.
                The normalizing factor is calculated via rejection sampling, so if you
                need speedier but unnormalized log posterior estimates set
                `norm_posterior_snpe=False`. The returned log posterior is set to
                $-\infty$ outside of the prior support regardless of this setting.

        Returns:
            Log posterior probability $p(\theta|x)$ for θ in the support of the prior,
            $-\infty$ (corresponding to zero probability) outside. Shape `(len(θ),)`.
        """

        # TODO Train exited here, entered after sampling?
        self.neural_net.eval()

        theta, x = utils.match_shapes_of_theta_and_x(
            theta, x, self.x_o, norm_posterior_snpe
        )

        try:
            log_prob_fn = getattr(self, f"_log_prob_{self._alg_family}")
        except AttributeError:
            raise ValueError(f"{self._alg_family} cannot evaluate probabilities.")

        if self._alg_family == "snpe":
            return log_prob_fn(theta, x, norm_posterior=norm_posterior_snpe)
        else:
            return log_prob_fn(theta, x)

    # TODO: Move _log_prob_X into the respective inference classes (X)?
    def _log_prob_snpe(self, theta: Tensor, x: Tensor, norm_posterior: bool) -> Tensor:
        r"""
        Return posterior log probability $p(\theta|x)$.

        The posterior probability will be only normalized if explictly requested,
        but it will always zeroed out($-\infty$ log-prob)outside of the prior support.
        """

        unnorm_log_prob = self.neural_net.log_prob(theta, x)
        is_prior_finite = torch.isfinite(self._prior.log_prob(theta))

        # Force probability to be zero outside prior support.
        masked_log_prob = torch.where(is_prior_finite, unnorm_log_prob, NEG_INF)

        log_factor = log(self.get_leakage_correction(x=x)) if norm_posterior else 0

        return masked_log_prob - log_factor

    def _log_prob_classifier(self, theta: Tensor, x: Tensor) -> Tensor:
        log_ratio = self.neural_net(torch.cat((theta, x)).reshape(1, -1))
        return log_ratio + self._prior.log_prob(theta)

    def _log_prob_sre(self, theta: Tensor, x: Tensor) -> Tensor:
        warn(
            "The log probability from SRE is only correct up to a normalizing constant."
        )
        return self._log_prob_classifier(theta, x)

    def _log_prob_aalr(self, theta: Tensor, x: Tensor) -> Tensor:
        if self._num_trained_rounds > 1:
            warn(
                "The log-probability from AALR beyond round 1 is only correct "
                "up to a normalizing constant."
            )
        return self._log_prob_classifier(theta, x)

    def get_leakage_correction(
        self,
        x: Tensor,
        num_rejection_samples: int = 10_000,
        force_update: bool = False,
        show_progressbar: bool = False,
    ) -> Tensor:
        r"""Return leakage correction factor for a leaky posterior density estimate.

        The factor is estimated from the acceptance probability during rejection 
        sampling from the posterior.

        NOTE: This is to avoid re-estimating the acceptance probability from scratch
              whenever `log_prob` is called and `norm_posterior_snpe=True`. Here, it
              is estimated only once for `self.x_o` and saved for later. We re-evaluate
              only whenever a new `x` is passed.

        Arguments:
            x: Conditioning context for posterior $p(\theta|x)$. Use `self.x_o` if None.
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
        x: Optional[Tensor] = None,
        show_progressbar: bool = False,
        **kwargs,
    ) -> Tensor:
        r"""
        Return samples from posterior distribution $p(\theta|x)$.

        Args:
            num_samples: Desired number of samples.
            x: Conditioning context for posterior $p(\theta|x)$, use self.x_o if None.
            show_progressbar: Whether to show sampling progress monitor.
            **kwargs: Additional parameters for the MCMC sampler (`thin` and `warmup`).

        Returns: Samples from posterior.
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
                # Rejection sampling.
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
        show_progressbar: bool = True,
    ) -> Tensor:
        r"""
        Return MCMC samples from posterior $p(\theta|x)$.

        Args:
            x: Conditioning context for posterior $p(\theta|x)$.
            num_samples: Desired number of samples.
            mcmc_method: Sampling method. Currently defaults to `slice_np` for a custom
                numpy implementation of slice sampling; select `hmc`, `nuts` or `slice`
                for Pyro-based sampling.
            thin: Thinning factor for the chain, e.g. for `thin=3` only every third
                sample will be returned, until a total of `num_samples`.
            show_progressbar: Whether to show a progressbar during sampling.

        Returns:
            Tensor of shape (num_samples, shape_of_single_theta).
        """

        # When using `slice_np` as mcmc sampler, we can only have a single chain.
        if mcmc_method == "slice_np" and num_chains > 1:
            warn("`slice_np` does not support multiple mcmc chains. Using just one.")

        # TODO Maybe get whole sampler instead of just potential function?
        potential_fn = self._get_potential_function(
            self._prior, self.neural_net, x, mcmc_method
        )
        if mcmc_method == "slice_np":
            samples = self.slice_np_mcmc(num_samples, potential_fn, x, thin, warmup)
        elif mcmc_method in ("hmc", "nuts", "slice"):
            samples = self.pyro_mcmc(
                num_samples=num_samples,
                potential_function=potential_fn,
                x=x,
                mcmc_method=mcmc_method,
                thin=thin,
                warmup=warmup,
                num_chains=num_chains,
                show_progressbar=show_progressbar,
            )

        return samples

    def slice_np_mcmc(
        self,
        num_samples: int,
        potential_function: Callable,
        x: Tensor,
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
            num_samples: Desired number of samples.
            potential_function: A callable **class**. A class, but not a function,
                is picklable for Pyro's MCMC to use it across chains in parallel,
                even when the potential function requires evaluating a neural network.
            x: Conditioning context for posterior $p(\theta|x)$.
            mcmc_method: One of `hmc`, `nuts` or `slice`.
            thin: Thinning (subsampling) factor.
            warmup_steps: Initial snumber of amples to discard.
            num_chains: Whether to sample in parallel. If None, use all but one CPU.
            show_progressbar: Whether to show a progressbar during sampling.

        Returns: Tensor of shape (num_samples, shape_of_single_theta).
        """

        num_chains = mp.cpu_count - 1 if num_chains is None else num_chains

        # TODO move outside function, and assert inside; remember return to train
        # Always sample in eval mode.
        self.neural_net.eval()

        kernels = dict(slice=Slice, hmc=HMC, nuts=NUTS)

        initial_params = self._prior.sample((num_chains,))

        sampler = MCMC(
            kernel=kernels[mcmc_method](potential_fn=potential_function),
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
