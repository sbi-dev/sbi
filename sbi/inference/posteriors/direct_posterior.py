# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Affero General Public License v3, see <https://www.gnu.org/licenses/>.

from typing import Any, Callable, Dict, List, Optional

import cma
import numpy as np
import torch
from torch import Tensor, log, nn, optim

from sbi import utils as utils
from sbi.inference.posteriors.base_posterior import NeuralPosterior
from sbi.types import ScalarFloat, Shape
from sbi.utils import del_entries
from sbi.utils.torchutils import (
    batched_first_of_batch,
    ensure_theta_batched,
    ensure_x_batched,
)


class DirectPosterior(NeuralPosterior):
    r"""Posterior $p(\theta|x)$ with `log_prob()` and `sample()` methods, obtained with
    SNPE.<br/><br/>
    SNPE trains a neural network to directly approximate the posterior distribution.
    However, for bounded priors, the neural network can have leakage: it puts non-zero
    mass in regions where the prior is zero. The `DirectPosterior` class wraps the
    trained network to deal with these cases.<br/><br/>
    Specifically, this class offers the following functionality:<br/>
    - correct the calculation of the log probability such that it compensates for the
      leakage.<br/>
    - reject samples that lie outside of the prior bounds.<br/>
    - alternatively, if leakage is very high (which can happen for multi-round SNPE),
      sample from the posterior with MCMC.<br/><br/>
    The neural network itself can be accessed via the `.net` attribute.
    """

    def __init__(
        self,
        method_family: str,
        neural_net: nn.Module,
        prior,
        x_shape: torch.Size,
        rejection_sampling_parameters: Optional[Dict[str, Any]] = None,
        sample_with_mcmc: bool = True,
        mcmc_method: str = "slice_np",
        mcmc_parameters: Optional[Dict[str, Any]] = None,
        device: str = "cpu",
    ):
        """
        Args:
            method_family: One of snpe, snl, snre_a or snre_b.
            neural_net: A classifier for SNRE, a density estimator for SNPE and SNL.
            prior: Prior distribution with `.log_prob()` and `.sample()`.
            x_shape: Shape of a single simulator output.
            rejection_sampling_parameters: Dictonary overriding the default parameters
                for rejection sampling. The following parameters are supported:
                `max_sampling_batch_size` to set the batch size for drawing new
                samples from the candidate distribution, e.g., the posterior. Larger
                batch size speeds up sampling.
            sample_with_mcmc: Whether to sample with MCMC. Will always be `True` for SRE
                and SNL, but can also be set to `True` for SNPE if MCMC is preferred to
                deal with leakage over rejection sampling.
            mcmc_method: Method used for MCMC sampling, one of `slice_np`, `slice`,
                `hmc`, `nuts`. Currently defaults to `slice_np` for a custom numpy
                implementation of slice sampling; select `hmc`, `nuts` or `slice` for
                Pyro-based sampling.
            mcmc_parameters: Dictionary overriding the default parameters for MCMC.
                The following parameters are supported: `thin` to set the thinning
                factor for the chain, `warmup_steps` to set the initial number of
                samples to discard, `num_chains` for the number of chains,
                `init_strategy` for the initialisation strategy for chains; `prior`
                will draw init locations from prior, whereas `sir` will use Sequential-
                Importance-Resampling using `init_strategy_num_candidates` to find init
                locations.
            device: Training device, e.g., cpu or cuda:0
        """

        kwargs = del_entries(
            locals(),
            entries=(
                "self",
                "__class__",
                "sample_with_mcmc",
                "rejection_sampling_parameters",
            ),
        )
        super().__init__(**kwargs)

        self.set_sample_with_mcmc(sample_with_mcmc)
        self.set_rejection_sampling_parameters(rejection_sampling_parameters)
        self._purpose = (
            "It allows to .sample() and .log_prob() the posterior and wraps the "
            "output of the .net to avoid leakage into regions with 0 prior probability."
        )

    @property
    def sample_with_mcmc(self) -> bool:
        """
        Return `True` if NeuralPosterior instance should use MCMC in `.sample()`.
        """
        return self._sample_with_mcmc

    @sample_with_mcmc.setter
    def sample_with_mcmc(self, value: bool) -> None:
        """See `set_sample_with_mcmc`."""
        self.set_sample_with_mcmc(value)

    def set_sample_with_mcmc(self, use_mcmc: bool) -> "NeuralPosterior":
        """Turns MCMC sampling on or off and returns `NeuralPosterior`.

        Args:
            use_mcmc: Flag to set whether or not MCMC sampling is used.

        Returns:
            `NeuralPosterior` for chainable calls.

        Raises:
            ValueError: on attempt to turn off MCMC sampling for family of methods that
                do not support rejection sampling.
        """
        self._sample_with_mcmc = use_mcmc
        return self

    @property
    def rejection_sampling_parameters(self) -> dict:
        """Returns rejection sampling parameter."""
        if self._rejection_sampling_parameters is None:
            return {}
        else:
            return self._rejection_sampling_parameters

    @rejection_sampling_parameters.setter
    def rejection_sampling_parameters(self, parameters: Dict[str, Any]) -> None:
        """See `set_rejection_sampling_parameters`."""
        self.set_rejection_sampling_parameters(parameters)

    def set_rejection_sampling_parameters(
        self, parameters: Dict[str, Any]
    ) -> "NeuralPosterior":
        """Sets parameters for rejection sampling and returns `NeuralPosterior`.

        Args:
            parameters: Dictonary overriding the default parameters
                for rejection sampling. The following parameters are supported:
                `max_sampling_batch_size` to the set the batch size for drawing new
                samples from the candidate distribution, e.g., the posterior. Larger
                batch size speeds up sampling.

        Returns:
            `NeuralPosterior for chainable calls.
        """
        self._rejection_sampling_parameters = parameters
        return self

    def log_prob(
        self,
        theta: Tensor,
        x: Optional[Tensor] = None,
        norm_posterior: bool = True,
        track_gradients: bool = False,
    ) -> Tensor:
        r"""
        Returns the log-probability of the posterior $p(\theta|x).$

        Args:
            theta: Parameters $\theta$.
            x: Conditioning context for posterior $p(\theta|x)$. If not provided,
                fall back onto `x` passed to `set_default_x()`.
            norm_posterior: Whether to enforce a normalized posterior density.
                Renormalization of the posterior is useful when some
                probability falls out or leaks out of the prescribed prior support.
                The normalizing factor is calculated via rejection sampling, so if you
                need speedier but unnormalized log posterior estimates set here
                `norm_posterior=False`. The returned log posterior is set to
                -∞ outside of the prior support regardless of this setting.
            track_gradients: Whether the returned tensor supports tracking gradients.
                This can be helpful for e.g. sensitivity analysis, but increases memory
                consumption.

        Returns:
            `(len(θ),)`-shaped log posterior probability $\log p(\theta|x)$ for θ in the
            support of the prior, -∞ (corresponding to 0 probability) outside.

        """

        # TODO Train exited here, entered after sampling?
        self.net.eval()

        theta, x = self._prepare_theta_and_x_for_log_prob_(theta, x)

        with torch.set_grad_enabled(track_gradients):

            # Evaluate on device, move back to cpu for comparison with prior.
            unnorm_log_prob = self.net.log_prob(
                theta.to(self._device), x.to(self._device)
            ).cpu()

            # Force probability to be zero outside prior support.
            is_prior_finite = torch.isfinite(self._prior.log_prob(theta))

            masked_log_prob = torch.where(
                is_prior_finite,
                unnorm_log_prob,
                torch.tensor(float("-inf"), dtype=torch.float32),
            )

            log_factor = (
                log(self.leakage_correction(x=batched_first_of_batch(x)))
                if norm_posterior
                else 0
            )

            return masked_log_prob - log_factor

    @torch.no_grad()
    def leakage_correction(
        self,
        x: Tensor,
        num_rejection_samples: int = 10_000,
        force_update: bool = False,
        show_progress_bars: bool = False,
        rejection_sampling_batch_size: int = 10_000,
    ) -> Tensor:
        r"""Return leakage correction factor for a leaky posterior density estimate.

        The factor is estimated from the acceptance probability during rejection
        sampling from the posterior.

        This is to avoid re-estimating the acceptance probability from scratch
        whenever `log_prob` is called and `norm_posterior=True`. Here, it
        is estimated only once for `self.default_x` and saved for later. We
        re-evaluate only whenever a new `x` is passed.

        Arguments:
            x: Conditioning context for posterior $p(\theta|x)$.
            num_rejection_samples: Number of samples used to estimate correction factor.
            force_update: Whether to force a reevaluation of the leakage correction even
                if the context `x` is the same as `self.default_x`. This is useful to
                enforce a new leakage estimate for rounds after the first (2, 3,..).
            show_progress_bars: Whether to show a progress bar during sampling.
            rejection_sampling_batch_size: Batch size for rejection sampling.

        Returns:
            Saved or newly-estimated correction factor (as a scalar `Tensor`).
        """

        def acceptance_at(x: Tensor) -> Tensor:
            return utils.sample_posterior_within_prior(
                self.net,
                self._prior,
                x.to(self._device),
                num_rejection_samples,
                show_progress_bars,
                sample_for_correction_factor=True,
                max_sampling_batch_size=rejection_sampling_batch_size,
            )[1]

        # Check if the provided x matches the default x (short-circuit on identity).
        is_new_x = self.default_x is None or (
            x is not self.default_x and (x != self.default_x).any()
        )

        not_saved_at_default_x = self._leakage_density_correction_factor is None

        if is_new_x:  # Calculate at x; don't save.
            return acceptance_at(x)
        elif not_saved_at_default_x or force_update:  # Calculate at default_x; save.
            self._leakage_density_correction_factor = acceptance_at(self.default_x)

        return self._leakage_density_correction_factor  # type:ignore

    def sample(
        self,
        sample_shape: Shape = torch.Size(),
        x: Optional[Tensor] = None,
        show_progress_bars: bool = True,
        sample_with_mcmc: Optional[bool] = None,
        mcmc_method: Optional[str] = None,
        mcmc_parameters: Optional[Dict[str, Any]] = None,
        rejection_sampling_parameters: Optional[Dict[str, Any]] = None,
    ) -> Tensor:
        r"""
        Return samples from posterior distribution $p(\theta|x)$.

        Samples are obtained either with rejection sampling or MCMC. Rejection sampling
        will be a lot faster if leakage is rather low. If leakage is high (e.g. over
        99%, which can happen in multi-round SNPE), MCMC can be faster than rejection
        sampling.

        Args:
            sample_shape: Desired shape of samples that are drawn from posterior. If
                sample_shape is multidimensional we simply draw `sample_shape.numel()`
                samples and then reshape into the desired shape.
            x: Conditioning context for posterior $p(\theta|x)$. If not provided,
                fall back onto `x` passed to `set_default_x()`.
            show_progress_bars: Whether to show sampling progress monitor.
            sample_with_mcmc: Optional parameter to override `self.sample_with_mcmc`.
            mcmc_method: Optional parameter to override `self.mcmc_method`.
            mcmc_parameters: Dictionary overriding the default parameters for MCMC.
                The following parameters are supported: `thin` to set the thinning
                factor for the chain, `warmup_steps` to set the initial number of
                samples to discard, `num_chains` for the number of chains,
                `init_strategy` for the initialisation strategy for chains; `prior`
                will draw init locations from prior, whereas `sir` will use Sequential-
                Importance-Resampling using `init_strategy_num_candidates` to find init
                locations.
            rejection_sampling_parameters: Dictionary overriding the default parameters
                for rejection sampling. The following parameters are supported:
                `max_sampling_batch_size` to set the batch size for drawing new
                samples from the candidate distribution, e.g., the posterior. Larger
                batch size speeds up sampling.
        Returns:
            Samples from posterior.
        """

        x, num_samples, mcmc_method, mcmc_parameters = self._prepare_for_sample(
            x, sample_shape, mcmc_method, mcmc_parameters
        )

        sample_with_mcmc = (
            sample_with_mcmc if sample_with_mcmc is not None else self.sample_with_mcmc
        )

        self.net.eval()

        if sample_with_mcmc:
            potential_fn_provider = PotentialFunctionProvider()
            samples = self._sample_posterior_mcmc(
                num_samples=num_samples,
                potential_fn=potential_fn_provider(
                    self._prior, self.net, x, mcmc_method
                ),
                init_fn=self._build_mcmc_init_fn(
                    self._prior,
                    potential_fn_provider(self._prior, self.net, x, "slice_np"),
                    **mcmc_parameters,
                ),
                mcmc_method=mcmc_method,
                show_progress_bars=show_progress_bars,
                **mcmc_parameters,
            )
        else:
            # Rejection sampling.
            samples, _ = utils.sample_posterior_within_prior(
                self.net,
                self._prior,
                x,
                num_samples=num_samples,
                show_progress_bars=show_progress_bars,
                **rejection_sampling_parameters
                if (rejection_sampling_parameters is not None)
                else self.rejection_sampling_parameters,
            )

        self.net.train(True)

        return samples.reshape((*sample_shape, -1))

    def sample_conditional(
        self,
        sample_shape: Shape,
        condition: Tensor,
        dims_to_sample: List[int],
        x: Optional[Tensor] = None,
        show_progress_bars: bool = True,
        mcmc_method: Optional[str] = None,
        mcmc_parameters: Optional[Dict[str, Any]] = None,
    ) -> Tensor:
        r"""
        Return samples from conditional posterior $p(\theta_i|\theta_j, x)$.

        In this function, we do not sample from the full posterior, but instead only
        from a few parameter dimensions while the other parameter dimensions are kept
        fixed at values specified in `condition`.

        Samples are obtained with MCMC.

        Args:
            sample_shape: Desired shape of samples that are drawn from posterior. If
                sample_shape is multidimensional we simply draw `sample_shape.numel()`
                samples and then reshape into the desired shape.
            condition: Parameter set that all dimensions not specified in
                `dims_to_sample` will be fixed to. Should contain dim_theta elements,
                i.e. it could e.g. be a sample from the posterior distribution.
                The entries at all `dims_to_sample` will be ignored.
            dims_to_sample: Which dimensions to sample from. The dimensions not
                specified in `dims_to_sample` will be fixed to values given in
                `condition`.
            x: Conditioning context for posterior $p(\theta|x)$. If not provided,
                fall back onto `x` passed to `set_default_x()`.
            show_progress_bars: Whether to show sampling progress monitor.
            mcmc_method: Optional parameter to override `self.mcmc_method`.
            mcmc_parameters: Dictionary overriding the default parameters for MCMC.
                The following parameters are supported: `thin` to set the thinning
                factor for the chain, `warmup_steps` to set the initial number of
                samples to discard, `num_chains` for the number of chains,
                `init_strategy` for the initialisation strategy for chains; `prior`
                will draw init locations from prior, whereas `sir` will use Sequential-
                Importance-Resampling using `init_strategy_num_candidates` to find init
                locations.

        Returns:
            Samples from conditional posterior.
        """

        return super().sample_conditional(
            PotentialFunctionProvider(),
            sample_shape,
            condition,
            dims_to_sample,
            x,
            show_progress_bars,
            mcmc_method,
            mcmc_parameters,
        )

    def map_estimate(
        self,
        x: Optional[Tensor] = None,
        num_iter: int = 500,
        learning_rate: float = 1e-2,
        num_init_samples: int = 10_000,
        num_to_optimize: int = 100,
        show_progress_bars: bool = False,
    ) -> Tensor:
        """
        Returns the maximum-a-posteriori estimate (MAP).

        The MAP is obtained by running gradient ascent from a given number of starting
        positions (sampled from the posterior) and then selecting the parameter set
        that has the highest log-probability after the optimization.

        Args:
            x: Conditioning context for posterior $p(\theta|x)$. If not provided,
                fall back onto `x` passed to `set_default_x()`.
            num_iter: Number of optimization steps that the algorithm takes
                to find the MAP.
            learning_rate: Learning rate of the optimizer.
            num_init_samples: Draw this number of samples from the posterior and
                evaluate the log-probability of all of them.
            num_to_optimize: From the drawn `num_init_samples`, use the
                `num_to_optimize` with highest log-probability as the initial points
                for the optimization.
            show_progress_bars: Whether or not to show a progressbar for sampling from
                the posterior.

        Returns: The MAP estimate.
        """

        if isinstance(self._prior, utils.BoxUniform):

            def expit(theta_t):
                ub = self._prior.support.upper_bound
                lb = self._prior.support.lower_bound
                r = ub - lb
                return r / (1 + torch.exp(-theta_t)) + lb

            def logit(theta):
                ub = self._prior.support.upper_bound
                lb = self._prior.support.lower_bound
                r = ub - lb
                theta_01 = (theta - lb) / r

                return torch.log(theta_01 / (1 - theta_01))

        else:

            def expit(theta_t):
                return theta_t

            def logit(theta):
                return theta

        # Find initial position.
        inits = self.sample((num_init_samples,), show_progress_bars=show_progress_bars)
        init_probs = self.log_prob(inits, x=x, norm_posterior=False)
        sort_indices = torch.argsort(init_probs, dim=0)

        # Pick the `num_to_optimize` best init locations.
        sorted_inits = inits[sort_indices]
        optimize_inits = sorted_inits[-num_to_optimize:]

        optimize_inits = logit(optimize_inits)

        # Optimize the init locations.
        optimize_inits.requires_grad_(True)
        optimizer = optim.Adam([optimize_inits], lr=learning_rate)

        for _ in range(num_iter):
            optimizer.zero_grad()
            probs = self.log_prob(
                expit(optimize_inits), x=x, norm_posterior=False, track_gradients=True
            ).squeeze()
            loss = -probs.sum()
            loss.backward()
            optimizer.step()

        # Evaluate the optimized locations and pick the best one.
        log_probs_of_optimized = self.log_prob(
            expit(optimize_inits), x=x, norm_posterior=False
        )
        best_theta = optimize_inits[torch.argmax(log_probs_of_optimized)]

        return expit(best_theta)

    def mle_estimate(
        self,
        x: Optional[Tensor] = None,
        num_iter: int = 500,
        learning_rate: float = 1e-2,
        num_init_samples: int = 10_000,
        num_to_optimize: int = 100,
        show_progress_bars: bool = False,
    ) -> Tensor:
        """
        Returns the maximum-likelihood estimate (MLE) on the prior support.

        The MLE is obtained by running gradient ascent from a given number of starting
        positions (sampled from the posterior) and then selecting the parameter set
        that has the highest log-probability after the optimization.

        Args:
            x: Conditioning context for posterior $p(\theta|x)$. If not provided,
                fall back onto `x` passed to `set_default_x()`.
            num_iter: Number of optimization steps that the algorithm takes
                to find the MAP.
            learning_rate: Learning rate of the optimizer.
            num_init_samples: Draw this number of samples from the posterior and
                evaluate the log-probability of all of them.
            num_to_optimize: From the drawn `num_init_samples`, use the
                `num_to_optimize` with highest log-probability as the initial points
                for the optimization.
            show_progress_bars: Whether or not to show a progressbar for sampling from
                the posterior.

        Returns: The MLE estimate.
        """

        if isinstance(self._prior, utils.BoxUniform):

            def expit(theta_t):
                ub = self._prior.support.upper_bound
                lb = self._prior.support.lower_bound
                r = ub - lb
                return r / (1 + torch.exp(-theta_t)) + lb

            def logit(theta):
                ub = self._prior.support.upper_bound
                lb = self._prior.support.lower_bound
                r = ub - lb
                theta_01 = (theta - lb) / r

                return torch.log(theta_01 / (1 - theta_01))

        else:

            def expit(theta_t):
                return theta_t

            def logit(theta):
                return theta

        # Find initial position.
        inits = self.sample((num_init_samples,), show_progress_bars=show_progress_bars)
        init_probs = self.log_prob(
            inits, x=x, norm_posterior=False
        ) - self._prior.log_prob(inits)
        sort_indices = torch.argsort(init_probs, dim=0)

        # Pick the `num_to_optimize` best init locations.
        sorted_inits = inits[sort_indices]
        optimize_inits = sorted_inits[-num_to_optimize:]

        optimize_inits = logit(optimize_inits)

        # Optimize the init locations.
        optimize_inits.requires_grad_(True)
        optimizer = optim.Adam([optimize_inits], lr=learning_rate)

        for _ in range(num_iter):
            optimizer.zero_grad()
            posterior_log_prob = self.log_prob(
                expit(optimize_inits), x=x, norm_posterior=False, track_gradients=True
            ).squeeze()

            prior_log_prob = self._prior.log_prob(expit(optimize_inits))

            likelihood = posterior_log_prob - prior_log_prob
            loss = -likelihood.sum()

            loss.backward()
            optimizer.step()

        # Evaluate the optimized locations and pick the best one.
        log_probs_of_optimized = self.log_prob(
            expit(optimize_inits), x=x, norm_posterior=False
        ) - self._prior.log_prob(expit(optimize_inits))
        best_theta = optimize_inits[torch.argmax(log_probs_of_optimized)]

        return expit(best_theta)


class PotentialFunctionProvider:
    """
    This class is initialized without arguments during the initialization of the
    Posterior class. When called, it specializes to the potential function appropriate
    to the requested `mcmc_method`.

    NOTE: Why use a class?
    ----------------------
    During inference, we use deepcopy to save untrained posteriors in memory. deepcopy
    uses pickle which can't serialize nested functions
    (https://stackoverflow.com/a/12022055).

    It is important to NOT initialize attributes upon instantiation, because we need the
     most current trained posterior neural net.

    Returns:
        Potential function for use by either numpy or pyro sampler
    """

    def __call__(
        self, prior, posterior_nn: nn.Module, x: Tensor, mcmc_method: str,
    ) -> Callable:
        """Return potential function.

        Switch on numpy or pyro potential function based on `mcmc_method`.
        """
        self.posterior_nn = posterior_nn
        self.prior = prior
        self.x = x

        if mcmc_method in ("slice", "hmc", "nuts"):
            return self.pyro_potential
        else:
            return self.np_potential

    def np_potential(self, theta: np.ndarray) -> ScalarFloat:
        r"""Return posterior theta log prob. $p(\theta|x)$, $-\infty$ if outside prior."

        Args:
            theta: Parameters $\theta$, batch dimension 1.

        Returns:
            Posterior log probability $\log(p(\theta|x))$.
        """
        theta = torch.as_tensor(theta, dtype=torch.float32)
        theta = ensure_theta_batched(theta)
        num_batch = theta.shape[0]

        x_batched = ensure_x_batched(self.x)
        # Repeat x over batch dim to match theta batch, accounting for multi-D x.
        x_repeated = x_batched.repeat(
            num_batch, *(1 for _ in range(x_batched.ndim - 1))
        )

        with torch.set_grad_enabled(False):
            target_log_prob = self.posterior_nn.log_prob(
                inputs=theta.to(self.x.device), context=x_repeated,
            )
            is_within_prior = torch.isfinite(self.prior.log_prob(theta))
            target_log_prob[~is_within_prior] = -float("Inf")

        return target_log_prob

    def pyro_potential(self, theta: Dict[str, Tensor]) -> Tensor:
        r"""Return posterior log prob. of theta $p(\theta|x)$, -inf where outside prior.

        Args:
            theta: Parameters $\theta$ (from pyro sampler).

        Returns:
            Posterior log probability $p(\theta|x)$, masked outside of prior.
        """

        theta = next(iter(theta.values()))

        # Notice opposite sign to numpy.
        # Move theta to device for evaluation.
        log_prob_posterior = -self.posterior_nn.log_prob(
            inputs=theta.to(self.x.device), context=self.x
        ).cpu()
        log_prob_prior = self.prior.log_prob(theta)

        within_prior = torch.isfinite(log_prob_prior)

        return torch.where(within_prior, log_prob_posterior, log_prob_prior)
