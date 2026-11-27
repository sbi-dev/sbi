# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

import warnings
from typing import Optional, Union

import torch
from torch import Tensor
from torch.distributions import Distribution

from sbi.inference.posteriors.base_posterior import (
    _IID_SAMPLERS,
    _LEAKAGE_HINT,
    _ONE_AT_A_TIME,
    NeuralPosterior,
    _normalize_by_support_mass,
)
from sbi.inference.potentials.posterior_based_potential import (
    posterior_estimator_based_potential,
)
from sbi.neural_nets.estimators.base import ConditionalDensityEstimator
from sbi.neural_nets.estimators.shape_handling import (
    reshape_to_batch_event,
    reshape_to_sample_batch_event,
)
from sbi.samplers.rejection import rejection
from sbi.sbi_types import Shape
from sbi.utils.sbiutils import warn_if_outside_prior_support, within_support
from sbi.utils.torchutils import (
    split_leading_dim,
)
from sbi.utils.user_input_checks import check_prior

_NPE_LEAKAGE_HINT = (
    f"{_LEAKAGE_HINT} After multi-round training, leakage often grows with every "
    "round and the estimate becomes poor; truncated proposals (TSNPE) can keep it "
    "from growing. See "
    "https://sbi.readthedocs.io/en/latest/faq/question_01_leakage.html"
)


class DirectPosterior(NeuralPosterior):
    r"""Posterior based on neural networks that directly estimate the posterior (NPE).

    NPE trains a neural network to directly approximate the posterior distribution.
    However, for bounded priors, the neural network can have leakage: it puts non-zero
    mass in regions where the prior is zero. The `DirectPosterior` class wraps the
    trained network to deal with these cases.

    Specifically, this class offers the following functionality:

    - correct the calculation of the log probability such that it compensates for the
      leakage.
    - reject samples that lie outside of the prior bounds.

    This class can not be used in combination with NLE or NRE.
    """

    # Suggested in the warning when rejection sampling accepts few samples.
    _alternative_sampling_method: Optional[str] = (
        "build_posterior(..., sample_with='mcmc')"
    )

    def __init__(
        self,
        posterior_estimator: ConditionalDensityEstimator,
        prior: Distribution,
        max_sampling_batch_size: int = 10_000,
        device: Optional[Union[str, torch.device]] = None,
        x_shape: Optional[torch.Size] = None,
        enable_transform: bool = True,
    ):
        """
        Args:
            prior: Prior distribution with `.log_prob()` and `.sample()`.
            posterior_estimator: The trained neural posterior.
            max_sampling_batch_size: Batchsize of samples being drawn from
                the proposal at every iteration.
            device: Training device, e.g., "cpu", "cuda" or "cuda:0". If None,
                `potential_fn.device` is used.
            x_shape: Deprecated, should not be passed.
            enable_transform: Whether to transform parameters to unconstrained space
                during MAP optimization. When False, an identity transform will be
                returned for `theta_transform`.
        """
        # Because `DirectPosterior` does not take the `potential_fn` as input, it
        # builds it itself. The `potential_fn` and `theta_transform` are used only for
        # obtaining the MAP.
        check_prior(prior)
        self.enable_transform = enable_transform
        self.x_shape = x_shape
        potential_fn, theta_transform = posterior_estimator_based_potential(
            posterior_estimator,
            prior,
            x_o=None,
            enable_transform=enable_transform,
        )

        super().__init__(
            potential_fn=potential_fn,
            theta_transform=theta_transform,
            device=device,
            x_shape=x_shape,
        )

        self.device = device
        self.prior = prior
        self.posterior_estimator = posterior_estimator

        self.max_sampling_batch_size = max_sampling_batch_size

        self._purpose = """It samples the posterior network and rejects samples that
            lie outside of the prior bounds."""

    @property
    def _one_observation_hint(self) -> str:
        return (
            f"{_ONE_AT_A_TIME}, or use `sample_batched()` and `log_prob_batched()`. "
            "To combine them as iid trials, use a posterior built with "
            f"{_IID_SAMPLERS}. For a direct posterior of several trials, train with "
            "a permutation-invariant embedding net on sets of trials, and pass each "
            "set as one observation."
        )

    def sample(
        self,
        sample_shape: Shape = torch.Size(),
        x: Optional[Tensor] = None,
        max_sampling_batch_size: Optional[int] = None,
        show_progress_bars: bool = True,
        reject_outside_prior: bool = True,
        max_sampling_time: Optional[float] = None,
        return_partial_on_timeout: bool = False,
    ) -> Tensor:
        r"""Draw samples from the approximate posterior distribution $p(\theta|x)$.

        Args:
            sample_shape: Desired shape of samples that are drawn from posterior. If
                sample_shape is multidimensional we simply draw `sample_shape.numel()`
                samples and then reshape into the desired shape.
            x: Conditioning observation $x_o$. If not provided, uses the default `x`
                set via `.set_default_x()`.
            max_sampling_batch_size: Maximum batch size for rejection sampling.
                If not provided, uses the value specified at initialization.
            show_progress_bars: Whether to show sampling progress monitor.
            reject_outside_prior: If True (default), rejection sampling is used to
                ensure samples lie within the prior support. If False, samples are drawn
                directly from the neural density estimator without rejection, which is
                faster but may include samples outside the prior support.
            max_sampling_time: Optional maximum allowed sampling time in seconds.
                If exceeded, sampling is aborted and a RuntimeError is raised. The
                limit is checked between sampling batches, so a running batch is not
                interrupted. Only applies when `reject_outside_prior=True` (no
                effect otherwise since direct sampling is fast).
            return_partial_on_timeout: If True and `max_sampling_time` is exceeded,
                return the samples collected so far instead of raising a RuntimeError.
                A warning will be issued. Only applies when `reject_outside_prior=True`
                (default).
        """
        num_samples = torch.Size(sample_shape).numel()
        x = self._x_else_default_x(x)
        self._check_num_observations(x.shape[0])

        max_sampling_batch_size = (
            self.max_sampling_batch_size
            if max_sampling_batch_size is None
            else max_sampling_batch_size
        )

        if reject_outside_prior:
            # Normal rejection behavior.
            samples = rejection.accept_reject_sample(
                proposal=self.posterior_estimator.sample,
                accept_reject_fn=lambda theta: within_support(self.prior, theta),
                num_samples=num_samples,
                show_progress_bars=show_progress_bars,
                max_sampling_batch_size=max_sampling_batch_size,
                proposal_sampling_kwargs={"condition": x},
                alternative_method=self._alternative_sampling_method,
                low_acceptance_hint=_NPE_LEAKAGE_HINT,
                max_sampling_time=max_sampling_time,
                return_partial_on_timeout=return_partial_on_timeout,
            )[0]
        else:
            # Bypass rejection sampling entirely.
            samples = self.posterior_estimator.sample(
                torch.Size([num_samples]),
                condition=x,
            )
            warn_if_outside_prior_support(self.prior, samples[:, 0])

        # Remove batch dimension.
        return self._reshape_to_sample_shape(samples[:, 0], sample_shape)

    @staticmethod
    def _reshape_to_sample_shape(samples: Tensor, sample_shape: Shape) -> Tensor:
        """Reshape the leading sample dimension of `samples` into `sample_shape`.

        A partial result on timeout holds fewer samples and stays flat.
        """
        if samples.shape[0] != torch.Size(sample_shape).numel():
            return samples
        return split_leading_dim(samples, sample_shape)

    def sample_batched(
        self,
        sample_shape: Shape,
        x: Tensor,
        max_sampling_batch_size: Optional[int] = None,
        show_progress_bars: bool = True,
        reject_outside_prior: bool = True,
        max_sampling_time: Optional[float] = None,
        return_partial_on_timeout: bool = False,
    ) -> Tensor:
        r"""Draw samples from the posteriors for a batch of different xs.

        Given a batch of observations `[x_1, ..., x_B]`, this method samples from
        posteriors $p(\theta|x_1), \ldots, p(\theta|x_B)$ in a vectorized manner.

        Args:
            sample_shape: Desired shape of samples that are drawn from the posterior
                given every observation.
            x: A batch of observations, of shape `(batch_dim, event_shape_x)`.
                `batch_dim` corresponds to the number of observations to be drawn.
            max_sampling_batch_size: Maximum batch size for rejection sampling.
                If not provided, uses the value specified at initialization.
            show_progress_bars: Whether to show sampling progress monitor.
            reject_outside_prior: If True (default), rejection sampling is used to
                ensure samples lie within the prior support. If False, samples are drawn
                directly from the neural density estimator without rejection, which is
                faster but may include samples outside the prior support.
            max_sampling_time: Optional maximum allowed sampling time in seconds.
                If exceeded, sampling is aborted and a RuntimeError is raised. The
                limit is checked between sampling batches, so a running batch is not
                interrupted. Only applies when `reject_outside_prior=True`.
            return_partial_on_timeout: If True and `max_sampling_time` is exceeded,
                return the samples collected so far instead of raising a RuntimeError.
                A warning will be issued. Only applies when `reject_outside_prior=True`.

        Returns:
            Samples from the posteriors of shape (*sample_shape, B, *input_shape)
        """
        x = self._process_x(x)
        num_samples = torch.Size(sample_shape).numel()
        num_xos = x.shape[0]

        # throw warning if num_x * num_samples is too large
        if num_xos * num_samples > 2**21:  # 2 million-ish
            warnings.warn(
                f"Note that for batched sampling, the direct posterior sampling "
                f"generates {num_xos} * {num_samples} = {num_xos * num_samples} "
                "samples. This can be slow and memory-intensive. Consider "
                "reducing the number of samples or batch size.",
                stacklevel=2,
            )

        max_sampling_batch_size = (
            self.max_sampling_batch_size
            if max_sampling_batch_size is None
            else max_sampling_batch_size
        )

        # Adjust max_sampling_batch_size to avoid excessive memory usage
        if max_sampling_batch_size * num_xos > 100_000:
            capped = max(1, 100_000 // num_xos)
            warnings.warn(
                f"Capping max_sampling_batch_size from {max_sampling_batch_size} "
                f"to {capped} to avoid excessive memory usage.",
                stacklevel=2,
            )
            max_sampling_batch_size = capped

        if reject_outside_prior:
            # Normal rejection behavior.
            samples = rejection.accept_reject_sample(
                proposal=self.posterior_estimator.sample,
                accept_reject_fn=lambda theta: within_support(self.prior, theta),
                num_samples=num_samples,
                show_progress_bars=show_progress_bars,
                max_sampling_batch_size=max_sampling_batch_size,
                proposal_sampling_kwargs={"condition": x},
                alternative_method=self._alternative_sampling_method,
                low_acceptance_hint=_NPE_LEAKAGE_HINT,
                max_sampling_time=max_sampling_time,
                return_partial_on_timeout=return_partial_on_timeout,
            )[0]
        else:
            # Bypass rejection sampling entirely.
            samples = self.posterior_estimator.sample(
                torch.Size([num_samples]),
                condition=x,
            )
            warn_if_outside_prior_support(self.prior, samples)

        return self._reshape_to_sample_shape(samples, sample_shape)

    def log_prob(
        self,
        theta: Tensor,
        x: Optional[Tensor] = None,
        norm_posterior: bool = True,
        track_gradients: bool = False,
        leakage_correction_params: Optional[dict] = None,
    ) -> Tensor:
        r"""Returns the log-probability of the posterior $p(\theta|x)$.

        Args:
            theta: Parameters $\theta$.
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
            leakage_correction_params: A `dict` of keyword arguments to override the
                default values of `leakage_correction()`. Possible options are:
                `num_rejection_samples`, `max_num_draws`, `force_update`,
                `show_progress_bars`, and `rejection_sampling_batch_size`.
                These parameters only have an effect if `norm_posterior=True`.

        Returns:
            `(len(θ),)`-shaped log posterior probability $\log p(\theta|x)$ for θ in the
            support of the prior, -∞ (corresponding to 0 probability) outside.
        """
        x = self._x_else_default_x(x)

        theta = self._theta_batched(theta).to(self._device)
        theta_density_estimator = reshape_to_sample_batch_event(
            theta, theta.shape[1:], leading_is_sample=True
        )
        self._check_num_observations(x.shape[0])

        self.posterior_estimator.eval()

        with torch.set_grad_enabled(track_gradients):
            # Evaluate on device, move back to cpu for comparison with prior.
            unnorm_log_prob = self.posterior_estimator.log_prob(
                theta_density_estimator, condition=x
            )
            # `log_prob` supports only a single observation (i.e. `batchsize==1`).
            # We now remove this additional dimension.
            unnorm_log_prob = unnorm_log_prob.squeeze(dim=1)

            # Force probability to be zero outside prior support.
            in_prior_support = within_support(self.prior, theta)

            masked_log_prob = torch.where(
                in_prior_support,
                unnorm_log_prob,
                torch.tensor(float("-inf"), dtype=torch.float32, device=self._device),
            )
            if not norm_posterior:
                return masked_log_prob
            mass = self.leakage_correction(x=x, **(leakage_correction_params or {}))
            return _normalize_by_support_mass(masked_log_prob, mass)

    def log_prob_batched(
        self,
        theta: Tensor,
        x: Tensor,
        norm_posterior: bool = True,
        track_gradients: bool = False,
        leakage_correction_params: Optional[dict] = None,
    ) -> Tensor:
        """Given a batch of observations [x_1, ..., x_B] and a batch of parameters \
            [$\theta_1$,..., $\theta_B$] this function evalautes the log-probabilities \
            of the posteriors $p(\theta_1|x_1)$, ..., $p(\theta_B|x_B)$ in a batched \
            (i.e. vectorized) manner.

        Args:
            theta: Batch of parameters $\theta$ of shape \
                `(*sample_shape, batch_dim, *theta_shape)`.
            x: Batch of observations $x$ of shape \
                `(batch_dim, *condition_shape)`.
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
            leakage_correction_params: A `dict` of keyword arguments to override the
                default values of `leakage_correction()`. Possible options are:
                `num_rejection_samples`, `max_num_draws`, `force_update`,
                `show_progress_bars`, and `rejection_sampling_batch_size`.
                These parameters only have an effect if `norm_posterior=True`.

        Returns:
            `(len(θ), B)`-shaped log posterior probability $\\log p(\theta|x)$\\ for θ \
            in the support of the prior, -∞ (corresponding to 0 probability) outside.
        """

        x = self._process_x(x)
        theta = self._theta_batched(theta).to(self._device)
        event_shape = self.posterior_estimator.input_shape
        # If theta has 1 leading dim (batch, event), treat it as batch (matching x).
        # otherwise, the leading is sample.
        num_leading = len(theta.shape) - len(event_shape)
        if num_leading > 2:
            flat_theta = theta.reshape(-1, *theta.shape[num_leading - 1 :])
            return self.log_prob_batched(
                flat_theta,
                x,
                norm_posterior,
                track_gradients,
                leakage_correction_params,
            ).reshape(theta.shape[:num_leading])
        theta_density_estimator = reshape_to_sample_batch_event(
            theta, event_shape, leading_is_sample=(num_leading > 1)
        )
        x_density_estimator = reshape_to_batch_event(
            x, event_shape=self.posterior_estimator.condition_shape
        )

        self.posterior_estimator.eval()

        with torch.set_grad_enabled(track_gradients):
            # Evaluate on device, move back to cpu for comparison with prior.
            unnorm_log_prob = self.posterior_estimator.log_prob(
                theta_density_estimator, condition=x_density_estimator
            )

            # Force probability to be zero outside prior support.
            in_prior_support = within_support(self.prior, theta)

            masked_log_prob = torch.where(
                in_prior_support,
                unnorm_log_prob,
                torch.tensor(float("-inf"), dtype=torch.float32, device=self._device),
            )
            if not norm_posterior:
                return masked_log_prob
            mass = self.leakage_correction(x=x, **(leakage_correction_params or {}))
            return _normalize_by_support_mass(masked_log_prob, mass)

    def leakage_correction(
        self,
        x: Tensor,
        num_rejection_samples: int = 10_000,
        force_update: bool = False,
        show_progress_bars: bool = False,
        rejection_sampling_batch_size: int = 10_000,
        max_num_draws: int = 1_000_000,
    ) -> Tensor:
        r"""Return leakage correction factor for a leaky posterior density estimate.

        The factor is the share of posterior draws inside the prior support. For each
        observation, it draws until `num_rejection_samples` draws are inside or
        `max_num_draws` draws are made. Without any draw inside, the factor is 0, with
        a warning, and `log_prob()` is NaN inside the prior support. The factor is
        saved for the last `x` and re-estimated whenever `x`, `num_rejection_samples`
        or `max_num_draws` changes. It is 1 for unbounded priors.

        Arguments:
            x: Observation at which to estimate the factor.
            num_rejection_samples: Number of draws inside the prior support after
                which the estimate stops.
            force_update: Whether to re-estimate the factor even if it is saved.
            show_progress_bars: Whether to show a progress bar during sampling.
            rejection_sampling_batch_size: Maximum number of draws at once, over all
                observations in `x`.
            max_num_draws: Number of draws per observation after which the estimate
                stops, whatever it found inside.

        Returns:
            Saved or newly-estimated correction factor, one per observation in `x`.
        """
        x = self._x_else_default_x(x)
        condition = reshape_to_batch_event(
            x, event_shape=self.posterior_estimator.condition_shape
        )

        def proposal(num_draws: int, indices: Tensor) -> Tensor:
            return self.posterior_estimator.sample(
                torch.Size((num_draws,)),
                condition=condition[indices.to(condition.device)],
            )

        return self._cached_leakage_factor(
            x,
            (num_rejection_samples, max_num_draws),
            self.prior,
            lambda: rejection.count_accepted(
                proposal=proposal,
                accept_reject_fn=lambda theta: within_support(self.prior, theta),
                num_xos=condition.shape[0],
                num_accepted=num_rejection_samples,
                max_num_draws=max_num_draws,
                max_sampling_batch_size=rejection_sampling_batch_size,
                show_progress_bars=show_progress_bars,
            ),
            _NPE_LEAKAGE_HINT,
            force_update,
        )

    def map(
        self,
        x: Optional[Tensor] = None,
        num_iter: int = 1_000,
        num_to_optimize: int = 100,
        learning_rate: float = 0.01,
        init_method: Union[str, Tensor] = "posterior",
        num_init_samples: int = 1_000,
        save_best_every: int = 10,
        show_progress_bars: bool = False,
        force_update: bool = False,
    ) -> Tensor:
        r"""Returns the maximum-a-posteriori estimate (MAP).

        The method can be interrupted (Ctrl-C) when the user sees that the
        log-probability converges. The best estimate will be saved in `self._map` and
        can be accessed with `self.map()`. The MAP is obtained by running gradient
        ascent from a given number of starting positions (samples from the posterior
        with the highest log-probability). After the optimization is done, we select the
        parameter set that has the highest log-probability after the optimization.

        Warning: The default values used by this function are not well-tested. They
        might require hand-tuning for the problem at hand.

        For developers: if the prior is a `BoxUniform`, we carry out the optimization
        in unbounded space and transform the result back into bounded space.

        Args:
            x: No longer supported; use `.set_default_x()` before `.map()`.
            num_iter: Number of optimization steps that the algorithm takes
                to find the MAP.
            learning_rate: Learning rate of the optimizer.
            init_method: How to select the starting parameters for the optimization. If
                it is a string, it must be `posterior`, which samples the
                posterior `num_init_samples` times. If it is a tensor, the tensor is
                used as the initial locations, so prior-based initialization can be
                done by passing prior samples.
            num_init_samples: Draw this number of samples from the posterior and
                evaluate the log-probability of all of them.
            num_to_optimize: From the drawn `num_init_samples`, use the
                `num_to_optimize` with highest log-probability as the initial points
                for the optimization.
            save_best_every: The best log-probability is computed, saved in the
                `map`-attribute, and printed every `save_best_every`-th iteration.
                Computing the best log-probability creates a significant overhead
                (thus, the default is `10`.)
            show_progress_bars: Whether to show a progressbar during sampling from the
                posterior.
            force_update: Whether to re-calculate the MAP when x is unchanged and
                have a cached value.

        Returns:
            The MAP estimate.
        """
        return super().map(
            x=x,
            num_iter=num_iter,
            num_to_optimize=num_to_optimize,
            learning_rate=learning_rate,
            init_method=init_method,
            num_init_samples=num_init_samples,
            save_best_every=save_best_every,
            show_progress_bars=show_progress_bars,
            force_update=force_update,
        )
