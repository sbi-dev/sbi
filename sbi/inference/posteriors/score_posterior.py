# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

from typing import Dict, Optional, Union

import torch
from torch import Tensor
from torch.distributions import Distribution

from sbi.inference.posteriors.base_posterior import NeuralPosterior
from sbi.inference.potentials.score_based_potential import (
    PosteriorScoreBasedPotential,
    score_estimator_based_potential,
)
from sbi.neural_nets.estimators.score_estimator import ConditionalScoreEstimator
from sbi.neural_nets.estimators.shape_handling import (
    reshape_to_batch_event,
)
from sbi.samplers.score import Corrector, Diffuser, Predictor
from sbi.sbi_types import Shape
from sbi.utils import check_prior
from sbi.utils.torchutils import ensure_theta_batched


class ScorePosterior(NeuralPosterior):
    r"""Posterior $p(\theta|x_o)$ with `log_prob()` and `sample()` methods. It samples
    from the diffusion model given the score_estimator and rejects samples that lie
    outside of the prior bounds.

    The posterior is defined by a score estimator and a prior. The score estimator
    provides the gradient of the log-posterior with respect to the parameters. The prior
    is used to reject samples that lie outside of the prior bounds.

    Sampling is done by running a diffusion process with a predictor and optionally a
    corrector.

    Log probabilities are obtained by calling the potential function, which in turn uses
    zuko probabilistic ODEs to compute the log-probability.
    """

    def __init__(
        self,
        score_estimator: ConditionalScoreEstimator,
        prior: Distribution,
        max_sampling_batch_size: int = 10_000,
        device: Optional[str] = None,
        enable_transform: bool = False,
        sample_with: str = "sde",
    ):
        """
        Args:
            prior: Prior distribution with `.log_prob()` and `.sample()`.
            score_estimator: The trained neural score estimator.
            max_sampling_batch_size: Batchsize of samples being drawn from
                the proposal at every iteration.
            device: Training device, e.g., "cpu", "cuda" or "cuda:0". If None,
                `potential_fn.device` is used.
            enable_transform: Whether to transform parameters to unconstrained space
                during MAP optimization. When False, an identity transform will be
                returned for `theta_transform`. True is not supported yet.
            sample_with: Whether to sample from the posterior using the ODE-based
                sampler or the SDE-based sampler.
        """

        check_prior(prior)
        potential_fn, theta_transform = score_estimator_based_potential(
            score_estimator,
            prior,
            x_o=None,
            enable_transform=enable_transform,
        )
        super().__init__(
            potential_fn=potential_fn,
            theta_transform=theta_transform,
            device=device,
        )
        # Set the potential function type.
        self.potential_fn: PosteriorScoreBasedPotential = potential_fn

        self.prior = prior
        self.score_estimator = score_estimator

        self.sample_with = sample_with
        assert self.sample_with in [
            "ode",
            "sde",
        ], f"sample_with must be 'ode' or 'sde', but is {self.sample_with}."
        self.max_sampling_batch_size = max_sampling_batch_size

        self._purpose = """It samples from the diffusion model given the \
            score_estimator."""

    def sample(
        self,
        sample_shape: Shape = torch.Size(),
        x: Optional[Tensor] = None,
        predictor: Union[str, Predictor] = "euler_maruyama",
        corrector: Optional[Union[str, Corrector]] = None,
        predictor_params: Optional[Dict] = None,
        corrector_params: Optional[Dict] = None,
        steps: int = 500,
        ts: Optional[Tensor] = None,
        max_sampling_batch_size: int = 10_000,
        sample_with: Optional[str] = None,
        show_progress_bars: bool = True,
    ) -> Tensor:
        r"""Return samples from posterior distribution $p(\theta|x)$.

        Args:
            sample_shape: Shape of the samples to be drawn.
            x: Deprecated - use `.set_default_x()` prior to `.sample()`.
            predictor: The predictor for the diffusion-based sampler. Can be a string or
                a custom predictor following the API in `sbi.samplers.score.predictors`.
                Currently, only `euler_maruyama` is implemented.
            corrector: The corrector for the diffusion-based sampler. Either of
                [None].
            predictor_params: Additional parameters passed to predictor.
            corrector_params: Additional parameters passed to corrector.
            steps: Number of steps to take for the Euler-Maruyama method.
            ts: Time points at which to evaluate the diffusion process. If None, a
                linear grid between t_max and t_min is used.
            max_sampling_batch_size: Maximum batch size for sampling.
            sample_with: Deprecated - use `.build_posterior(sample_with=...)` prior to
                `.sample()`.
            show_progress_bars: Whether to show a progress bar during sampling.
        """

        if sample_with is not None:
            raise ValueError(
                f"You set `sample_with={sample_with}`. As of sbi v0.18.0, setting "
                f"`sample_with` is no longer supported. You have to rerun "
                f"`.build_posterior(sample_with={sample_with}).`"
            )

        x = self._x_else_default_x(x)
        x = reshape_to_batch_event(x, self.score_estimator.condition_shape)
        self.potential_fn.set_x(x)

        if self.sample_with == "ode":
            samples = self.sample_via_zuko(sample_shape=sample_shape, x=x)
        elif self.sample_with == "sde":
            samples = self._sample_via_diffusion(
                sample_shape=sample_shape,
                predictor=predictor,
                corrector=corrector,
                predictor_params=predictor_params,
                corrector_params=corrector_params,
                steps=steps,
                ts=ts,
                max_sampling_batch_size=max_sampling_batch_size,
                show_progress_bars=show_progress_bars,
            )

        return samples

    def _sample_via_diffusion(
        self,
        sample_shape: Shape = torch.Size(),
        predictor: Union[str, Predictor] = "euler_maruyama",
        corrector: Optional[Union[str, Corrector]] = None,
        predictor_params: Optional[Dict] = None,
        corrector_params: Optional[Dict] = None,
        steps: int = 500,
        ts: Optional[Tensor] = None,
        max_sampling_batch_size: int = 10_000,
        show_progress_bars: bool = True,
    ) -> Tensor:
        r"""Return samples from posterior distribution $p(\theta|x)$.

        Args:
            sample_shape: Shape of the samples to be drawn.
            x: Deprecated - use `.set_default_x()` prior to `.sample()`.
            predictor: The predictor for the diffusion-based sampler. Can be a string or
                a custom predictor following the API in `sbi.samplers.score.predictors`.
                Currently, only `euler_maruyama` is implemented.
            corrector: The corrector for the diffusion-based sampler. Either of
                [None].
            steps: Number of steps to take for the Euler-Maruyama method.
            ts: Time points at which to evaluate the diffusion process. If None, a
                linear grid between t_max and t_min is used.
            max_sampling_batch_size: Maximum batch size for sampling.
            sample_with: Deprecated - use `.build_posterior(sample_with=...)` prior to
                `.sample()`.
            show_progress_bars: Whether to show a progress bar during sampling.
        """

        num_samples = torch.Size(sample_shape).numel()

        max_sampling_batch_size = (
            self.max_sampling_batch_size
            if max_sampling_batch_size is None
            else max_sampling_batch_size
        )

        if ts is None:
            t_max = self.score_estimator.t_max
            t_min = self.score_estimator.t_min
            ts = torch.linspace(t_max, t_min, steps)

        diffuser = Diffuser(
            self.potential_fn,
            predictor=predictor,
            corrector=corrector,
            predictor_params=predictor_params,
            corrector_params=corrector_params,
        )
        max_sampling_batch_size = min(max_sampling_batch_size, num_samples)
        samples = []
        num_iter = num_samples // max_sampling_batch_size
        num_iter = (
            num_iter + 1 if (num_samples % max_sampling_batch_size) != 0 else num_iter
        )
        for _ in range(num_iter):
            samples.append(
                diffuser.run(
                    num_samples=max_sampling_batch_size,
                    ts=ts,
                    show_progress_bars=show_progress_bars,
                )
            )
        samples = torch.cat(samples, dim=0)[:num_samples]

        return samples.reshape(sample_shape + self.score_estimator.input_shape)

    def sample_via_zuko(
        self,
        x: Tensor,
        sample_shape: Shape = torch.Size(),
    ) -> Tensor:
        r"""Return samples from posterior distribution with probability flow ODE.

        This build the probability flow ODE and then samples from the corresponding
        flow. This is implemented via the zuko library.

        Args:
            x: Condition.
            sample_shape: The shape of the samples to be returned.

        Returns:
            Samples.
        """
        num_samples = torch.Size(sample_shape).numel()

        flow = self.potential_fn.get_continuous_normalizing_flow(condition=x)
        samples = flow.sample(torch.Size((num_samples,)))

        return samples.reshape(sample_shape + self.score_estimator.input_shape)

    def log_prob(
        self,
        theta: Tensor,
        x: Optional[Tensor] = None,
        track_gradients: bool = False,
        atol: float = 1e-5,
        rtol: float = 1e-6,
        exact: bool = True,
    ) -> Tensor:
        r"""Returns the log-probability of the posterior $p(\theta|x)$.

        This requires building and evaluating the probability flow ODE.

        Args:
            theta: Parameters $\theta$.
            x: Observed data $x_o$. If None, the default $x_o$ is used.
            track_gradients: Whether the returned tensor supports tracking gradients.
                This can be helpful for e.g. sensitivity analysis, but increases memory
                consumption.
            atol: Absolute tolerance for the ODE solver.
            rtol: Relative tolerance for the ODE solver.
            exact: Whether to use the exact Jacobian of the transformation or an
                stochastic approximation, which is faster but less accurate.

        Returns:
            `(len(θ),)`-shaped log posterior probability $\log p(\theta|x)$ for θ in the
            support of the prior, -∞ (corresponding to 0 probability) outside.
        """
        self.potential_fn.set_x(self._x_else_default_x(x))

        theta = ensure_theta_batched(torch.as_tensor(theta))
        return self.potential_fn(
            theta.to(self._device),
            track_gradients=track_gradients,
            atol=atol,
            rtol=rtol,
            exact=exact,
        )

    def sample_batched(
        self,
        sample_shape: torch.Size,
        x: Tensor,
        max_sampling_batch_size: int = 10000,
        show_progress_bars: bool = True,
    ) -> Tensor:
        raise NotImplementedError(
            "Batched sampling is not implemented for ScorePosterior."
        )

    def map(
        self,
        x: Optional[Tensor] = None,
        num_iter: int = 1000,
        num_to_optimize: int = 1000,
        learning_rate: float = 1e-5,
        init_method: Union[str, Tensor] = "posterior",
        num_init_samples: int = 1000,
        save_best_every: int = 1000,
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
            x: Deprecated - use `.set_default_x()` prior to `.map()`.
            num_iter: Number of optimization steps that the algorithm takes
                to find the MAP.
            num_to_optimize: From the drawn `num_init_samples`, use the
                `num_to_optimize` with highest log-probability as the initial points
                for the optimization.
            learning_rate: Learning rate of the optimizer.
            init_method: How to select the starting parameters for the optimization. If
                it is a string, it can be either [`posterior`, `prior`], which samples
                the respective distribution `num_init_samples` times. If it is a
                tensor, the tensor will be used as init locations.
            num_init_samples: Draw this number of samples from the posterior and
                evaluate the log-probability of all of them.
            save_best_every: The best log-probability is computed, saved in the
                `map`-attribute, and printed every `save_best_every`-th iteration.
                Computing the best log-probability creates a significant overhead
                (thus, the default is `10`.)
            show_progress_bars: Whether to show a progressbar during sampling from
                the posterior.
            force_update: Whether to re-calculate the MAP when x is unchanged and
                have a cached value.

        Returns:
            The MAP estimate.
        """
        raise NotImplementedError(
            "MAP estimation is currently not working accurately for ScorePosterior."
        )
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
