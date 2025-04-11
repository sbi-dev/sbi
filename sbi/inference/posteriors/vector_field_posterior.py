# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

from typing import Dict, Literal, Optional, Union

import torch
from torch import Tensor
from torch.distributions import Distribution

from sbi.inference.posteriors.base_posterior import NeuralPosterior
from sbi.inference.potentials.vector_field_potential import (
    CallableDifferentiablePotentialFunction,
    VectorFieldBasedPotential,
    vector_field_estimator_based_potential,
)
from sbi.neural_nets.estimators.base import ConditionalVectorFieldEstimator
from sbi.neural_nets.estimators.shape_handling import (
    reshape_to_batch_event,
)
from sbi.samplers.rejection import rejection
from sbi.samplers.score.correctors import Corrector
from sbi.samplers.score.diffuser import Diffuser
from sbi.samplers.score.predictors import Predictor
from sbi.sbi_types import Shape
from sbi.utils import check_prior
from sbi.utils.sbiutils import gradient_ascent, within_support
from sbi.utils.torchutils import ensure_theta_batched


class VectorFieldPosterior(NeuralPosterior):
    r"""Posterior based on flow- or score-matching estimators.

    This posterior samples from the vector field model - typically a score-based or a
    flow matching model - given the `vector_field_estimator` and rejects samples that
    lie outside of the prior bounds.

    The posterior is defined by a vector field estimator and a prior. The vector field
    estimator defines a continuous transformation from a base distribution to the
    approximated posterior distribution. Sampling is done by running either
    an ordinary differential equation (ODE) or a stochastic differential equation
    (SDE) defined by the vector field estimator with the starting points sampled from
    the base distribution.

    Log probabilities are obtained by calling the potential function, which in turn uses
    the ODE to compute the log-probability.
    """

    def __init__(
        self,
        vector_field_estimator: ConditionalVectorFieldEstimator,
        prior: Distribution,  # type: ignore
        max_sampling_batch_size: int = 10_000,
        device: Optional[Union[str, torch.device]] = None,
        enable_transform: bool = True,
        sample_with: str = "sde",
        **kwargs,
    ):
        """
        Args:
            prior: Prior distribution with `.log_prob()` and `.sample()`.
            vector_field_estimator: The trained vector field estimator.
            max_sampling_batch_size: Batchsize of samples being drawn from
                the proposal at every iteration.
            device: Training device, e.g., "cpu", "cuda" or "cuda:0". If None,
                `potential_fn.device` is used.
            enable_transform: Whether to transform parameters to unconstrained space
                during MAP optimization. When False, an identity transform will be
                returned for `theta_transform`. True is not supported yet.
            sample_with: Whether to sample from the posterior using the ODE-based
                sampler or the SDE-based sampler.
            **kwargs: Additional keyword arguments passed to
                `VectorFieldBasedPotential`.
        """

        check_prior(prior)
        potential_fn, theta_transform = vector_field_estimator_based_potential(
            vector_field_estimator,
            prior,
            x_o=None,
            enable_transform=enable_transform,
            **kwargs,
        )
        super().__init__(
            potential_fn=potential_fn,
            theta_transform=theta_transform,
            device=device,
        )
        # Set the potential function type.
        self.potential_fn: VectorFieldBasedPotential = potential_fn

        self.prior = prior
        self.enable_transform = enable_transform
        self.vector_field_estimator = vector_field_estimator
        self.device = device

        self.sample_with = sample_with
        assert self.sample_with in [
            "ode",
            "sde",
        ], f"sample_with must be 'ode' or 'sde', but is {self.sample_with}."
        self.max_sampling_batch_size = max_sampling_batch_size

        self._purpose = """It samples from the vector field model given the \
            vector_field_estimator."""

    def to(self, device: Union[str, torch.device]) -> None:
        """Move posterior to device.

        Args:
            device: device where to move the posterior to.
        """
        self.device = device
        if hasattr(self.prior, "to"):
            self.prior.to(device)  # type: ignore
        else:
            raise ValueError("""Prior has no attribute to(device).""")
        if hasattr(self.vector_field_estimator, "to"):
            self.vector_field_estimator.to(device)
        else:
            raise ValueError("""Posterior estimator has no attribute to(device).""")

        potential_fn, theta_transform = vector_field_estimator_based_potential(
            self.vector_field_estimator,
            self.prior,
            x_o=None,
            enable_transform=self.enable_transform,
        )
        x_o = None
        if hasattr(self, "_x") and (self._x is not None):
            x_o = self._x.to(device)
        super().__init__(
            potential_fn=potential_fn,
            theta_transform=theta_transform,
            device=device,
        )
        # super().__init__ erases the self._x, so we need to set it again
        if x_o is not None:
            self.set_default_x(x_o)

        self.potential_fn: VectorFieldBasedPotential = potential_fn

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
        iid_method: Literal["fnpe", "gauss", "auto_gauss", "jac_gauss"] = "auto_gauss",
        iid_params: Optional[Dict] = None,
        max_sampling_batch_size: int = 10_000,
        sample_with: Optional[str] = None,
        show_progress_bars: bool = True,
    ) -> Tensor:
        r"""Return samples from posterior distribution $p(\theta|x)$.

        Args:
            sample_shape: Shape of the samples to be drawn.
            predictor: The predictor for the vector field sampler. Can be a string or
                a custom predictor following the API in `sbi.samplers.score.predictors`.
                Currently, only `euler_maruyama` is implemented.
            corrector: The corrector for the vector field sampler. Either of
                [None].
            predictor_params: Additional parameters passed to predictor.
            corrector_params: Additional parameters passed to corrector.
            steps: Number of steps to take for the Euler-Maruyama method.
                If `sample_with` is "ode", this is ignored.
            ts: Time points at which to evaluate the vector field process. If None, a
                linear grid between t_max and t_min is used. If `sample_with` is "ode",
                this is ignored.
            iid_method: Which method to use for computing the score in the iid setting.
                We currently support "fnpe", "gauss", "auto_gauss", "jac_gauss". The
                fnpe method is simple and generally applicable. However, it can become
                inaccurate already for quite a few iid samples (as it based on heuristic
                approximations), and should be used at best only with a `corrector`. The
                "gauss" methods are more accurate, by aiming for an efficient
                approximation of the correct marginal score in the iid case. This
                however requires estimating some hyperparamters, which is done in a
                systematic way in the "auto_gauss" (initial overhead) and "jac_gauss"
                (iterative jacobian computations are expensive). We default to
                "auto_gauss" for these reasons. Note that in order to use the iid
                method, the vector field estimator must support it and have
                SCORE_DEFINED and MARGINALS_DEFINED class attributes set to True.
            iid_params: Additional parameters passed to the iid method. See the specific
                `IIDScoreFunction` child class for details.
            max_sampling_batch_size: Maximum batch size for sampling.
            sample_with: Sampling method to use - 'ode' or 'sde'. Note that in order to
                use the 'sde' sampling method, the vector field estimator must support
                it and have the SCORE_DEFINED class attribute set to True.
            show_progress_bars: Whether to show a progress bar during sampling.
        """

        if sample_with is None:
            sample_with = self.sample_with

        x = self._x_else_default_x(x)
        x = reshape_to_batch_event(x, self.vector_field_estimator.condition_shape)
        is_iid = x.shape[0] > 1
        self.potential_fn.set_x(
            x, x_is_iid=is_iid, iid_method=iid_method, iid_params=iid_params
        )

        num_samples = torch.Size(sample_shape).numel()

        if sample_with == "ode":
            samples = rejection.accept_reject_sample(
                proposal=self.sample_via_ode,
                accept_reject_fn=lambda theta: within_support(self.prior, theta),
                num_samples=num_samples,
                show_progress_bars=show_progress_bars,
                max_sampling_batch_size=max_sampling_batch_size,
            )[0]
        elif sample_with == "sde":
            proposal_sampling_kwargs = {
                "predictor": predictor,
                "corrector": corrector,
                "predictor_params": predictor_params,
                "corrector_params": corrector_params,
                "steps": steps,
                "ts": ts,
                "max_sampling_batch_size": max_sampling_batch_size,
                "show_progress_bars": show_progress_bars,
            }
            samples = rejection.accept_reject_sample(
                proposal=self._sample_via_diffusion,
                accept_reject_fn=lambda theta: within_support(self.prior, theta),
                num_samples=num_samples,
                show_progress_bars=show_progress_bars,
                max_sampling_batch_size=max_sampling_batch_size,
                proposal_sampling_kwargs=proposal_sampling_kwargs,
            )[0]
        else:
            raise ValueError(
                f"Expected sample_with to be 'ode' or 'sde', but got {sample_with}."
            )

        samples = samples.reshape(
            sample_shape + self.vector_field_estimator.input_shape
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

        NOTE: this method can be unsupported for some vector field estimators, e.g.,
        if the vector field estimator was trained with a custom flow matching routine
        for which the corresponding score is not defined.

        Args:
            sample_shape: Shape of the samples to be drawn.
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

        if not self.vector_field_estimator.SCORE_DEFINED:
            raise ValueError(
                "The vector field estimator does not support the 'sde' sampling method."
            )

        num_samples = torch.Size(sample_shape).numel()

        max_sampling_batch_size = (
            self.max_sampling_batch_size
            if max_sampling_batch_size is None
            else max_sampling_batch_size
        )

        # TODO: the time schedule should be provided by the estimator, see issue #1437
        if ts is None:
            t_max = self.vector_field_estimator.t_max
            t_min = self.vector_field_estimator.t_min
            ts = torch.linspace(t_max, t_min, steps)
        ts = ts.to(self.device)

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

        return samples

    def sample_via_ode(
        self,
        sample_shape: Shape = torch.Size(),
        **kwargs,
    ) -> Tensor:
        r"""
        Return samples from posterior distribution with probability flow ODE.

        This builds the probability flow ODE and then samples from the corresponding
        flow.

        Args:
            sample_shape: The shape of the samples to be returned.
            **kwargs: Additional keyword arguments for the ODE solver that
                depend on the used ODE backend.

        Returns:
            Samples from the approximated posterior distribution
                :math:`\theta \sim p(\theta|x)`.
        """
        num_samples = torch.Size(sample_shape).numel()

        samples = self.potential_fn.neural_ode(self.potential_fn.x_o, **kwargs).sample(
            torch.Size((num_samples,))
        )

        return samples

    def log_prob(
        self,
        theta: Tensor,
        x: Optional[Tensor] = None,
        track_gradients: bool = False,
        ode_kwargs: Optional[Dict] = None,
    ) -> Tensor:
        r"""Returns the log-probability of the posterior $p(\theta|x)$.

        This requires building and evaluating the probability flow ODE.

        Args:
            theta: Parameters $\theta$.
            x: Observed data $x_o$. If None, the default $x_o$ is used.
            track_gradients: Whether the returned tensor supports tracking gradients.
                This can be helpful for e.g. sensitivity analysis, but increases memory
                consumption.
            ode_kwargs: Additional keyword arguments for the ODE solver.

        Returns:
            `(len(θ),)`-shaped log posterior probability $\log p(\theta|x)$ for θ in the
            support of the prior, -∞ (corresponding to 0 probability) outside.
        """
        self.potential_fn.set_x(self._x_else_default_x(x), **(ode_kwargs or {}))

        theta = ensure_theta_batched(torch.as_tensor(theta))
        return self.potential_fn(
            theta.to(self._device),
            track_gradients=track_gradients,
        )

    def sample_batched(
        self,
        sample_shape: torch.Size,
        x: Tensor,
        predictor: Union[str, Predictor] = "euler_maruyama",
        corrector: Optional[Union[str, Corrector]] = None,
        predictor_params: Optional[Dict] = None,
        corrector_params: Optional[Dict] = None,
        steps: int = 500,
        ts: Optional[Tensor] = None,
        max_sampling_batch_size: int = 10000,
        show_progress_bars: bool = True,
    ) -> Tensor:
        r"""Given a batch of observations [x_1, ..., x_B] this function samples from
        posteriors $p(\theta|x_1)$, ... ,$p(\theta|x_B)$, in a batched (i.e. vectorized)
        manner.

        Args:
            sample_shape: Desired shape of samples that are drawn from the posterior
                given every observation.
            x: A batch of observations, of shape `(batch_dim, event_shape_x)`.
                `batch_dim` corresponds to the number of observations to be
                drawn.
            predictor: The predictor for the diffusion-based sampler. Can be a string or
                a custom predictor following the API in `sbi.samplers.score.predictors`.
                Currently, only `euler_maruyama` is implemented.
            corrector: The corrector for the diffusion-based sampler.
            predictor_params: Additional parameters passed to predictor.
            corrector_params: Additional parameters passed to corrector.
            steps: Number of steps to take for the Euler-Maruyama method.
            ts: Time points at which to evaluate the diffusion process. If None, a
                linear grid between t_max and t_min is used.
            max_sampling_batch_size: Maximum batch size for sampling.
            show_progress_bars: Whether to show sampling progress monitor.

        Returns:
            Samples from the posteriors of shape (*sample_shape, B, *input_shape)
        """
        num_samples = torch.Size(sample_shape).numel()
        x = reshape_to_batch_event(x, self.vector_field_estimator.condition_shape)
        condition_dim = len(self.vector_field_estimator.condition_shape)
        batch_shape = x.shape[:-condition_dim]
        batch_size = batch_shape.numel()
        self.potential_fn.set_x(x)

        max_sampling_batch_size = (
            self.max_sampling_batch_size
            if max_sampling_batch_size is None
            else max_sampling_batch_size
        )

        if self.sample_with == "ode":
            samples = rejection.accept_reject_sample(
                proposal=self.sample_via_ode,
                accept_reject_fn=lambda theta: within_support(self.prior, theta),
                num_samples=num_samples,
                num_xos=batch_size,
                show_progress_bars=show_progress_bars,
                max_sampling_batch_size=max_sampling_batch_size,
            )[0]
            samples = samples.reshape(
                sample_shape + batch_shape + self.vector_field_estimator.input_shape
            )
        elif self.sample_with == "sde":
            proposal_sampling_kwargs = {
                "predictor": predictor,
                "corrector": corrector,
                "predictor_params": predictor_params,
                "corrector_params": corrector_params,
                "steps": steps,
                "ts": ts,
                "max_sampling_batch_size": max_sampling_batch_size,
                "show_progress_bars": show_progress_bars,
            }
            samples = rejection.accept_reject_sample(
                proposal=self._sample_via_diffusion,
                accept_reject_fn=lambda theta: within_support(self.prior, theta),
                num_samples=num_samples,
                num_xos=batch_size,
                show_progress_bars=show_progress_bars,
                max_sampling_batch_size=max_sampling_batch_size,
                proposal_sampling_kwargs=proposal_sampling_kwargs,
            )[0]
            samples = samples.reshape(
                sample_shape + batch_shape + self.vector_field_estimator.input_shape
            )

        return samples

    def map(
        self,
        x: Optional[Tensor] = None,
        num_iter: int = 1000,
        num_to_optimize: int = 1000,
        learning_rate: float = 0.01,
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
        if x is not None:
            raise ValueError(
                "Passing `x` directly to `.map()` has been deprecated."
                "Use `.self_default_x()` to set `x`, and then run `.map()` "
            )

        if self.default_x is None:
            raise ValueError(
                "Default `x` has not been set."
                "To set the default, use the `.set_default_x()` method."
            )

        if self._map is None or force_update:
            # rebuild coarse flow fast for MAP optimization.
            self.potential_fn.set_x(self.default_x, atol=1e-2, rtol=1e-3, exact=True)
            callable_potential_fn = CallableDifferentiablePotentialFunction(
                self.potential_fn
            )
            if init_method == "posterior":
                inits = self.sample((num_init_samples,))
            elif init_method == "proposal":
                inits = self.proposal.sample((num_init_samples,))  # type: ignore
            elif isinstance(init_method, Tensor):
                inits = init_method
            else:
                raise ValueError

            self._map = gradient_ascent(
                potential_fn=callable_potential_fn,
                inits=inits,
                theta_transform=self.theta_transform,
                num_iter=num_iter,
                num_to_optimize=num_to_optimize,
                learning_rate=learning_rate,
                save_best_every=save_best_every,
                show_progress_bars=show_progress_bars,
            )[0]

        return self._map
