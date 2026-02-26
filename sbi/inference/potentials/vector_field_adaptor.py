# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

import functools
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, Optional, Type, TypeVar, Union

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.distributions import Distribution

from sbi.neural_nets.estimators import ConditionalVectorFieldEstimator
from sbi.utils.torchutils import ensure_theta_batched
from sbi.utils.vector_field_utils import (
    add_diag_or_dense,
    denoise,
    fit_gmm_ratio,
    marginalize,
    mv_diag_or_dense,
    solve_diag_or_dense,
)

IID_METHODS: dict[str, Type["IIDScoreFunction"]] = {}
GUIDANCE_METHODS: dict[str, tuple[Callable[..., "ScoreAdaptation"], Type[object]]] = {}

ScoreAdaptationT = TypeVar("ScoreAdaptationT", bound="ScoreAdaptation")
IIDScoreFunctionT = TypeVar("IIDScoreFunctionT", bound="IIDScoreFunction")


def get_iid_method(name: str) -> Type["IIDScoreFunction"]:
    r"""
    Retrieves the IID method by name.

    Args:
        name: The name of the IID method.

    Returns:
        The IID method class.
    """
    if name not in IID_METHODS:
        raise NotImplementedError(
            f"Method {name} for iid score accumulation not implemented."
        )
    return IID_METHODS[name]


def get_guidance_method(
    name: str,
) -> tuple[Callable[..., "ScoreAdaptation"], Type[object]]:
    r"""
    Retrieves the guidance method by name.

    Args:
        name: The name of the guidance method.

    Returns:
        A tuple of the guidance method class and its default configuration class.
    """
    if name not in GUIDANCE_METHODS:
        raise NotImplementedError(f"Method {name} for guidance not implemented.")
    return GUIDANCE_METHODS[name]


def register_guidance_method(
    name: str, default_config: Type[object]
) -> Callable[[Type[ScoreAdaptationT]], Type[ScoreAdaptationT]]:
    r"""
    Registers a guidance method and its default configuration.

    Args:
        name: The name of the guidance method.
        default_config: The default configuration class for the guidance method.

    Returns:
        A decorator function to register the guidance method class.
    """

    def decorator(cls: Type[ScoreAdaptationT]) -> Type[ScoreAdaptationT]:
        GUIDANCE_METHODS[name] = (cls, default_config)
        return cls

    return decorator


def register_iid_method(
    name: str,
) -> Callable[[Type[IIDScoreFunctionT]], Type[IIDScoreFunctionT]]:
    r"""
    Registers an IID method.

    Args:
        name: The name of the IID method.

    Returns:
        A decorator function to register the IID method class.
    """

    def decorator(cls: Type[IIDScoreFunctionT]) -> Type[IIDScoreFunctionT]:
        IID_METHODS[name] = cls
        return cls

    return decorator


class ScoreAdaptation(ABC):
    """Abstract base class for manipulating score estimators to impose additional
    constraints on the posterior via guidance.
    """

    def __init__(
        self,
        vf_estimator: ConditionalVectorFieldEstimator,
        prior: Optional[Distribution],
        device: Union[str, torch.device] = "cpu",
    ) -> None:
        """Initializes the score adaptation with an estimator, prior, and device.

        Args:
            vf_estimator: The vector field estimator.
            prior: The prior distribution.
            device: The device on which to evaluate the potential.
        """
        self.vf_estimator = vf_estimator
        self.prior = prior
        self.device = device

    @abstractmethod
    def __call__(
        self, input: Tensor, condition: Tensor, time: Optional[Tensor] = None
    ) -> Tensor:
        pass

    def score(
        self, input: Tensor, condition: Tensor, t: Optional[Tensor] = None
    ) -> Tensor:
        return self.__call__(input, condition, t)

    @property
    def SCORE_DEFINED(self) -> bool:
        return self.vf_estimator.SCORE_DEFINED

    @property
    def MARGINALS_DEFINED(self) -> bool:
        return self.vf_estimator.MARGINALS_DEFINED

    def to(self, device: Union[str, torch.device]) -> "ScoreAdaptation":
        self.device = device
        self.vf_estimator.to(device)
        if self.prior is not None and hasattr(self.prior, "to"):
            self.prior.to(device)  # type: ignore
        return self

    def eval(self) -> "ScoreAdaptation":
        self.vf_estimator.eval()
        return self

    def train(self, mode: bool = True) -> "ScoreAdaptation":
        self.vf_estimator.train(mode)
        return self

    def __getattr__(self, name: str):
        return getattr(self.vf_estimator, name)


@dataclass
class AffineClassifierFreeGuidanceConfig:
    """Configuration for :class:`AffineClassifierFreeGuidance`.

    Attributes:
        prior_scale: Multiplicative factor applied to the prior score.
        prior_shift: Additive shift applied to the prior score.
        likelihood_scale: Multiplicative factor applied to the likelihood score.
        likelihood_shift: Additive shift applied to the likelihood score.
    """

    prior_scale: Union[float, Tensor] = 1.0
    prior_shift: Union[float, Tensor] = 0.0
    likelihood_scale: Union[float, Tensor] = 1.0
    likelihood_shift: Union[float, Tensor] = 0.0


@register_guidance_method("affine_classifier_free", AffineClassifierFreeGuidanceConfig)
class AffineClassifierFreeGuidance(ScoreAdaptation):
    r"""Classifier-free guidance via affine transformation of prior and likelihood
    scores.

    Decomposes the posterior score into a prior and likelihood component, then applies
    independent scale and shift to each. This allows tempering the posterior, e.g.,
    sharpening or flattening it.

    Example:
    --------

    ::

        from sbi.inference.potentials.vector_field_adaptor import (
            AffineClassifierFreeGuidance,
            AffineClassifierFreeGuidanceConfig,
        )

        config = AffineClassifierFreeGuidanceConfig(likelihood_scale=2.0)
        guidance = AffineClassifierFreeGuidance(
            vf_estimator, prior, config, device="cpu"
        )
        adjusted_score = guidance(input, condition, time)

    References:
    - [1] Classifier-Free Diffusion Guidance (2022)
    - [2] All-in-one simulation-based inference (2024)
    """

    def __init__(
        self,
        vf_estimator: ConditionalVectorFieldEstimator,
        prior: Optional[Distribution],
        config: AffineClassifierFreeGuidanceConfig,
        device: Union[str, torch.device] = "cpu",
    ) -> None:
        """Initializes affine classifier-free guidance.

        Args:
            vf_estimator: The vector field estimator.
            prior: The prior distribution (required).
            config: Scale and shift parameters for the prior and likelihood.
            device: The device on which to evaluate the potential.
        """

        if prior is None:
            raise ValueError(
                "Prior is required for classifier-free guidance, please"
                " provide at least an improper empirical prior."
            )

        self.prior_scale = torch.tensor(config.prior_scale, device=device)
        self.prior_shift = torch.tensor(config.prior_shift, device=device)
        self.likelihood_scale = torch.tensor(config.likelihood_scale, device=device)
        self.likelihood_shift = torch.tensor(config.likelihood_shift, device=device)
        super().__init__(vf_estimator, prior, device)

    def marginal_prior_score(self, theta: Tensor, time: Tensor) -> Tensor:
        """Computes the marginal prior score analytically (or approximately)."""
        m = self.vf_estimator.mean_t_fn(time)
        std = self.vf_estimator.std_fn(time)
        marginal_prior = marginalize(self.prior, m, std)  # type: ignore
        marginal_prior_score = compute_score(marginal_prior, theta)
        return marginal_prior_score

    def __call__(
        self, input: Tensor, condition: Tensor, time: Optional[Tensor] = None
    ) -> Tensor:
        if time is None:
            time = torch.tensor([self.vf_estimator.t_min])

        posterior_score = self.vf_estimator(input=input, condition=condition, time=time)
        prior_score = self.marginal_prior_score(input, time)
        ll_score = posterior_score - prior_score
        ll_score_mod = ll_score * self.likelihood_scale + self.likelihood_shift
        prior_score_mod = prior_score * self.prior_scale + self.prior_shift

        return ll_score_mod + prior_score_mod


@dataclass
class UniversalGuidanceConfig:
    """Configuration for :class:`UniversalGuidance`.

    Attributes:
        guidance_fn: Differentiable scalar guidance function.
        guidance_fn_score: Optional analytic score of ``guidance_fn``.
    """

    guidance_fn: Callable[[Tensor, Tensor, Tensor, Tensor], Tensor]
    guidance_fn_score: Optional[Callable[[Tensor, Tensor, Tensor, Tensor], Tensor]] = (
        None
    )


@register_guidance_method("universal", UniversalGuidanceConfig)
class UniversalGuidance(ScoreAdaptation):
    r"""Score guidance using a user-supplied differentiable guidance function.

    Adds a guidance term to the score by applying Tweedie's formula to denoise the
    input and then computing the gradient of a custom guidance function. If no
    analytic gradient is provided, it is obtained via ``torch.autograd``.

    Example:
    --------

    ::

        from sbi.inference.potentials.vector_field_adaptor import (
            UniversalGuidance,
            UniversalGuidanceConfig,
        )

        def my_guidance(input, condition, m, std):
            return -((input - condition) ** 2).sum(-1, keepdim=True)

        config = UniversalGuidanceConfig(guidance_fn=my_guidance)
        guidance = UniversalGuidance(vf_estimator, prior, config)
        adjusted_score = guidance(input, condition, time)

    References:
    - [1] Universal Guidance for Diffusion Models (2022)
    """

    def __init__(
        self,
        vf_estimator: ConditionalVectorFieldEstimator,
        prior: Optional[Distribution],
        config: UniversalGuidanceConfig,
        device: Union[str, torch.device] = "cpu",
    ) -> None:
        """Initializes universal guidance with a custom guidance function.

        Args:
            vf_estimator: The vector field estimator.
            prior: The prior distribution.
            config: Configuration containing the guidance function.
            device: The device on which to evaluate the potential.
        """
        self.guidance_fn = config.guidance_fn

        if config.guidance_fn_score is None:

            def guidance_fn_score(
                input: Tensor, condition: Tensor, m: Tensor, std: Tensor
            ) -> Tensor:
                with torch.enable_grad():
                    input = input.detach().clone().requires_grad_(True)
                    score = torch.autograd.grad(
                        config.guidance_fn(input, condition, m, std).sum(),
                        input,
                        create_graph=True,
                    )[0]
                return score

            self.guidance_fn_score = guidance_fn_score
        else:
            self.guidance_fn_score = config.guidance_fn_score

        super().__init__(vf_estimator, prior, device)

    def __call__(
        self, input: Tensor, condition: Tensor, time: Optional[Tensor] = None
    ) -> Tensor:
        if time is None:
            time = torch.tensor([self.vf_estimator.t_min])
        score = self.vf_estimator(input, condition, time)
        m = self.vf_estimator.mean_t_fn(time)
        std = self.vf_estimator.std_fn(time)

        # Tweedie's formula for denoising
        denoised_input = (input + std**2 * score) / m
        guidance_score = self.guidance_fn_score(denoised_input, condition, m, std)

        return score + guidance_score


@dataclass
class IntervalGuidanceConfig:
    """Configuration for :class:`IntervalGuidance`.

    Attributes:
        lower_bound: Optional lower bound for parameters.
        upper_bound: Optional upper bound for parameters.
        mask: Optional boolean mask selecting constrained dimensions.
        scale_factor: Soft-constraint strength.
    """

    lower_bound: Optional[Union[float, Tensor]]
    upper_bound: Optional[Union[float, Tensor]]
    mask: Optional[Tensor] = None
    scale_factor: float = 0.5

    def __post_init__(self) -> None:
        if self.lower_bound is None and self.upper_bound is None:
            raise ValueError("At least one of lower_bound or upper_bound is required.")
        if (
            self.lower_bound is not None
            and self.upper_bound is not None
            and self.lower_bound >= self.upper_bound
        ):
            raise ValueError(
                f"lower_bound ({self.lower_bound}) must be less than "
                f"upper_bound ({self.upper_bound})."
            )


@register_guidance_method("interval", IntervalGuidanceConfig)
class IntervalGuidance(UniversalGuidance):
    r"""Guidance that constrains parameters to lie within specified bounds.

    Uses soft log-sigmoid penalties to steer samples toward an interval
    ``[lower_bound, upper_bound]``. Built on top of
    :class:`UniversalGuidance`.

    Example:
    --------

    ::

        from sbi.inference.potentials.vector_field_adaptor import (
            IntervalGuidance,
            IntervalGuidanceConfig,
        )

        config = IntervalGuidanceConfig(lower_bound=0.0, upper_bound=1.0)
        guidance = IntervalGuidance(vf_estimator, prior, config)
        adjusted_score = guidance(input, condition, time)

    References:
    - [1] All-in-one simulation-based inference (2024)
    """

    def __init__(
        self,
        vf_estimator: ConditionalVectorFieldEstimator,
        prior: Optional[Distribution],
        config: IntervalGuidanceConfig,
        device: Union[str, torch.device] = "cpu",
    ) -> None:
        """Initializes interval guidance with bound constraints.

        Args:
            vf_estimator: The vector field estimator.
            prior: The prior distribution.
            config: Configuration specifying the interval bounds.
            device: The device on which to evaluate the potential.
        """

        def interval_fn(
            input: Tensor, condition: Tensor, m: Tensor, std: Tensor
        ) -> Tensor:
            del condition
            scale = config.scale_factor / (m**2 * std**2 + 1e-6)
            upper_bound = (
                F.logsigmoid(-scale * (input - config.upper_bound))
                if config.upper_bound is not None
                else torch.zeros_like(input)
            )
            lower_bound = (
                F.logsigmoid(scale * (input - config.lower_bound))
                if config.lower_bound is not None
                else torch.zeros_like(input)
            )
            out = upper_bound + lower_bound
            if config.mask is not None:
                if config.mask.shape != out.shape:
                    config.mask = config.mask.unsqueeze(0).expand_as(out)
                out = torch.where(config.mask, out, torch.zeros_like(out))
            return out

        super().__init__(
            vf_estimator,
            prior,
            UniversalGuidanceConfig(guidance_fn=interval_fn),
            device=device,
        )


@dataclass
class PriorGuideConfig:
    """Configuration for :class:`PriorGuide`.

    Attributes:
        train_prior: Prior used during training.
        test_prior: Prior used at inference time.
        K: Number of GMM components.
        num_steps: Optimization steps for ratio fitting.
        batch_size: Batch size for ratio fitting.
        lr: Learning rate for ratio fitting.
        covariance_type: Covariance parameterization (``"diag"`` or ``"full"``).
        min_cov: Minimum covariance value used during fitting.
        max_log_ratio: Maximum log-ratio clipping value.
        nugget: Numerical safety term for matrix and division stabilization.
    """

    train_prior: Distribution
    test_prior: Distribution
    K: int = 5
    num_steps: int = 10_000
    batch_size: int = 1_000
    lr: float = 1e-2
    covariance_type: str = "diag"
    min_cov: float = 1e-4
    max_log_ratio: float = 50.0
    nugget: float = 1e-6

    def __post_init__(self) -> None:
        if self.K < 1:
            raise ValueError(f"K must be at least 1, got {self.K}.")
        if self.covariance_type not in ("diag", "full"):
            raise ValueError(
                f"covariance_type must be 'diag' or 'full', "
                f"got '{self.covariance_type}'."
            )
        if self.nugget < 0:
            raise ValueError(f"nugget must be non-negative, got {self.nugget}.")


@register_guidance_method("prior_guide", PriorGuideConfig)
class PriorGuide(ScoreAdaptation):
    r"""Prior guidance via a GMM approximation of the prior ratio.

    Fits a Gaussian mixture model to the ratio between a training prior and a
    test prior, then leverages the backward kernel of the diffusion process to
    accurately compute the marginal adjusted posterior score.

    Example:
    --------

    ::

        from sbi.inference.potentials.vector_field_adaptor import (
            PriorGuide,
            PriorGuideConfig,
        )

        config = PriorGuideConfig(train_prior=train_prior, test_prior=test_prior, K=5)
        guidance = PriorGuide(vf_estimator, train_prior, config)
        adjusted_score = guidance(input, condition, time)

    References:
    - [1] "Prior Guidance for Diffusion Models" (arXiv:2510.13763)
    """

    def __init__(
        self,
        vf_estimator: ConditionalVectorFieldEstimator,
        prior: Optional[Distribution],
        config: PriorGuideConfig,
        device: Union[str, torch.device] = "cpu",
    ) -> None:
        """Initializes prior guidance and fits the GMM ratio approximation.

        Args:
            vf_estimator: The vector field estimator.
            prior: The training prior distribution (required).
            config: PriorGuide configuration, including train/test priors and K.
            device: The device on which to evaluate the potential.
        """
        if prior is None:
            raise ValueError(
                "Prior is required for PriorGuide to match the training distribution."
            )
        if not vf_estimator.MARGINALS_DEFINED:
            raise ValueError(
                "Marginals are not defined for this vector field estimator."
            )
        self.config = config
        self.weights, self.means, self.cov = fit_gmm_ratio(
            config.train_prior,
            config.test_prior,
            int(config.K),
            num_steps=config.num_steps,
            batch_size=config.batch_size,
            lr=config.lr,
            covariance_type=config.covariance_type,
            min_cov=config.min_cov,
            max_log_ratio=config.max_log_ratio,
            device=device,
        )
        super().__init__(vf_estimator, prior, device)

    def _sigma0_t2(self, std: Tensor) -> Tensor:
        # From PriorGuide Eq. (12): sigma(t)^2 / (1 + sigma(t)^2)
        # NOTE: This can be improved similar to the AutoGauss or
        # JAC methods for IIDScoreFunctions i.e. this assumes
        # identity covariance but one can also use an estimator
        # of the posterior covariance.
        return std**2 / (1.0 + std**2)

    def __call__(
        self, input: Tensor, condition: Tensor, time: Optional[Tensor] = None
    ) -> Tensor:
        if time is None:
            time = torch.tensor([self.vf_estimator.t_min])

        input_ = input.detach()
        score = self.vf_estimator(input=input_, condition=condition, time=time)
        m = self.vf_estimator.mean_t_fn(time)
        std = self.vf_estimator.std_fn(time)
        # Standard Tweedie's formula for denoising
        mu0 = (input_ + std**2 * score) / m

        # Prior adjustment
        sigma0_t2 = self._sigma0_t2(std)
        sigma0_t2 = sigma0_t2.reshape(-1)
        if sigma0_t2.numel() == 1:
            sigma0_t2 = sigma0_t2.expand(mu0.shape[-1])
        nugget = self.config.nugget
        if self.config.covariance_type == "diag":
            cov_e = self.cov + sigma0_t2 + nugget
            diff = mu0[..., None, :] - self.means
            log_det = torch.log(cov_e).sum(-1)
            quad = (diff**2 / cov_e).sum(-1)
            d = diff.shape[-1]
            log_comp = -0.5 * (d * math.log(2 * math.pi) + log_det + quad)
        else:
            nugget_eye = nugget * torch.eye(
                sigma0_t2.shape[-1], device=sigma0_t2.device
            )
            cov_e = self.cov + torch.diag_embed(sigma0_t2) + nugget_eye
            diff = mu0[..., None, :] - self.means
            chol = torch.linalg.cholesky(cov_e)
            diff_col = diff[..., None]
            solve = torch.cholesky_solve(diff_col, chol)
            quad = (diff_col * solve).sum(-2).squeeze(-1)
            log_det = 2.0 * torch.log(torch.diagonal(chol, dim1=-2, dim2=-1)).sum(-1)
            d = diff.shape[-1]
            log_comp = -0.5 * (d * math.log(2 * math.pi) + log_det + quad)

        numerator = self.weights[None, :] * torch.exp(log_comp)
        denom = numerator.sum(dim=1, keepdim=True)
        denom = torch.clamp(denom, min=nugget)
        w_tilde = numerator / denom

        diff = self.means - mu0[..., None, :]
        if self.config.covariance_type == "diag":
            v_i = diff / cov_e
        else:
            # Solve per-component for full covariance; support shared or
            # per-component cov.
            v_i = torch.linalg.solve(cov_e, diff.unsqueeze(-1)).squeeze(-1)

        v = torch.sum(w_tilde[..., None] * v_i, dim=-2)
        mu0_adjusted = mu0 + std**2 * v
        score_adjusted = (mu0_adjusted * m - input_) / (std**2 + nugget)

        return score_adjusted


class IIDScoreFunction(ABC):
    r"""Abstract base class for IID score accumulation methods.

    In the IID setting the posterior for N observations can be represented as a
    product of N "local" posteriors, divided by N-1 prior terms. Subclasses
    implement different strategies for computing the marginal posterior score at
    time :math:`t`, which does not factorize even when the true posterior does.
    """

    def __init__(
        self,
        vector_field_estimator: Union[ConditionalVectorFieldEstimator, ScoreAdaptation],
        prior: Distribution,  # type: ignore
        device: Union[str, torch.device] = "cpu",
    ) -> None:
        """Initializes the IID score function with validation checks.

        Args:
            vector_field_estimator: The neural network modeling the score.
            prior: The prior distribution.
            device: The device on which to evaluate the potential. Defaults to "cpu".
        """

        if not vector_field_estimator.SCORE_DEFINED:
            raise ValueError(
                "Score is not defined for this vector field estimator. "
                "You are probably using a vector field estimator that does not "
                "implement the score function, e.g. an optimal transport-based "
                "flow matching. IID methods require the score function to be defined "
                "so they are not applicable to this vector field estimator. "
                "If you have implemented a custom vector field "
                "estimator with score defined, set SCORE_DEFINED to True."
            )
        if not vector_field_estimator.MARGINALS_DEFINED:
            raise ValueError(
                "Marginals are not defined for this vector field estimator. "
                "You are probably using a vector field estimator that does not "
                "implement the marginals mean_t_fn and std_fn, e.g. an "
                "optimal transport-based flow matching. IID methods require the "
                "marginals to be defined so they are not applicable to this "
                "vector field estimator. "
                "If you have implemented a custom vector field "
                "estimator with marginals defined, set MARGINALS_DEFINED to True."
            )

        self.vector_field_estimator = vector_field_estimator.to(device).eval()
        self.prior = prior
        self.device = device

    def to(self, device: Union[str, torch.device]) -> None:
        """
        Moves score_estimator and prior to the given device.

        It also sets the device attribute to the given device.

        Args:
            device: Device to move the score_estimator and prior to.
        """
        self.device = device
        self.vector_field_estimator.to(device)
        if self.prior:
            self.prior.to(device)  # type: ignore

    @abstractmethod
    def __call__(
        self,
        inputs: Tensor,
        conditions: Tensor,
        time: Optional[Tensor] = None,
    ) -> Tensor:
        r"""
        Abstract method to be implemented by subclasses to compute the score function.

        Args:
            inputs: The parameters at which to evaluate the potential of size [b,iid,d]
            conditions: The sequence of observations of size [iid,...]
            time: The time in the diffusion process to specify the target marginal.

        Returns:
            The score of the inputs given N observations of shape (batch,d).
        """
        pass


@register_iid_method("fnpe")
class FactorizedNPEScoreFunction(IIDScoreFunction):
    r"""Factorized Neural Posterior Estimation for score-based models.

    Accumulates per-observation scores with a weighted prior subtraction. Does not
    apply corrections for the marginal score at :math:`t > 0`, so post-hoc adjustment
    (e.g., predictor-corrector samplers) may be needed for many observations.

    Example:
    --------

    ::

        from sbi.inference.potentials.vector_field_adaptor import (
            FactorizedNPEScoreFunction,
        )

        iid_score = FactorizedNPEScoreFunction(vf_estimator, prior)
        score = iid_score(inputs, conditions, time)  # inputs: [b, iid, d]

    References:
    - [1] Compositional Score Modeling for Simulation-Based Inference
        (https://arxiv.org/abs/2209.14249)
    """

    def __init__(
        self,
        vector_field_estimator: Union[ConditionalVectorFieldEstimator, ScoreAdaptation],
        prior: Distribution,
        device: Union[str, torch.device] = "cpu",
        prior_score_weight: Optional[Callable[[Tensor], Tensor]] = None,
    ) -> None:
        """Initializes factorized NPE score accumulation.

        Args:
            vector_field_estimator: The neural network modeling the score.
            prior: The prior distribution.
            device: The device on which to evaluate the potential. Defaults to "cpu".
            prior_score_weight: A function to weight the prior score. Defaults to the
                linear interpolation between zero (at t=0) and one (at t=t_max).
        """
        super().__init__(vector_field_estimator, prior, device)
        if prior_score_weight is None:
            t_max = vector_field_estimator.t_max

            def prior_score_weight_fn(t: Tensor) -> Tensor:
                return (t_max - t) / t_max

        self.prior_score_weight_fn = prior_score_weight_fn

    def __call__(
        self,
        inputs: Tensor,
        conditions: Tensor,
        time: Optional[Tensor] = None,
    ) -> Tensor:
        r"""
        Computes the score function for score-based methods on multiple observations.

        Args:
            inputs: The input parameters at which to evaluate the potential.
            conditions: The observed data at which to evaluate the posterior.
            time: The time at which to evaluate the score. Defaults to None.

        Returns:
            The computed score function.
        """

        assert inputs.ndim == 3, "Inputs must have shape [b,iid,d]"
        assert conditions.ndim == 2, "Conditions must have shape [iid,...]"

        if time is None:
            time = torch.tensor([self.vector_field_estimator.t_min])

        N = conditions.shape[0]

        # Compute the per-sample score
        inputs = ensure_theta_batched(inputs)
        base_score = self.vector_field_estimator.score(inputs, conditions, time)

        # Compute the prior score

        prior_score = self.prior_score_weight_fn(time) * compute_score(
            self.prior, inputs
        )

        # Accumulate
        score = (1 - N) * prior_score + base_score.sum(-2, keepdim=True)

        return score


class BaseGaussCorrectedScoreFunction(IIDScoreFunction):
    r"""Base class for Gauss-corrected IID score accumulation.

    Derives an analytic correction for the marginal scores under Gaussian
    assumptions. Subclasses provide different strategies for estimating the
    posterior precision (see :class:`GaussCorrectedScoreFn`,
    :class:`AutoGaussCorrectedScoreFn`, :class:`JacCorrectedScoreFn`).

    References:
    - [1] Diffusion posterior sampling for simulation-based inference in tall data
        settings (https://arxiv.org/abs/2404.07593)
    - [2] Compositional simulation-based inference for time series
        (https://arxiv.org/abs/2411.02728)
    """

    def __init__(
        self,
        vector_field_estimator: Union[ConditionalVectorFieldEstimator, ScoreAdaptation],
        prior: Distribution,
        ensure_lam_psd: bool = True,
        lam_psd_nugget: float = 0.01,
        device: Union[str, torch.device] = "cpu",
    ) -> None:
        """Initializes Gauss-corrected score accumulation.

        Args:
            vector_field_estimator: The neural network modelling the score.
            prior: The prior distribution.
            ensure_lam_psd: Whether to ensure the precision matrix is positive definite.
            lam_psd_nugget: The nugget value to ensure positive definiteness.
            device: The device on which to evaluate the potential. Defaults to "cpu".
        """
        super().__init__(vector_field_estimator, prior, device)
        self.ensure_lam_psd = ensure_lam_psd
        self.lam_psd_nugget = lam_psd_nugget

    @abstractmethod
    def posterior_precision_est_fn(self, conditions: Tensor) -> Tensor:
        r"""Abstract method for estimating the posterior precision.

        This can be seen as an important hyperparameter which can be estimated in
        different ways leading to different methods (see child classes).

        Args:
            conditions: Observed data.

        Returns:
            Estimated posterior precision.
        """
        pass

    def marginal_denoising_posterior_precision_est_fn(
        self,
        time: Tensor,
        inputs: Optional[Tensor],
        conditions: Tensor,
    ) -> Tensor:
        r"""Estimates the marginal posterior precision.

        Args:
            time: Time tensor.
            inputs: Parameters tensor.
            conditions: Observed data.

        Returns:
            Estimated marginal posterior precision.
        """
        del inputs
        precisions_posteriors = self.posterior_precision_est_fn(conditions)

        # Denoising posterior via Bayes rule
        m = self.vector_field_estimator.mean_t_fn(time)
        std = self.vector_field_estimator.std_fn(time)

        if precisions_posteriors.ndim == 4:
            Ident = torch.eye(precisions_posteriors.shape[-1], device=self.device)
        else:
            Ident = torch.ones_like(precisions_posteriors, device=self.device)

        marginal_precisions = m**2 / std**2 * Ident + precisions_posteriors
        return marginal_precisions

    def marginal_prior_score_fn(self, time: Tensor, inputs: Tensor) -> Tensor:
        r"""Computes the score of the marginal prior distribution.

        Args:
            time: Time tensor.
            inputs: Parameters tensor.

        Returns:
            Marginal prior score.
        """
        # NOTE: This is for the uniform distribution and distributions that do not
        # implement a log_prob.
        try:
            with torch.enable_grad():
                inputs = inputs.clone().detach().requires_grad_(True)
                m = self.vector_field_estimator.mean_t_fn(time)
                std = self.vector_field_estimator.std_fn(time)
                p = marginalize(self.prior, m, std)
                log_p = p.log_prob(inputs)
                prior_score = torch.autograd.grad(
                    log_p,
                    inputs,
                    grad_outputs=torch.ones_like(log_p),
                    create_graph=True,
                )[0].detach()
        except Exception:
            prior_score = torch.zeros_like(inputs)

        return prior_score

    def marginal_denoising_prior_precision_fn(
        self, time: Tensor, inputs: Tensor
    ) -> Tensor:
        r"""Computes the precision of the marginal denoising prior.

        Args:
            time: Time tensor.
            inputs: Parameters tensor.

        Returns:
            Marginal denoising prior precision.
        """
        m = self.vector_field_estimator.mean_t_fn(time)
        std = self.vector_field_estimator.std_fn(time)

        p_denoise = denoise(self.prior, m, std, inputs)

        if hasattr(p_denoise, "covariance_matrix"):
            inv_cov = torch.inverse(p_denoise.covariance_matrix)  # type: ignore
            return inv_cov.reshape(inputs.shape + inputs.shape[-1:])
        else:
            try:
                precision = 1 / p_denoise.variance
                return precision.reshape(inputs.shape)
            except Exception as e:
                msg = """This iid_method tries to denoise the prior distribution
                analytically. For custom prior distributions (i.e. which do not
                implement the variance/covariance_matrix method) but inherit from
                standard prior distributions i.e. Normal or Uniform, this might lead to
                errors. If you encounter this error, please raise an issue on the sbi
                repository.
                """
                raise NotImplementedError(msg) from e

    def __call__(
        self,
        inputs: Tensor,
        conditions: Tensor,
        time: Optional[Tensor] = None,
        **kwargs,
    ) -> Tensor:
        r"""
        Returns the corrected score function.

        Args:
            inputs: Parameter tensor.
            conditions: Observed data.
            time: Time tensor.

        Returns:
            Corrected score function.
        """
        assert inputs.ndim == 3, "Inputs must have shape [b,iid,d]"
        assert conditions.ndim == 2, "Conditions must have shape [iid,...]"

        N, *_ = conditions.shape

        if time is None:
            time = torch.tensor([self.vector_field_estimator.t_min])

        base_score = self.vector_field_estimator.score(
            inputs, conditions, time, **kwargs
        )
        prior_score = self.marginal_prior_score_fn(time, inputs)

        # Marginal prior precision
        prior_precision = self.marginal_denoising_prior_precision_fn(time, inputs)
        # Marginal posterior variance estimates
        posterior_precisions = self.marginal_denoising_posterior_precision_est_fn(
            time,
            inputs,
            conditions,
        )

        if self.ensure_lam_psd:
            prior_precision, posterior_precisions = ensure_lam_positive_definite(
                prior_precision,
                posterior_precisions,
                N,
                precision_nugget=self.lam_psd_nugget,
            )

        # Total precision
        term1 = (1 - N) * prior_precision
        term2 = torch.sum(posterior_precisions, dim=1, keepdim=True)
        Lam = add_diag_or_dense(term1, term2, batch_dims=2)

        # Weighted scores
        weighted_prior_score = mv_diag_or_dense(
            prior_precision, prior_score, batch_dims=2
        )
        weighted_posterior_scores = mv_diag_or_dense(
            posterior_precisions, base_score, batch_dims=2
        )

        # Accumulate the scores
        score = (1 - N) * weighted_prior_score.sum(dim=1) + torch.sum(
            weighted_posterior_scores, dim=1
        )

        # Solve the linear system
        score = solve_diag_or_dense(Lam.squeeze(1), score, batch_dims=1)

        return score.reshape(inputs.shape)


@register_iid_method("gauss")
class GaussCorrectedScoreFn(BaseGaussCorrectedScoreFunction):
    r"""Gauss-corrected IID scores with a heuristic posterior precision estimate.

    Estimates the posterior precision by scaling the prior precision by a fixed
    factor, assuming that the data is informative enough for the posterior to be
    narrower than the prior.

    Example:
    --------

    ::

        from sbi.inference.potentials.vector_field_adaptor import (
            GaussCorrectedScoreFn,
        )

        iid_score = GaussCorrectedScoreFn(vf_estimator, prior)
        score = iid_score(inputs, conditions, time)  # inputs: [b, iid, d]
    """

    def __init__(
        self,
        vector_field_estimator: Union[ConditionalVectorFieldEstimator, ScoreAdaptation],
        prior: Distribution,
        posterior_precision: Optional[Tensor] = None,
        scale_from_prior_precision: float = 2.0,
        enable_lam_psd: bool = False,
        lam_psd_nugget: float = 0.01,
        device: Union[str, torch.device] = "cpu",
    ) -> None:
        """Initializes Gauss-corrected scores with a heuristic precision estimate.

        Args:
            vector_field_estimator: The neural network modeling the score.
            prior: The prior distribution.
            posterior_precision: Optional preset posterior precision.
            scale_from_prior_precision: Factor to scale the prior precision by when
                ``posterior_precision`` is not provided.
            enable_lam_psd: Whether to ensure the precision matrix is positive definite.
            lam_psd_nugget: The nugget value to ensure positive definiteness.
            device: The device on which to evaluate the potential. Defaults to "cpu".
        """
        super().__init__(
            vector_field_estimator, prior, enable_lam_psd, lam_psd_nugget, device=device
        )

        if posterior_precision is None:
            posterior_precision = self.estimate_prior_precision(
                prior, scale_from_prior_precision
            )
        assert posterior_precision is not None
        self.posterior_precision = posterior_precision

    def posterior_precision_est_fn(self, conditions: Tensor) -> Tensor:
        r"""
        Estimates the posterior precision.

        Args:
            conditions: Observed data.

        Returns:
            Estimated posterior precision.
        """
        precision = self.posterior_precision
        precision = torch.broadcast_to(
            precision, (1, conditions.shape[0], *precision.shape)
        )
        return precision

    @classmethod
    @functools.lru_cache()
    def estimate_prior_precision(
        cls, prior: Distribution, scale_from_prior_precision: float
    ) -> Tensor:
        r"""
        Estimates the prior precision.

        Args:
            prior: The prior distribution.
            scale_from_prior_precision: Scaling factor for the posterior precision if
                not provided.

        Returns:
            Estimated prior precision.
        """
        try:
            prior_precision = 1 / prior.variance
            posterior_precision = scale_from_prior_precision * prior_precision
        except Exception:
            d = prior.event_shape[0]
            num_samples = int(math.sqrt(d) * 1000)
            prior_samples = prior.sample(torch.Size((num_samples,)))
            prior_precision_estimate = 1 / torch.var(prior_samples, dim=0)
            posterior_precision = scale_from_prior_precision * prior_precision_estimate
        return posterior_precision


@register_iid_method("auto_gauss")
class AutoGaussCorrectedScoreFn(BaseGaussCorrectedScoreFunction):
    r"""Gauss-corrected IID scores with automatic posterior precision estimation.

    Estimates the posterior precision by drawing approximate samples from each
    per-observation posterior using the vector field estimator. Has a slight
    initialization overhead but is generally more accurate than a fixed heuristic.

    Example:
    --------

    ::

        from sbi.inference.potentials.vector_field_adaptor import (
            AutoGaussCorrectedScoreFn,
        )

        iid_score = AutoGaussCorrectedScoreFn(vf_estimator, prior)
        score = iid_score(inputs, conditions, time)  # inputs: [b, iid, d]
    """

    def __init__(
        self,
        vector_field_estimator: Union[ConditionalVectorFieldEstimator, ScoreAdaptation],
        prior: Distribution,
        enable_lam_psd: bool = True,
        lam_psd_nugget: float = 0.01,
        precision_est_only_diag: bool = False,
        precision_est_budget: Optional[int] = None,
        precision_initial_sampler_steps: int = 100,
        device: Union[str, torch.device] = "cpu",
    ) -> None:
        """Initializes Gauss-corrected scores with automatic precision estimation.

        Args:
            vector_field_estimator: The neural network modeling the score.
            prior: The prior distribution.
            enable_lam_psd: Whether to ensure the precision matrix is positive definite.
            lam_psd_nugget: The nugget value to ensure positive definiteness.
            precision_est_only_diag: Whether to estimate only the diagonal of the
                precision matrix.
            precision_est_budget: The budget for the precision estimation.
            precision_initial_sampler_steps: The number of initial sampler steps.
            device: The device on which to evaluate the potential. Defaults to "cpu".
        """
        super().__init__(
            vector_field_estimator, prior, enable_lam_psd, lam_psd_nugget, device=device
        )
        self.precision_est_only_diag = precision_est_only_diag
        self.precision_est_budget = precision_est_budget
        self.precision_initial_sampler_steps = precision_initial_sampler_steps

    def posterior_precision_est_fn(self, conditions: Tensor) -> Tensor:
        r"""
        Estimates the posterior precision.

        Args:
            conditions: Observed data.

        Returns:
            Estimated posterior precision.
        """
        return self.estimate_posterior_precision(
            self.vector_field_estimator,
            self.prior,
            conditions,
            precision_est_only_diag=self.precision_est_only_diag,
            precision_est_budget=self.precision_est_budget,
            precision_initial_sampler_steps=self.precision_initial_sampler_steps,
        )

    @classmethod
    @functools.lru_cache()
    def estimate_posterior_precision(
        cls,
        vector_field_estimator: ConditionalVectorFieldEstimator,
        prior: Distribution,
        conditions: Tensor,
        precision_est_only_diag: bool = False,
        precision_est_budget: Optional[int] = None,
        precision_initial_sampler_steps: int = 100,
    ) -> Tensor:
        r"""
        Estimates the posterior precision by sampling from the posteriors p(theta|x_i)
        for each observation, then empirically estimating the precision matrix.

        Args:
            vector_field_estimator: The score estimator.
            prior: The prior distribution.
            conditions: Observed data.
            precision_est_only_diag: Whether to estimate only the diagonal of the
                precision matrix.
            precision_est_budget: The budget for the precision estimation i.e. number
                of samples.
            precision_initial_sampler_steps: The number of sampler steps to get the
                per observation posterior samples.

        Returns:
            Estimated posterior precision.
        """
        # NOTE: This assumes that the objects don't change between calls to cache
        # the results efficiently.

        # NOTE: To avoid circular imports :(
        from sbi.inference.posteriors.vector_field_posterior import VectorFieldPosterior

        posterior = VectorFieldPosterior(
            vector_field_estimator, prior, device=conditions.device
        )

        if precision_est_budget is None:
            if precision_est_only_diag:
                precision_est_budget = int(math.sqrt(prior.event_shape[0]) * 1000)
            else:
                precision_est_budget = min(int(prior.event_shape[0] * 1000), 5000)

        thetas = posterior.sample_batched(
            sample_shape=torch.Size([precision_est_budget]),
            x=conditions,
            show_progress_bars=False,
            steps=precision_initial_sampler_steps,
        )

        if precision_est_only_diag:
            variances = torch.var(thetas, dim=0)
            precisions = 1 / variances
        else:
            cov = torch.einsum("bnd,bne->nde", thetas, thetas) / (
                precision_est_budget - 1
            )
            precisions = torch.inverse(cov)

        return precisions.unsqueeze(0)


@register_iid_method("jac_gauss")
class JacCorrectedScoreFn(BaseGaussCorrectedScoreFunction):
    """
    This method extends the BaseGaussCorrectedScoreFunction by using Tweedie's moment
    projection to estimate the marginal posterior covariance at each time step.

    This method theoretically provides the most accurate estimates for the marginal,
    however, it requires the computation of the Jacobian of the score function at each
    iteration which can be computationally expensive and lead to numerical instabilities
    in some cases.
    """

    def posterior_precision_est_fn(self, conditions: Tensor) -> Tensor:
        raise ValueError("This method does not use the posterior precision estimation.")

    def marginal_denoising_posterior_precision_est_fn(
        self,
        time: Tensor,
        inputs: Tensor,
        conditions: Tensor,
    ) -> Tensor:
        r"""
        Estimates the marginal posterior precision using the Jacobian of the score
        function.

        Based on Tweedie's denoising covariance estimator.

        Args:
            time: Time tensor.
            inputs: Parameter tensor.
            conditions: Observed data.

        Returns:
            Estimated marginal posterior precision.
        """
        d = inputs.shape[-1]
        with torch.enable_grad():
            # NOTE: torch.func can be relatively unstable...
            jac_fn = torch.func.jacrev(  # type: ignore
                lambda x: self.vector_field_estimator.score(x, conditions, time)
            )
            jac_fn = torch.func.vmap(torch.func.vmap(jac_fn))  # type: ignore
            jac = jac_fn(inputs).squeeze(1)

        # Must be symmetrical
        jac = 0.5 * (jac + jac.transpose(-1, -2))

        m = self.vector_field_estimator.mean_t_fn(time)
        std = self.vector_field_estimator.std_fn(time)
        cov0 = std**2 * jac + torch.eye(d, device=self.device)[None, None, :, :]

        denoising_posterior_precision = m**2 / std**2 * torch.inverse(cov0)

        return denoising_posterior_precision


def compute_score(p: Distribution, inputs: Tensor) -> Tensor:
    """Computes the score (gradient of log-probability) of a distribution.

    Falls back to zero if the distribution does not support ``log_prob`` or
    autodiff (e.g., uniform priors).

    Args:
        p: The distribution to compute the score for.
        inputs: The points at which to evaluate the score.

    Returns:
        The score tensor, same shape as ``inputs``.
    """
    try:
        with torch.enable_grad():
            inputs = inputs.detach().clone().requires_grad_(True)
            log_prob = p.log_prob(inputs)
            score = torch.autograd.grad(
                log_prob,
                inputs,
                grad_outputs=torch.ones_like(log_prob),
                create_graph=True,
            )[0].detach()
    except Exception:
        score = torch.zeros_like(inputs)
    return score


def ensure_lam_positive_definite(
    denoising_prior_precision: torch.Tensor,
    denoising_posterior_precision: torch.Tensor,
    N: int,
    precision_nugget: float = 0.1,
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""
    Ensure that the matrix is positive definite.

    Args:
        denoising_prior_precision: The prior precision tensor.
        denoising_posterior_precision: The posterior precision tensor.
        N: The scaling factor used in the correction.
        precision_nugget: The nugget value to ensure positive definiteness.

    Returns:
        A tuple of (denoising_prior_precision, denoising_posterior_precision) where the
        posterior precision has been adjusted to be positive definite.
    """
    d = denoising_prior_precision.shape[-1]

    term1 = (1 - N) * denoising_prior_precision
    term2 = torch.sum(denoising_posterior_precision, axis=1, keepdim=True)  # type: ignore
    Lam = add_diag_or_dense(term1, term2, batch_dims=2)

    is_diag = Lam.ndim == 3
    if d > 1 and not is_diag:
        eigenvalues, eigenvectors = torch.linalg.eigh(Lam)
        corrected_eigs = torch.where(
            eigenvalues <= 0, -eigenvalues, torch.zeros_like(eigenvalues)
        )
        corrected_eigs = corrected_eigs / (N - 1)

        Lam_corr = torch.einsum(
            '...ij,...j,...kj->...ik', eigenvectors, corrected_eigs, eigenvectors
        )
        Lam_corr += precision_nugget * torch.eye(
            Lam.shape[-1], device=Lam.device, dtype=Lam.dtype
        )

        denoising_posterior_precision = add_diag_or_dense(
            denoising_posterior_precision, Lam_corr, batch_dims=2
        )
    else:
        # For one-dimensional case, treat Lam as a vector by putting it on the diagonal.
        Lam_diag = Lam
        average_diff = (
            torch.where(Lam_diag > 0, torch.zeros_like(Lam_diag), -Lam_diag) / (N - 1)
            + precision_nugget
        )
        Lam_corr = average_diff
        denoising_posterior_precision = denoising_posterior_precision + Lam_corr

    return denoising_prior_precision, denoising_posterior_precision
