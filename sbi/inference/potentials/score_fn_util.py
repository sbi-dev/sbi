import functools
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, Optional, Type, Union

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.distributions import Distribution

from sbi.neural_nets.estimators.score_estimator import ConditionalScoreEstimator
from sbi.utils.score_utils import (
    add_diag_or_dense,
    denoise,
    marginalize,
    mv_diag_or_dense,
    solve_diag_or_dense,
)
from sbi.utils.torchutils import ensure_theta_batched

IID_METHODS = {}
GUIDANCE_METHODS = {}


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


def get_guidance_method(name: str) -> Type["ScoreAdaptation"]:
    r"""
    Retrieves the guidance method by name.

    Args:
        name: The name of the guidance method.

    Returns:
        The guidance method class and its default configuration.
    """
    if name not in GUIDANCE_METHODS:
        raise NotImplementedError(f"Method {name} for guidance not implemented.")
    return GUIDANCE_METHODS[name]


def register_guidance_method(name: str, default_cfg: Optional[Type] = None) -> Callable:
    r"""
    Registers a guidance method and its default configuration.

    Args:
        name: The name of the guidance method.
        default_cfg: The default configuration class for the guidance method.

    Returns:
        A decorator function to register the guidance method class.
    """

    def decorator(cls: Type["ScoreAdaptation"]) -> Type["ScoreAdaptation"]:
        GUIDANCE_METHODS[name] = (cls, default_cfg)
        return cls

    return decorator


def register_iid_method(name: str) -> Callable:
    r"""
    Registers an IID method.

    Args:
        name: The name of the IID method.

    Returns:
        A decorator function to register the IID method class.
    """

    def decorator(cls: Type["IIDScoreFunction"]) -> Type["IIDScoreFunction"]:
        IID_METHODS[name] = cls
        return cls

    return decorator


class ScoreAdaptation(ABC):
    def __init__(
        self,
        score_estimator: ConditionalScoreEstimator,
        prior: Optional[Distribution],
        device: str = "cpu",
    ):
        """This class manages manipulating the score estimator to impose additional
        constraints on the posterior via guidance.

        Args:
            score_estimator: The score estimator.
            prior: The prior distribution.
            device: The device on which to evaluate the potential.
        """
        self.score_estimator = score_estimator
        self.prior = prior
        self.device = device

    @abstractmethod
    def __call__(self, theta: Tensor, x_o: Tensor, time: Optional[Tensor] = None):
        pass


@dataclass
class AffineClassifierFreeGuidanceCfg:
    prior_scale: Union[float, Tensor] = 1.0
    prior_shift: Union[float, Tensor] = 0.0
    likelihood_scale: Union[float, Tensor] = 1.0
    likelihood_shift: Union[float, Tensor] = 0.0


@register_guidance_method("affine_classifier_free", AffineClassifierFreeGuidanceCfg)
class AffineClassifierFreeGuidance(ScoreAdaptation):
    def __init__(
        self,
        score_estimator: ConditionalScoreEstimator,
        prior: Optional[Distribution],
        cfg: AffineClassifierFreeGuidanceCfg,
        device: str = "cpu",
    ):
        """This class manages manipulating the score estimator to temper or shift the
        prior and likelihood.

        This is usually known as classifier-free guidance. And works by decomposing the
        posterior score into a prior and likelihood component. These can then be scaled
        and shifted to impose change the posterior to p

        Args:
            score_estimator: The score estimator.
            prior: The prior distribution.
            device: The device on which to evaluate the potential.
        """

        if prior is None:
            raise ValueError(
                "Prior is required for classifier-free guidance, please"
                " provide as least an improper empirical prior."
            )

        self.prior_scale = torch.tensor(cfg.prior_scale, device=device)
        self.prior_shift = torch.tensor(cfg.prior_shift, device=device)
        self.likelihood_scale = torch.tensor(cfg.likelihood_scale, device=device)
        self.likelihood_shift = torch.tensor(cfg.likelihood_shift, device=device)
        super().__init__(score_estimator, prior, device)

    def marginal_prior_score(self, theta: Tensor, time: Tensor):
        """Computes the marginal prior score analyticaly (or approximatly)"""
        m = self.score_estimator.mean_t_fn(time)
        std = self.score_estimator.std_fn(time)
        marginal_prior = marginalize(self.prior, m, std)  # type: ignore
        marginal_prior_score = compute_score(marginal_prior, theta)
        return marginal_prior_score

    def __call__(self, input: Tensor, condition: Tensor, time: Optional[Tensor] = None):
        if time is None:
            time = torch.tensor([self.score_estimator.t_min])

        posterior_score = self.score_estimator(
            input=input, condition=condition, time=time
        )
        prior_score = self.marginal_prior_score(input, time)
        ll_score = posterior_score - prior_score
        ll_score_mod = ll_score * self.likelihood_scale + self.likelihood_shift
        prior_score_mod = prior_score * self.prior_scale + self.prior_shift

        return ll_score_mod + prior_score_mod


@dataclass
class UniversalGuidanceCfg:
    guidance_fn: Callable[[Tensor, Tensor, Tensor, Tensor], Tensor]
    guidance_fn_score: Optional[Callable[[Tensor, Tensor, Tensor, Tensor], Tensor]] = (
        None
    )

@register_guidance_method("universal", UniversalGuidanceCfg)
class UniversalGuidance(ScoreAdaptation):
    def __init__(
        self,
        score_estimator: ConditionalScoreEstimator,
        prior: Optional[Distribution],
        cfg: UniversalGuidanceCfg,
        device: str = "cpu",
    ):
        """This class manages manipulating the score estimator using a custom guidance
        function.

        Args:
            score_estimator: The score estimator.
            prior: The prior distribution.
            cfg: Configuration for the universal guidance.
            device: The device on which to evaluate the potential.
        """
        self.guidance_fn = cfg.guidance_fn

        if cfg.guidance_fn_score is None:

            def guidance_fn_score(input, condition, m, std):
                with torch.enable_grad():
                    input = input.detach().clone().requires_grad_(True)
                    score = torch.autograd.grad(
                        cfg.guidance_fn(input, condition, m, std).sum(),
                        input,
                        create_graph=True,
                    )[0]
                return score

            self.guidance_fn_score = guidance_fn_score
        else:
            self.guidance_fn_score = cfg.guidance_fn_score

        super().__init__(score_estimator, prior, device)

    def __call__(self, input: Tensor, condition: Tensor, time: Optional[Tensor] = None):
        if time is None:
            time = torch.tensor([self.score_estimator.t_min])
        score = self.score_estimator(input, condition, time)
        m = self.score_estimator.mean_t_fn(time)
        std = self.score_estimator.std_fn(time)

        # Tweedie's formula for denoising
        denoised_input = (input + std**2 * score) / m
        guidance_score = self.guidance_fn_score(denoised_input, condition, m, std)

        return score + guidance_score


@dataclass
class IntervalGuidanceCfg:
    lower_bound: Optional[Union[float, Tensor]]
    upper_bound: Optional[Union[float, Tensor]]
    mask: Optional[Tensor] = None
    scale_factor: float = 0.5

@register_guidance_method("interval", IntervalGuidanceCfg)
class IntervalGuidance(UniversalGuidance):
    def __init__(
        self,
        score_estimator: ConditionalScoreEstimator,
        prior: Optional[Distribution],
        cfg: IntervalGuidanceCfg,
        device: str = "cpu",
    ):
        """Implements interval guidance to constrain parameters within bounds.

        Args:
            score_estimator: The score estimator.
            prior: The prior distribution.
            cfg: Configuration specifying the interval bounds.
            device: The device on which to evaluate the potential.
        """

        def interval_fn(input, condition, m, std):
            if cfg.lower_bound is None and cfg.upper_bound is None:
                raise ValueError(
                    "At least one of lower_bound or upper_bound is required. Otherwise"
                    " the guidance function has no effect."
                )

            scale = cfg.scale_factor / (m**2 * std**2)
            upper_bound = (
                F.logsigmoid(scale * (input - cfg.upper_bound))
                if cfg.upper_bound is not None
                else torch.zeros_like(input)
            )
            lower_bound = (
                F.logsigmoid(-scale * (input - cfg.lower_bound))
                if cfg.lower_bound is not None
                else torch.zeros_like(input)
            )
            out = upper_bound + lower_bound
            if cfg.mask is not None:
                if cfg.mask.shape != out.shape:
                    cfg.mask = cfg.mask.unsqueeze(0).expand_as(out)
                out = torch.where(cfg.mask, out, torch.zeros_like(out))
            return out

        super().__init__(
            score_estimator,
            prior,
            UniversalGuidanceCfg(guidance_fn=interval_fn),
            device=device,
        )


class IIDScoreFunction(ABC):
    def __init__(
        self,
        score_estimator: "ConditionalScoreEstimator",
        prior: Distribution,
        device: str = "cpu",
    ) -> None:
        r"""
        This is a abstract base class wrapper for score estimators.

        Subclasses are used to implement different methods for factorized distributions.
        For example, in the IID setting the posterior for N observations can be
        represented as a product of N "local" posteriors, divided by N-1 prior terms.
        This allows to efficiently extend "single" observation score estimators to a
        sequence of IID observtions. Unfortunatly, this is not as simple as just summing
        the scores minus the prior score, as in diffusion models the we also need to
        represent the "marginal" posterior scores at time $t$. Even if the true
        posterior factorizes, the marginal true posterior at time $t>0$ does not and
        this requires some adjustments.

        Args:
            score_estimator: The neural network modeling the score.
            prior: The prior distribution.
            device: The device on which to evaluate the potential. Defaults to "cpu".
        """

        self.score_estimator = score_estimator.to(device).eval()
        self.prior = prior

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
class FNPEScoreFunction(IIDScoreFunction):
    def __init__(
        self,
        score_estimator: "ConditionalScoreEstimator",
        prior: Distribution,
        device: str = "cpu",
        prior_score_weight: Optional[Callable[[Tensor], Tensor]] = None,
    ) -> None:
        r"""
        The FNPEScoreFunction implments the "Factorized Neural Posterior Estimation"
        method for score-based models [1].

        This method does not apply the necessary corrections for the score function, but
        instead uses a simple weighting of the prior score. This is generally applicable
        and simple but does in general not return the correct marginal score for any
        $t > 0$.
        For a moderate number of factors, this hence does require post-hoc adjustment
        through e.g. predictor-corrector samplers to ensure stable convergence to the
        correct terminal distiribution at $t=0$.

        Literature:
        - [1] Compositional Score Modeling for Simulation-Based Inference
            (https://arxiv.org/abs/2209.14249)

        Args:
            score_estimator: The neural network modeling the score.
            prior: The prior distribution.
            device: The device on which to evaluate the potential. Defaults to "cpu".
            prior_score_weight: A function to weight the prior score. Defaults to the
                linear interpolation between zero (at t=0) and one (at t=t_max).
        """
        super().__init__(score_estimator, prior, device)
        if prior_score_weight is None:
            t_max = score_estimator.t_max

            def prior_score_weight_fn(t):
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
            time = torch.tensor([self.score_estimator.t_min])

        N = conditions.shape[0]

        # Compute the per-sample score
        inputs = ensure_theta_batched(inputs)
        base_score = self.score_estimator(inputs, conditions, time)

        # Compute the prior score

        prior_score = self.prior_score_weight_fn(time) * compute_score(
            self.prior, inputs
        )

        # Accumulate
        score = (1 - N) * prior_score + base_score.sum(-2, keepdim=True)

        return score


class BaseGaussCorrectedScoreFunction(IIDScoreFunction):
    def __init__(
        self,
        score_estimator: "ConditionalScoreEstimator",
        prior: Distribution,
        ensure_lam_psd: bool = True,
        lam_psd_nugget: float = 0.01,
        device: str = "cpu",
    ) -> None:
        r"""Base class for Gauss-corrected score function as proposed in [1].

        Specificially a simple analytic correction for the marginal scores is derived
        using Gaussian assumptions. This is a simple and efficient method to correct
        the score function for the marginal posterior, which was also shown to scale
        well to large sequence settings [1,2].

        A limitation of the method is that it requires following inputs:
        - The marginal prior distribution (which might be non-trivial to compute)
        - The marginal posterior precision (which is generally not available, and needs
            to be estimated).

        Within this library we have an extensive set of tools to obtain analytic (or
        good approximations) for the most common prior distributions. The marginal
        posterior precision can be treated as a "hyperparameter" different estimation
        methods are available, which will be subclased from this class.

        Literature:
        - [1] Diffusion posterior sampling for simulation-based inference in tall data
            settings (https://arxiv.org/abs/2404.07593)
        - [2] Compositional simulation-based inference for time series
            (https://arxiv.org/abs/2411.02728)

        Args:
            score_estimator: The neural network modelling the score.
            prior: The prior distribution.
            ensure_lam_psd: Whether to ensure the precision matrix is positive definite.
            lam_psd_nugget: The nugget value to ensure positive definiteness.
            device: The device on which to evaluate the potential. Defaults to "cpu".
        """
        super().__init__(score_estimator, prior, device)
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
        m = self.score_estimator.mean_t_fn(time)
        std = self.score_estimator.std_fn(time)

        if precisions_posteriors.ndim == 4:
            Ident = torch.eye(precisions_posteriors.shape[-1])
        else:
            Ident = torch.ones_like(precisions_posteriors)

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
        m = self.score_estimator.mean_t_fn(time)
        std = self.score_estimator.std_fn(time)
        p = marginalize(self.prior, m, std)
        prior_score = compute_score(p, inputs)
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
        m = self.score_estimator.mean_t_fn(time)
        std = self.score_estimator.std_fn(time)

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
                implemented the variance/covariance_matrix method) but inherit from
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
            time = torch.tensor([self.score_estimator.t_min])

        base_score = self.score_estimator(inputs, conditions, time, **kwargs)
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
    def __init__(
        self,
        score_estimator: "ConditionalScoreEstimator",
        prior: Distribution,
        posterior_precision: Optional[Tensor] = None,
        scale_from_prior_precision: float = 2.0,
        enable_lam_psd: bool = False,
        lam_psd_nugget: float = 0.01,
        device: str = "cpu",
    ) -> None:
        r"""
        This extends the BaseGaussCorrectedScoreFunction to provide a simple method to
        heuristically estimate the posterior precision. Assuming that data is
        informative enough, the posterior precision should be higher than the prior
        precision. This method simply scales the prior precision by a factor.

        Args:
            score_estimator: The neural network modeling the score.
            prior: The prior distribution.
            posterior_precision: Optional preset posterior precision.
            scale_from_prior_precision: If not provided it simply increases the prior
                precision by this factor.
            enable_lam_psd: Whether to ensure the precision matrix is positive definite.
            lam_psd_nugget: The nugget value to ensure positive definiteness.
            device: The device on which to evaluate the potential. Defaults to "cpu".
        """
        super().__init__(
            score_estimator, prior, enable_lam_psd, lam_psd_nugget, device=device
        )

        if posterior_precision is None:
            posterior_precision = self.estimate_prior_precision(
                prior, scale_from_prior_precision
            )

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
            prior_samples = prior.sample((num_samples,))
            prior_precision_estimate = 1 / torch.var(prior_samples, dim=0)
            posterior_precision = scale_from_prior_precision * prior_precision_estimate
        return posterior_precision


@register_iid_method("auto_gauss")
class AutoGaussCorrectedScoreFn(BaseGaussCorrectedScoreFunction):
    def __init__(
        self,
        score_estimator: "ConditionalScoreEstimator",
        prior: Distribution,
        enable_lam_psd: bool = True,
        lam_psd_nugget: float = 0.01,
        precision_est_only_diag: bool = False,
        precision_est_budget: Optional[int] = None,
        precision_initial_sampler_steps: int = 100,
        device: str = "cpu",
    ) -> None:
        r"""
        This method extends the by estimating the posterior precision using
        approximate posterior samples obtained from a diffusion model (using the
        score_estimator) [1]. This method has a slight initialization overhead, but
        generally provides more accurate results than simple heuristics.

        Args:
            score_estimator: The neural network modeling the score.
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
            score_estimator, prior, enable_lam_psd, lam_psd_nugget, device=device
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
            self.score_estimator,
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
        score_estimator: "ConditionalScoreEstimator",
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
            score_estimator: The score estimator.
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
        # NOTE: This assumes that the objects dont change between calls to cache
        # the results efficiently.

        # NOTE: To avoid circular imports :(
        from sbi.inference.posteriors.score_posterior import ScorePosterior

        posterior = ScorePosterior(score_estimator, prior)

        if precision_est_budget is None:
            if precision_est_only_diag:
                precision_est_budget = int(math.sqrt(prior.event_shape[0]) * 1000)
            else:
                precision_est_budget = min(int(prior.event_shape[0] * 1000), 5000)

        thetas = posterior.sample_batched(
            torch.Size([precision_est_budget]),
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
            # NOTE: torch.func can be realtively unstable...
            jac_fn = torch.func.jacrev(  # type: ignore
                lambda x: self.score_estimator(x, conditions, time)
            )
            jac_fn = torch.func.vmap(torch.func.vmap(jac_fn))  # type: ignore
            jac = jac_fn(inputs).squeeze(1)

        # Must be symmetrical
        jac = 0.5 * (jac + jac.transpose(-1, -2))

        m = self.score_estimator.mean_t_fn(time)
        std = self.score_estimator.std_fn(time)
        cov0 = std**2 * jac + torch.eye(d)[None, None, :, :]

        denoising_posterior_precision = m**2 / std**2 + torch.inverse(cov0)

        return denoising_posterior_precision


def compute_score(p: Distribution, inputs: Tensor):
    # NOTE The try except is for unifrom priors which do not have a grad, and
    # implementations that do not implement the log_prob method.
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
) -> (torch.Tensor, torch.Tensor):
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
