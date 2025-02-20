import functools
import math
from abc import abstractmethod
from typing import Callable, Optional, Type

import torch
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


def get_iid_method(name: str) -> Type["ScoreFnIID"]:
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


def register_iid_method(name: str) -> Callable:
    r"""
    Registers an IID method.

    Args:
        name: The name of the IID method.

    Returns:
        A decorator function to register the IID method class.
    """

    def decorator(cls: Type["ScoreFnIID"]) -> Type["ScoreFnIID"]:
        IID_METHODS[name] = cls
        return cls

    return decorator


class ScoreFnIID:
    def __init__(
        self,
        score_estimator: "ConditionalScoreEstimator",
        prior: Distribution,
        device: str = "cpu",
    ) -> None:
        r"""
        Initializes the ScoreFnIID class.

        Args:
            score_estimator: The neural network modeling the score.
            prior: The prior distribution.
            device: The device on which to evaluate the potential. Defaults to "cpu".
        """

        self.score_estimator = score_estimator.to(device)
        self.prior = prior
        self.score_estimator.eval()

    @abstractmethod
    def __call__(
        self,
        theta: Tensor,
    ) -> Tensor:
        r"""
        Abstract method to be implemented by subclasses to compute the score function.

        Args:
            theta: The parameters at which to evaluate the potential.

        Returns:
            The computed score function.
        """
        pass

    def prior_score_fn(self, theta: Tensor) -> Tensor:
        r"""
        Computes the score of the prior distribution.

        Args:
            theta: The parameters at which to evaluate the prior score.

        Returns:
            The computed prior score.
        """
        with torch.enable_grad():
            theta = theta.detach().clone().requires_grad_(True)
            prior_log_prob = self.prior.log_prob(theta)
            prior_score = torch.autograd.grad(
                prior_log_prob,
                theta,
                grad_outputs=torch.ones_like(prior_log_prob),
            )[0]
        return prior_score


@register_iid_method("fnpe")
class FNPEScoreFn(ScoreFnIID):
    def __init__(
        self,
        score_estimator: "ConditionalScoreEstimator",
        prior: Distribution,
        device: str = "cpu",
        prior_score_weight: Optional[Callable[[Tensor], Tensor]] = None,
    ) -> None:
        r"""
        Initializes the FNPEScoreFn class.

        Paper: Compositional Score Modeling for Simulation-Based Inference
        - https://arxiv.org/abs/2209.14249

        Args:
            score_estimator: The neural network modeling the score.
            prior: The prior distribution.
            device: The device on which to evaluate the potential. Defaults to "cpu".
            prior_score_weight: A function to weight the prior score. Defaults to None.
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
        Computes the score function for score-based methods.

        Args:
            inputs: The input parameters at which to evaluate the potential.
            conditions: The observed data at which to evaluate the posterior.
            time: The time at which to evaluate the score. Defaults to None.

        Returns:
            The computed score function.
        """
        if time is None:
            time = torch.tensor([self.score_estimator.t_min])

        N = conditions.shape[0]

        # Compute the per-sample score
        inputs = ensure_theta_batched(inputs)
        base_score = self.score_estimator(inputs, conditions, time)

        # Compute the prior score
        prior_score = self.prior_score_fn(inputs)
        prior_score = self.prior_score_weight_fn(time) * prior_score

        # Accumulate
        score = (1 - N) * prior_score + base_score.sum(-2, keepdim=True)

        return score


class AbstractGaussCorrectedScoreFn(ScoreFnIID):
    def __init__(
        self,
        score_estimator: "ConditionalScoreEstimator",
        prior: Distribution,
        ensure_lam_psd: bool = True,
        lam_psd_nugget: float = 0.01,
        device: str = "cpu",
    ) -> None:
        r"""Initializes the AbstractGaussCorrectedScoreFn class.

        Method for appropriately correcting the score function using Gaussian
        approximations.

        Papers:
        - https://arxiv.org/abs/2106.05399
        - https://arxiv.org/abs/2411.02728

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
        r"""Abstract method to estimate the posterior precision.

        Args:
            conditions: Observed data.

        Returns:
            Estimated posterior precision.
        """
        pass

    def denoising_prior(self, m: Tensor, std: Tensor, inputs: Tensor) -> Distribution:
        r"""Denoise the prior distribution.

        Args:
            m: Mean tensor.
            std: Standard deviation tensor.
            inputs: Parameters tensor.

        Returns:
            Denoised prior distribution.
        """
        return denoise(self.prior, m, std, inputs)

    def marginal_prior(self, time: Tensor, inputs: Tensor) -> Distribution:
        r"""Compute the marginal prior distribution.

        Args:
            time: Time tensor.
            inputs: Parameters tensor.

        Returns:
            Marginal prior distribution.
        """
        m = self.score_estimator.mean_t_fn(time)
        std = self.score_estimator.std_fn(time)
        return marginalize(self.prior, m, std)

    def marginal_denoising_posterior_precision_est_fn(
        self, time: Tensor, inputs: Tensor, conditions: Tensor, N: int
    ) -> Tensor:
        r"""Estimates the marginal posterior precision.

        Args:
            time: Time tensor.
            inputs: Parameters tensor.
            conditions: Observed data.
            N: Number of samples.

        Returns:
            Estimated marginal posterior precision.
        """
        precisions_posteriors = self.posterior_precision_est_fn(conditions)

        print(f"precisions_posteriors: {precisions_posteriors.shape}")
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
        # NOTE: This is a bit slow, one could add a score function for priors that
        # explicitly computes it.
        with torch.enable_grad():
            inputs = inputs.clone().detach().requires_grad_(True)
            p = self.marginal_prior(time, inputs)
            log_p = p.log_prob(inputs)
            return torch.autograd.grad(
                log_p,
                inputs,
                grad_outputs=torch.ones_like(log_p),
            )[0]

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

        p_denoise = self.denoising_prior(m, std, inputs)

        if hasattr(p_denoise, "covariance_matrix"):
            # We currently only support diagonal covariances
            inv_cov = torch.inverse(p_denoise.covariance_matrix)  # type: ignore
            return inv_cov.reshape(inputs.shape + inputs.shape[-1:])
        else:
            precision = 1 / p_denoise.variance
            return precision.reshape(inputs.shape)

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
        # NOTE We can assume here a fixed 3dim shape of inputs [b,N,d]
        # NOTE We can assume here a fixed format of conditions [N,...]
        N, *_ = conditions.shape

        if time is None:
            time = torch.tensor([self.score_estimator.t_min])

        base_score = self.score_estimator(inputs, conditions, time, **kwargs)
        prior_score = self.marginal_prior_score_fn(time, inputs)

        # Marginal prior precision
        prior_precision = self.marginal_denoising_prior_precision_fn(time, inputs)
        # Marginal posterior variance estimates
        posterior_precisions = self.marginal_denoising_posterior_precision_est_fn(
            time, inputs, conditions, N
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
class GaussCorrectedScoreFn(AbstractGaussCorrectedScoreFn):
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
        Initializes the GaussCorrectedScoreFn class.

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

    def posterior_precision_est_fn(self, x_o: Tensor) -> Tensor:
        r"""
        Estimates the posterior precision.

        Args:
            x_o: Observed data.

        Returns:
            Estimated posterior precision.
        """
        precision = self.posterior_precision
        precision = torch.broadcast_to(precision, (1, x_o.shape[0], *precision.shape))
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
        finally:
            d = prior.event_shape[0]
            num_samples = int(math.sqrt(d) * 1000)
            prior_samples = prior.sample((num_samples,))
            prior_precision_estimate = 1 / torch.var(prior_samples, dim=0)
            posterior_precision = scale_from_prior_precision * prior_precision_estimate
        return posterior_precision.squeeze()


@register_iid_method("auto_gauss")
class AutoGaussCorrectedScoreFn(AbstractGaussCorrectedScoreFn):
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
        Initializes the AutoGaussCorrectedScoreFn class.

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

        # NOTE: To avoid circular imports
        from sbi.inference.posteriors.score_posterior import ScorePosterior

        posterior = ScorePosterior(score_estimator, prior)

        if precision_est_budget is None:
            if precision_est_only_diag:
                precision_est_budget = int(math.sqrt(prior.event_shape[0]) * 1000)
            else:
                precision_est_budget = int(prior.event_shape[0] * 2000)

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
            cov = torch.einsum("...ij,...kj->...ik", thetas, thetas)
            cov = cov / (precision_est_budget - 1)
            precisions = torch.inverse(cov)

        return precisions.unsqueeze(0)


@register_iid_method("jac_gauss")
class JacCorrectedScoreFn(AbstractGaussCorrectedScoreFn):
    def posterior_precision_est_fn(self, conditions: Tensor) -> Tensor:
        raise ValueError("This method is not used for JacCorrectedScoreFn.")

    def marginal_denoising_posterior_precision_est_fn(
        self, time: Tensor, inputs: Tensor, conditions: Tensor, N: int
    ) -> Tensor:
        r"""
        Estimates the marginal posterior precision using the Jacobian of the score
        function.

        Based on Tweedie's denoising covariance estimator.

        Args:
            time: Time tensor.
            inputs: Parameter tensor.
            conditions: Observed data.
            N: Number of samples.

        Returns:
            Estimated marginal posterior precision.
        """
        d = inputs.shape[-1]
        with torch.enable_grad():
            # NOTE: torch.func can be realtively unstable...
            jac_fn = torch.func.jacrev(
                lambda x: self.score_estimator(x, conditions, time)
            )
            jac_fn = torch.func.vmap(torch.func.vmap(jac_fn))
            jac = jac_fn(inputs).squeeze(1)

        # Must be symmetrical
        jac = 0.5 * (jac + jac.transpose(-1, -2))

        m = self.score_estimator.mean_t_fn(time)
        std = self.score_estimator.std_fn(time)
        cov0 = std**2 * jac + torch.eye(d)[None, None, :, :]

        denoising_posterior_precision = m**2 / std**2 + torch.inverse(cov0)

        return denoising_posterior_precision


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
