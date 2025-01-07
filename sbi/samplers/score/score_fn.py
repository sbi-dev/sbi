import re
import torch
from torch import Tensor
from torch.distributions import Distribution
from typing import Optional, Callable
from abc import abstractmethod

from sbi.neural_nets.estimators import ConditionalScoreEstimator
from sbi.utils.torchutils import ensure_theta_batched
from sbi.samplers.score.denoise_and_marginalize import denoise, marginalize


def mv_diag_or_dense(A_diag_or_dense: Tensor, b: Tensor) -> Tensor:
    """Dot product of a diagonal matrix and a dense matrix.

    Args:
        A_diag_or_dense (Tensor): Diagonal matrix.
        b (Tensor): Dense matrix.

    Returns:
        Tensor: Dot product.
    """
    if A_diag_or_dense.ndim == 1:
        return A_diag_or_dense * b
    else:
        return torch.matmul(A_diag_or_dense, b)


def solve_diag_or_dense(A_diag_or_dense: Tensor, b: Tensor) -> Tensor:
    """Solve a linear system with a diagonal matrix or a dense matrix.

    Args:
        A_diag_or_dense (Tensor): Diagonal matrix or dense matrix.
        b (Tensor): Dense matrix.

    Returns:
        Tensor: Solution to the linear system.
    """
    if A_diag_or_dense.ndim == 1:
        return b / A_diag_or_dense
    else:
        return torch.linalg.solve(A_diag_or_dense, b)


def add_diag_or_dense(A_diag_or_dense: Tensor, B_diag_or_dense: Tensor) -> Tensor:
    """Add two diagonal matrices or two dense matrices.

    Args:
        A_diag_or_dense (Tensor): Diagonal matrix or dense matrix.
        B_diag_or_dense (Tensor): Diagonal matrix or dense matrix.

    Returns:
        Tensor: Sum of the matrices.
    """
    if (
        A_diag_or_dense.ndim == 1
        and B_diag_or_dense.ndim == 1
        or A_diag_or_dense.ndim == 2
        and B_diag_or_dense.ndim == 2
    ):
        return A_diag_or_dense + B_diag_or_dense
    elif A_diag_or_dense.ndim == 2 and B_diag_or_dense.ndim == 1:
        return A_diag_or_dense + torch.diag(B_diag_or_dense)
    elif A_diag_or_dense.ndim == 1 and B_diag_or_dense.ndim == 2:
        return torch.diag(A_diag_or_dense) + B_diag_or_dense
    else:
        raise ValueError("Incompatible dimensions for addition")


class ScoreFnIID:
    def __init__(
        self,
        score_estimator: ConditionalScoreEstimator,
        prior: Distribution,
        device: str = "cpu",
    ):
        r"""Initializes the ScoreFnIID class.

        Args:
            score_estimator: The neural network modelling the score.
            prior: The prior distribution.
            device: The device on which to evaluate the potential.
        """
        self.score_estimator = score_estimator.to(device)
        self.prior = prior
        self.score_estimator.eval()

    @abstractmethod
    def __call__(
        self,
        theta: Tensor,
    ) -> Tensor:
        r"""Abstract method to be implemented by subclasses.

        Args:
            theta: The parameters at which to evaluate the potential.

        Returns:
            The score function.
        """
        pass

    def prior_score_fn(self, theta: Tensor) -> Tensor:
        r"""Computes the score of the prior distribution.

        Args:
            theta: The parameters at which to evaluate the prior score.

        Returns:
            The prior score.
        """
        theta = theta.detach().clone().requires_grad_(True)
        prior_log_prob = self.prior.log_prob(theta)
        prior_score = torch.autograd.grad(
            prior_log_prob,
            theta,
            grad_outputs=torch.ones_like(prior_log_prob),
            create_graph=True,
        )[0]
        return prior_score


class FNPEScoreFN(ScoreFnIID):
    def __init__(
        self,
        score_estimator: ConditionalScoreEstimator,
        prior: Distribution,
        device: str = "cpu",
        prior_score_weight: Optional[Callable] = None,
    ):
        r"""Initializes the FNPEScoreFN class.

        Args:
            score_estimator: The neural network modelling the score.
            prior: The prior distribution.
            device: The device on which to evaluate the potential.
            prior_score_weight: Optional callable to weight the prior score.
        """
        super().__init__(score_estimator, prior, device)

        if prior_score_weight is None:
            t_max = score_estimator.t_max

            def prior_score_weight_fn(t):
                return (t_max - t) / t_max

        self.prior_score_weight_fn = prior_score_weight_fn

    def __call__(
        self,
        theta: Tensor,
        time: Optional[Tensor] = None,
    ) -> Tensor:
        r"""Returns the score function for score-based methods.

        Args:
            theta: The parameters at which to evaluate the potential.
            time: Optional time tensor.

        Returns:
            The score function.
        """
        if time is None:
            time = torch.tensor([self.score_estimator.t_min])

        N = theta.shape[0]

        # Compute the per-sample score
        theta = ensure_theta_batched(theta)
        base_score = self.score_estimator(theta)

        # Compute the prior score
        prior_score = self.prior_score_fn(theta)
        prior_score = self.prior_score_weight_fn(time) * prior_score

        # Accumulate
        score = (1 - N) * prior_score + base_score.sum(0)

        return score


class AbstractGaussCorrectedScoreFn(ScoreFnIID):
    def __init__(
        self,
        score_estimator: ConditionalScoreEstimator,
        prior: Distribution,
    ) -> None:
        r"""Initializes the AbstractGaussCorrectedScoreFn class.

        Args:
            score_estimator: The neural network modelling the score.
            prior: The prior distribution.
        """
        super().__init__(score_estimator, prior)

    @abstractmethod
    def posterior_precision_est_fn(self, x_o: Tensor) -> Tensor:
        r"""Abstract method to estimate the posterior precision.

        Args:
            x_o: Observed data.

        Returns:
            Estimated posterior precision.
        """
        pass

    def denoising_prior(self, m: Tensor, std: Tensor, theta: Tensor) -> Distribution:
        r"""Denoise the prior distribution.

        Args:
            m: Mean tensor.
            std: Standard deviation tensor.
            theta: Parameters tensor.

        Returns:
            Denoised prior distribution.
        """
        return denoise(self.prior, m, std, theta)

    def marginal_prior(self, a: Tensor, theta: Tensor) -> Distribution:
        r"""Compute the marginal prior distribution.

        Args:
            a: Auxiliary variable tensor.
            theta: Parameters tensor.

        Returns:
            Marginal prior distribution.
        """
        m = self.score_estimator.mean_t_fn(a)
        std = self.score_estimator.std_t_fn(a)
        return marginalize(self.prior, m, std)

    def marginal_posterior_precision_est_fn(
        self, a: Tensor, theta: Tensor, x_o: Tensor, N: int
    ) -> Tensor:
        r"""Estimates the marginal posterior precision.

        Args:
            a: Auxiliary variable tensor.
            theta: Parameters tensor.
            x_o: Observed data.
            N: Number of samples.

        Returns:
            Estimated marginal posterior precision.
        """
        precisions_posteriors = self.posterior_precision_est_fn(x_o)
        precisions_posteriors = torch.atleast_2d(precisions_posteriors)

        # If one constant precision is given, tile it
        if precisions_posteriors.shape[0] < N:
            precisions_posteriors = precisions_posteriors.repeat(N, 1)

        # Denoising posterior via Bayes rule
        m = self.score_estimator.mean_t_fn(a)
        std = self.score_estimator.std_t_fn(a)

        if precisions_posteriors.ndim == 3:
            I = torch.eye(precisions_posteriors.shape[-1])
        else:
            I = torch.ones_like(precisions_posteriors)

        marginal_precisions = m**2 / std**2 * I + precisions_posteriors
        return marginal_precisions

    def marginal_prior_score_fn(self, a: Tensor, theta: Tensor) -> Tensor:
        r"""Computes the score of the marginal prior distribution.

        Args:
            a: Auxiliary variable tensor.
            theta: Parameters tensor.

        Returns:
            Marginal prior score.
        """
        p = self.marginal_prior(a, theta)
        log_p = p.log_prob(theta)
        return torch.autograd.grad(
            log_p,
            theta,
            grad_outputs=torch.ones_like(log_p),
            create_graph=True,
        )[0]

    def marginal_denoising_prior_precision_fn(self, a: Tensor, theta: Tensor) -> Tensor:
        r"""Computes the precision of the marginal denoising prior.

        Args:
            a: Auxiliary variable tensor.
            theta: Parameters tensor.

        Returns:
            Marginal denoising prior precision.
        """
        m = self.score_estimator.mean_t_fn(a)
        std = self.score_estimator.std_t_fn(a)
        p_denoise = self.denoising_prior(m, std, theta)
        return 1 / p_denoise.variance

    def __call__(self, a: Tensor, theta: Tensor, x_o: Tensor, **kwargs) -> Tensor:
        r"""Returns the corrected score function.

        Args:
            a: Auxiliary variable tensor.
            theta: Parameters tensor.
            x_o: Observed data.

        Returns:
            Corrected score function.
        """
        base_score = self.score_estimator(a, theta, x_o, **kwargs)
        prior_score = self.marginal_prior_score_fn(a, theta)
        N = base_score.shape[0]

        # Marginal prior precision
        prior_precision = self.marginal_denoising_prior_precision_fn(a, theta)
        # Marginal posterior variance estimates
        posterior_precisions = self.marginal_posterior_precision_est_fn(
            a, theta, x_o, N
        )

        # Total precision
        term1 = (1 - N) * prior_precision
        term2 = torch.sum(posterior_precisions, dim=0)
        Lam = add_diag_or_dense(term1, term2)

        # Weighted scores
        weighted_prior_score = mv_diag_or_dense(prior_precision, prior_score)
        weighted_posterior_scores = torch.stack([
            mv_diag_or_dense(p, base_score[i])
            for i, p in enumerate(posterior_precisions)
        ])

        # Accumulate the scores
        score = (1 - N) * weighted_prior_score + torch.sum(
            weighted_posterior_scores, dim=0
        )

        # Solve the linear system
        score = solve_diag_or_dense(Lam, score)

        return score


class GaussCorrectedScoreFn(AbstractGaussCorrectedScoreFn):
    def __init__(
        self,
        score_estimator: ConditionalScoreEstimator,
        prior: Distribution,
        posterior_precision: Tensor,
    ) -> None:
        r"""Initializes the GaussCorrectedScoreFn class.

        Args:
            score_estimator: The neural network modelling the score.
            prior: The prior distribution.
        """
        super().__init__(score_estimator, prior)
        self.posterior_precision = posterior_precision

    def posterior_precision_est_fn(self, x_o: Tensor) -> Tensor:
        r"""Estimates the posterior precision.

        Args:
            x_o: Observed data.

        Returns:
            Estimated posterior precision.
        """
        return self.posterior_precision