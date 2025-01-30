from typing import Callable, Optional, Union
from sbi.neural_nets.estimators.score_estimator import ConditionalScoreEstimator
from torch.distributions import Distribution

from abc import abstractmethod
import torch
from torch import Tensor

from sbi.utils.torchutils import ensure_theta_batched


class ScoreFnIID:
    def __init__(
        self,
        score_estimator: ConditionalScoreEstimator,
        prior: Distribution,
        device: str = "cpu",
    ):
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
        theta = theta.detach().clone().requires_grad_(True)
        prior_log_prob = self.prior.log_prob(theta)
        prior_score = torch.autograd.grad(
            prior_log_prob,
            theta,
            grad_outputs=torch.ones_like(prior_log_prob),
            create_graph=True,
        )[0]
        return prior_score


class FNPEScoreFn(ScoreFnIID):
    def __init__(
        self,
        score_estimator: ConditionalScoreEstimator,
        prior: Distribution,
        device: str = "cpu",
        prior_score_weight: Optional[Callable[[Tensor], Tensor]] = None,
    ):
        r"""
        Initializes the FNPEScoreFn class.

        Paper: Compositional Score Modeling for Simulation-Based Inference
        - https://arxiv.org/abs/2106.05399

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

        # NOTE: If this always works needs to be testd

        N = inputs.shape[-2]

        # Compute the per-sample score
        inputs = ensure_theta_batched(inputs)
        base_score = self.score_estimator(inputs, conditions, time)

        # Compute the prior score
        prior_score = self.prior_score_fn(inputs)
        prior_score = self.prior_score_weight_fn(time) * prior_score

        # Accumulate
        score = (1 - N) * prior_score + base_score.sum(-2, keepdim=True)

        return score