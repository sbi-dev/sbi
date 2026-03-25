# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

from typing import Protocol, Tuple

import torch
from torch import Tensor, eye, nn, ones

from sbi.neural_nets.ratio_estimators import RatioEstimator
from sbi.utils.torchutils import assert_all_finite, repeat_rows


class NRELossStrategy(Protocol):
    """Protocol defining the interface for all Neural Ratio Estimation loss strategies.

    A strategy must implement a forward pass (__call__) mapping parameters (theta),
    observations (x), and algorithm-specific keyword arguments (like num_atoms) to a
    scalar loss tensor.
    """
    def __call__(
        self,
        neural_net: RatioEstimator,
        device: str,
        theta: Tensor,
        x: Tensor,
        **kwargs
    ) -> Tensor: ...


def _classifier_logits(
    neural_net: RatioEstimator, theta: Tensor, x: Tensor, num_atoms: int
) -> Tensor:
    """Return logits obtained through classifier forward pass.

    The logits are obtained from atomic sets of (theta,x) pairs.
    """
    batch_size = theta.shape[0]
    repeated_x = repeat_rows(x, num_atoms)

    # Choose `1` or `num_atoms - 1` thetas from the rest of the batch for each x.
    probs = (
        ones(batch_size, batch_size, device=theta.device)
        * (1 - eye(batch_size, device=theta.device))
        / (batch_size - 1)
    )

    choices = torch.multinomial(probs, num_samples=num_atoms - 1, replacement=False)

    contrasting_theta = theta[choices]

    atomic_theta = torch.cat((theta[:, None, :], contrasting_theta), dim=1).reshape(
        batch_size * num_atoms, -1
    )

    return neural_net(atomic_theta, repeated_x)


class AALRLoss:
    """Neural Ratio Estimation (NRE-A / AALR) loss strategy.

    Returns the binary cross-entropy loss for the trained classifier.
    """
    def __call__(self, neural_net: RatioEstimator, device: str, theta: Tensor, x: Tensor, num_atoms: int, **kwargs) -> Tensor:
        assert theta.shape[0] == x.shape[0], "Batch sizes for theta and x must match."
        batch_size = theta.shape[0]

        logits = _classifier_logits(neural_net, theta, x, num_atoms)
        likelihood = torch.sigmoid(logits).squeeze()

        # Alternating pairs where there is one sampled from the joint and one
        # sampled from the marginals. The first element is sampled from the
        # joint p(theta, x) and is labelled 1. The second element is sampled
        # from the marginals p(theta)p(x) and is labelled 0. And so on.
        labels = ones(2 * batch_size, device=device)  # two atoms
        labels[1::2] = 0.0

        # Binary cross entropy to learn the likelihood (AALR-specific)
        loss = nn.BCELoss()(likelihood, labels)
        assert_all_finite(loss, "NRE-A loss")
        return loss


class SRELoss:
    """Neural Ratio Estimation (NRE-B / SRE) loss strategy.

    Returns cross-entropy (via softmax activation) loss for 1-out-of-`num_atoms`
    classification.
    """
    def __call__(self, neural_net: RatioEstimator, device: str, theta: Tensor, x: Tensor, num_atoms: int, **kwargs) -> Tensor:
        assert theta.shape[0] == x.shape[0], "Batch sizes for theta and x must match."
        batch_size = theta.shape[0]
        logits = _classifier_logits(neural_net, theta, x, num_atoms)

        # For 1-out-of-`num_atoms` classification each datapoint consists
        # of `num_atoms` points, with one of them being the correct one.
        # We have a batch of `batch_size` such datapoints.
        logits = logits.reshape(batch_size, num_atoms)

        # Index 0 is the theta-x-pair sampled from the joint p(theta,x) and hence the
        # "correct" one for the 1-out-of-N classification.
        log_prob = logits[:, 0] - torch.logsumexp(logits, dim=-1)

        loss = -torch.mean(log_prob)
        assert_all_finite(loss, "NRE-B loss")
        return loss


class CNRELoss:
    """Neural Ratio Estimation (NRE-C / CNRE) loss strategy.

    Returns cross-entropy loss (via 'multi-class sigmoid' activation) for
    1-out-of-`K + 1` classification.
    """
    @staticmethod
    def _get_prior_probs_marginal_and_joint(gamma: float) -> Tuple[float, float]:
        r"""Return a tuple (p_marginal, p_joint) where `p_marginal := `$p_0$,
        `p_joint := `$p_K \cdot K$.
        """
        p_joint = gamma / (1 + gamma)
        p_marginal = 1 / (1 + gamma)
        return p_marginal, p_joint

    def __call__(
        self,
        neural_net: RatioEstimator,
        device: str,
        theta: Tensor,
        x: Tensor,
        num_atoms: int,
        gamma: float,
        **kwargs
    ) -> Tensor:
        # Reminder: K = num_classes
        num_classes = num_atoms - 1
        assert num_classes >= 1, f"num_classes = {num_classes} must be greater than 1."

        assert theta.shape[0] == x.shape[0], "Batch sizes for theta and x must match."
        batch_size = theta.shape[0]

        logits_marginal = _classifier_logits(
            neural_net, theta, x, num_classes + 1
        ).reshape(batch_size, num_classes + 1)

        logits_joint = _classifier_logits(
            neural_net, theta, x, num_classes
        ).reshape(batch_size, num_classes)

        dtype = logits_marginal.dtype

        # Index 0 is the theta-x-pair sampled from the joint p(theta,x) and hence
        # we remove the jointly drawn sample from the logits_marginal
        logits_marginal = logits_marginal[:, 1:]

        # To use logsumexp, we extend the denominator logits with loggamma
        loggamma = torch.tensor(gamma, dtype=dtype, device=device).log()
        logK = torch.tensor(num_classes, dtype=dtype, device=device).log()
        denominator_marginal = torch.concat(
            [loggamma + logits_marginal, logK.expand((batch_size, 1))],
            dim=-1,
        )
        denominator_joint = torch.concat(
            [loggamma + logits_joint, logK.expand((batch_size, 1))],
            dim=-1,
        )

        # Compute the contributions to the loss from each term in the classification.
        log_prob_marginal = logK - torch.logsumexp(denominator_marginal, dim=-1)
        log_prob_joint = (
            loggamma + logits_joint[:, 0] - torch.logsumexp(denominator_joint, dim=-1)
        )

        # relative weights. p_marginal := p_0, and p_joint := p_K * K from the notation.
        p_marginal, p_joint = CNRELoss._get_prior_probs_marginal_and_joint(gamma)

        loss = -torch.mean(p_marginal * log_prob_marginal + p_joint * log_prob_joint)
        assert_all_finite(loss, "NRE-C loss")
        return loss


class BNRELoss:
    """Balanced Neural Ratio Estimation (BNRE) loss strategy.

    A variation of NRE-A that adds a balancing regularizer to the 
    binary cross-entropy loss.
    """
    def __call__(
        self,
        neural_net: RatioEstimator,
        device: str,
        theta: Tensor,
        x: Tensor,
        num_atoms: int,
        regularization_strength: float,
        **kwargs
    ) -> Tensor:
        assert theta.shape[0] == x.shape[0], "Batch sizes for theta and x must match."
        batch_size = theta.shape[0]

        logits = _classifier_logits(neural_net, theta, x, num_atoms)
        likelihood = torch.sigmoid(logits).squeeze()

        # Alternating pairs where there is one sampled from the joint and one
        # sampled from the marginals. The first element is sampled from the
        # joint p(theta, x) and is labelled 1. The second element is sampled
        # from the marginals p(theta)p(x) and is labelled 0. And so on.
        labels = ones(2 * batch_size, device=device)  # two atoms
        labels[1::2] = 0.0

        # Binary cross entropy to learn the likelihood (AALR-specific)
        bce = nn.BCELoss()(likelihood, labels)

        # Balancing regularizer
        regularizer = (
            (torch.sigmoid(logits[0::2]) + torch.sigmoid(logits[1::2]) - 1)
            .mean()
            .square()
        )

        loss = bce + regularization_strength * regularizer
        assert_all_finite(loss, "BNRE loss")
        return loss
