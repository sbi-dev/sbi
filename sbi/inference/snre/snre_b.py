from __future__ import annotations

import torch
from torch import Tensor

from sbi.inference.snre.snre_base import RatioEstimator


class SNRE_B(RatioEstimator):
    def _loss(self, theta: Tensor, x: Tensor, num_atoms: int) -> Tensor:
        """
        Return cross-entropy loss for 1-out-of-`num_atoms` classification.

        The classifier takes as input `num_atoms` $(\theta,x)$ pairs. Out of these
        pairs, one pair was sampled from the joint $p(\theta,x)$ and all others from the
        marginals $p(\theta)p(x)$. The classifier is trained to predict which of the
        pairs was sampled from the joint $p(\theta,x)$.
        """

        assert theta.shape[0] == x.shape[0], "Batch sizes for theta and x must match."
        batch_size = theta.shape[0]
        logits = self._classifier_logits(theta, x, num_atoms)

        # For 1-out-of-`num_atoms` classification each datapoint consists
        # of `num_atoms` points, with one of them being the correct one.
        # We have a batch of `batch_size` such datapoints.
        logits = logits.reshape(batch_size, num_atoms)

        # Index 0 is the theta-x-pair sampled from the joint p(theta,x) and hence the
        # "correct" one for the 1-out-of-N classification.
        log_prob = logits[:, 0] - torch.logsumexp(logits, dim=-1)

        return -torch.mean(log_prob)
