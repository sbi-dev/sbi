from __future__ import annotations

import torch
from torch import Tensor

from sbi.inference.snre.snre_base import RatioEstimator


class SNRE_B(RatioEstimator):
    def _loss(
        self, theta: Tensor, x: Tensor, clipped_batch_size: int, num_atoms: int
    ) -> Tensor:
        """Return cross-entropy loss for 1-out-of-`num_atoms` classification."""

        logits = self._classifier_logits(theta, x, clipped_batch_size, num_atoms)

        # For 1-out-of-`num_atoms` classification each datapoint consists
        # of `num_atoms` points, with one of them being the correct one.
        # We have a batch of `clipped_batch_size` such datapoints.
        logits = logits.reshape(clipped_batch_size, num_atoms)

        # Index 0 is the theta sampled from the joint.
        # The first is the correct one for the 1-out-of-N classification.
        log_prob = logits[:, 0] - torch.logsumexp(logits, dim=-1)

        return -torch.mean(log_prob)
