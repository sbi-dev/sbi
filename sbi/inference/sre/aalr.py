from __future__ import annotations

import torch
from torch import nn, ones

from sbi.inference.sre.re_base import RatioEstimationBase


class AALR(RatioEstimationBase):
    def _loss(self, theta, x, clipped_batch_size):

        logits = self._classifier_logits(theta, x, clipped_batch_size)
        likelihood = torch.sigmoid(logits).squeeze()

        # Alternating pairs where there is one sampled from the joint and one
        # sampled from the marginals. The first element is sampled from the
        # joint p(theta, x) and is labelled 1. The second element is sampled
        # from the marginals p(theta)p(x) and is labelled 0. And so on.
        labels = ones(2 * clipped_batch_size)  # two atoms
        labels[1::2] = 0.0

        # Binary cross entropy to learn the likelihood (AALR-specific)
        return nn.BCELoss()(likelihood, labels)
