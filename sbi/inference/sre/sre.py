from __future__ import annotations

import torch
from sbi.inference.sre.re_base import RatioEstimationBase


class SRE(RatioEstimationBase):
    def _loss(self, theta, x, clipped_batch_size):

        logits = self._classifier_logits(theta, x, clipped_batch_size).reshape(
            clipped_batch_size, self._used_atoms(clipped_batch_size)
        )

        # Index 0 is the theta sampled from the joint.
        log_prob = logits[:, 0] - torch.logsumexp(logits, dim=-1)

        return -torch.mean(log_prob)
