from __future__ import annotations

from typing import Optional
from sbi.types import OneOrMore

import torch
from torch import nn, ones, Tensor

from sbi.inference.posteriors.sbi_posterior import NeuralPosterior
from sbi.inference.snre.snre_base import RatioEstimator
from sbi.utils import del_entries


class SNRE_A(RatioEstimator):
    "AALR, here known as SRE_A."

    def __call__(
        self,
        num_rounds: int,
        num_simulations_per_round: OneOrMore[int],
        x_o: Optional[Tensor] = None,
        batch_size: int = 100,
        learning_rate: float = 5e-4,
        validation_fraction: float = 0.1,
        stop_after_epochs: int = 20,
        max_num_epochs: Optional[int] = None,
        clip_max_norm: Optional[float] = 5.0,
    ) -> NeuralPosterior:

        # AALR only supports exactly `num_atoms=2`.
        kwargs = del_entries(locals(), entries=("self", "__class__"))
        return super().__call__(**kwargs, num_atoms=2)

    def _loss(self, theta, x, clipped_batch_size, num_atoms):

        logits = self._classifier_logits(theta, x, clipped_batch_size, num_atoms)
        likelihood = torch.sigmoid(logits).squeeze()

        # Alternating pairs where there is one sampled from the joint and one
        # sampled from the marginals. The first element is sampled from the
        # joint p(theta, x) and is labelled 1. The second element is sampled
        # from the marginals p(theta)p(x) and is labelled 0. And so on.
        labels = ones(2 * clipped_batch_size)  # two atoms
        labels[1::2] = 0.0

        # Binary cross entropy to learn the likelihood (AALR-specific)
        return nn.BCELoss()(likelihood, labels)
