from __future__ import annotations

from typing import Optional
from sbi.types import OneOrMore

import torch
from torch import nn, ones, Tensor

from sbi.inference.posterior import NeuralPosterior
from sbi.inference.snre.snre_base import RatioEstimator
from sbi.utils import del_entries


class SNRE_A(RatioEstimator):
    """AALR[1], here known as SNRE_A.

    [1] _Likelihood-free MCMC with Amortized Approximate Likelihood Ratios_, Hermans et
        al., ICML 2020, https://arxiv.org/abs/1903.04057
    """

    def __call__(
        self,
        num_rounds: int,
        num_simulations_per_round: OneOrMore[int],
        x_o: Optional[Tensor] = None,
        batch_size: int = 50,
        learning_rate: float = 5e-4,
        validation_fraction: float = 0.1,
        stop_after_epochs: int = 20,
        max_num_epochs: Optional[int] = None,
        clip_max_norm: Optional[float] = 5.0,
        exclude_invalid_x: bool = True,
        discard_prior_samples: bool = False,
        retrain_from_scratch_each_round: bool = False,
    ) -> NeuralPosterior:

        # AALR is defined for `num_atoms=2`.
        # Proxy to `super().__call__` to ensure right parameter.
        kwargs = del_entries(locals(), entries=("self", "__class__"))
        return super().__call__(**kwargs, num_atoms=2)

    def _loss(self, theta: Tensor, x: Tensor, num_atoms: int) -> Tensor:
        """
        Returns the binary cross-entropy loss for the trained classifier.

        The classifier takes as input a $(\theta,x)$ pair. It is trained to predict 1
        if the pair was sampled from the joint $p(\theta,x)$, and to predict 0 if the
        pair was sampled from the marginals $p(\theta)p(x)$.
        """

        assert theta.shape[0] == x.shape[0], "Batch sizes for theta and x must match."
        batch_size = theta.shape[0]

        logits = self._classifier_logits(theta, x, num_atoms)
        likelihood = torch.sigmoid(logits).squeeze()

        # Alternating pairs where there is one sampled from the joint and one
        # sampled from the marginals. The first element is sampled from the
        # joint p(theta, x) and is labelled 1. The second element is sampled
        # from the marginals p(theta)p(x) and is labelled 0. And so on.
        labels = ones(2 * batch_size)  # two atoms
        labels[1::2] = 0.0

        # Binary cross entropy to learn the likelihood (AALR-specific)
        return nn.BCELoss()(likelihood, labels)
