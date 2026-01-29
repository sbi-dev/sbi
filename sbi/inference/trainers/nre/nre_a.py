# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

from typing import Dict, Optional, Union

import torch
from torch import Tensor, nn, ones
from torch.distributions import Distribution

from sbi.inference.trainers._contracts import LossArgsNRE_A
from sbi.inference.trainers.nre.nre_base import (
    RatioEstimatorTrainer,
)
from sbi.neural_nets.estimators.base import ConditionalEstimatorBuilder
from sbi.neural_nets.ratio_estimators import RatioEstimator
from sbi.sbi_types import TensorBoardSummaryWriter
from sbi.utils.sbiutils import del_entries
from sbi.utils.torchutils import assert_all_finite


class NRE_A(RatioEstimatorTrainer):
    """AALR, here known as Neural Ratio Estimation algorithm (NRE-A) [1].

    [1] *Likelihood-free MCMC with Amortized Approximate Likelihood Ratios*, Hermans
        et al., ICML 2020, https://arxiv.org/abs/1903.04057
    """

    def __init__(
        self,
        prior: Optional[Distribution] = None,
        classifier: Union[str, ConditionalEstimatorBuilder[RatioEstimator]] = "resnet",
        device: str = "cpu",
        logging_level: Union[int, str] = "warning",
        summary_writer: Optional[TensorBoardSummaryWriter] = None,
        show_progress_bars: bool = True,
    ):
        r"""Initialize NRE_A.

        Args:
            prior: A probability distribution that expresses prior knowledge about the
                parameters, e.g. which ranges are meaningful for them. If `None`, the
                prior must be passed to `.build_posterior()`.
            classifier: Classifier trained to approximate likelihood ratios. If it is
                a string, use a pre-configured network of the provided type (one of
                linear, mlp, resnet), or a callable that implements the
                `ConditionalEstimatorBuilder` protocol. The callable will
                be called with the first batch of simulations (theta, x), which can
                thus be used for shape inference and potentially for z-scoring. It
                returns a `RatioEstimator`.
            device: Training device, e.g., "cpu", "cuda" or "cuda:{0, 1, ...}".
            logging_level: Minimum severity of messages to log. One of the strings
                INFO, WARNING, DEBUG, ERROR and CRITICAL.
            summary_writer: A tensorboard `SummaryWriter` to control, among others, log
                file location (default is `<current working directory>/logs`.)
            show_progress_bars: Whether to show a progressbar during simulation and
                sampling.
        """

        kwargs = del_entries(locals(), entries=("self", "__class__"))
        super().__init__(**kwargs)

    def train(
        self,
        training_batch_size: int = 200,
        learning_rate: float = 5e-4,
        validation_fraction: float = 0.1,
        stop_after_epochs: int = 20,
        max_num_epochs: int = 2**31 - 1,
        clip_max_norm: Optional[float] = 5.0,
        resume_training: bool = False,
        discard_prior_samples: bool = False,
        retrain_from_scratch: bool = False,
        show_train_summary: bool = False,
        dataloader_kwargs: Optional[Dict] = None,
        loss_kwargs: Optional[LossArgsNRE_A] = None,
    ) -> RatioEstimator:
        r"""Return classifier that approximates the ratio $p(\theta,x)/p(\theta)p(x)$.

        Args:
            training_batch_size: Training batch size.
            learning_rate: Learning rate for Adam optimizer.
            validation_fraction: The fraction of data to use for validation.
            stop_after_epochs: The number of epochs to wait for improvement on the
                validation set before terminating training.
            max_num_epochs: Maximum number of epochs to run. If reached, we stop
                training even when the validation loss is still decreasing. Otherwise,
                we train until validation loss increases (see also `stop_after_epochs`).
            clip_max_norm: Value at which to clip the total gradient norm in order to
                prevent exploding gradients. Use None for no clipping.
            resume_training: Can be used in case training time is limited, e.g. on a
                cluster. If `True`, the split between train and validation set, the
                optimizer, the number of epochs, and the best validation log-prob will
                be restored from the last time `.train()` was called.
            discard_prior_samples: Whether to discard samples simulated in round 1, i.e.
                from the prior. Training may be sped up by ignoring such less targeted
                samples.
            retrain_from_scratch: Whether to retrain the conditional density
                estimator for the posterior from scratch each round.
            show_train_summary: Whether to print the number of epochs and validation
                loss and leakage after the training.
            dataloader_kwargs: Additional or updated kwargs to be passed to the training
                and validation dataloaders (like, e.g., a collate_fn)
            loss_kwargs: Additional or updated kwargs to be passed to the self._loss fn.

        Returns:
            Classifier that approximates the ratio $p(\theta,x)/p(\theta)p(x)$.
        """

        kwargs = del_entries(locals(), entries=("self", "__class__"))

        if loss_kwargs is None:
            kwargs["loss_kwargs"] = LossArgsNRE_A()
        elif not issubclass(type(loss_kwargs), LossArgsNRE_A):
            raise TypeError(
                "Expected loss_kwargs to be a subclass of LossArgsNRE_A,"
                f" but got {type(loss_kwargs)}"
            )

        return super().train(**kwargs)

    def _loss(self, theta: Tensor, x: Tensor, num_atoms: int) -> Tensor:
        """Returns the binary cross-entropy loss for the trained classifier.

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
        labels = ones(2 * batch_size, device=self._device)  # two atoms
        labels[1::2] = 0.0

        # Binary cross entropy to learn the likelihood (AALR-specific)
        loss = nn.BCELoss()(likelihood, labels)
        assert_all_finite(loss, "NRE-A loss")
        return loss
