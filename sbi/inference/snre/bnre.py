from typing import Callable, Dict, Optional, Union

import torch
from torch import Tensor, nn, ones
from torch.distributions import Distribution

from sbi.inference.snre.snre_a import SNRE_A
from sbi.types import TensorboardSummaryWriter
from sbi.utils import del_entries


class BNRE(SNRE_A):
    def __init__(
        self,
        prior: Optional[Distribution] = None,
        classifier: Union[str, Callable] = "resnet",
        device: str = "cpu",
        logging_level: Union[int, str] = "warning",
        summary_writer: Optional[TensorboardSummaryWriter] = None,
        show_progress_bars: bool = True,
    ):

        r"""Balanced neural ratio estimation (BNRE)[1]. BNRE is a variation of NRE
        aiming to produce more conservative posterior approximations

        [1] Delaunoy, A., Hermans, J., Rozet, F., Wehenkel, A., & Louppe, G..
        Towards Reliable Simulation-Based Inference with Balanced Neural Ratio
        Estimation.
        NeurIPS 2022. https://arxiv.org/abs/2208.13624

        Args:
            prior: A probability distribution that expresses prior knowledge about the
                parameters, e.g. which ranges are meaningful for them. If `None`, the
                prior must be passed to `.build_posterior()`.
            classifier: Classifier trained to approximate likelihood ratios. If it is
                a string, use a pre-configured network of the provided type (one of
                linear, mlp, resnet). Alternatively, a function that builds a custom
                neural network can be provided. The function will be called with the
                first batch of simulations $(\theta, x)$, which can thus be used for
                shape inference and potentially for z-scoring. It needs to return a
                PyTorch `nn.Module` implementing the classifier.
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
        regularization_strength: float = 100.0,
        training_batch_size: int = 50,
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
    ) -> nn.Module:
        r"""Return classifier that approximates the ratio $p(\theta,x)/p(\theta)p(x)$.
        Args:

            regularization_strength: The multiplicative coefficient applied to the
                balancing regularizer ($\lambda$).
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
            exclude_invalid_x: Whether to exclude simulation outputs `x=NaN` or `x=±∞`
                during training. Expect errors, silent or explicit, when `False`.
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
        Returns:
            Classifier that approximates the ratio $p(\theta,x)/p(\theta)p(x)$.
        """
        kwargs = del_entries(locals(), entries=("self", "__class__"))
        kwargs["loss_kwargs"] = {
            "regularization_strength": kwargs.pop("regularization_strength")
        }
        return super().train(**kwargs)

    def _loss(
        self, theta: Tensor, x: Tensor, num_atoms: int, regularization_strength: float
    ) -> Tensor:
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
        bce = nn.BCELoss()(likelihood, labels)

        # Balancing regularizer
        regularizer = (
            (torch.sigmoid(logits[0::2]) + torch.sigmoid(logits[1::2]) - 1)
            .mean()
            .square()
        )

        return bce + regularization_strength * regularizer
