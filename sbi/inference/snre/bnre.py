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
        regularization_strength = 100,
    ):

        r"""Balenced neural ratio estimation (BNRE)[1].

        [1] Delaunoy, A., Hermans, J., Rozet, F., Wehenkel, A., & Louppe, G.. 
        Towards Reliable Simulation-Based Inference with Balanced Neural Ratio Estimation. 
        NeurIPS 2022. https://arxiv.org/abs/2208.13624

        Args:
            prior: A probability distribution that expresses prior knowledge about the
                parameters, e.g. which ranges are meaningful for them. If `None`, the
                prior must be passed to `.build_posterior()`.
            classifier: Classifier trained to approximate likelihood ratios. If it is
                a string, use a pre-configured network of the provided type (one of
                linear, mlp, resnet). Alternatively, a function that builds a custom
                neural network can be provided. The function will be called with the
                first batch of simulations $(\theta, x)$, which can thus be used for shape
                inference and potentially for z-scoring. It needs to return a PyTorch
                `nn.Module` implementing the classifier.
            device: Training device, e.g., "cpu", "cuda" or "cuda:{0, 1, ...}".
            logging_level: Minimum severity of messages to log. One of the strings
                INFO, WARNING, DEBUG, ERROR and CRITICAL.
            summary_writer: A tensorboard `SummaryWriter` to control, among others, log
                file location (default is `<current working directory>/logs`.)
            show_progress_bars: Whether to show a progressbar during simulation and
                sampling.
            regularization_strength: The multiplicative coefficient applied to the 
                balancing regularizer ($\lambda$ in the paper)
        """

        self.regularization_strength = regularization_strength
        kwargs = del_entries(locals(), entries=("self", "__class__", "regularization_strength"))
        super().__init__(**kwargs)

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
        bce = nn.BCELoss()(likelihood, labels)

        # Balancing regularizer
        regularizer = (torch.sigmoid(logits[0::2]) + torch.sigmoid(logits[1::2]) - 1).mean().square()

        return bce + self.regularization_strength * regularizer