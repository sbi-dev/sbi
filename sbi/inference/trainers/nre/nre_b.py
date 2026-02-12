# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

from typing import Dict, Optional, Union

import torch
from torch import Tensor
from torch.distributions import Distribution
from torch.utils.tensorboard.writer import SummaryWriter

from sbi.inference.trainers._contracts import LossArgsNRE
from sbi.inference.trainers.nre.nre_base import (
    RatioEstimatorTrainer,
)
from sbi.neural_nets.estimators.base import ConditionalEstimatorBuilder
from sbi.neural_nets.ratio_estimators import RatioEstimator
from sbi.sbi_types import Tracker
from sbi.utils.sbiutils import del_entries
from sbi.utils.torchutils import assert_all_finite


class NRE_B(RatioEstimatorTrainer):
    r"""Neural Ratio Estimation (NRE-B) as in Durkan et al. (2020) [1].

    NRE-B trains a classifier using contrastive learning to estimate the ratio
    $r(\theta, x)$. It contrasts one joint sample against multiple marginal samples
    using a multi-class classification loss.

    [1] *On Contrastive Learning for Likelihood-free Inference*, Durkan et al.,
        ICML 2020, https://arxiv.org/pdf/2002.03712

    Example:
    --------

    .. code-block:: python

        import torch
        from sbi.inference import NRE_B
        from sbi.utils import BoxUniform

        prior = BoxUniform(low=torch.zeros(3), high=torch.ones(3))
        theta = prior.sample((100,))
        x = torch.randn(100, 10)

        inference = NRE_B(prior=prior)
        ratio_estimator = inference.append_simulations(theta, x).train()
        posterior = inference.build_posterior(ratio_estimator)

        samples = posterior.sample((100,), x=x[0])
    """

    def __init__(
        self,
        prior: Optional[Distribution] = None,
        classifier: Union[str, ConditionalEstimatorBuilder[RatioEstimator]] = "resnet",
        device: str = "cpu",
        logging_level: Union[int, str] = "warning",
        summary_writer: Optional[SummaryWriter] = None,
        tracker: Optional[Tracker] = None,
        show_progress_bars: bool = True,
    ):
        r"""Initialize NRE_B.

        Args:
            prior: A probability distribution that expresses prior knowledge about the
                parameters, e.g. which ranges are meaningful for them. If `None`, the
                prior must be passed to `.build_posterior()`.
            classifier: Classifier trained to approximate likelihood ratios. If it is
                a string, use a pre-configured network of the provided type (one of
                linear, mlp, resnet), or a callable that implements the
                `ConditionalEstimatorBuilder` protocol. The callable will
                be called with the first batch of simulations (theta, x), which can thus
                be used for shape inference and potentially for z-scoring. It returns a
                `RatioEstimator`.
            device: Training device, e.g., "cpu", "cuda" or "cuda:{0, 1, ...}".
            logging_level: Minimum severity of messages to log. One of the strings
                INFO, WARNING, DEBUG, ERROR and CRITICAL.
            summary_writer: Deprecated alias for the TensorBoard summary writer.
                Use ``tracker`` instead.
            tracker: Tracking adapter used to log training metrics. If None, a
                TensorBoard tracker is used with a default log directory.
            show_progress_bars: Whether to show a progressbar during simulation and
                sampling.
        """

        kwargs = del_entries(locals(), entries=("self", "__class__"))
        super().__init__(**kwargs)

    def train(
        self,
        num_atoms: int = 10,
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
    ) -> RatioEstimator:
        r"""Return classifier that approximates the ratio $p(\theta,x)/p(\theta)p(x)$.

        Args:
            num_atoms: Number of atoms to use for classification.
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

        Returns:
            Classifier that approximates the ratio $p(\theta,x)/p(\theta)p(x)$.
        """
        kwargs = del_entries(locals(), entries=("self", "__class__"))
        kwargs["loss_kwargs"] = LossArgsNRE(num_atoms=kwargs.pop("num_atoms"))

        return super().train(**kwargs)

    def _loss(self, theta: Tensor, x: Tensor, num_atoms: int) -> Tensor:
        r"""Return cross-entropy (via softmax activation) loss for 1-out-of-`num_atoms`
        classification.

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

        loss = -torch.mean(log_prob)
        assert_all_finite(loss, "NRE-B loss")
        return loss
