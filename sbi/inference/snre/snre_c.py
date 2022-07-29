from typing import Callable, Dict, Optional, Union

import torch
from torch import Tensor, nn
from torch.distributions import Distribution

from sbi.inference.snre.snre_base import RatioEstimator
from sbi.types import TensorboardSummaryWriter
from sbi.utils import del_entries, repeat_rows


class SNRE_C(RatioEstimator):
    def __init__(
        self,
        prior: Optional[Distribution] = None,
        classifier: Union[str, Callable] = "resnet",
        device: str = "cpu",
        logging_level: Union[int, str] = "warning",
        summary_writer: Optional[TensorboardSummaryWriter] = None,
        show_progress_bars: bool = True,
    ):
        r"""In progress SNRE_C.

        TODO SNRE_C

        Args:
            prior: A probability distribution that expresses prior knowledge about the
                parameters, e.g. which ranges are meaningful for them. If `None`, the
                prior must be passed to `.build_posterior()`.
            classifier: Classifier trained to approximate likelihood ratios. If it is
                a string, use a pre-configured network of the provided type (one of
                linear, mlp, resnet). Alternatively, a function that builds a custom
                neural network can be provided. The function will be called with the
                first batch of simulations (theta, x), which can thus be used for shape
                inference and potentially for z-scoring. It needs to return a PyTorch
                `nn.Module` implementing the classifier.
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
        K: int = 10,
        gamma: float = 1.0,
        training_batch_size: int = 50,
        learning_rate: float = 5e-4,
        validation_fraction: float = 0.1,
        stop_after_epochs: int = 20,
        max_num_epochs: int = 2**31 - 1,
        clip_max_norm: Optional[float] = 5.0,
        exclude_invalid_x: bool = True,
        resume_training: bool = False,
        discard_prior_samples: bool = False,
        retrain_from_scratch: bool = False,
        show_train_summary: bool = False,
        dataloader_kwargs: Optional[Dict] = None,
    ) -> nn.Module:
        r"""Return classifier that approximates the ratio $p(\theta,x)/p(\theta)p(x)$.

        Args:
            K: number of classes to draw from the joint distribution plus another drawn independently / marginally.
            gamma: sum of the prior weight given to the dependently drawn samples over the weight given to the independently drawn one.
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
        kwargs["num_atoms"] = kwargs.pop("K")
        kwargs["loss_kwargs"] = {"gamma": kwargs.pop("gamma")}
        return super().train(**kwargs)

    def _loss(
        self, 
        theta: Tensor, 
        x: Tensor, 
        K: int, 
        gamma: float
    ) -> torch.Tensor:
        r"""Return cross-entropy loss (via ''multi-class sigmoid'' activation) for 1-out-of-`K` classification.

        The classifier takes as input `K` $(\theta,x)$ pairs. Out of these
        pairs, one pair was sampled from the joint $p(\theta,x)$ and all others from the
        marginals $p(\theta)p(x)$. The classifier is trained to predict whether a pair was sampled from
        the independent distribution $p(\theta_0)\cdots p(\theta_K)p(x)$ or the dependent $p(\theta_0)\cdots p(\theta_K)p(x \mid \theta_k)$
        distribution. If dependently drawn, the classifier must predict which $\theta_k$ produced the $x$.
        """
        assert K >= 2
        assert theta.shape[0] == x.shape[0], "Batch sizes for theta and x must match."
        batch_size = theta.shape[0]
        logits_marginal = self._classifier_logits(theta, x, K)
        logits_joint = self._classifier_logits(theta, x, K)

        dtype = logits_marginal.dtype
        device = logits_marginal.device

        # For 1-out-of-`num_atoms` classification each datapoint consists
        # of `num_atoms` points, with one of them being the correct one.
        # We have a batch of `batch_size` such datapoints.
        logits_marginal = logits_marginal.reshape(batch_size, K)
        logits_joint = logits_joint.reshape(batch_size, K)

        # Index 0 is the theta-x-pair sampled from the joint p(theta,x) and hence the
        # "correct" one for the 1-out-of-N classification.
        # We remove the jointly drawn sample from the logits_marginal
        logits_marginal = logits_marginal[:, 1:]
        # ... and retain it in the logits_joint.
        logits_joint = logits_joint[:, :-1]

        # To use logsumexp, we extend the denominator logits with loggamma
        loggamma = torch.tensor(gamma, dtype=dtype, device=device).log()
        logK = torch.tensor(K, dtype=dtype, device=device).log()
        denominator_marginal = torch.concat(
            [loggamma + logits_marginal, logK.expand((batch_size, 1))],
            dim=-1,
        )
        denominator_joint = torch.concat(
            [loggamma + logits_joint, logK.expand((batch_size, 1))],
            dim=-1,
        )

        # Index 0 is the theta-x-pair sampled from the joint p(theta,x) and hence the
        # "correct" one for the 1-out-of-N classification.
        log_prob_marginal = logK - torch.logsumexp(denominator_marginal, dim=-1)
        log_prob_joint = (
            loggamma + logits_joint[:, 0] - torch.logsumexp(denominator_joint, dim=-1)
        )

        # relative weights
        p_marginal, p_joint = self._get_prior_probs_marginal_and_joint(K, gamma)
        return -torch.mean(p_marginal * log_prob_marginal + p_joint * K * log_prob_joint)

    @staticmethod
    def _get_prior_probs_marginal_and_joint(K: int, gamma: float) -> float:
        """Return a tuple of (prior_marginal_probability, prior_single_class_joint_probability). 
        We let the joint (dependently drawn) class to be equally likely across K options."""
        assert K >= 1
        p_joint = gamma / (1 + gamma * K)
        p_marginal = 1 / (1 + gamma * K)
        return p_marginal, p_joint
