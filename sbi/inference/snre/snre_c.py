from typing import Callable, Dict, Optional, Tuple, Union

import torch
from torch import Tensor, nn
from torch.distributions import Distribution

from sbi.inference.snre.snre_base import RatioEstimator
from sbi.types import TensorboardSummaryWriter
from sbi.utils import del_entries


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
        r"""NRE-C[1] is a generalization of the non-sequential (amortized) versions of
        SNRE_A and SNRE_B. We call the algorithm SNRE_C within `sbi`.

        NRE-C:
        (1) like SNRE_B, features a "multiclass" loss function where several marginally
            drawn parameter-data pairs are contrasted against a jointly drawn pair.
        (2) like AALR/NRE_A, i.e., the non-sequential version of SNRE_A, it encourages
            the approximate ratio $p(\theta,x)/p(\theta)p(x)$, accessed through
            `.potential()` within `sbi`, to be exact at optimum. This addresses the
            issue that SNRE_B estimates this ratio only up to an arbitrary function
            (normalizing constant) of the data $x$.

        Just like for all ratio estimation algorithms, the sequential version of SNRE_C
        will be estimated only up to a function (normalizing constant) of the data $x$
        in rounds after the first.

        [1] _Contrastive Neural Ratio Estimation_, Benajmin Kurt Miller, et. al.,
            NeurIPS 2022, https://arxiv.org/abs/2210.06170

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
        num_classes: int = 5,
        gamma: float = 1.0,
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
            num_classes: Number of theta to classify against, corresponds to $K$ in
                _Contrastive Neural Ratio Estimation_. Minimum value is 1. Similar to
                `num_atoms` for SNRE_B except SNRE_C has an additional independently
                drawn sample. The total number of alternative parameters `NRE-C` "sees"
                is $2K-1$ or `2 * num_classes - 1` divided between two loss terms.
            gamma: Determines the relative weight of the sum of all $K$ dependently
                drawn classes against the marginally drawn one. Specifically,
                $p(y=k) :=p_K$, $p(y=0) := p_0$, $p_0 = 1 - K p_K$, and finally
                $\gamma := K p_K / p_0$.
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
        kwargs["num_atoms"] = kwargs.pop("num_classes") + 1
        kwargs["loss_kwargs"] = {"gamma": kwargs.pop("gamma")}
        return super().train(**kwargs)

    def _loss(
        self, theta: Tensor, x: Tensor, num_atoms: int, gamma: float
    ) -> torch.Tensor:
        r"""Return cross-entropy loss (via ''multi-class sigmoid'' activation) for
        1-out-of-`K + 1` classification.

        At optimum, this loss function returns the exact likelihood-to-evidence ratio
        in the first round.
        Details of loss computation are described in Contrastive Neural Ratio
        Estimation[1]. The paper does not discuss the sequential case.

        [1] _Contrastive Neural Ratio Estimation_, Benajmin Kurt Miller, et. al.,
            NeurIPS 2022, https://arxiv.org/abs/2210.06170
        """

        # Reminder: K = num_classes
        # The algorithm is written with K, so we convert back to K format rather than
        # reasoning in num_atoms.
        num_classes = num_atoms - 1
        assert num_classes >= 1, f"num_classes = {num_classes} must be greater than 1."

        assert theta.shape[0] == x.shape[0], "Batch sizes for theta and x must match."
        batch_size = theta.shape[0]

        # We append an contrastive theta to the marginal case because we will remove
        # the jointly drawn
        # sample in the logits_marginal[:, 0] position. That makes the remaining sample
        # marginally drawn.
        # We have a batch of `batch_size` datapoints.
        logits_marginal = self._classifier_logits(theta, x, num_classes + 1).reshape(
            batch_size, num_classes + 1
        )
        logits_joint = self._classifier_logits(theta, x, num_classes).reshape(
            batch_size, num_classes
        )

        dtype = logits_marginal.dtype
        device = logits_marginal.device

        # Index 0 is the theta-x-pair sampled from the joint p(theta,x) and hence
        # we remove the jointly drawn sample from the logits_marginal
        logits_marginal = logits_marginal[:, 1:]
        # ... and retain it in the logits_joint. Now we have two arrays with K choices.

        # To use logsumexp, we extend the denominator logits with loggamma
        loggamma = torch.tensor(gamma, dtype=dtype, device=device).log()
        logK = torch.tensor(num_classes, dtype=dtype, device=device).log()
        denominator_marginal = torch.concat(
            [loggamma + logits_marginal, logK.expand((batch_size, 1))],
            dim=-1,
        )
        denominator_joint = torch.concat(
            [loggamma + logits_joint, logK.expand((batch_size, 1))],
            dim=-1,
        )

        # Compute the contributions to the loss from each term in the classification.
        log_prob_marginal = logK - torch.logsumexp(denominator_marginal, dim=-1)
        log_prob_joint = (
            loggamma + logits_joint[:, 0] - torch.logsumexp(denominator_joint, dim=-1)
        )

        # relative weights. p_marginal := p_0, and p_joint := p_K from the notation.
        p_marginal, p_joint = self._get_prior_probs_marginal_and_joint(
            num_classes, gamma
        )
        return -torch.mean(
            p_marginal * log_prob_marginal + p_joint * num_classes * log_prob_joint
        )

    @staticmethod
    def _get_prior_probs_marginal_and_joint(
        num_classes: int, gamma: float
    ) -> Tuple[float, float]:
        """Return a tuple (p_marginal, p_joint) where `p_marginal := `$p_0$,
        `p_joint := `$p_K$.

        We let the joint (dependently drawn) class to be equally likely across K
        options. The marginal class is therefore restricted to get the remaining
        probability.
        """
        assert num_classes >= 1
        p_joint = gamma / (1 + gamma * num_classes)
        p_marginal = 1 / (1 + gamma * num_classes)
        return p_marginal, p_joint
