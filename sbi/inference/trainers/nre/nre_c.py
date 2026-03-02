# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

from typing import Dict, Optional, Sequence, Tuple, Union

import torch
from torch import Tensor
from torch.distributions import Distribution
from torch.utils.tensorboard.writer import SummaryWriter

from sbi.inference.trainers._contracts import LossArgs, LossArgsNRE_C
from sbi.inference.trainers.nre.nre_base import (
    RatioEstimatorTrainer,
)
from sbi.neural_nets.estimators.base import ConditionalEstimatorBuilder
from sbi.neural_nets.ratio_estimators import RatioEstimator
from sbi.sbi_types import Tracker
from sbi.utils.sbiutils import del_entries
from sbi.utils.torchutils import assert_all_finite


class NRE_C(RatioEstimatorTrainer):
    r"""Neural Ratio Estimation (NRE-C / CNRE) as in Miller et al. (2022) [1].

    NRE-C generalizes NRE-A and NRE-B using a "multi-class sigmoid" loss that
    ensures the estimated ratio $p(\theta,x)/p(\theta)p(x)$ is exact at optimum
    in the first round. This addresses the issue that NRE-B's ratio is only defined
    up to an arbitrary function of $x$. NRE-C provides more accurate ratio estimates
    while maintaining the benefits of contrastive learning.

    [1] *Contrastive Neural Ratio Estimation*, Benjamin Kurt Miller, et al.,
        NeurIPS 2022, https://arxiv.org/abs/2210.06170

    Example:
    --------

    ::

        import torch
        from sbi.inference import NRE_C
        from sbi.utils import BoxUniform

        # 1. Setup prior and simulate data
        prior = BoxUniform(low=torch.zeros(3), high=torch.ones(3))
        theta = prior.sample((100,))
        x = theta + torch.randn_like(theta) * 0.1

        # 2. Train ratio estimator
        inference = NRE_C(prior=prior)
        ratio_estimator = inference.append_simulations(theta, x).train(num_classes=5)

        # 3. Build posterior
        posterior = inference.build_posterior(ratio_estimator)

        # 4. Sample from posterior
        x_o = torch.randn(1, 3)
        samples = posterior.sample((1000,), x=x_o)
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
        r"""Initialize NRE-C.

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
        num_classes: int = 5,
        gamma: float = 1.0,
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
        kwargs["loss_kwargs"] = LossArgsNRE_C(
            num_atoms=kwargs.pop("num_classes") + 1, gamma=kwargs.pop("gamma")
        )

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

        # We append a contrastive theta to the marginal case because we will remove
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

        # relative weights. p_marginal := p_0, and p_joint := p_K * K from the notation.
        p_marginal, p_joint = self._get_prior_probs_marginal_and_joint(gamma)

        loss = -torch.mean(p_marginal * log_prob_marginal + p_joint * log_prob_joint)
        assert_all_finite(loss, "NRE-C loss")
        return loss

    @staticmethod
    def _get_prior_probs_marginal_and_joint(gamma: float) -> Tuple[float, float]:
        r"""Return a tuple (p_marginal, p_joint) where `p_marginal := `$p_0$,
        `p_joint := `$p_K \cdot K$.

        We let the joint (dependently drawn) class to be equally likely across K
        options. The marginal class is therefore restricted to get the remaining
        probability.
        """
        p_joint = gamma / (1 + gamma)
        p_marginal = 1 / (1 + gamma)
        return p_marginal, p_joint

    def _get_losses(self, batch: Sequence[Tensor], loss_args: LossArgs) -> Tensor:
        """Overrides the parent class method to check the type of loss_args."""

        if not isinstance(loss_args, LossArgsNRE_C):
            raise TypeError(
                "Expected type of loss_args to be LossArgsNRE_C,"
                f" but got {type(loss_args)}"
            )

        return super()._get_losses(batch=batch, loss_args=loss_args)
