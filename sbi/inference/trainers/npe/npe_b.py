# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

from typing import Any, Callable, Optional, Union

import torch
from torch import Tensor
from torch.distributions import Distribution

import sbi.utils as utils
from sbi.inference.trainers.npe.npe_base import PosteriorEstimator
from sbi.neural_nets.estimators.shape_handling import reshape_to_sample_batch_event
from sbi.sbi_types import TensorboardSummaryWriter
from sbi.utils.sbiutils import del_entries


class NPE_B(PosteriorEstimator):
    """Neural Posterior Estimation algorithm (NPE-B) as in Lueckmann et al. (2017)."""

    def __init__(
        self,
        prior: Optional[Distribution] = None,
        density_estimator: Union[str, Callable] = "maf",
        device: str = "cpu",
        logging_level: Union[int, str] = "WARNING",
        summary_writer: Optional[TensorboardSummaryWriter] = None,
        show_progress_bars: bool = True,
    ):
        r"""NPE-B [1].

        [1] *Flexible statistical inference for mechanistic models of neural
        dynamics*, Lueckmann, Gonçalves et al., NeurIPS 2017. https://arxiv.org/abs/171


        Like all NPE methods, this method trains a deep neural density estimator to
        directly approximate the posterior. Also like all other NPE methods, in the
        first round, this density estimator is trained with a maximum-likelihood loss.

        This class implements NPE-B. NPE-B trains across multiple rounds with a
        an importance-weighted log-loss. Unlike NPE-A the loss will make training
        directly converge to the true posterior.
        Thus, SNPE-B is not limited to Gaussian proposal.

        Args:
            prior: A probability distribution that expresses prior knowledge about the
                parameters, e.g. which ranges are meaningful for them.
            density_estimator: If it is a string, use a pre-configured network of the
                provided type (one of nsf, maf, mdn, made). Alternatively, a function
                that builds a custom neural network can be provided. The function will
                be called with the first batch of simulations (theta, x), which can
                thus be used for shape inference and potentially for z-scoring. It
                needs to return a PyTorch `nn.Module` implementing the density
                estimator. The density estimator needs to provide the methods
                `.log_prob` and `.sample()`.
            device: Training device, e.g., "cpu", "cuda" or "cuda:{0, 1, ...}".
            logging_level: Minimum severity of messages to log. One of the strings
                INFO, WARNING, DEBUG, ERROR and CRITICAL.
            summary_writer: A tensorboard `SummaryWriter` to control, among others, log
                file location (default is `<current working directory>/logs`.)
            show_progress_bars: Whether to show a progressbar during training.
        """

        kwargs = del_entries(locals(), entries=("self", "__class__"))
        super().__init__(**kwargs)

    def _log_prob_proposal_posterior(
        self,
        theta: Tensor,
        x: Tensor,
        masks: Tensor,
        proposal: Optional[Any],
    ) -> Tensor:
        """
        Return importance-weighted log probability (Lueckmann, Goncalves et al., 2017).

        Args:
            theta: Batch of parameters θ.
            x: Batch of data.
            masks: Mask that is True for prior samples in the batch in order to train
                them with prior loss.
            proposal: Proposal distribution.

        Returns:
            Importance-weighted log-probability of the proposal posterior.
        """

        # Evaluate prior
        # we accept prior log prob to be -Inf at theta
        # meaning that theta is out of the prior range (the weight is thus 0)
        utils.assert_not_nan_or_plus_inf(
            self._prior.log_prob(theta), "prior log probs of proposal samples"
        )
        prior = torch.exp(self._prior.log_prob(theta))

        # Evaluate proposal
        # (as theta comes from prior and proposal from previous rounds,
        # the last proposal is actually a mixture of the prior
        # and of all the previous proposals with coefficients representing
        # the proportion of the new theta added at each round)
        prop = torch.zeros(self._round + 1, device=theta.device)
        nb_samples = 0  # total number of theta from all the rounds

        for k in range(self._round + 1):
            nb_samples += self._theta_roundwise[k].size(0)
            # the number of new theta sampled in the round k
            prop[k] = self._theta_roundwise[k].size(0)

        prop /= nb_samples
        log_prop = torch.log(prop).repeat(theta.size(0), 1)

        log_previous_proposals = torch.zeros(
            (theta.size(0), self._round + 1), device=theta.device
        )
        for k, density in enumerate(self._proposal_roundwise):
            # we accept the k th proposal log prob to be -Inf at theta
            # meaning that theta is out of the k th proposal range
            log_previous_proposals[:, k] = density.log_prob(theta)
            utils.assert_not_nan_or_plus_inf(
                log_previous_proposals[:, k], "proposal log probs of proposal samples"
            )

        log_proposal = torch.logsumexp(log_prop + log_previous_proposals, dim=1)
        proposal = torch.exp(log_proposal)

        # Construct the importance weights and normalize them
        importance_weights = prior / proposal
        importance_weights /= importance_weights.sum()

        theta = reshape_to_sample_batch_event(theta, theta.shape[1:])
        # Reshape the density estimator log probs
        # from (sample_shape, batch_shape) to (batch_shape)
        posterior_log_probs = self._neural_net.log_prob(theta, x).squeeze(dim=0)

        return importance_weights * posterior_log_probs
