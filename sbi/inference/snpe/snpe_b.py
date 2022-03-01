# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Affero General Public License v3, see <https://www.gnu.org/licenses/>.


from typing import Callable, Optional, Union

import torch
from torch import Tensor
from torch.distributions import Distribution

import sbi.utils as utils
from sbi.inference.snpe.snpe_base import PosteriorEstimator
from sbi.types import TensorboardSummaryWriter
from sbi.utils import del_entries


class SNPE_B(PosteriorEstimator):
    def __init__(
        self,
        prior: Optional[Distribution] = None,
        density_estimator: Union[str, Callable] = "maf",
        device: str = "cpu",
        logging_level: Union[int, str] = "WARNING",
        summary_writer: Optional[TensorboardSummaryWriter] = None,
        show_progress_bars: bool = True,
    ):
        r"""SNPE-B [1]. CURRENTLY NOT IMPLEMENTED.

        [1] _Flexible statistical inference for mechanistic models of neural dynamics_,
            Lueckmann, Gonçalves et al., NeurIPS 2017, https://arxiv.org/abs/1711.01861.

        See docstring of `PosteriorEstimator` class for all other arguments.
        """

        raise NotImplementedError(
            "SNPE-B is not yet implemented in the sbi package, see issue #199."
        )

        kwargs = del_entries(locals(), entries=("self", "__class__"))
        super().__init__(**kwargs)

    def _log_prob_proposal_posterior(
        self, theta: Tensor, x: Tensor, masks: Tensor
    ) -> Tensor:
        """
        Return importance-weighted log probability (Lueckmann, Goncalves et al., 2017).

        Args:
            theta: Batch of parameters θ.
            x: Batch of data.
            masks: Whether to retrain with prior loss (for each prior sample).

        Returns:
            Log probability of proposal posterior.
        """

        raise NotImplementedError

        batch_size = theta.shape[0]

        # Evaluate posterior.
        log_prob_posterior = self._posterior.net.log_prob(theta, x)
        log_prob_posterior = log_prob_posterior.reshape(batch_size)
        utils.assert_all_finite(log_prob_posterior, "posterior eval")

        # Evaluate prior.
        log_prob_prior = self._prior.log_prob(theta).reshape(batch_size)
        utils.assert_all_finite(log_prob_prior, "prior eval.")

        # Evaluate proposal.
        log_prob_proposal = self._model_bank[-1].net.log_prob(theta, x)
        utils.assert_all_finite(log_prob_proposal, "proposal posterior eval")

        # Compute log prob with importance weights.
        log_prob = torch.exp(log_prob_prior - log_prob_proposal) * log_prob_posterior

        return log_prob
