# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Affero General Public License v3, see <https://www.gnu.org/licenses/>.


from typing import Callable, Optional, Union, Dict, Any, Tuple, Union, cast, List, Sequence, TypeVar

import torch
from torch import Tensor, nn
from torch.utils.tensorboard import SummaryWriter

from sbi.inference.snpe.snpe_base import PosteriorEstimator


class SNPE_B(PosteriorEstimator):
    def __init__(
        self,
        simulator: Callable,
        prior,
        num_workers: int = 1,
        simulation_batch_size: Optional[int] = 1,
        density_estimator: Union[str, Callable] = "mdn",
        calibration_kernel: Optional[Callable] = None,
        retrain_from_scratch_each_round: bool = False,
        discard_prior_samples: bool = False,
        exclude_invalid_x: bool = True,
        device: str = "cpu",
        logging_level: Union[int, str] = "WARNING",
        summary_writer: Optional[SummaryWriter] = None,
        show_progress_bars: bool = True,
        show_round_summary: bool = False,
    ):
        r"""SNPE-B [1]. CURRENTLY NOT IMPLEMENTED.

        [1] _Flexible statistical inference for mechanistic models of neural dynamics_,
            Lueckmann, Gonçalves et al., NeurIPS 2017, https://arxiv.org/abs/1711.01861.

        See docstring of `PosteriorEstimator` class for all other arguments.
        """

        raise NotImplementedError(
            "SNPE-B is not yet implemented in the sbi package, see issue #199."
        )

        super().__init__(
            simulator=simulator,
            prior=prior,
            num_workers=num_workers,
            simulation_batch_size=simulation_batch_size,
            density_estimator=density_estimator,
            calibration_kernel=calibration_kernel,
            retrain_from_scratch_each_round=retrain_from_scratch_each_round,
            discard_prior_samples=discard_prior_samples,
            exclude_invalid_x=exclude_invalid_x,
            device=device,
            logging_level=logging_level,
            show_progress_bars=show_progress_bars,
            show_round_summary=show_round_summary,
        )

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

        batch_size = theta.shape[0]

        # Evaluate posterior.
        log_prob_posterior = self._posterior.net.log_prob(theta, x)
        log_prob_posterior = log_prob_posterior.reshape(batch_size)
        self._assert_all_finite(log_prob_posterior, "posterior eval")

        # Evaluate prior.
        log_prob_prior = self._prior.log_prob(theta).reshape(batch_size)
        self._assert_all_finite(log_prob_prior, "prior eval.")

        # Evaluate proposal.
        log_prob_proposal = self._model_bank[-1].net.log_prob(theta, x)
        self._assert_all_finite(log_prob_proposal, "proposal posterior eval")

        # Compute log prob with importance weights.
        log_prob = torch.exp(log_prob_prior - log_prob_proposal) * log_prob_posterior

        return log_prob
