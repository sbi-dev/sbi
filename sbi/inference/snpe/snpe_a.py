# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Affero General Public License v3, see <https://www.gnu.org/licenses/>.


from typing import Callable, Optional, Union, Dict, Any, Tuple, Union, cast, List, Sequence, TypeVar

import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from sbi.inference.snpe.snpe_base import PosteriorEstimator


class SNPE_A(PosteriorEstimator):
    def __init__(
        self,
        simulator: Callable,
        prior,
        num_workers: int = 1,
        simulation_batch_size: int = 1,
        density_estimator: Union[str, Callable] = "mdn",
        calibration_kernel: Optional[Callable] = None,
        exclude_invalid_x: bool = True,
        device: str = "cpu",
        logging_level: Union[int, str] = "WARNING",
        summary_writer: Optional[SummaryWriter] = None,
        show_progress_bars: bool = True,
        show_round_summary: bool = False,
    ):
        """SNPE-A [1]. CURRENTLY NOT IMPLEMENTED.

        [1] _Fast epsilon-free Inference of Simulation Models with Bayesian Conditional
            Density Estimation_, Papamakarios et al., NeurIPS 2016,
            https://arxiv.org/abs/1605.06376.

        """

        raise NotImplementedError
