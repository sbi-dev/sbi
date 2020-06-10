from __future__ import annotations
from typing import Callable, Optional, Union

import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from sbi.inference.snpe.snpe_base import PosteriorEstimator
from sbi.utils.torchutils import get_default_device


class SNPE_A(PosteriorEstimator):
    def __init__(
        self,
        simulator: Callable,
        prior,
        x_shape: Optional[torch.Size] = None,
        num_workers: int = 1,
        simulation_batch_size: int = 1,
        density_estimator: Union[str, nn.Module] = "mdn",
        calibration_kernel: Optional[Callable] = None,
        z_score_x: bool = True,
        z_score_min_std: float = 1e-7,
        exclude_invalid_x: bool = True,
        device: Union[torch.device, str] = get_default_device(),
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
