from abc import ABC
import os.path
from typing import Callable, Optional
import warnings

import torch
from torch import Tensor
from torch.distributions import Uniform
from torch.utils.tensorboard import SummaryWriter

from sbi.simulators.simutils import prepare_sbi_problem
from sbi.utils import get_log_root, get_timestamp
from sbi.utils.torchutils import get_default_device


class NeuralInference(ABC):
    """Abstract base class for neural inference methods."""

    def __init__(
        self,
        simulator: Callable,
        prior,
        x_o: Tensor,
        simulation_batch_size: int = 1,
        device: Optional[torch.device] = None,
        summary_writer: Optional[SummaryWriter] = None,
        simulator_name: Optional[str] = "simulator",
    ):

        r"""
        Args:

            simulator: a regular function parameter->result
                Both parameters and result can be multi-dimensional.         
            prior: distribution-like object with `log_prob`and `sample` methods.
            x_o: tensor containing the observation $x_o$.
                If it has more than one dimension, the leading dimension will be
                 interpreted as a batch dimension but *currently* only the first batch
                 element will be used to condition on.
            simulation_batch_size: number of parameter sets that the
                simulator accepts and converts to data x at once. If -1, we simulate all
                 parameter sets at the same time. If >= 1, the simulator has to process
                 data of shape (simulation_batch_size, parameter_dimension).
            summary_writer: an optional SummaryWriter to control, among others, log     
                file location (default is <current working directory>/logs.)
            device: torch.device on which to compute (optional).
            mcmc_method: MCMC method to use for posterior sampling, one of 
                ['slice', 'hmc', 'nuts'].
        """

        self._simulator, self._prior, self._x_o = prepare_sbi_problem(
            simulator, prior, x_o
        )

        self._simulation_batch_size = simulation_batch_size

        self._device = get_default_device() if device is None else device

        # Initialize roundwise (theta, x) for (parameters, simulation outputs) storage.
        self._theta_bank, self._x_bank = [], []

        # XXX We could instantiate here the Posterior for all children. Two problems:
        # XXX 1. We must dispatch to right PotentialProvider for mcmc based on name
        # XXX 2. `alg_family` cannot be resolved only from `self.__class__.__name__`,
        # XXX     since SRE, AALR demand different handling but are both in SRE class.

        if summary_writer is None:
            log_dir = os.path.join(
                get_log_root(),
                self.__class__.__name__,
                simulator_name,
                get_timestamp(),
            )
            self._summary_writer = SummaryWriter(log_dir)
        else:
            self._summary_writer = summary_writer

        # Logging during training (by SummaryWriter).
        self._summary = dict(
            mmds=[],
            median_observation_distances=[],
            negative_log_probs_true_parameters=[],
            neural_net_fit_times=[],  # XXX unused elsewhere
            epochs=[],
            best_validation_log_probs=[],
        )
