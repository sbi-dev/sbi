from abc import ABC
import os.path
from typing import Callable, Dict, List, Optional, Union

import torch
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter

from sbi.simulators.simutils import simulate_in_batches
from sbi.user_input.user_input_checks import prepare_sbi_problem
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
        skip_input_checks: bool = False,
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
                simulator accepts and converts to data x at once. If -1, we simulate 
                all parameter sets at the same time. If >= 1, the simulator has to 
                process data of shape (simulation_batch_size, parameter_dimension).
            device: torch.device on which to compute (optional).
            summary_writer: an optional SummaryWriter to control, among others, log     
                file location (default is <current working directory>/logs.)
            skip_simulator_checks: Flag to turn off input checks,
                e.g., for saving simulation budget as the input checks run the
                simulator a couple of times.
        """

        self._simulator, self._prior, self._x_o = prepare_sbi_problem(
            simulator, prior, x_o, skip_input_checks
        )

        self._batched_simulator = lambda theta: simulate_in_batches(
            self._simulator, theta, simulation_batch_size
        )

        self._device = get_default_device() if device is None else device

        # Initialize roundwise (theta, x) for storage of parameters and simulations.
        # XXX rename self._roundwise_theta, self._roundwise_x = [], []
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

    @staticmethod
    def _ensure_list(
        num_simulations_per_round: Union[List, int], num_rounds: int
    ) -> List[int]:
        """Return `num_simulations_per_round` as a list of length `num_rounds`.
        """
        try:
            assert (
                len(num_simulations_per_round) == num_rounds
            ), "Please provide list with number of simulations per round for each round, or a single integer to be used for all rounds."
        except TypeError:
            num_simulations_per_round = [num_simulations_per_round] * num_rounds
        return num_simulations_per_round

    @staticmethod
    def _describe_round(round_: int, summary: Dict[str, list]) -> str:
        epochs = summary["epochs"][-1]
        best_validation_log_probs = summary["best_validation_log_probs"][-1]

        description = f"""
        -------------------------
        ||||| ROUND {round_ + 1} STATS |||||:
        -------------------------
        Epochs trained: {epochs}
        Best validation performance: {best_validation_log_probs:.4f}


        """

        return description
