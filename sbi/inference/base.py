from abc import ABC
import os.path
from typing import Callable, Dict, List, Optional, Union

import torch
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter

from copy import deepcopy
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
        simulation_batch_size: Optional[int] = 1,
        device: Optional[torch.device] = None,
        summary_writer: Optional[SummaryWriter] = None,
        simulator_name: str = "simulator",
        skip_input_checks: bool = False,
        show_progressbar: bool = True,
        show_round_summary: bool = False,
    ):
        r"""
        Args:
            simulator: A regular callable $f(\theta)\to x$. Both parameter $\theta$ and
                simulation $x$ can be multi-dimensional.
            prior: Distribution-like object with `log_prob`and `sample` methods.
            x_o: Observation $x_o$. If it has more than one dimension, the leading
                dimension will be interpreted as a batch dimension but *currently* only the first batch element will be used to condition on.
            simulation_batch_size: Number of parameter sets that the
                simulator accepts maps to data x at once. If None, we simulate
                all parameter sets at the same time. If >= 1, the simulator has to
                process data of shape (simulation_batch_size, parameter_dimension).
            device: torch.device on which to compute (optional).
            summary_writer: An optional SummaryWriter to control, among others, log
                file location (default is <current working directory>/logs.)
            skip_input_checks: Whether to disable input checks. This saves simulation
                time because they test-run the simulator to ensure it's correct.
            show_progressbar: Whether to show a progressbar during simulation, training,
                and sampling.
            show_round_summary: Whether to show the validation loss and leakage after
                each round.
        """

        self._simulator, self._prior, self._x_o = prepare_sbi_problem(
            simulator, prior, x_o, skip_input_checks
        )

        self._show_progressbar = show_progressbar
        self._show_round_summary = show_round_summary

        self._batched_simulator = lambda theta: simulate_in_batches(
            self._simulator, theta, simulation_batch_size, self._show_progressbar
        )

        self._device = get_default_device() if device is None else device

        # Initialize roundwise (theta, x) for storage of parameters and simulations.
        # XXX Rename self._roundwise_* or self._rounds_*
        self._theta_bank, self._x_bank = [], []

        # XXX We could instantiate here the Posterior for all children. Two problems:
        #     1. We must dispatch to right PotentialProvider for mcmc based on name
        #     2. `alg_family` cannot be resolved only from `self.__class__.__name__`,
        #         since SRE, AALR demand different handling but are both in SRE class.

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
            epochs=[],
            best_validation_log_probs=[],
        )

    def _has_converged(self, epoch: int, stop_after_epochs: int) -> bool:
        """Return whether the training converged yet and save best model state so far.

        Checks for improvement in validation performance over previous epochs.

        Args:
            epoch: Current epoch in training.
            stop_after_epochs: How many fruitless epochs to let pass before stopping.

        Returns:
            Whether the training has stopped improving, i.e. has converged.
        """
        converged = False

        posterior_nn = self._neural_posterior.neural_net

        # (Re)-start the epoch count with the first epoch or any improvement.
        if epoch == 0 or self._val_log_prob > self._best_val_log_prob:
            self._best_val_log_prob = self._val_log_prob
            self._epochs_since_last_improvement = 0
            self._best_model_state_dict = deepcopy(posterior_nn.state_dict())
        else:
            self._epochs_since_last_improvement += 1

        # If no validation improvement over many epochs, stop training.
        if self._epochs_since_last_improvement > stop_after_epochs - 1:
            posterior_nn.load_state_dict(self._best_model_state_dict)
            converged = True

        return converged

    @staticmethod
    def _ensure_list(
        num_simulations_per_round: Union[List[int], int], num_rounds: int
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
        if "rejection_sampling_acceptance_rates" in summary:
            # if this key exists, we are using SNPE
            posterior_acceptance_prob = summary["rejection_sampling_acceptance_rates"][
                -1
            ]
        else:
            # for all other methods, rejection_sampling_acceptance_rates is not logged
            # because the acceptance probability is by definition 1.0
            posterior_acceptance_prob = 1.0

        description = f"""
        -------------------------
        ||||| ROUND {round_ + 1} STATS |||||:
        -------------------------
        Epochs trained: {epochs}
        Best validation performance: {best_validation_log_probs:.4f}
        Leakage: {1.-posterior_acceptance_prob:.4f}
        -------------------------
        """

        return description

    @staticmethod
    def _assert_all_finite(quantity: Tensor, description: str = "tensor") -> None:
        """Raise if tensor quantity contains any NaN or Inf element."""

        msg = f"NaN/Inf present in {description}."
        assert torch.isfinite(quantity).all(), msg
