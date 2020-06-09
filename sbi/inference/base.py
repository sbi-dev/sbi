from abc import ABC
import os.path
from typing import Callable, Dict, List, Optional, Union
from warnings import warn

import torch
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter

from copy import deepcopy
from sbi.simulators.simutils import simulate_in_batches
from sbi.user_input.user_input_checks import prepare_sbi_problem
from sbi.utils import get_log_root, get_timestamp
from sbi.utils.torchutils import get_default_device
from sbi.utils.plot.plot import samples_nd
from sbi.user_input.user_input_checks import process_x_o


class NeuralInference(ABC):
    """Abstract base class for neural inference methods."""

    def __init__(
        self,
        simulator: Callable,
        prior,
        x_shape: Optional[torch.Size] = None,
        simulation_batch_size: int = 1,
        device: Optional[torch.device] = None,
        summary_writer: Optional[SummaryWriter] = None,
        simulator_name: str = "simulator",
        num_workers: int = 1,
        skip_input_checks: bool = False,
        show_progressbar: bool = True,
        show_round_summary: bool = False,
        logging_level: Union[int, str] = "warning",
        exclude_invalid_x: bool = False,
    ):
        r"""
        Args:
            simulator: A regular callable $f(\theta)\to x$. Both parameter $\theta$ and
                simulation $x$ can be multi-dimensional.
            prior: Distribution-like object with `log_prob`and `sample` methods.
            x_shape: Shape of a single simulation output $x$. Can be either (1, N) or
                (N). Multidimensional observations (e.g. images) are not supported yet.
                If None, the shape will be inferred by running a single simulation.
            simulation_batch_size: Number of parameter sets that the simulator
                maps to data x at once. If None, we simulate all parameter sets at the
                same time. If >= 1, the simulator has to process data of shape
                (simulation_batch_size, parameter_dimension).
            device: torch.device on which to compute (optional).
            summary_writer: An optional SummaryWriter to control, among others, log
                file location (default is <current working directory>/logs.)
            num_workers: Number of parallel workers to start.
            overhead from starting workers frequently. A higher value leads to the
                simulation progressbar being updated less frequently (updates only
                happen after a worker is finished).
            skip_input_checks: Whether to disable input checks. This saves simulation
                time because they test-run the simulator to ensure it's correct.
            show_progressbar: Whether to show a progressbar during simulation, training,
                and sampling.
            show_round_summary: Whether to print the validation loss and leakage after
                each round.
            logging_level: Minimum severity of messages to log. One of the strings
                "info", "warning", "debug", "error" and "critical". Currently only
                applied when parallelization is requested for the simulator.
            exclude_invalid_x: If True, simulations containing NaN or infinite values
                are excluded from training, if False a warning is raised if NaNs or
                infinite values occur.
        """

        self._simulator, self._prior, self._x_shape = prepare_sbi_problem(
            simulator, prior, x_shape, skip_input_checks
        )

        self._skip_input_checks = skip_input_checks
        self._show_progressbar = show_progressbar
        self._show_round_summary = show_round_summary

        self._batched_simulator = lambda theta: simulate_in_batches(
            self._simulator,
            theta,
            simulation_batch_size,
            num_workers,
            self._show_progressbar,
        )

        self._device = get_default_device() if device is None else device

        # Initialize roundwise (theta, x) for storage of parameters and simulations.
        # XXX Rename self._roundwise_* or self._rounds_*
        self._theta_bank, self._x_bank = [], []
        self.exclude_invalid_x = exclude_invalid_x

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
            median_observation_distances=[], epochs=[], best_validation_log_probs=[],
        )

    def _converged(self, epoch: int, stop_after_epochs: int) -> bool:
        """Return whether the training converged yet and save best model state so far.

        Checks for improvement in validation performance over previous epochs.

        Args:
            epoch: Current epoch in training.
            stop_after_epochs: How many fruitless epochs to let pass before stopping.

        Returns:
            Whether the training has stopped improving, i.e. has converged.
        """
        converged = False

        posterior_nn = self._posterior.net

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
        Acceptance rate: {posterior_acceptance_prob:.4f}
        -------------------------
        """

        return description

    @staticmethod
    def _maybe_show_progress(show=bool, epoch=int) -> None:
        if show:
            # end="\r" deletes the print statement when a new one appears.
            # https://stackoverflow.com/questions/3419984/
            print("Training neural network. Epochs trained: ", epoch, end="\r")

    def _report_convergence_at_end(
        self, epoch: int, stop_after_epochs: int, max_num_epochs: int
    ) -> None:
        if self._converged(epoch, stop_after_epochs):
            print(f"Neural network successfully converged after {epoch} epochs.")
        elif max_num_epochs == epoch:
            warn(
                "Maximum number of epochs `max_num_epochs={max_num_epochs}` reached,"
                "but network has not yet fully converged. Consider increasing it."
            )

    @staticmethod
    def _assert_all_finite(quantity: Tensor, description: str = "tensor") -> None:
        """Raise if tensor quantity contains any NaN or Inf element."""

        msg = f"NaN/Inf present in {description}."
        assert torch.isfinite(quantity).all(), msg

    def _summarize(
        self,
        round_: int,
        x_o: Union[Tensor, None],
        theta_bank: List[Tensor],
        x_bank: List[Tensor],
        posterior_samples_acceptance_rate: Optional[Tensor] = None,
    ) -> None:
        """Update the summary_writer with statistics for a given round.

        Statistics are extracted from the arguments and from entries in self._summary
        created during training.
        """

        # NB. This is a subset of the logging from the conormdurkan/lfi. A big
        # part of the logging was removed because of API changes, e.g., logging
        # comparisons to ground truth parameters and samples.

        # Median |x - x0| for most recent round.
        if x_o is not None:
            median_observation_distance = torch.median(
                torch.sqrt(torch.sum((x_bank[-1] - x_o.reshape(1, -1)) ** 2, dim=-1,))
            )
            self._summary["median_observation_distances"].append(
                median_observation_distance.item()
            )

            self._summary_writer.add_scalar(
                tag="median_observation_distance",
                scalar_value=self._summary["median_observation_distances"][-1],
                global_step=round_ + 1,
            )

        # Rejection sampling acceptance rate, only for SNPE.
        if posterior_samples_acceptance_rate is not None:
            self._summary["rejection_sampling_acceptance_rates"].append(
                posterior_samples_acceptance_rate.item()
            )

            self._summary_writer.add_scalar(
                tag="rejection_sampling_acceptance_rate",
                scalar_value=self._summary["rejection_sampling_acceptance_rates"][-1],
                global_step=round_ + 1,
            )

        # Plot most recently sampled parameters.
        # XXX: need more plotting kwargs, e.g., prior limits.
        parameters = theta_bank[-1]

        figure, axes = samples_nd(parameters.numpy())

        self._summary_writer.add_figure(
            tag="posterior_samples", figure=figure, global_step=round_ + 1
        )

        # Add most recent training stats to summary writer.
        self._summary_writer.add_scalar(
            tag="epochs_trained",
            scalar_value=self._summary["epochs"][-1],
            global_step=round_ + 1,
        )

        self._summary_writer.add_scalar(
            tag="best_validation_log_prob",
            scalar_value=self._summary["best_validation_log_probs"][-1],
            global_step=round_ + 1,
        )

        self._summary_writer.flush()

    @property
    def summary(self):
        return self._summary

    def _handle_x_o_wrt_amortization(self, x_o: Tensor, num_rounds: int) -> None:
        """
        Check if `x_o` is consistent with `num_rounds` and maybe set it as default.

        For multi-round inference, set `x_o` as the default observation.

        Warns: if an `x_o` is provided when doing amortized inference (`num_rounds==1`).

        Raises: if `x_o` is absent but but multi-round inference is requested.
        """

        if num_rounds == 1 and x_o is not None:
            warn("Providing `x_o` in a single-round scenario has no effect.")
        elif num_rounds > 1 and x_o is None:
            raise ValueError(
                "Observed data `x_o` is required if `num_rounds>1` since"
                "samples in all rounds after the first one are drawn from"
                "the posterior `p(theta|x_o)`."
            )

        if num_rounds > 1:
            self._posterior.x_o = (
                x_o if self._skip_input_checks else process_x_o(x_o, self._x_shape)
            )
