# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Affero General Public License v3, see <https://www.gnu.org/licenses/>.

from __future__ import annotations

from abc import ABC
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional, Union, cast
from warnings import warn

import torch
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter

import sbi.inference
from sbi.inference.posterior import NeuralPosterior
from sbi.simulators.simutils import simulate_in_batches
from sbi.user_input.user_input_checks import prepare_for_sbi, process_x
from sbi.utils import get_log_root
from sbi.utils.plot import pairplot
from sbi.utils.torchutils import configure_default_device


def infer(
    simulator: Callable, prior, method: str, num_simulations: int, num_workers: int = 1
) -> NeuralPosterior:
    r"""
    Return posterior distribution by running simulation-based inference.

    This function provides a simple interface to run sbi. Inference is run for a single
    round and hence the returned posterior $p(\theta|x)$ can be sampled and evaluated
    for any $x$ (i.e. it is amortized).

    The scope of this function is limited to the most essential features of sbi. For
    more flexibility (e.g. multi-round inference, different density estimators) please
    use the flexible interface described here:
    https://www.mackelab.org/sbi/tutorial/03_flexible_interface/

    Args:
        simulator: A function that takes parameters $\theta$ and maps them to
            simulations, or observations, `x`, $\mathrm{sim}(\theta)\to x$. Any
            regular Python callable (i.e. function or class with `__call__` method)
            can be used.
        prior: A probability distribution that expresses prior knowledge about the
            parameters, e.g. which ranges are meaningful for them. Any
            object with `.log_prob()`and `.sample()` (for example, a PyTorch
            distribution) can be used.
        method: What inference method to use. Either of SNPE, SNLE or SNRE.
        num_simulations: Number of simulation calls. More simulations means a longer
            runtime, but a better posterior estimate.
        num_workers: Number of parallel workers to use for simulations.

    Returns: Posterior over parameters conditional on observations (amortized).
    """

    try:
        method_fun: Callable = getattr(sbi.inference, method.upper())
    except AttributeError:
        raise NameError(
            "Method not available. `method` must be one of 'SNPE', 'SNLE', 'SNRE'."
        )

    simulator, prior = prepare_for_sbi(simulator, prior)

    infer_ = method_fun(simulator, prior, num_workers=num_workers)
    posterior = infer_(num_rounds=1, num_simulations_per_round=num_simulations)

    return posterior


class NeuralInference(ABC):
    """Abstract base class for neural inference methods."""

    def __init__(
        self,
        simulator: Callable,
        prior,
        num_workers: int = 1,
        simulation_batch_size: int = 1,
        device: str = "cpu",
        logging_level: Union[int, str] = "WARNING",
        summary_writer: Optional[SummaryWriter] = None,
        show_progress_bars: bool = True,
        show_round_summary: bool = False,
    ):
        r"""
        Base class for inference methods.

        Args:
            simulator: A function that takes parameters $\theta$ and maps them to
                simulations, or observations, `x`, $\mathrm{sim}(\theta)\to x$. Any
                regular Python callable (i.e. function or class with `__call__` method)
                can be used.
            prior: A probability distribution that expresses prior knowledge about the
                parameters, e.g. which ranges are meaningful for them. Any
                object with `.log_prob()`and `.sample()` (for example, a PyTorch
                distribution) can be used.
            num_workers: Number of parallel workers to use for simulations.
            simulation_batch_size: Number of parameter sets that the simulator
                maps to data x at once. If None, we simulate all parameter sets at the
                same time. If >= 1, the simulator has to process data of shape
                (simulation_batch_size, parameter_dimension).
            device: torch device on which to compute, e.g. gpu or cpu.
            logging_level: Minimum severity of messages to log. One of the strings
               "INFO", "WARNING", "DEBUG", "ERROR" and "CRITICAL".
            summary_writer: A `SummaryWriter` to control, among others, log
                file location (default is `<current working directory>/logs`.)
            show_progress_bars: Whether to show a progressbar during simulation and
                sampling.
            show_round_summary: Whether to show the validation loss and leakage after
                each round.
        """

        # We set the device globally by setting the default tensor type for all tensors.
        assert device in (
            "gpu",
            "cpu",
        ), "Currently, only 'gpu' or 'cpu' are supported as devices."

        self._device = configure_default_device(device)

        self._simulator, self._prior = simulator, prior

        self._show_progress_bars = show_progress_bars
        self._show_round_summary = show_round_summary

        self._batched_simulator = lambda theta: simulate_in_batches(
            self._simulator,
            theta,
            simulation_batch_size,
            num_workers,
            self._show_progress_bars,
        )

        # Initialize roundwise (theta, x) for storage of parameters and simulations.
        # XXX Rename self._roundwise_* or self._rounds_*
        self._theta_bank, self._x_bank = [], []

        # XXX We could instantiate here the Posterior for all children. Two problems:
        #     1. We must dispatch to right PotentialProvider for mcmc based on name
        #     2. `method_family` cannot be resolved only from `self.__class__.__name__`,
        #         since SRE, AALR demand different handling but are both in SRE class.

        self._summary_writer = (
            self._default_summary_writer() if summary_writer is None else summary_writer
        )

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

    def _default_summary_writer(self) -> SummaryWriter:
        """Return summary writer logging to method- and simulator-specific directory."""
        try:
            simulator = self._simulator.__name__
        except AttributeError:
            simulator = self._simulator.__class__.__name__

        method = self.__class__.__name__
        logdir = Path(
            get_log_root(),
            simulator,
            method,
            datetime.now().isoformat().replace(":", "_"),
        )
        return SummaryWriter(logdir)

    @staticmethod
    def _ensure_list(
        num_simulations_per_round: Union[List[int], int], num_rounds: int
    ) -> List[int]:
        """Return `num_simulations_per_round` as a list of length `num_rounds`.
        """
        try:
            assert len(num_simulations_per_round) == num_rounds, (
                "Please provide a list with number of simulations per round for each "
                "round, or a single integer to be used for all rounds."
            )
        except TypeError:
            num_simulations_per_round: List = [num_simulations_per_round] * num_rounds

        return cast(list, num_simulations_per_round)

    @staticmethod
    def _describe_round(round_: int, summary: Dict[str, list]) -> str:
        epochs = summary["epochs"][-1]
        best_validation_log_probs = summary["best_validation_log_probs"][-1]
        if "rejection_sampling_acceptance_rates" in summary:
            # If this key exists, we are using SNPE.
            posterior_acceptance_prob = summary["rejection_sampling_acceptance_rates"][
                -1
            ]
        else:
            # For all other methods, `rejection_sampling_acceptance_rates` is not logged
            # because the acceptance probability is by definition 1.0.
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

        # NB. This is a subset of the logging as done in `GH:conormdurkan/lfi`. A big
        # part of the logging was removed because of API changes, e.g., logging
        # comparisons to ground-truth parameters and samples.

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

        figure, axes = pairplot(parameters.to("cpu").numpy())

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

    def _handle_x_o_wrt_amortization(
        self, x_o: Optional[Tensor], x_shape: torch.Size, num_rounds: int
    ) -> None:
        """
        Check whether provided `x_o` is consistent with `num_rounds`. For multi-round
        inference, set `x_o` as the default observation in the posterior.

        Warns:
            When `x_o` is provided, yet `num_rounds=1`, i.e. inference is amortized.

        Raises:
            When `x_o` is absent, yet `num_rounds>1`, i.e. inference is focused.
        """

        if num_rounds == 1 and x_o is not None:
            warn("Providing `x_o` in a single-round scenario has no effect.")
        elif num_rounds > 1 and x_o is None:
            raise ValueError(
                "Observed data `x_o` is required if `num_rounds>1` since "
                "samples in all rounds after the first one are drawn from "
                "the posterior `p(theta|x_o)`."
            )

        if num_rounds > 1:
            processed_x = process_x(x_o, x_shape)
            self._posterior._x_o_training_focused_on = processed_x
            self._posterior.default_x = processed_x
