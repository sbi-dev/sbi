# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Affero General Public License v3, see <https://www.gnu.org/licenses/>.

from abc import ABC, abstractmethod
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple, Union
from warnings import warn

import torch
from torch import Tensor
from torch.distributions import Distribution
from torch.utils import data
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.tensorboard.writer import SummaryWriter

import sbi.inference
from sbi.inference.posteriors.base_posterior import NeuralPosterior
from sbi.simulators.simutils import simulate_in_batches
from sbi.utils import check_prior, get_log_root
from sbi.utils.sbiutils import get_simulations_since_round
from sbi.utils.torchutils import check_if_prior_on_device, process_device
from sbi.utils.user_input_checks import prepare_for_sbi


def infer(
    simulator: Callable,
    prior: Distribution,
    method: str,
    num_simulations: int,
    num_workers: int = 1,
) -> NeuralPosterior:
    r"""Runs simulation-based inference and returns the posterior.

    This function provides a simple interface to run sbi. Inference is run for a single
    round and hence the returned posterior $p(\theta|x)$ can be sampled and evaluated
    for any $x$ (i.e. it is amortized).

    The scope of this function is limited to the most essential features of sbi. For
    more flexibility (e.g. multi-round inference, different density estimators) please
    use the flexible interface described here:
    https://www.mackelab.org/sbi/tutorial/02_flexible_interface/

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

    inference = method_fun(prior=prior)
    theta, x = simulate_for_sbi(
        simulator=simulator,
        proposal=prior,
        num_simulations=num_simulations,
        num_workers=num_workers,
    )
    _ = inference.append_simulations(theta, x).train()
    posterior = inference.build_posterior()

    return posterior


class NeuralInference(ABC):
    """Abstract base class for neural inference methods."""

    def __init__(
        self,
        prior: Optional[Distribution] = None,
        device: str = "cpu",
        logging_level: Union[int, str] = "WARNING",
        summary_writer: Optional[SummaryWriter] = None,
        show_progress_bars: bool = True,
    ):
        r"""Base class for inference methods.

        Args:
            prior: A probability distribution that expresses prior knowledge about the
                parameters, e.g. which ranges are meaningful for them. Must be a PyTorch
                distribution, see FAQ for details on how to use custom distributions.
            device: torch device on which to train the neural net and on which to
                perform all posterior operations, e.g. gpu or cpu.
            logging_level: Minimum severity of messages to log. One of the strings
               "INFO", "WARNING", "DEBUG", "ERROR" and "CRITICAL".
            summary_writer: A `SummaryWriter` to control, among others, log
                file location (default is `<current working directory>/logs`.)
            show_progress_bars: Whether to show a progressbar during simulation and
                sampling.
        """

        self._device = process_device(device)
        check_prior(prior)
        check_if_prior_on_device(self._device, prior)
        self._prior = prior

        self._posterior = None
        self._neural_net = None
        self._x_shape = None

        self._show_progress_bars = show_progress_bars

        # Initialize roundwise (theta, x, prior_masks) for storage of parameters,
        # simulations and masks indicating if simulations came from prior.
        self._theta_roundwise = []
        self._x_roundwise = []
        self._prior_masks = []
        self._model_bank = []

        # Initialize list that indicates the round from which simulations were drawn.
        self._data_round_index = []

        self._round = 0
        self._val_log_prob = float("-Inf")

        # XXX We could instantiate here the Posterior for all children. Two problems:
        #     1. We must dispatch to right PotentialProvider for mcmc based on name
        #     2. `method_family` cannot be resolved only from `self.__class__.__name__`,
        #         since SRE, AALR demand different handling but are both in SRE class.

        self._summary_writer = (
            self._default_summary_writer() if summary_writer is None else summary_writer
        )

        # Logging during training (by SummaryWriter).
        self._summary = dict(
            epochs_trained=[],
            best_validation_log_prob=[],
            validation_log_probs=[],
            training_log_probs=[],
            epoch_durations_sec=[],
        )

    def get_simulations(
        self,
        starting_round: int = 0,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        r"""Returns all $\theta$, $x$, and prior_masks from rounds >= `starting_round`.

        If requested, do not return invalid data.

        Args:
            starting_round: The earliest round to return samples from (we start counting
                from zero).
            exclude_invalid_x: Whether to exclude simulation outputs `x=NaN` or `x=±∞`
                during training.
            warn_on_invalid: Whether to give out a warning if invalid simulations were
                found.

        Returns: Parameters, simulation outputs, prior masks.
        """

        theta = get_simulations_since_round(
            self._theta_roundwise, self._data_round_index, starting_round
        )
        x = get_simulations_since_round(
            self._x_roundwise, self._data_round_index, starting_round
        )
        prior_masks = get_simulations_since_round(
            self._prior_masks, self._data_round_index, starting_round
        )

        return theta, x, prior_masks

    @abstractmethod
    def train(
        self,
        training_batch_size: int = 50,
        learning_rate: float = 5e-4,
        validation_fraction: float = 0.1,
        stop_after_epochs: int = 20,
        max_num_epochs: Optional[int] = None,
        clip_max_norm: Optional[float] = 5.0,
        calibration_kernel: Optional[Callable] = None,
        exclude_invalid_x: bool = True,
        discard_prior_samples: bool = False,
        retrain_from_scratch: bool = False,
        show_train_summary: bool = False,
    ) -> NeuralPosterior:
        raise NotImplementedError

    def get_dataloaders(
        self,
        starting_round: int = 0,
        training_batch_size: int = 50,
        validation_fraction: float = 0.1,
        resume_training: bool = False,
        dataloader_kwargs: Optional[dict] = None,
    ) -> Tuple[data.DataLoader, data.DataLoader]:
        """Return dataloaders for training and validation.

        Args:
            dataset: holding all theta and x, optionally masks.
            training_batch_size: training arg of inference methods.
            resume_training: Whether the current call is resuming training so that no
                new training and validation indices into the dataset have to be created.
            dataloader_kwargs: Additional or updated kwargs to be passed to the training
                and validation dataloaders (like, e.g., a collate_fn).

        Returns:
            Tuple of dataloaders for training and validation.

        """

        #
        theta, x, prior_masks = self.get_simulations(starting_round)

        dataset = data.TensorDataset(theta, x, prior_masks)

        # Get total number of training examples.
        num_examples = theta.size(0)
        # Select random train and validation splits from (theta, x) pairs.
        num_training_examples = int((1 - validation_fraction) * num_examples)
        num_validation_examples = num_examples - num_training_examples

        if not resume_training:
            # Seperate indicies for training and validation
            permuted_indices = torch.randperm(num_examples)
            self.train_indices, self.val_indices = (
                permuted_indices[:num_training_examples],
                permuted_indices[num_training_examples:],
            )

        # Create training and validation loaders using a subset sampler.
        # Intentionally use dicts to define the default dataloader args
        # Then, use dataloader_kwargs to override (or add to) any of these defaults
        # https://stackoverflow.com/questions/44784577/in-method-call-args-how-to-override-keyword-argument-of-unpacked-dict
        train_loader_kwargs = {
            "batch_size": min(training_batch_size, num_training_examples),
            "drop_last": True,
            "sampler": SubsetRandomSampler(self.train_indices.tolist()),
        }
        val_loader_kwargs = {
            "batch_size": min(training_batch_size, num_validation_examples),
            "shuffle": False,
            "drop_last": True,
            "sampler": SubsetRandomSampler(self.val_indices.tolist()),
        }
        if dataloader_kwargs is not None:
            train_loader_kwargs = dict(train_loader_kwargs, **dataloader_kwargs)
            val_loader_kwargs = dict(val_loader_kwargs, **dataloader_kwargs)

        train_loader = data.DataLoader(dataset, **train_loader_kwargs)
        val_loader = data.DataLoader(dataset, **val_loader_kwargs)

        return train_loader, val_loader

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

        assert self._neural_net is not None
        neural_net = self._neural_net

        # (Re)-start the epoch count with the first epoch or any improvement.
        if epoch == 0 or self._val_log_prob > self._best_val_log_prob:
            self._best_val_log_prob = self._val_log_prob
            self._epochs_since_last_improvement = 0
            self._best_model_state_dict = deepcopy(neural_net.state_dict())
        else:
            self._epochs_since_last_improvement += 1

        # If no validation improvement over many epochs, stop training.
        if self._epochs_since_last_improvement > stop_after_epochs - 1:
            neural_net.load_state_dict(self._best_model_state_dict)
            converged = True

        return converged

    def _default_summary_writer(self) -> SummaryWriter:
        """Return summary writer logging to method- and simulator-specific directory."""

        method = self.__class__.__name__
        logdir = Path(
            get_log_root(), method, datetime.now().isoformat().replace(":", "_")
        )
        return SummaryWriter(logdir)

    @staticmethod
    def _describe_round(round_: int, summary: Dict[str, list]) -> str:
        epochs = summary["epochs_trained"][-1]
        best_validation_log_prob = summary["best_validation_log_prob"][-1]

        description = f"""
        -------------------------
        ||||| ROUND {round_ + 1} STATS |||||:
        -------------------------
        Epochs trained: {epochs}
        Best validation performance: {best_validation_log_prob:.4f}
        -------------------------
        """

        return description

    @staticmethod
    def _maybe_show_progress(show: bool, epoch: int) -> None:
        if show:
            # end="\r" deletes the print statement when a new one appears.
            # https://stackoverflow.com/questions/3419984/. `\r` in the beginning due
            # to #330.
            print("\r", f"Training neural network. Epochs trained: {epoch}", end="")

    def _report_convergence_at_end(
        self, epoch: int, stop_after_epochs: int, max_num_epochs: int
    ) -> None:
        if self._converged(epoch, stop_after_epochs):
            print(
                "\r",
                f"Neural network successfully converged after {epoch} epochs.",
                end="",
            )
        elif max_num_epochs == epoch:
            warn(
                "Maximum number of epochs `max_num_epochs={max_num_epochs}` reached,"
                "but network has not yet fully converged. Consider increasing it."
            )

    def _summarize(
        self,
        round_: int,
    ) -> None:
        """Update the summary_writer with statistics for a given round.

        During training several performance statistics are added to the summary, e.g.,
        using `self._summary['key'].append(value)`. This function writes these values
        into summary writer object.

        Args:
            round: index of round

        Scalar tags:
            - epochs_trained:
                number of epochs trained
            - best_validation_log_prob:
                best validation log prob (for each round).
            - validation_log_probs:
                validation log probs for every epoch (for each round).
            - training_log_probs
                training log probs for every epoch (for each round).
            - epoch_durations_sec
                epoch duration for every epoch (for each round)

        """

        # Add most recent training stats to summary writer.
        self._summary_writer.add_scalar(
            tag="epochs_trained",
            scalar_value=self._summary["epochs_trained"][-1],
            global_step=round_ + 1,
        )

        self._summary_writer.add_scalar(
            tag="best_validation_log_prob",
            scalar_value=self._summary["best_validation_log_prob"][-1],
            global_step=round_ + 1,
        )

        # Add validation log prob for every epoch.
        # Offset with all previous epochs.
        offset = (
            torch.tensor(self._summary["epochs_trained"][:-1], dtype=torch.int)
            .sum()
            .item()
        )
        for i, vlp in enumerate(self._summary["validation_log_probs"][offset:]):
            self._summary_writer.add_scalar(
                tag="validation_log_probs",
                scalar_value=vlp,
                global_step=offset + i,
            )

        for i, tlp in enumerate(self._summary["training_log_probs"][offset:]):
            self._summary_writer.add_scalar(
                tag="training_log_probs",
                scalar_value=tlp,
                global_step=offset + i,
            )

        for i, eds in enumerate(self._summary["epoch_durations_sec"][offset:]):
            self._summary_writer.add_scalar(
                tag="epoch_durations_sec",
                scalar_value=eds,
                global_step=offset + i,
            )

        self._summary_writer.flush()

    @property
    def summary(self):
        return self._summary

    def __getstate__(self) -> Dict:
        """Returns the state of the object that is supposed to be pickled.

        Attributes that can not be serialized are set to `None`.

        Returns:
            Dictionary containing the state.
        """
        warn(
            "When the inference object is pickled, the behaviour of the loaded object "
            "changes in the following two ways: "
            "1) `.train(..., retrain_from_scratch=True)` is not supported. "
            "2) When the loaded object calls the `.train()` method, it generates a new "
            "tensorboard summary writer (instead of appending to the current one)."
        )
        dict_to_save = {}
        unpicklable_attributes = ["_summary_writer", "_build_neural_net"]
        for key in self.__dict__.keys():
            if key in unpicklable_attributes:
                dict_to_save[key] = None
            else:
                dict_to_save[key] = self.__dict__[key]
        return dict_to_save

    def __setstate__(self, state_dict: Dict):
        """Sets the state when being loaded from pickle.

        Also creates a new summary writer (because the previous one was set to `None`
        during serializing, see `__get_state__()`).

        Args:
            state_dict: State to be restored.
        """
        state_dict["_summary_writer"] = self._default_summary_writer()
        self.__dict__ = state_dict


def simulate_for_sbi(
    simulator: Callable,
    proposal: Any,
    num_simulations: int,
    num_workers: int = 1,
    simulation_batch_size: int = 1,
    show_progress_bar: bool = True,
) -> Tuple[Tensor, Tensor]:
    r"""Returns ($\theta, x$) pairs obtained from sampling the proposal and simulating.

    This function performs two steps:

    - Sample parameters $\theta$ from the `proposal`.
    - Simulate these parameters to obtain $x$.

    Args:
        simulator: A function that takes parameters $\theta$ and maps them to
            simulations, or observations, `x`, $\text{sim}(\theta)\to x$. Any
            regular Python callable (i.e. function or class with `__call__` method)
            can be used.
        proposal: Probability distribution that the parameters $\theta$ are sampled
            from.
        num_simulations: Number of simulations that are run.
        num_workers: Number of parallel workers to use for simulations.
        simulation_batch_size: Number of parameter sets that the simulator
            maps to data x at once. If None, we simulate all parameter sets at the
            same time. If >= 1, the simulator has to process data of shape
            (simulation_batch_size, parameter_dimension).
        show_progress_bar: Whether to show a progress bar for simulating. This will not
            affect whether there will be a progressbar while drawing samples from the
            proposal.

    Returns: Sampled parameters $\theta$ and simulation-outputs $x$.
    """

    theta = proposal.sample((num_simulations,))

    x = simulate_in_batches(
        simulator, theta, simulation_batch_size, num_workers, show_progress_bar
    )

    return theta, x


def check_if_proposal_has_default_x(proposal: Any):
    """Check for validity of the provided proposal distribution.

    If the proposal is a `NeuralPosterior`, we check if the default_x is set and
    if it matches the `_x_o_training_focused_on`.
    """
    if isinstance(proposal, NeuralPosterior):
        if proposal.default_x is None:
            raise ValueError(
                "`proposal.default_x` is None, i.e. there is no "
                "x_o for training. Set it with "
                "`posterior.set_default_x(x_o)`."
            )
