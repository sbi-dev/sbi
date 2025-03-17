# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

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

from sbi.inference.posteriors.base_posterior import NeuralPosterior
from sbi.utils import (
    check_prior,
    get_log_root,
    handle_invalid_x,
    mask_sims_from_prior,
    nle_nre_apt_msg_on_invalid_x,
    validate_theta_and_x,
    warn_if_zscoring_changes_data,
)
from sbi.utils.sbiutils import get_simulations_since_round
from sbi.utils.simulation_utils import simulate_for_sbi
from sbi.utils.torchutils import check_if_prior_on_device, process_device
from sbi.utils.user_input_checks import (
    check_sbi_inputs,
    process_prior,
    process_simulator,
)


def infer(
    simulator: Callable,
    prior: Distribution,
    method: str,
    num_simulations: int,
    num_workers: int = 1,
    init_kwargs: Optional[Dict] = None,
    train_kwargs: Optional[Dict] = None,
    build_posterior_kwargs: Optional[Dict] = None,
) -> NeuralPosterior:
    r"""Runs simulation-based inference and returns the posterior.

    This function provides a simple interface to run sbi. Inference is run for a single
    round and hence the returned posterior $p(\theta|x)$ can be sampled and evaluated
    for any $x$ (i.e. it is amortized).

    The scope of this function is limited to the most essential features of sbi. For
    more flexibility (e.g. multi-round inference, different density estimators) please
    use the flexible interface described here:
    https://sbi-dev.github.io/sbi/latest/tutorials/02_multiround_inference/

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
        init_kwargs: Additional keyword arguments for the inference method
            which are passed to `__init__`.
        train_kwargs: Additional keyword arguments for training the density estimator.
        build_posterior_kwargs: Additional keyword arguments for `build_posterior`.

    Returns: Posterior over parameters conditional on observations (amortized).
    """

    try:
        # Moved here to avoid circular imports at initialization.
        import sbi.inference  # noqa: R0401

        method_fun: Callable = getattr(sbi.inference, method.upper())
    except AttributeError as err:
        raise NameError(
            "Method not available. `method` must be one of 'SNPE', 'SNLE', 'SNRE'."
        ) from err

    if (
        init_kwargs is not None
        or build_posterior_kwargs is not None
        or train_kwargs is not None
    ):
        warn(
            "We discourage the use the simple interface in more complicated settings. "
            "Have a look into the flexible interface, e.g. in our tutorial "
            "(https://sbi-dev.github.io/sbi/latest/tutorials/00_getting_started).",
            stacklevel=2,
        )
    # Set variables to empty dicts to be able to pass them
    # to the functions later on (if necessary).
    if build_posterior_kwargs is None:
        build_posterior_kwargs = {}
    if train_kwargs is None:
        train_kwargs = {}
    if init_kwargs is None:
        init_kwargs = {}

    prior, _, prior_returns_numpy = process_prior(prior)
    simulator = process_simulator(simulator, prior, prior_returns_numpy)
    check_sbi_inputs(simulator, prior)

    inference = method_fun(prior=prior, **init_kwargs)
    theta, x = simulate_for_sbi(
        simulator=simulator,
        proposal=prior,
        num_simulations=num_simulations,
        num_workers=num_workers,
    )
    _ = inference.append_simulations(theta, x).train(**train_kwargs)
    posterior = inference.build_posterior(**build_posterior_kwargs)

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
        self._val_loss = float("Inf")

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
            best_validation_loss=[],
            validation_loss=[],
            training_loss=[],
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
    def append_simulations(
        self,
        theta: Tensor,
        x: Tensor,
        exclude_invalid_x: bool = False,
        from_round: int = 0,
        algorithm: Optional[str] = None,
        data_device: Optional[str] = None,
    ) -> "NeuralInference":
        r"""Store parameters and simulation outputs to use them for later training.

        Data are stored as entries in lists for each type of variable (parameter/data).

        Stores $\theta$, $x$, prior_masks (indicating if simulations are coming from the
        prior or not) and an index indicating which round the batch of simulations came
        from.

        Args:
            theta: Parameter sets.
            x: Simulation outputs.
            exclude_invalid_x: Whether invalid simulations are discarded during
                training. If `False`, The inference algorithm raises an error when
                invalid simulations are found. If `True`, invalid simulations are
                discarded and training can proceed, but this gives systematically wrong
                results.
            from_round: Which round the data stemmed from. Round 0 means from the prior.
                With default settings, this is not used at all for the inference
                algorithm. Only when the user later on requests
                `.train(discard_prior_samples=True)`, we use these indices to find which
                training data stemmed from the prior.
            algorithm: Which algorithm is used. This is used to give a more informative
                warning or error message when invalid simulations are found.
            data_device: Where to store the data, default is on the same device where
                the training is happening. If training a large dataset on a GPU with not
                much VRAM can set to 'cpu' to store data on system memory instead.
        Returns:
            NeuralInference object (returned so that this function is chainable).
        """

        is_valid_x, num_nans, num_infs = handle_invalid_x(x, exclude_invalid_x)

        x = x[is_valid_x]
        theta = theta[is_valid_x]

        # Check for problematic z-scoring
        warn_if_zscoring_changes_data(x)
        nle_nre_apt_msg_on_invalid_x(
            num_nans, num_infs, exclude_invalid_x, algorithm or type(self).__name__
        )

        if data_device is None:
            data_device = self._device
        theta, x = validate_theta_and_x(
            theta, x, data_device=data_device, training_device=self._device
        )

        prior_masks = mask_sims_from_prior(int(from_round), theta.size(0))

        self._theta_roundwise.append(theta)
        self._x_roundwise.append(x)
        self._prior_masks.append(prior_masks)

        self._data_round_index.append(int(from_round))

        return self

    @abstractmethod
    def train(
        self,
        training_batch_size: int = 200,
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
        training_batch_size: int = 200,
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
        if epoch == 0 or self._val_loss < self._best_val_loss:
            self._best_val_loss = self._val_loss
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
        best_validation_loss = summary["best_validation_loss"][-1]

        description = f"""
        -------------------------
        ||||| ROUND {round_ + 1} STATS |||||:
        -------------------------
        Epochs trained: {epochs}
        Best validation performance: {best_validation_loss:.4f}
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
                "but network has not yet fully converged. Consider increasing it.",
                stacklevel=2,
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
            - best_validation_loss:
                best validation loss (for each round).
            - validation_loss:
                validation loss for every epoch (for each round).
            - training_loss
                training loss for every epoch (for each round).
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
            tag="best_validation_loss",
            scalar_value=self._summary["best_validation_loss"][-1],
            global_step=round_ + 1,
        )

        # Add validation loss for every epoch.
        # Offset with all previous epochs.
        offset = (
            torch.tensor(self._summary["epochs_trained"][:-1], dtype=torch.int)
            .sum()
            .item()
        )
        for i, vlp in enumerate(self._summary["validation_loss"][offset:]):
            self._summary_writer.add_scalar(
                tag="validation_loss",
                scalar_value=vlp,
                global_step=offset + i,
            )

        for i, tlp in enumerate(self._summary["training_loss"][offset:]):
            self._summary_writer.add_scalar(
                tag="training_loss",
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
            "tensorboard summary writer (instead of appending to the current one).",
            stacklevel=2,
        )
        dict_to_save = {}
        unpicklable_attributes = ["_summary_writer", "_build_neural_net"]
        for key in self.__dict__:
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


def check_if_proposal_has_default_x(proposal: Any):
    """Check for validity of the provided proposal distribution.

    If the proposal is a `NeuralPosterior`, we check if the default_x is set and
    if it matches the `_x_o_training_focused_on`.
    """
    if isinstance(proposal, NeuralPosterior) and proposal.default_x is None:
        raise ValueError(
            "`proposal.default_x` is None, i.e. there is no "
            "x_o for training. Set it with "
            "`posterior.set_default_x(x_o)`."
        )
