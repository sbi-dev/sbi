# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

import time
import warnings
from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    List,
    Literal,
    Optional,
    Sequence,
    Tuple,
    Union,
    overload,
)
from warnings import warn

import torch
from torch import Tensor
from torch.distributions import Distribution
from torch.nn.utils.clip_grad import clip_grad_norm_
from torch.optim.adam import Adam
from torch.utils import data
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.tensorboard.writer import SummaryWriter
from typing_extensions import Self

from sbi.inference.posteriors.base_posterior import NeuralPosterior
from sbi.inference.posteriors.direct_posterior import DirectPosterior
from sbi.inference.posteriors.importance_posterior import ImportanceSamplingPosterior
from sbi.inference.posteriors.mcmc_posterior import MCMCPosterior
from sbi.inference.posteriors.posterior_parameters import (
    DirectPosteriorParameters,
    ImportanceSamplingPosteriorParameters,
    MCMCPosteriorParameters,
    PosteriorParameters,
    RejectionPosteriorParameters,
    VIPosteriorParameters,
    VectorFieldPosteriorParameters,
)
from sbi.inference.posteriors.rejection_posterior import RejectionPosterior
from sbi.inference.posteriors.vector_field_posterior import VectorFieldPosterior
from sbi.inference.posteriors.vi_posterior import VIPosterior
from sbi.inference.potentials.base_potential import BasePotential
from sbi.inference.trainers._contracts import LossArgs, StartIndexContext, TrainConfig
from sbi.neural_nets.estimators.base import (
    ConditionalDensityEstimator,
    ConditionalEstimator,
    ConditionalEstimatorType,
    ConditionalVectorFieldEstimator,
)
from sbi.sbi_types import TorchTransform
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


class NeuralInference(ABC, Generic[ConditionalEstimatorType]):
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
        self._best_val_loss = float("Inf")
        self._epochs_since_last_improvement = 0

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

    @property
    def summary(self):
        return self._summary

    @abstractmethod
    def append_simulations(
        self,
        theta: Tensor,
        x: Tensor,
        exclude_invalid_x: bool = False,
        from_round: int = 0,
        algorithm: Optional[str] = None,
        data_device: Optional[str] = None,
    ) -> Self:
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
    ) -> ConditionalEstimatorType: ...

    @abstractmethod
    def _initialize_neural_network(
        self, retrain_from_scratch: bool, start_idx: int
    ) -> None:
        """Initialize (or reinitialize) the neural network for the current round."""
        ...

    @abstractmethod
    def _get_start_index(self, context: StartIndexContext) -> int:
        """Get the starting index for the current round."""
        ...

    @overload
    def _get_losses(self, batch: Sequence[Tensor]) -> Tensor:
        """
        Called when the trainer does not require additional loss arguments
        (e.g., NLE).
        """
        ...

    @overload
    def _get_losses(self, batch: Sequence[Tensor], loss_args: LossArgs) -> Tensor:
        """Called when the trainer requires additional loss parameters via loss_args."""
        ...

    @abstractmethod
    def _get_losses(
        self, batch: Sequence[Tensor], loss_args: LossArgs | None = None
    ) -> Tensor:
        """Return per-sample loss tensor for a training/validation batch."""
        ...

    @abstractmethod
    def _get_potential_function(
        self,
        prior: Distribution,
        estimator: ConditionalEstimator,
    ) -> Tuple[BasePotential, TorchTransform]:
        """Subclass-specific potential creation"""
        ...

    @abstractmethod
    def _loss(self, *args, **kwargs) -> Tensor:
        """Compute scalar loss given subclass-specific inputs."""
        ...

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
            # Separate indices for training and validation
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

    def build_posterior(
        self,
        estimator: Optional[ConditionalEstimator],
        prior: Optional[Distribution],
        sample_with: Literal[
            "mcmc", "rejection", "vi", "importance", "direct", "sde", "ode"
        ],
        posterior_parameters: Optional[PosteriorParameters],
        **kwargs,
    ) -> NeuralPosterior:
        r"""Method for building posteriors.

        This method serves as a base method for constructing a posterior based
        on a given estimator and prior. The posterior can be sampled using one of
        several inference methods specified by `sample_with`.

        Args:
            estimator: The estimator that the posterior is based on.
            prior: A probability distribution that expresses prior knowledge about the
                parameters, e.g. which ranges are meaningful for them. Must be a PyTorch
                distribution, see FAQ for details on how to use custom distributions.
            sample_with: The inference method to use. Must be one of:
                - "mcmc"
                - "rejection"
                - "vi"
                - "importance"
                - "direct"
                - "sde"
                - "ode"
            posterior_parameters: Configuration passed to the init method for the
                posterior. Must be of type PosteriorParameters.
            **kwargs: Additional method-specific parameters.

        Returns:
            NeuralPosterior object.
        """

        prior = self._resolve_prior(prior)
        estimator, device = self._resolve_estimator(estimator)

        posterior_parameters = self._resolve_posterior_parameters(
            sample_with, posterior_parameters, **kwargs
        )

        self._posterior = self._create_posterior(
            estimator,
            prior,
            sample_with,
            device,
            posterior_parameters,
        )

        # Store models at end of each round.
        self._model_bank.append(deepcopy(self._posterior))

        return deepcopy(self._posterior)

    def _resolve_prior(self, prior: Optional[Distribution]) -> Distribution:
        """
        Resolves the prior distribution to use.

        If a prior is passed, it is validated and returned.
        If not passed, attempts to use the stored `self._prior`.
        Raises a ValueError if no valid prior is available.

        Args:
            prior: Optional prior distribution to resolve.

        Returns:
            A valid prior distribution.
        """

        if prior is None:
            if self._prior is None:
                cls_name = self.__class__.__name__
                raise ValueError(
                    f"""You did not pass a prior. You have to pass the prior either at
                    initialization `inference = {cls_name}(prior)` or to `
                    .build_posterior (prior=prior)`."""
                )
            prior = self._prior
        else:
            check_prior(prior)

        return prior

    def _resolve_estimator(
        self, estimator: Optional[ConditionalEstimator]
    ) -> Tuple[ConditionalEstimator, str]:
        """
        Resolves the estimator and determines its device.

        If no estimator is provided, the internal neural net (`self._neural_net`)
        is used and the device is taken from `self._device`. Otherwise, validates
        the passed estimator and infers its device.

        Args:
            estimator: Optional estimator to use.

        Returns:
            A tuple of (estimator, device).
        """

        if estimator is None:
            assert self._neural_net is not None, (
                "Provide an estimator or initialize self._neural_net."
            )
            estimator = self._neural_net
            # If internal net is used device is defined.
            device = self._device
        else:
            if not isinstance(estimator, ConditionalEstimator):
                raise TypeError(
                    "estimator must be ConditionalEstimator,"
                    f" got {type(estimator).__name__}",
                )
            # Otherwise, infer it from the device of the net parameters.
            device = str(next(estimator.parameters()).device)

        return estimator, device

    def _resolve_posterior_parameters(
        self,
        sample_with: Literal[
            "mcmc", "rejection", "vi", "importance", "direct", "sde", "ode"
        ],
        posterior_parameters: Optional[PosteriorParameters],
        **kwargs,
    ) -> PosteriorParameters:
        """
        Resolve posterior parameters based on the sampling strategy.

        If `posterior_parameters` is provided, it is returned directly.

        If `posterior_parameters` is not provided, this method extracts
        sampling-specific parameters from `kwargs` using predefined keys
        to instantiate the appropriate posterior parameters dataclass.

        Raises:
            NotImplementedError: If an unsupported `sample_with` method is provided.
            ValueError: If posterior_parameter and a configuration dictionary are passed
                together.

        Args:
            sample_with: The posterior sampling method to use.
            posterior_parameters: Optional preconstructed posterior parameter object.
            **kwargs: Additional parameters to construct the posterior parameters.

        Returns:
            A dataclass instance containing the resolved posterior
            parameters.
        """

        deprecated_params = self._resolve_deprecated_posterior_parameters(**kwargs)

        if posterior_parameters is not None:
            self._validate_no_duplicate_parameters(deprecated_params)
            self._validate_posterior_parameters_consistency(
                posterior_parameters, **kwargs
            )
        else:
            self._raise_deprecation_warning(deprecated_params, **kwargs)
            posterior_parameters = self._build_posterior_parameters(
                sample_with, **kwargs
            )

        return posterior_parameters

    def _build_posterior_parameters(
        self,
        sample_with: Literal[
            "mcmc", "rejection", "vi", "importance", "direct", "sde", "ode"
        ],
        **kwargs,
    ) -> PosteriorParameters:
        """
        Resolve parameters passed through kwargs and convert into a
        subclass of PosteriorParameters.

        Args:
            sample_with: The posterior sampling method to use.
            **kwargs: Additional parameters to construct the posterior parameters.
        Returns
            A dataclass instance containing the resolved posterior
            parameters.
        """

        if sample_with == "direct":
            params = kwargs.get("direct_sampling_parameters", {}) or {}
            posterior_parameters = DirectPosteriorParameters(**params)
        elif sample_with == "mcmc":
            params = kwargs.get("mcmc_parameters", {}) or {}
            posterior_parameters = MCMCPosteriorParameters(
                method=kwargs.get("mcmc_method", "slice_np_vectorized"), **params
            )
        elif sample_with in ("ode", "sde"):
            params = kwargs.get("vectorfield_sampling_parameters", {}) or {}
            posterior_parameters = VectorFieldPosteriorParameters(**params)
        elif sample_with == "rejection":
            params = kwargs.get("rejection_sampling_parameters", {}) or {}
            posterior_parameters = RejectionPosteriorParameters(**params)
        elif sample_with == "vi":
            params = kwargs.get("vi_parameters", {}) or {}
            posterior_parameters = VIPosteriorParameters(
                vi_method=kwargs.get("vi_method", "rKL"), **params
            )
        elif sample_with == "importance":
            params = kwargs.get("importance_sampling_parameters", {}) or {}
            posterior_parameters = ImportanceSamplingPosteriorParameters(**params)
        else:
            raise NotImplementedError(
                "Posterior parameter construction not implemented for",
                f"'{sample_with}'",
            )

        return posterior_parameters

    def _resolve_deprecated_posterior_parameters(self, **kwargs) -> List[str]:
        """
        Identify deprecated posterior construction parameters
        provided to the method.

        Args:
            **kwargs: Keyword arguments potentially containing deprecated
                      posterior parameters.

        Returns:
            A list of names of deprecated posterior parameters that were provided.
        """

        deprecated_params = {
            "direct_sampling_parameters",
            "mcmc_parameters",
            "vectorfield_sampling_parameters",
            "rejection_sampling_parameters",
            "vi_parameters",
            "importance_sampling_parameters",
        }

        # Check if any deprecated parameters were provided
        provided_deprecated_params = [
            param for param in deprecated_params if kwargs.get(param) is not None
        ]

        return provided_deprecated_params

    def _raise_deprecation_warning(
        self, deprecated_params: List[str], **kwargs
    ) -> None:
        """
        Raise a deprecation warning if a deprecated posterior parameters or
        non-default arguments are used.

        Args:
            deprecated_params: List of deprecated posterior parameter names provided.
            **kwargs: Additional keyword arguments.
        """

        deprecated_params = deprecated_params.copy()
        default_mcmc_method = "slice_np_vectorized"
        default_vi_method = "rKL"

        # Check if deprecated parameters are used
        if (
            kwargs.get("mcmc_method") is not None
            and kwargs.get("mcmc_method") != default_mcmc_method
        ):
            deprecated_params.append("mcmc_method")
        if (
            kwargs.get("vi_method") is not None
            and kwargs.get("vi_method") != default_vi_method
        ):
            deprecated_params.append("vi_method")

        if deprecated_params:
            warnings.warn(
                f"The following arguments are deprecated and"
                " will be removed in a future version: "
                f"{', '.join(deprecated_params)}. Please use `posterior_parameters`"
                " instead. Refer to this guide for details:\n"
                "https://sbi.readthedocs.io/en/latest/how_to_guide/19_posterior_parameters.html#",
                FutureWarning,
                stacklevel=2,
            )

    def _validate_no_duplicate_parameters(self, deprecated_params: List[str]) -> None:
        """
        Validate that deprecated and new-style posterior parameters are not used
        together.

        Args:
            deprecated_params: List of deprecated posterior parameter names provided.

        Raises:
            ValueError: If both deprecated parameters and new-style
                        `posterior_parameters`are used in the same call.
        """

        if deprecated_params:
            raise ValueError(
                f"Cannot use both old-style parameters {deprecated_params} "
                f"and new-style posterior_parameters. Please use only one approach."
            )

    def _validate_posterior_parameters_consistency(
        self, posterior_parameters: PosteriorParameters, **kwargs
    ) -> None:
        """
        This method raises a warning for mismatches between values passed in
        mcmc_method and MCMCPosteriorParameters.method, or vi_method and
        VIPosteriorParameters.vi_method.

        Args:
            posterior_parameters: Configuration passed to the init method for the
                posterior.
            kwargs: keyword arguments passed from build_posterior method.
        """

        if not isinstance(posterior_parameters, PosteriorParameters):
            raise TypeError(
                "posterior_parameters must be PosteriorParameters,"
                f" got {type(posterior_parameters).__name__}",
            )
        elif isinstance(posterior_parameters, MCMCPosteriorParameters):
            mcmc_method = kwargs.get("mcmc_method")
            if (
                mcmc_method != "slice_np_vectorized"
                and posterior_parameters.method != mcmc_method
            ):
                warnings.warn(
                    f"Conflicting mcmc_method='{mcmc_method}' ignored in favor of "
                    f"posterior_parameters.method='{posterior_parameters.method}'",
                    stacklevel=2,
                )
        elif isinstance(posterior_parameters, VIPosteriorParameters):
            vi_method = kwargs.get("vi_method")
            if vi_method != "rKL" and posterior_parameters.vi_method != vi_method:
                warnings.warn(
                    f"Conflicting vi_method='{vi_method}' ignored in favor of "
                    f"posterior_parameters.vi_method='{posterior_parameters.vi_method}'",
                    stacklevel=2,
                )

    def _create_posterior(
        self,
        estimator: ConditionalEstimator,
        prior: Distribution,
        sample_with: Literal[
            "mcmc", "rejection", "vi", "importance", "direct", "sde", "ode"
        ],
        device: Union[str, torch.device],
        posterior_parameters: PosteriorParameters,
    ) -> NeuralPosterior:
        """
        Create a posterior object using the specified inference method.

        Depending on the value of `sample_with`, this method instantiates one of the
        supported posterior inference strategies.

        Args:
            estimator: The estimator that the posterior is based on.
            prior: A probability distribution that expresses prior knowledge about the
                parameters, e.g. which ranges are meaningful for them. Must be a PyTorch
                distribution, see FAQ for details on how to use custom distributions.
            sample_with: The inference method to use. Must be one of:
                - "mcmc"
                - "rejection"
                - "vi"
                - "importance"
                - "direct"
                - "sde"
                - "ode"
            device: torch device on which to train the neural net and on which to
                perform all posterior operations, e.g. gpu or cpu.
            posterior_parameters: Configuration passed to the init method for the
                posterior. Must be of type PosteriorParameters.

        Returns:
            NeuralPosterior object.
        """

        if isinstance(posterior_parameters, DirectPosteriorParameters):
            posterior_estimator = estimator
            if not isinstance(posterior_estimator, ConditionalDensityEstimator):
                raise TypeError(
                    f"Expected posterior_estimator to be an instance of "
                    " ConditionalDensityEstimator, "
                    f"but got {type(posterior_estimator).__name__} instead."
                )
            posterior = DirectPosterior(
                posterior_estimator=posterior_estimator,
                prior=prior,
                device=device,
                **asdict(posterior_parameters),
            )
        elif isinstance(posterior_parameters, VectorFieldPosteriorParameters):
            vector_field_estimator = estimator
            if not isinstance(vector_field_estimator, ConditionalVectorFieldEstimator):
                raise TypeError(
                    f"Expected vector_field_estimator to be an instance of "
                    " ConditionalVectorFieldEstimator, "
                    f"but got {type(vector_field_estimator).__name__} instead."
                )
            if sample_with not in ("ode", "sde"):
                raise ValueError(
                    "`sample_with` must be either",
                    f" 'ode' or 'sde', got '{sample_with}'",
                )
            posterior = VectorFieldPosterior(
                vector_field_estimator=vector_field_estimator,
                prior=prior,
                device=device,
                sample_with=sample_with,
                **asdict(posterior_parameters),
            )
        else:
            # Posteriors requiring potential_fn and theta_transform
            potential_fn, theta_transform = self._get_potential_function(
                prior, estimator
            )
            if isinstance(posterior_parameters, MCMCPosteriorParameters):
                posterior = MCMCPosterior(
                    potential_fn=potential_fn,
                    theta_transform=theta_transform,
                    proposal=prior,
                    device=device,
                    **asdict(posterior_parameters),
                )
            elif isinstance(posterior_parameters, RejectionPosteriorParameters):
                posterior = RejectionPosterior(
                    potential_fn=potential_fn,
                    proposal=prior,
                    device=device,
                    **asdict(posterior_parameters),
                )
            elif isinstance(posterior_parameters, VIPosteriorParameters):
                posterior = VIPosterior(
                    potential_fn=potential_fn,
                    theta_transform=theta_transform,
                    prior=prior,
                    device=device,
                    **asdict(posterior_parameters),
                )
            elif isinstance(
                posterior_parameters, ImportanceSamplingPosteriorParameters
            ):
                posterior = ImportanceSamplingPosterior(
                    potential_fn=potential_fn,
                    proposal=prior,
                    device=device,
                    **asdict(posterior_parameters),
                )
            else:
                raise NotImplementedError(
                    "Sampling method not implemented for",
                    f"'{posterior_parameters}'",
                )
        return posterior

    def _run_training_loop(
        self,
        train_loader: data.DataLoader,
        val_loader: data.DataLoader,
        train_config: TrainConfig,
        loss_args: LossArgs | None = None,
        summarization_kwargs: Optional[Dict[str, Any]] = None,
    ) -> ConditionalEstimatorType:
        """
        Run the main training loop for the neural network, including epoch-wise
        training, validation, and convergence checking.

        Args:
            train_loader: Dataloader for training.
            val_loader: Dataloader for validation.
            train_config: TrainConfig dataclass configuration for the core training
                path.
            loss_args: Additional arguments passed to self._loss fn.
            summarization_kwargs: Additional kwargs passed to self._summarize_epoch fn.
        """

        if summarization_kwargs is None:
            summarization_kwargs = {}

        assert self._neural_net is not None

        # Move entire net to device for training.
        self._neural_net.to(self._device)

        if not train_config.resume_training:
            self.optimizer = Adam(
                list(self._neural_net.parameters()),
                lr=train_config.learning_rate,
            )
            self.epoch, self.val_loss = 0, float("Inf")

        while self.epoch <= train_config.max_num_epochs and not self._converged(
            self.epoch, train_config.stop_after_epochs
        ):
            # Train for a single epoch.
            self._neural_net.train()
            epoch_start_time = time.time()
            train_loss = self._train_epoch(
                train_loader, train_config.clip_max_norm, loss_args
            )

            # Calculate validation performance.
            self._neural_net.eval()

            self._val_loss = self._validate_epoch(val_loader, loss_args)

            self._summarize_epoch(
                train_loss, self._val_loss, epoch_start_time, summarization_kwargs
            )

            self.epoch += 1
            self._maybe_show_progress(self._show_progress_bars, self.epoch)

        self._report_convergence_at_end(
            self.epoch, train_config.stop_after_epochs, train_config.max_num_epochs
        )

        # Update summary.
        self._summary["epochs_trained"].append(self.epoch)
        self._summary["best_validation_loss"].append(self._best_val_loss)

        # Update TensorBoard and summary dict.
        self._summarize(round_=self._round)

        # Update description for progress bar.
        if train_config.show_train_summary:
            print(self._describe_round(self._round, self._summary))

        # Avoid keeping the gradients in the resulting network, which can
        # cause memory leakage when benchmarking.
        self._neural_net.zero_grad(set_to_none=True)

        return deepcopy(self._neural_net)

    def _train_epoch(
        self,
        train_loader: data.DataLoader,
        clip_max_norm: Optional[float],
        loss_args: LossArgs | None,
    ) -> float:
        """
        Perform a single training epoch over the provided training data.

        Args:
            train_loader: Dataloader for training.
            clip_max_norm: Value at which to clip the total gradient norm in order to
                prevent exploding gradients. Use None for no clipping.
            loss_args: Additional arguments passed to self._loss fn.

        Returns:
            The average training loss over all samples in the epoch.
        """

        assert self._neural_net is not None

        train_loss_sum = 0
        for batch in train_loader:
            self.optimizer.zero_grad()
            if loss_args is None:
                train_losses = self._get_losses(batch=batch)
            else:
                train_losses = self._get_losses(batch=batch, loss_args=loss_args)
            train_loss = torch.mean(train_losses)
            train_loss_sum += train_losses.sum().item()

            train_loss.backward()
            if clip_max_norm is not None:
                clip_grad_norm_(
                    self._neural_net.parameters(),
                    max_norm=clip_max_norm,
                )
            self.optimizer.step()

        train_loss_average = train_loss_sum / (
            len(train_loader) * train_loader.batch_size  # type: ignore
        )

        return train_loss_average

    def _validate_epoch(
        self,
        val_loader: data.DataLoader,
        loss_args: LossArgs | None,
    ) -> float:
        """
        Perform a single validation epoch over the provided validation data.

        Args:
            val_loader: Dataloader for validation.
            loss_args: Additional arguments passed to self._loss fn.

        Returns:
            The average validation loss over all samples in the epoch.
        """

        val_loss_sum = 0
        with torch.no_grad():
            for batch in val_loader:
                if loss_args is None:
                    val_losses = self._get_losses(batch=batch)
                else:
                    val_losses = self._get_losses(batch=batch, loss_args=loss_args)
                val_loss_sum += val_losses.sum().item()

        # Take mean over all validation samples.
        val_loss = val_loss_sum / (
            len(val_loader) * val_loader.batch_size  # type: ignore
        )

        return val_loss

    def _summarize_epoch(
        self,
        train_loss: float,
        val_loss: float,
        epoch_start_time: float,
        summarization_kwargs: Dict[str, Any],
    ) -> None:
        """
        Update internal summaries after a single training epoch.

        Records training and validation losses, as well as the duration of the epoch,
        in `self._summary` dictionary.

        Args:
            train_loss: The average training loss for the epoch.
            val_loss: The average validation loss for the epoch.
            epoch_start_time: Timestamp when the epoch started, used to compute
                duration.
            summarization_kwargs: Additional keyword arguments for customizing
                the summarization.
        """

        self._summary["training_loss"].append(train_loss)
        # Log validation loss for every epoch.
        self._summary["validation_loss"].append(val_loss)
        self._summary["epoch_durations_sec"].append(time.time() - epoch_start_time)

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
                f"Maximum number of epochs `max_num_epochs={max_num_epochs}` reached,"
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
