# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

from copy import deepcopy
from dataclasses import asdict
from typing import Any, Dict, Literal, Optional, Sequence, Tuple, Union

from torch import Tensor, ones
from torch.distributions import Distribution
from torch.utils.tensorboard.writer import SummaryWriter
from typing_extensions import Self

from sbi import utils as utils
from sbi.inference.posteriors.base_posterior import NeuralPosterior
from sbi.inference.posteriors.direct_posterior import DirectPosterior
from sbi.inference.posteriors.filtered_direct_posterior import FilteredDirectPosterior
from sbi.inference.posteriors.posterior_parameters import (
    DirectPosteriorParameters,
    FilteredDirectPosteriorParameters,
    ImportanceSamplingPosteriorParameters,
    RejectionPosteriorParameters,
)
from sbi.inference.potentials import posterior_estimator_based_potential
from sbi.inference.potentials.posterior_based_potential import PosteriorBasedPotential
from sbi.inference.trainers._contracts import LossArgs, StartIndexContext
from sbi.inference.trainers.base import NeuralInference
from sbi.neural_nets import posterior_nn
from sbi.neural_nets.estimators.base import (
    ConditionalDensityEstimator,
    ConditionalEstimatorBuilder,
)
from sbi.neural_nets.estimators.tabpfn_flow import TabPFNFlow
from sbi.sbi_types import TorchTransform, Tracker
from sbi.utils import (
    handle_invalid_x,
    npe_msg_on_invalid_x,
    validate_theta_and_x,
    warn_if_invalid_for_zscoring,
)
from sbi.utils.sbiutils import ImproperEmpirical, mask_sims_from_prior


class NPE_PFN(NeuralInference[ConditionalDensityEstimator]):
    r"""Neural Posterior Estimation with TabPFN (NPE-PFN).

    NPE-PFN is a training-free NPE variant based on in-context learning with
    `TabPFNFlow`. Simulations are stored as context pairs `(theta, x)` and the
    posterior is built directly from this context without gradient-based training.

    The current implementation supports single-round inference only.
    """

    def __init__(
        self,
        prior: Optional[Distribution] = None,
        density_estimator: Optional[ConditionalEstimatorBuilder[TabPFNFlow]] = None,
        device: str = "cpu",
        logging_level: Union[int, str] = "WARNING",
        summary_writer: Optional[SummaryWriter] = None,
        tracker: Optional[Tracker] = None,
        show_progress_bars: bool = True,
    ):
        r"""Initialize NPE-PFN.

        Args:
            prior: A probability distribution that expresses prior knowledge about the
                parameters, e.g. which ranges are meaningful for them.
            density_estimator: Optional custom builder for the density estimator.
                When `None`, a `TabPFNFlow` estimator is constructed via `posterior_nn`.
                Otherwise, a function that builds such a estimator needs to be provided.
            device: Training device, e.g., "cpu", "cuda" or "cuda:{0, 1, ...}".
            logging_level: Minimum severity of messages to log. One of the strings
                INFO, WARNING, DEBUG, ERROR and CRITICAL.
            summary_writer: Deprecated alias for the TensorBoard summary writer.
                Use ``tracker`` instead.
            tracker: Tracking adapter used to log training metrics. If None, a
                TensorBoard tracker is used with a default log directory.
            show_progress_bars: Whether to show a progressbar during training.
        """

        super().__init__(
            prior=prior,
            device=device,
            logging_level=logging_level,
            summary_writer=summary_writer,
            tracker=tracker,
            show_progress_bars=show_progress_bars,
        )

        if density_estimator is None:
            self._build_neural_net = posterior_nn(
                model="tabpfn",
                z_score_theta="none",
                z_score_x="none",
            )
        else:
            self._build_neural_net = density_estimator

        self._proposal_roundwise = []

    def append_simulations(
        self,
        theta: Tensor,
        x: Tensor,
        proposal: Optional[DirectPosterior] = None,
        exclude_invalid_x: Optional[bool] = None,
        data_device: Optional[str] = None,
    ) -> Self:
        r"""Store parameters and simulation outputs for posterior construction.

        Data are stored as entries in lists for each type of variable (parameter/data).

        Stores $\theta$, $x$, prior_masks (indicating if simulations are coming from the
        prior or not) and an index indicating which round the batch of simulations came
        from.

        Args:
            theta: Parameter sets.
            x: Simulation outputs.
            proposal: The distribution that the parameters $\theta$ were sampled from.
                Pass `None` if the parameters were sampled from the prior. Multi-round
                training is not yet implemented, so anything other than `None` will
                raise an error.
            exclude_invalid_x: Whether invalid simulations are discarded during
                training. For single-round training, it is fine to discard invalid
                simulations, but for multi-round sequential (atomic) training,
                discarding invalid simulations gives systematically wrong results. If
                `None`, it will be `True` in the first round and `False` in later
                rounds. Note that multi-round training is not yet implemented.
            data_device: Where to store the data, default is on the same device where
                the training is happening. If training a large dataset on a GPU with not
                much VRAM can set to 'cpu' to store data on system memory instead.

        Returns:
            `self` (returned so that this function is chainable).
        """
        inference_name = self.__class__.__name__
        assert proposal is None, (
            f"Multi-round {inference_name} is not yet implemented. "
            f"Please use single-round {inference_name}."
        )
        current_round = 0

        if exclude_invalid_x is None:
            exclude_invalid_x = current_round == 0

        if data_device is None:
            data_device = self._device

        theta, x = validate_theta_and_x(
            theta, x, data_device=data_device, training_device=self._device
        )

        is_valid_x, num_nans, num_infs = handle_invalid_x(
            x, exclude_invalid_x=exclude_invalid_x
        )

        x = x[is_valid_x]
        theta = theta[is_valid_x]

        # Check for problematic z-scoring
        warn_if_invalid_for_zscoring(x)

        npe_msg_on_invalid_x(
            num_nans,
            num_infs,
            exclude_invalid_x,
            algorithm=f"Single-round {inference_name}",
        )

        self._data_round_index.append(current_round)
        prior_masks = mask_sims_from_prior(int(current_round > 0), theta.size(0))

        self._theta_roundwise.append(theta)
        self._x_roundwise.append(x)
        self._prior_masks.append(prior_masks)

        self._proposal_roundwise.append(proposal)

        if self._prior is None or isinstance(self._prior, ImproperEmpirical):
            theta_prior = self.get_simulations()[0].to(self._device)
            self._prior = ImproperEmpirical(
                theta_prior, ones(theta_prior.shape[0], device=self._device)
            )

        return self

    def train(self) -> None:
        r"""NPE-PFN is training-free and therefore has no training step."""
        pass

    def build_posterior(
        self,
        density_estimator: Optional[TabPFNFlow] = None,
        prior: Optional[Distribution] = None,
        sample_with: Literal[
            "direct", "filtered_direct", "rejection", "importance"
        ] = "filtered_direct",
        direct_sampling_parameters: Optional[Dict[str, Any]] = None,
        filtered_direct_sampling_parameters: Optional[Dict[str, Any]] = None,
        rejection_sampling_parameters: Optional[Dict[str, Any]] = None,
        importance_sampling_parameters: Optional[Dict[str, Any]] = None,
        posterior_parameters: Optional[
            Union[
                DirectPosteriorParameters,
                FilteredDirectPosteriorParameters,
                RejectionPosteriorParameters,
                ImportanceSamplingPosteriorParameters,
            ]
        ] = None,
        discard_prior_samples: bool = False,
    ) -> NeuralPosterior:
        r"""Build a posterior from a `TabPFNFlow` estimator.

        For `sample_with="filtered_direct"`, the returned posterior dynamically
        filters the context for each queried observation before evaluating
        probabilities or drawing samples.

        Args:
            density_estimator: The density estimator that the posterior is based on.
                If `None`, use the latest neural density estimator that was trained.
            prior: Prior distribution.
            sample_with: Method used for posterior sampling. One of
                [`direct`, `filtered_direct`, `rejection`, `importance`].
            direct_sampling_parameters: Additional kwargs passed to `DirectPosterior`.
            filtered_direct_sampling_parameters: Additional kwargs passed to
                `FilteredDirectPosterior`. Context tensors are derived from stored
                simulations and combined with these overrides. Supported keys include
                `filter_size` and `filter_type` (`'knn'`, `'first'`, or a callable
                returning indices).
            rejection_sampling_parameters: Additional kwargs passed to
                `RejectionPosterior`.
            importance_sampling_parameters: Additional kwargs passed to
                `ImportanceSamplingPosterior`.
            posterior_parameters: Configuration passed to the init method for the
                posterior. Must be one of the following
                - `ImportanceSamplingPosteriorParameters`
                - `DirectPosteriorParameters`
                - `FilteredDirectPosteriorParameters`
                - `RejectionPosteriorParameters`
            discard_prior_samples: Whether to discard samples simulated in round 0 of
                a multi-round procedure. Should only be `True` in a multi-round setting.

        Returns:
            Posterior $p(\theta|x)$  with `.sample()` and `.log_prob()` methods
            (the returned log-probability is unnormalized).
        """

        if len(self._data_round_index) == 0:
            raise RuntimeError(
                "No simulations found. You must call .append_simulations() "
                "before building the NPE_PFN posterior."
            )

        start_idx = self._get_start_index(
            context=StartIndexContext(
                discard_prior_samples=discard_prior_samples,
            )
        )

        full_theta, full_x, _ = self.get_simulations(starting_round=start_idx)
        if density_estimator is None:
            self._neural_net = self._build_neural_net(
                full_theta.to("cpu"), full_x.to("cpu")
            )

        estimator, device = self._resolve_estimator(density_estimator)
        if not isinstance(estimator, TabPFNFlow):
            raise TypeError(
                f"Expected estimator to be TabPFNFlow, got {type(estimator).__name__}."
            )

        if sample_with == "filtered_direct":
            full_context_input, full_context_condition = full_theta, full_x

            prior = self._resolve_prior(prior, sample_with)
            estimator = deepcopy(estimator)

            resolved_params = self._resolve_posterior_parameters(
                sample_with,
                posterior_parameters,
                filtered_direct_sampling_parameters=filtered_direct_sampling_parameters,
            )

            self._posterior = FilteredDirectPosterior(
                estimator=estimator,
                prior=prior,
                full_context_input=full_context_input,
                full_context_condition=full_context_condition,
                device=device,
                **asdict(resolved_params),
            )
            return self._posterior

        estimator.set_context(full_theta, full_x)

        return super().build_posterior(
            estimator,
            prior,
            sample_with,
            posterior_parameters,
            direct_sampling_parameters=direct_sampling_parameters,
            rejection_sampling_parameters=rejection_sampling_parameters,
            importance_sampling_parameters=importance_sampling_parameters,
        )

    def _get_start_index(self, context: StartIndexContext) -> int:
        """
        Determine the starting index for stored simulation rounds.

        Args:
            context: `StartIndexContext` values used to determine the starting
                index in the stored simulation data.
        Returns:
            `1` to skip round-0 samples, otherwise `0`.
        """

        # Load data from most recent round.
        self._round = max(self._data_round_index)

        # Starting index for the training set (1 = discard round-0 samples).
        start_idx = int(context.discard_prior_samples and self._round > 0)

        return start_idx

    def _get_potential_function(
        self,
        prior: Distribution,
        estimator: ConditionalDensityEstimator,
    ) -> Tuple[PosteriorBasedPotential, TorchTransform]:
        r"""Gets the potential for posterior-based methods.

        It also returns a transformation that can be used to transform the potential
        into unconstrained space.

        The potential is the same as the log-probability of the `posterior_estimator`,
        but it is set to $-\inf$ outside of the prior bounds.

        Args:
            prior: The prior distribution.
            estimator: The neural network modelling the posterior.

        Returns:
            The potential function and a transformation that maps
            to unconstrained space.
        """

        potential_fn, theta_transform = posterior_estimator_based_potential(
            posterior_estimator=estimator,
            prior=prior,
            x_o=None,
        )
        return potential_fn, theta_transform

    def _get_losses(self, batch: Sequence[Tensor], loss_args: LossArgs) -> Tensor:
        """Not implemented because NPE-PFN does not optimize a training loss."""
        raise NotImplementedError(
            "NPE-PFN is training-free. Fine-tuning is currently not implemented."
        )

    def _loss(self, *args, **kwargs) -> Tensor:
        """Not implemented because NPE-PFN does not perform gradient training."""
        raise NotImplementedError(
            "NPE-PFN is training-free. Fine-tuning is currently not implemented."
        )

    def _initialize_neural_network(self, retrain_from_scratch, start_idx):
        """Not implemented because the estimator is rebuilt from context data."""
        raise NotImplementedError(
            "NPE-PFN is based on in-context learning. "
            "The underlying density estimator will automatically "
            "reflect updates to the training dataset "
            "after rebuilding the posterior."
        )
