# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

from typing import Any, Dict, Literal, Optional, Sequence, Tuple, Union

import torch
from torch import Tensor, ones
from torch.distributions import Distribution
from torch.utils.tensorboard.writer import SummaryWriter
from typing_extensions import Self

from sbi import utils as utils
from sbi.inference.posteriors.base_posterior import NeuralPosterior
from sbi.inference.posteriors.direct_posterior import DirectPosterior
from sbi.inference.posteriors.posterior_parameters import (
    DirectPosteriorParameters,
    ImportanceSamplingPosteriorParameters,
    MCMCPosteriorParameters,
    FilteredDirectPosteriorParameters,
    RejectionPosteriorParameters,
    VIPosteriorParameters,
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
from sbi.neural_nets.estimators.shape_handling import (
    reshape_to_batch_event,
    reshape_to_sample_batch_event,
)
from sbi.sbi_types import TorchTransform, Tracker
from sbi.utils import (
    check_estimator_arg,
    handle_invalid_x,
    npe_msg_on_invalid_x,
    test_posterior_net_for_multi_d_x,
    validate_theta_and_x,
    warn_if_invalid_for_zscoring,
)
from sbi.utils.sbiutils import ImproperEmpirical, mask_sims_from_prior


class NPE_PFN(NeuralInference[ConditionalDensityEstimator]):
    r"""Neural Posterior Estimation algorithm (NPE-C) as in Greenberg et al. (2019) [1].

    NPE-C (also known as APT - Automatic Posterior Transformation, aka SNPE-C) trains
    a neural network over multiple rounds to directly approximate the posterior for a
    specific observation x_o. In the first round, NPE-C is equivalent to other NPE
    methods and is fully amortized (direct inference for any new observation). After
    the first round, NPE-C automatically selects between two loss variants depending
    on the chosen density estimator: the non-atomic loss (for Mixture of Gaussians)
    which is stable and avoids leakage, or the atomic loss (for flows) which is more
    flexible but may suffer from leakage issues.

    For single-round inference, NPE-A, NPE-B, and NPE-C are equivalent and use
    plain NLL loss.

    [1] *Automatic Posterior Transformation for Likelihood-free Inference*,
        Greenberg et al., ICML 2019, https://arxiv.org/abs/1905.07488.

    Example:
    --------

    ::

        import torch
        from sbi.inference import NPE_C
        from sbi.utils import BoxUniform

        # 1. Setup simulator, prior, and observation
        prior = BoxUniform(low=torch.zeros(3), high=torch.ones(3))
        x_o = torch.randn(1, 3)  # Observed data

        def simulator(theta):
            return theta + torch.randn_like(theta) * 0.1

        # 2. Multi-round inference
        inference = NPE_C(prior=prior)
        proposal = prior

        for round_idx in range(5):
            theta = proposal.sample((100,))
            x = simulator(theta)
            density_estimator = inference.append_simulations(theta, x).train()
            posterior = inference.build_posterior(density_estimator)
            proposal = posterior.set_default_x(x_o)

        # 3. Sample from final posterior
        samples = posterior.sample((1000,), x=x_o)
    """

    def __init__(
        self,
        prior: Optional[Distribution] = None,
        density_estimator: Optional[
            ConditionalEstimatorBuilder[ConditionalDensityEstimator]
        ] = None,
        device: str = "cpu",
        logging_level: Union[int, str] = "WARNING",
        summary_writer: Optional[SummaryWriter] = None,
        tracker: Optional[Tracker] = None,
        show_progress_bars: bool = True,
    ):
        r"""Initialize NPE-C.

        Args:
            prior: A probability distribution that expresses prior knowledge about the
                parameters, e.g. which ranges are meaningful for them.
            density_estimator: If it is a string, use a pre-configured network of the
                provided type (one of nsf, maf, mdn, made). Alternatively, a function
                that builds a custom neural network, which adheres to
                `ConditionalEstimatorBuilder` protocol can be provided. The function
                will be called with the first batch of simulations (theta, x), which can
                thus be used for shape inference and potentially for z-scoring. The
                density estimator needs to provide the methods `.log_prob` and
                `.sample()` and must return a `ConditionalDensityEstimator`.
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

        # As detailed in the docstring, `density_estimator` is either a string or
        # a callable. The function creating the neural network is attached to
        # `_build_neural_net`. It will be called in the first round and receive
        # thetas and xs as inputs, so that they can be used for shape inference and
        # potentially for z-scoring.

        # TODO add tailored check?
        # check_estimator_arg(density_estimator)
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
        r"""Store parameters and simulation outputs to use them for later training.

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
            VectorFieldTrainer object (returned so that this function is chainable).
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

    def train(
        self,
        retrain_from_scratch: bool = False,
        discard_prior_samples: bool = False,
        force_first_round_loss: bool = False,
    ) -> ConditionalDensityEstimator:
        r"""Return density estimator that approximates the distribution $p(\theta|x)$.

        Args:
            num_atoms: Number of atoms to use for classification.
            training_batch_size: Training batch size.
            learning_rate: Learning rate for Adam optimizer.
            validation_fraction: The fraction of data to use for validation.
            stop_after_epochs: The number of epochs to wait for improvement on the
                validation set before terminating training.
            max_num_epochs: Maximum number of epochs to run. If reached, we stop
                training even when the validation loss is still decreasing. Otherwise,
                we train until validation loss increases (see also
                ``stop_after_epochs``).
            clip_max_norm: Value at which to clip the total gradient norm in order to
                prevent exploding gradients. Use None for no clipping.
            calibration_kernel: A function to calibrate the loss with respect to the
                simulations ``x``. See Lueckmann, Gonçalves et al., NeurIPS 2017.
            resume_training: Can be used in case training time is limited, e.g. on a
                cluster. If ``True``, the split between train and validation set, the
                optimizer, the number of epochs, and the best validation log-prob will
                be restored from the last time ``.train()`` was called.
            force_first_round_loss: If ``True``, train with maximum likelihood,
                i.e., potentially ignoring the correction for using a proposal
                distribution different from the prior.
            discard_prior_samples: Whether to discard samples simulated in round 1, i.e.
                from the prior. Training may be sped up by ignoring such less targeted
                samples.
            use_combined_loss: Whether to train the neural net also on prior samples
                using maximum likelihood in addition to training it on all samples using
                atomic loss. The extra MLE loss helps prevent density leaking with
                bounded priors.
            retrain_from_scratch: Whether to retrain the conditional density
                estimator for the posterior from scratch each round.
            show_train_summary: Whether to print the number of epochs and validation
                loss and leakage after the training.
            dataloader_kwargs: Additional or updated kwargs to be passed to the training
                and validation dataloaders (like, e.g., a collate_fn)

        Returns:
            Density estimator that approximates the distribution $p(\theta|x)$.
        """

        if len(self._data_round_index) == 0:
            raise RuntimeError(
                "No simulations found. You must call .append_simulations() "
                "before calling .train()."
            )

        start_idx = self._get_start_index(
            context=StartIndexContext(
                discard_prior_samples=discard_prior_samples,
                force_first_round_loss=force_first_round_loss,
            )
        )

        # TODO mock these for now
        theta, x, masks = self.get_simulations(start_idx)
        self.train_indices = torch.arange(theta.shape[0])

        # TODO this function is a bit questionable in the context of NPE-PFN
        # The tabpfn builder now already adds the batch used to infer dimensionality to get a working estimator.
        # In initizlize_neural_network, actually all train indices are used, so its the whole dataset.
        # This will however be called only if not yet initizliaed or if retrain from sratch.
        # So its important so set context here (and also more intuitive)
        # Another alternative would be to just set retrain from scracth to True.
        # Some decisions need to be made here
        self._initialize_neural_network(retrain_from_scratch, start_idx)
        self._neural_net.set_context(theta, x)

        return self._neural_net

    # TODO can we even support all posteriors here?
    def build_posterior(
        self,
        density_estimator: Optional[ConditionalDensityEstimator] = None,
        prior: Optional[Distribution] = None,
        sample_with: Literal[
            "mcmc", "rejection", "vi", "importance", "direct", "filtered_direct"
        ] = "filtered_direct",
        mcmc_method: Literal[
            "slice_np",
            "slice_np_vectorized",
            "hmc_pyro",
            "nuts_pyro",
            "slice_pymc",
            "hmc_pymc",
            "nuts_pymc",
        ] = "slice_np_vectorized",
        vi_method: Literal["rKL", "fKL", "IW", "alpha"] = "rKL",
        direct_sampling_parameters: Optional[Dict[str, Any]] = None,
        mcmc_parameters: Optional[Dict[str, Any]] = None,
        vi_parameters: Optional[Dict[str, Any]] = None,
        filtered_direct_sampling_parameters: Optional[Dict[str, Any]] = None,
        rejection_sampling_parameters: Optional[Dict[str, Any]] = None,
        importance_sampling_parameters: Optional[Dict[str, Any]] = None,
        posterior_parameters: Optional[
            Union[
                DirectPosteriorParameters,
                MCMCPosteriorParameters,
                VIPosteriorParameters,
                RejectionPosteriorParameters,
                ImportanceSamplingPosteriorParameters,
                FilteredDirectPosteriorParameters,
            ]
        ] = None,
    ) -> NeuralPosterior:
        r"""Build posterior from the neural density estimator.

        For SNPE, the posterior distribution that is returned here implements the
        following functionality over the raw neural density estimator:
        - correct the calculation of the log probability such that it compensates for
            the leakage.
        - reject samples that lie outside of the prior bounds.
        - alternatively, if leakage is very high (which can happen for multi-round
            SNPE), sample from the posterior with MCMC.

        Args:
            density_estimator: The density estimator that the posterior is based on.
                If `None`, use the latest neural density estimator that was trained.
            prior: Prior distribution.
            sample_with: Method to use for sampling from the posterior. Must be one of
                [`direct` | `filtered_direct` | `mcmc` | `rejection` | `vi` | `importance`].
            mcmc_method: Method used for MCMC sampling, one of `slice_np`,
                `slice_np_vectorized`, `hmc_pyro`, `nuts_pyro`, `slice_pymc`,
                `hmc_pymc`, `nuts_pymc`. `slice_np` is a custom
                numpy implementation of slice sampling. `slice_np_vectorized` is
                identical to `slice_np`, but if `num_chains>1`, the chains are
                vectorized for `slice_np_vectorized` whereas they are run sequentially
                for `slice_np`. The samplers ending on `_pyro` are using Pyro, and
                likewise the samplers ending on `_pymc` are using PyMC.
            vi_method: Method used for VI, one of [`rKL`, `fKL`, `IW`, `alpha`]. Note
                some of the methods admit a `mode seeking` property (e.g. rKL) whereas
                some admit a `mass covering` one (e.g fKL).
            direct_sampling_parameters: Additional kwargs passed to `DirectPosterior`.
            filtered_direct_sampling_parameters: Additional kwargs passed to
                `FilteredDirectPosterior`. If `posterior_parameters` is not
                provided and `sample_with='filtered_direct'`, context tensors are
                derived from stored simulations and combined with these overrides.
            mcmc_parameters: Additional kwargs passed to `MCMCPosterior`.
            vi_parameters: Additional kwargs passed to `VIPosterior`.
            rejection_sampling_parameters: Additional kwargs passed to
                `RejectionPosterior`.
            importance_sampling_parameters: Additional kwargs passed to
                `ImportanceSamplingPosterior`.
            posterior_parameters: Configuration passed to the init method for the
                posterior. Must be one of the following
                - `VIPosteriorParameters`
                - `ImportanceSamplingPosteriorParameters`
                - `MCMCPosteriorParameters`
                - `DirectPosteriorParameters`
                - `RejectionPosteriorParameters`
                - `FilteredDirectPosteriorParameters`

        Returns:
            Posterior $p(\theta|x)$  with `.sample()` and `.log_prob()` methods
            (the returned log-probability is unnormalized).
        """

        if sample_with == "filtered_direct":
            if (
                posterior_parameters is not None
                and filtered_direct_sampling_parameters is not None
            ):
                raise ValueError(
                    "Cannot pass both `posterior_parameters` and "
                    "`filtered_direct_sampling_parameters` for "
                    "sample_with='filtered_direct'."
                )

            if posterior_parameters is not None and not isinstance(
                posterior_parameters, FilteredDirectPosteriorParameters
            ):
                raise TypeError(
                    "For sample_with='filtered_direct', posterior_parameters must be "
                    "an instance of FilteredDirectPosteriorParameters."
                )

            if posterior_parameters is None:
                if len(self._data_round_index) == 0:
                    raise RuntimeError(
                        "No simulations found. You must call "
                        ".append_simulations() before calling "
                        ".build_posterior(sample_with='filtered_direct')."
                    )

                full_context_input, full_context_condition, _ = self.get_simulations(
                    starting_round=0
                )

                if full_context_input.shape[0] == 0:
                    raise RuntimeError(
                        "No valid simulations available to build "
                        "`FilteredDirectPosteriorParameters` context."
                    )

                default_filtered_direct_params = {
                    "full_context_input": full_context_input,
                    "full_context_condition": full_context_condition,
                    "max_sampling_batch_size": 10_000,
                    "enable_transform": True,
                    "context_nn_k": min(2048, int(full_context_input.shape[0])),
                    "context_nn_enabled": True,
                }
                overrides = filtered_direct_sampling_parameters or {}
                posterior_parameters = FilteredDirectPosteriorParameters(
                    **dict(default_filtered_direct_params, **overrides)
                )
                # Avoid old/new style parameter conflict in base validation.
                filtered_direct_sampling_parameters = None

        self._check_prior_for_rejection_sampling(
            prior, sample_with, posterior_parameters
        )

        return super().build_posterior(
            density_estimator,
            prior,
            sample_with,
            posterior_parameters,
            mcmc_method=mcmc_method,
            vi_method=vi_method,
            mcmc_parameters=mcmc_parameters,
            vi_parameters=vi_parameters,
            filtered_direct_sampling_parameters=filtered_direct_sampling_parameters,
            rejection_sampling_parameters=rejection_sampling_parameters,
            importance_sampling_parameters=importance_sampling_parameters,
            direct_sampling_parameters=direct_sampling_parameters,
        )

    def _get_start_index(self, context: StartIndexContext) -> int:
        """
        Determine the starting index for training based on previous rounds.

        Args:
            context: StartIndexContext dataclass values used to determine the starting
                index of the training set.
        Returns:
            The method will return 1 to skip samples from round 0; otherwise,
            it returns 0.
        """

        # Load data from most recent round.
        self._round = max(self._data_round_index)

        if self._round == 0 and self._neural_net is not None:
            assert context.force_first_round_loss, (
                # TODO adapt message
                "You have already trained this neural network. After you had trained "
                "the network, you again appended simulations with `append_simulations"
                "(theta, x)`, but you did not provide a proposal. If the new "
                "simulations are sampled from the prior, you can set "
                "`.train(..., force_first_round_loss=True`). However, if the new "
                "simulations were not sampled from the prior, you should pass the "
                "proposal, i.e. `append_simulations(theta, x, proposal)`. If "
                "your samples are not sampled from the prior and you do not pass a "
                "proposal and you set `force_first_round_loss=True`, the result of "
                "NPSE will not be the true posterior. Instead, it will be the proposal "
                "posterior, which (usually) is more narrow than the true posterior."
            )

        # Starting index for the training set (1 = discard round-0 samples).
        start_idx = int(context.discard_prior_samples and self._round > 0)

        return start_idx

    def _initialize_neural_network(
        self,
        retrain_from_scratch: bool,
        start_idx: int,
    ) -> None:
        """
        Initialize the neural network if it is None or retraining from scratch.

        Args:
            retrain_from_scratch: Whether to retrain the conditional density
                estimator for the posterior from scratch each round.
            start_idx: The index of the first round to retrieve simulation data from.
        """

        # First round or if retraining from scratch:
        # Call the `self._build_neural_net` with the rounds' thetas and xs as
        # arguments, which will build the neural network.
        # This is passed into NeuralPosterior, to create a neural posterior which
        # can `sample()` and `log_prob()`. The network is accessible via `.net`.
        if self._neural_net is None or retrain_from_scratch:
            # Get theta,x to initialize NN
            theta, x, _ = self.get_simulations(starting_round=start_idx)
            # Use only training data for building the neural net (z-scoring transforms)

            self._neural_net = self._build_neural_net(
                theta[self.train_indices].to("cpu"),
                x[self.train_indices].to("cpu"),
            )

            theta = reshape_to_sample_batch_event(
                theta.to("cpu"), self._neural_net.input_shape
            )
            x = reshape_to_batch_event(x.to("cpu"), self._neural_net.condition_shape)
            test_posterior_net_for_multi_d_x(self._neural_net, theta, x)

            del theta, x

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
        raise NotImplementedError(
            "NPE-PFN is training-free. Currenlty not implemented. Finetuning not yet supported."
        )

    def _loss(self, *args, **kwargs) -> Tensor:
        raise NotImplementedError(
            "NPE-PFN is training-free. Currenlty not implemented. Finetuning not yet supported."
        )

    def _check_prior_for_rejection_sampling(
        self,
        prior: Optional[Distribution],
        sample_with: Literal[
            "mcmc",
            "rejection",
            "vi",
            "importance",
            "direct",
            "filtered_direct",
        ],
        posterior_parameters: Optional[
            Union[
                DirectPosteriorParameters,
                MCMCPosteriorParameters,
                VIPosteriorParameters,
                RejectionPosteriorParameters,
                ImportanceSamplingPosteriorParameters,
                FilteredDirectPosteriorParameters,
            ]
        ],
    ) -> None:
        """
        Validates that when using rejection sampling, a prior distribution
        is explicitly provided.

        Args:
            prior: Prior distribution.
            sample_with: The sampling method used. Must be one of
                "mcmc", "rejection", "vi", "importance", "direct", or
                "filtered_direct".
            posterior_parameters: Configuration for building the posterior.
        """

        if (
            sample_with == "rejection"
            or isinstance(posterior_parameters, RejectionPosteriorParameters)
        ) and prior is None:
            raise ValueError(
                "You indicated sampling via rejection sampling but "
                "haven't passed a prior. As of sbi v0.23.0, you either have"
                " to pass a prior to perform rejection sampling using the prior"
                " as proposal, or to use the posterior as proposal, you have to"
                " use a DirectPosterior via `sample_with='direct' or"
                " `posterior_parameters=DirectPosteriorParameters`."
            )
