# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

import warnings
from abc import ABC
from typing import Any, Dict, Literal, Optional, Sequence, Tuple, Union

from torch import Tensor
from torch.distributions import Distribution
from torch.utils.tensorboard.writer import SummaryWriter
from typing_extensions import Self

from sbi.inference.posteriors.base_posterior import NeuralPosterior
from sbi.inference.posteriors.posterior_parameters import (
    ImportanceSamplingPosteriorParameters,
    MCMCPosteriorParameters,
    RejectionPosteriorParameters,
    VIPosteriorParameters,
)
from sbi.inference.potentials import likelihood_estimator_based_potential
from sbi.inference.potentials.likelihood_based_potential import LikelihoodBasedPotential
from sbi.inference.trainers._contracts import StartIndexContext, TrainConfig
from sbi.inference.trainers.base import NeuralInference
from sbi.neural_nets import likelihood_nn
from sbi.neural_nets.estimators import ConditionalDensityEstimator
from sbi.neural_nets.estimators.base import ConditionalEstimatorBuilder
from sbi.neural_nets.estimators.shape_handling import (
    reshape_to_batch_event,
)
from sbi.sbi_types import TorchTransform
from sbi.utils import check_estimator_arg, x_shape_from_simulation
from sbi.utils.torchutils import assert_all_finite


class LikelihoodEstimatorTrainer(NeuralInference[ConditionalDensityEstimator], ABC):
    def __init__(
        self,
        prior: Optional[Distribution] = None,
        density_estimator: Union[
            Literal["nsf", "maf", "mdn", "made"],
            ConditionalEstimatorBuilder[ConditionalDensityEstimator],
        ] = "maf",
        device: str = "cpu",
        logging_level: Union[int, str] = "WARNING",
        summary_writer: Optional[SummaryWriter] = None,
        show_progress_bars: bool = True,
    ):
        r"""Base class for `Neural Likelihood Estimation` methods.

        Args:
            prior: A probability distribution that expresses prior knowledge about the
                parameters, e.g. which ranges are meaningful for them. Any
                object with `.log_prob()`and `.sample()` (for example, a PyTorch
                distribution) can be used.
            density_estimator: If it is a string, use a pre-configured network of the
                provided type (one of nsf, maf, mdn, made). Alternatively, a function
                that builds a custom neural network, which adheres to
                `ConditionalEstimatorBuilder` protocol can be provided. The function
                will be called with the first batch of simulations (theta, x), which can
                thus be used for shape inference and potentially for z-scoring. The
                density estimator needs to provide the methods `.log_prob` and
                `.sample()` and must return a `ConditionalDensityEstimator`.

        See docstring of `NeuralInference` class for all other arguments.
        """

        super().__init__(
            prior=prior,
            device=device,
            logging_level=logging_level,
            summary_writer=summary_writer,
            show_progress_bars=show_progress_bars,
        )

        # As detailed in the docstring, `density_estimator` is either a string or
        # a callable. The function creating the neural network is attached to
        # `_build_neural_net`. It will be called in the first round and receive
        # thetas and xs as inputs, so that they can be used for shape inference and
        # potentially for z-scoring.
        check_estimator_arg(density_estimator)
        if isinstance(density_estimator, str):
            self._build_neural_net = likelihood_nn(model=density_estimator)
        else:
            self._build_neural_net = density_estimator

    def append_simulations(
        self,
        theta: Tensor,
        x: Tensor,
        exclude_invalid_x: bool = False,
        from_round: int = 0,
        algorithm: str = "SNLE",
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
                training. If `False`, NLE raises an error when invalid simulations are
                found. If `True`, invalid simulations are discarded and training
                can proceed, but this gives systematically wrong results.
            from_round: Which round the data stemmed from. Round 0 means from the prior.
                With default settings, this is not used at all for `NLE`. Only when
                the user later on requests `.train(discard_prior_samples=True)`, we
                use these indices to find which training data stemmed from the prior.
            algorithm: Which algorithm is used. This is used to give a more informative
                warning or error message when invalid simulations are found.
            data_device: Where to store the data, default is on the same device where
                the training is happening. If training a large dataset on a GPU with not
                much VRAM can set to 'cpu' to store data on system memory instead.
        Returns:
            NeuralInference object (returned so that this function is chainable).
        """

        if exclude_invalid_x:
            warnings.warn(
                "NLE gives systematically wrong results when exclude_invalid_x=True.",
                stacklevel=2,
            )

        return super().append_simulations(
            theta=theta,
            x=x,
            exclude_invalid_x=exclude_invalid_x,
            from_round=from_round,
            algorithm=algorithm,
            data_device=data_device,
        )

    def train(
        self,
        training_batch_size: int = 200,
        learning_rate: float = 5e-4,
        validation_fraction: float = 0.1,
        stop_after_epochs: int = 20,
        max_num_epochs: int = 2**31 - 1,
        clip_max_norm: Optional[float] = 5.0,
        resume_training: bool = False,
        discard_prior_samples: bool = False,
        retrain_from_scratch: bool = False,
        show_train_summary: bool = False,
        dataloader_kwargs: Optional[Dict] = None,
    ) -> ConditionalDensityEstimator:
        r"""Train the density estimator to learn the distribution $p(x|\theta)$.

        Args:
            training_batch_size: Training batch size.
            learning_rate: Learning rate for Adam optimizer.
            validation_fraction: The fraction of data to use for validation.
            stop_after_epochs: The number of epochs to wait for improvement on the
                validation set before terminating training.
            max_num_epochs: Maximum number of epochs to run. If reached, we stop
                training even when the validation loss is still decreasing. Otherwise,
                we train until validation loss increases (see also `stop_after_epochs`).
            clip_max_norm: Value at which to clip the total gradient norm in order to
                prevent exploding gradients. Use None for no clipping.
            resume_training: Can be used in case training time is limited, e.g. on a
                cluster. If `True`, the split between train and validation set, the
                optimizer, the number of epochs, and the best validation log-prob will
                be restored from the last time `.train()` was called.
            discard_prior_samples: Whether to discard samples simulated in round 1, i.e.
                from the prior. Training may be sped up by ignoring such less targeted
                samples.
            retrain_from_scratch: Whether to retrain the conditional density
                estimator for the posterior from scratch each round.
            show_train_summary: Whether to print the number of epochs and validation
                loss after the training.
            dataloader_kwargs: Additional or updated kwargs to be passed to the training
                and validation dataloaders (like, e.g., a collate_fn)

        Returns:
            Density estimator that has learned the distribution $p(x|\theta)$.
        """

        start_idx = self._get_start_index(
            context=StartIndexContext(discard_prior_samples=discard_prior_samples)
        )

        train_config = TrainConfig(
            max_num_epochs=max_num_epochs,
            stop_after_epochs=stop_after_epochs,
            learning_rate=learning_rate,
            resume_training=resume_training,
            show_train_summary=show_train_summary,
            training_batch_size=training_batch_size,
            retrain_from_scratch=retrain_from_scratch,
            validation_fraction=validation_fraction,
            clip_max_norm=clip_max_norm,
        )

        train_loader, val_loader = self.get_dataloaders(
            start_idx,
            train_config.training_batch_size,
            train_config.validation_fraction,
            train_config.resume_training,
            dataloader_kwargs=dataloader_kwargs,
        )

        self._initialize_neural_network(
            retrain_from_scratch=train_config.retrain_from_scratch,
            start_idx=start_idx,
        )

        return self._run_training_loop(
            train_loader=train_loader, val_loader=val_loader, train_config=train_config
        )

    def build_posterior(
        self,
        density_estimator: Optional[ConditionalDensityEstimator] = None,
        prior: Optional[Distribution] = None,
        sample_with: Literal["mcmc", "rejection", "vi", "importance"] = "mcmc",
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
        mcmc_parameters: Optional[Dict[str, Any]] = None,
        vi_parameters: Optional[Dict[str, Any]] = None,
        rejection_sampling_parameters: Optional[Dict[str, Any]] = None,
        importance_sampling_parameters: Optional[Dict[str, Any]] = None,
        posterior_parameters: Optional[
            Union[
                MCMCPosteriorParameters,
                VIPosteriorParameters,
                RejectionPosteriorParameters,
                ImportanceSamplingPosteriorParameters,
            ]
        ] = None,
    ) -> NeuralPosterior:
        r"""Build posterior from the neural density estimator.

        SNLE trains a neural network to approximate the likelihood $p(x|\theta)$. The
        posterior wraps the trained network such that one can directly evaluate the
        unnormalized posterior log probability $p(\theta|x) \propto p(x|\theta) \cdot
        p(\theta)$ and draw samples from the posterior with MCMC or rejection sampling.

        Args:
            density_estimator: The density estimator that the posterior is based on.
                If `None`, use the latest neural density estimator that was trained.
            prior: Prior distribution.
            sample_with: Method to use for sampling from the posterior. Must be one of
                [`mcmc` | `rejection` | `vi` | `importance`].
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
                - `RejectionPosteriorParameters`
        Returns:
            Posterior $p(\theta|x)$  with `.sample()` and `.log_prob()` methods
            (the returned log-probability is unnormalized).
        """

        return super().build_posterior(
            density_estimator,
            prior,
            sample_with,
            posterior_parameters,
            mcmc_method=mcmc_method,
            vi_method=vi_method,
            mcmc_parameters=mcmc_parameters,
            vi_parameters=vi_parameters,
            rejection_sampling_parameters=rejection_sampling_parameters,
            importance_sampling_parameters=importance_sampling_parameters,
        )

    def _get_potential_function(
        self, prior: Distribution, estimator: ConditionalDensityEstimator
    ) -> Tuple[LikelihoodBasedPotential, TorchTransform]:
        r"""Gets potential :math:`\log(p(x_o|\theta)p(\theta))` for
        likelihood estimator.

        It also returns a transformation that can be used to transform the potential
        into unconstrained space.

        Args:
            prior: The prior distribution.
            estimator: The density estimator modelling the likelihood.

        Returns:
            The potential function $p(x_o|\theta)p(\theta)$ and a transformation that
            maps to unconstrained space.
        """
        potential_fn, theta_transform = likelihood_estimator_based_potential(
            likelihood_estimator=estimator,
            prior=prior,
            x_o=None,
        )
        return potential_fn, theta_transform

    def _loss(self, theta: Tensor, x: Tensor) -> Tensor:
        r"""Return loss for SNLE, which is the likelihood of $-\log q(x_i | \theta_i)$.

        Returns:
            Negative log prob.
        """
        theta = reshape_to_batch_event(
            theta, event_shape=self._neural_net.condition_shape
        )
        x = reshape_to_batch_event(x, event_shape=self._neural_net.input_shape)
        loss = self._neural_net.loss(x, condition=theta)
        assert_all_finite(loss, "NLE loss")
        return loss

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
        # arguments, which will build the neural network
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

            assert len(x_shape_from_simulation(x.to("cpu"))) < 3, (
                "SNLE cannot handle multi-dimensional simulator output."
            )
            del theta, x

    def _get_losses(self, batch: Sequence[Tensor]) -> Tensor:
        """
        Compute losses for a batch of data.

        Args:
            batch: A batch of data.

        Returns:
            A tensor containing the computed losses for each sample in the batch.
        """

        # Get batches on current device.
        theta_batch, x_batch = (
            batch[0].to(self._device),
            batch[1].to(self._device),
        )

        losses = self._loss(theta_batch, x_batch)

        return losses
