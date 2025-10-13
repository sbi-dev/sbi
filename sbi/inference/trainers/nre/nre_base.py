# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

import warnings
from abc import ABC, abstractmethod
from dataclasses import asdict, replace
from typing import Any, Dict, Literal, Optional, Sequence, Tuple, Union

import torch
from torch import Tensor, eye, ones
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
from sbi.inference.potentials import ratio_estimator_based_potential
from sbi.inference.potentials.ratio_based_potential import RatioBasedPotential
from sbi.inference.trainers._contracts import (
    LossArgsNRE,
    StartIndexContext,
    TrainConfig,
)
from sbi.inference.trainers.base import (
    LossArgs,
    NeuralInference,
)
from sbi.neural_nets import classifier_nn
from sbi.neural_nets.estimators.base import ConditionalEstimatorBuilder
from sbi.neural_nets.ratio_estimators import RatioEstimator
from sbi.sbi_types import TorchTransform
from sbi.utils import (
    check_estimator_arg,
    clamp_and_warn,
)
from sbi.utils.torchutils import repeat_rows


class RatioEstimatorTrainer(NeuralInference[RatioEstimator], ABC):
    def __init__(
        self,
        prior: Optional[Distribution] = None,
        classifier: Union[str, ConditionalEstimatorBuilder[RatioEstimator]] = "resnet",
        device: str = "cpu",
        logging_level: Union[int, str] = "warning",
        summary_writer: Optional[SummaryWriter] = None,
        show_progress_bars: bool = True,
    ):
        r"""Neural Ratio Estimation.

        We implement three inference methods in the respective subclasses.

        - SNRE_A / AALR is limited to `num_atoms=2`, but allows for density evaluation
          when training for one round.
        - SNRE_B / SRE can use more than two atoms, potentially boosting performance,
          but allows for posterior evaluation **only up to a normalizing constant**,
          even when training only one round.
        - BNRE is a variation of SNRE_A aiming to produce more conservative posterior
          approximations.
        - SNRE_C / NRE-C is a generalization of SNRE_A and SNRE_B which can use multiple
          classes (similar to atoms) but encourages an exact likelihood-to-evidence
          ratio (density evaluation) by introducing an independently drawn class.
          Addressing the issue in SNRE_B which only estimates the ratio up to a function
          (normalizing constant) of the data $x$.

        Args:
            classifier: Classifier trained to approximate likelihood ratios. If it is
                a string, use a pre-configured network of the provided type (one of
                linear, mlp, resnet), or a callable that implements the
                `ConditionalEstimatorBuilder` protocol. The callable will
                be called with the first batch of simulations (theta, x), which can thus
                be used for shape inference and potentially for z-scoring. It returns a
                `RatioEstimator`.

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
        check_estimator_arg(classifier)
        if isinstance(classifier, str):
            self._build_neural_net = classifier_nn(model=classifier)
        else:
            self._build_neural_net = classifier

    @abstractmethod
    def _loss(self, theta: Tensor, x: Tensor, num_atoms: int) -> Tensor: ...

    def append_simulations(
        self,
        theta: Tensor,
        x: Tensor,
        exclude_invalid_x: bool = False,
        from_round: int = 0,
        algorithm: str = "SNRE",
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
                training. If `False`, NRE raises an error when invalid simulations are
                found. If `True`, invalid simulations are discarded and training
                can proceed, but this gives systematically wrong results.
            from_round: Which round the data stemmed from. Round 0 means from the prior.
                With default settings, this is not used at all for `NRE`. Only when
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
                "NRE gives systematically wrong results when exclude_invalid_x=True.",
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
        num_atoms: Optional[int] = None,
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
        loss_kwargs: Optional[LossArgsNRE] = None,
    ) -> RatioEstimator:
        r"""Return classifier that approximates the ratio $p(\theta,x)/p(\theta)p(x)$.

        Args:
            num_atoms: Number of atoms to use for classification, this value must be
                passed through the loss_kwargs argument.
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
                and validation dataloaders (like, e.g., a collate_fn).
            loss_kwargs: Additional or updated kwargs to be passed to the self._loss fn.

        Returns:
            Classifier that approximates the ratio $p(\theta,x)/p(\theta)p(x)$.
        """

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

        if num_atoms is not None:
            warnings.warn(
                "num_atoms value is ignored. The value must be passed through"
                " the loss_kwargs argument.",
                stacklevel=2,
            )

        if loss_kwargs is None:
            loss_kwargs = LossArgsNRE()
            warnings.warn(
                "No value provided for loss_kwargs. NRE loss arguments like "
                "num_atoms should be set via `LossArgsNRE. A default of  "
                "num_atoms={loss_kwargs.num_atoms} will be used.",
                stacklevel=2,
            )

        if not issubclass(type(loss_kwargs), LossArgsNRE):
            raise TypeError(
                "Expected loss_kwargs to be a subclass of LossArgsNRE,"
                f" but got {type(loss_kwargs)}"
            )

        start_idx = self._get_start_index(
            context=StartIndexContext(discard_prior_samples=discard_prior_samples)
        )

        train_loader, val_loader = self.get_dataloaders(
            start_idx,
            train_config.training_batch_size,
            train_config.validation_fraction,
            train_config.resume_training,
            dataloader_kwargs=dataloader_kwargs,
        )

        clipped_batch_size = min(
            train_config.training_batch_size,
            val_loader.batch_size,  # type: ignore
        )

        num_atoms = int(
            clamp_and_warn(
                "num_atoms",
                loss_kwargs.num_atoms,
                min_val=2,
                max_val=clipped_batch_size,
            )
        )

        if num_atoms != loss_kwargs.num_atoms:
            # Update num_atoms field in loss_kwargs
            loss_kwargs = replace(loss_kwargs, **dict(num_atoms=num_atoms))

        self._initialize_neural_network(
            retrain_from_scratch=train_config.retrain_from_scratch,
            start_idx=start_idx,
        )

        return self._run_training_loop(
            train_loader=train_loader,
            val_loader=val_loader,
            train_config=train_config,
            loss_args=loss_kwargs,
        )

    def build_posterior(
        self,
        density_estimator: Optional[RatioEstimator] = None,
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

        SNRE trains a neural network to approximate likelihood ratios. The
        posterior wraps the trained network such that one can directly evaluate the
        unnormalized posterior log probability $p(\theta|x) \propto p(x|\theta) \cdot
        p(\theta)$ and draw samples from the posterior with MCMC or rejection sampling.
        Note that, in the case of single-round SNRE_A / AALR, it is possible to
        evaluate the log-probability of the **normalized** posterior, but sampling
        still requires MCMC (or rejection sampling).

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
                that some of the methods admit a `mode seeking` property (e.g. rKL)
                whereas some admit a `mass covering` one (e.g fKL).
            mcmc_parameters: Additional kwargs passed to `MCMCPosterior`.
            vi_parameters: Additional kwargs passed to `VIPosterior`.
            rejection_sampling_parameters: Additional kwargs passed to
                `RejectionPosterior`.
            importance_sampling_parameters: Additional kwargs passed to
                `ImportanceSamplingPosterior`.
            posterior_parameters: Configuration passed to the init method for the
                posterior. Must be one of the following:
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

    def _classifier_logits(self, theta: Tensor, x: Tensor, num_atoms: int) -> Tensor:
        """Return logits obtained through classifier forward pass.

        The logits are obtained from atomic sets of (theta,x) pairs.
        """
        batch_size = theta.shape[0]
        repeated_x = repeat_rows(x, num_atoms)

        # Choose `1` or `num_atoms - 1` thetas from the rest of the batch for each x.
        probs = ones(batch_size, batch_size) * (1 - eye(batch_size)) / (batch_size - 1)

        choices = torch.multinomial(probs, num_samples=num_atoms - 1, replacement=False)

        contrasting_theta = theta[choices]

        atomic_theta = torch.cat((theta[:, None, :], contrasting_theta), dim=1).reshape(
            batch_size * num_atoms, -1
        )

        return self._neural_net(atomic_theta, repeated_x)

    def _get_potential_function(
        self, prior: Distribution, estimator: RatioEstimator
    ) -> Tuple[RatioBasedPotential, TorchTransform]:
        r"""Gets the potential for ratio-based methods.

        It also returns a transformation that can be used to transform the potential
        into unconstrained space.

        Args:
            prior: The prior distribution.
            estimator: The neural network modelling likelihood-to-evidence ratio.

        Returns:
            The potential function and a transformation that maps
            to unconstrained space.
        """
        potential_fn, theta_transform = ratio_estimator_based_potential(
            ratio_estimator=estimator,
            prior=prior,
            x_o=None,
        )

        return potential_fn, theta_transform

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
        self, retrain_from_scratch: bool, start_idx: int
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

            del x, theta

    def _get_losses(self, batch: Sequence[Tensor], loss_args: LossArgs) -> Tensor:
        """
        Compute losses for a batch of data.

        Args:
            batch: A batch of data.
            loss_args: Additional arguments passed to self._loss fn.

        Returns:
            A tensor containing the computed losses for each sample in the batch.
        """

        theta_batch, x_batch = (
            batch[0].to(self._device),
            batch[1].to(self._device),
        )

        if not issubclass(type(loss_args), LossArgsNRE):
            raise TypeError(
                "Expected loss_args to be a subclass of LossArgsNRE,"
                f" but got {type(loss_args)}"
            )

        losses = self._loss(theta_batch, x_batch, **asdict(loss_args))

        return losses
