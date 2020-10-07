# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Affero General Public License v3, see <https://www.gnu.org/licenses/>.


from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Any, Callable, Dict, Optional, Union, cast

import torch
from torch import Tensor, ones, optim
from torch.nn.utils import clip_grad_norm_
from torch.utils import data
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.tensorboard import SummaryWriter

from sbi import utils as utils
from sbi.inference import NeuralInference
from sbi.inference.posteriors.direct_posterior import DirectPosterior
from sbi.utils import (
    check_estimator_arg,
    x_shape_from_simulation,
    test_posterior_net_for_multi_d_x,
)


class PosteriorEstimator(NeuralInference, ABC):
    def __init__(
        self,
        simulator: Callable,
        prior,
        num_workers: int = 1,
        simulation_batch_size: int = 1,
        density_estimator: Union[str, Callable] = "maf",
        sample_with_mcmc: bool = False,
        mcmc_method: str = "slice_np",
        mcmc_parameters: Optional[Dict[str, Any]] = None,
        rejection_sampling_parameters: Optional[Dict[str, Any]] = None,
        device: str = "cpu",
        logging_level: Union[int, str] = "WARNING",
        summary_writer: Optional[SummaryWriter] = None,
        show_progress_bars: bool = True,
        show_round_summary: bool = False,
    ):
        """Base class for Sequential Neural Posterior Estimation methods.

        Args:
            density_estimator: If it is a string, use a pre-configured network of the
                provided type (one of nsf, maf, mdn, made). Alternatively, a function
                that builds a custom neural network can be provided. The function will
                be called with the first batch of simulations (theta, x), which can
                thus be used for shape inference and potentially for z-scoring. It
                needs to return a PyTorch `nn.Module` implementing the density
                estimator. The density estimator needs to provide the methods
                `.log_prob` and `.sample()`.
            sample_with_mcmc: Whether to sample with MCMC. MCMC can be used to deal
                with high leakage.
            mcmc_method: Method used for MCMC sampling, one of `slice_np`, `slice`,
                `hmc`, `nuts`. Currently defaults to `slice_np` for a custom numpy
                implementation of slice sampling; select `hmc`, `nuts` or `slice` for
                Pyro-based sampling.
            mcmc_parameters: Dictionary overriding the default parameters for MCMC.
                The following parameters are supported: `thin` to set the thinning
                factor for the chain, `warmup_steps` to set the initial number of
                samples to discard, `num_chains` for the number of chains,
                `init_strategy` for the initialisation strategy for chains; `prior` will
                draw init locations from prior, whereas `sir` will use
                Sequential-Importance-Resampling using `init_strategy_num_candidates`
                to find init locations.
            rejection_sampling_parameters: Dictonary overriding the default parameters
                for rejection sampling. The following parameters are supported:
                `max_sampling_batch_size` to set the batch size for drawing new
                samples from the candidate distribution, e.g., the posterior. Larger
                batch size speeds up sampling.

        See docstring of `NeuralInference` class for all other arguments.
        """

        super().__init__(
            simulator=simulator,
            prior=prior,
            num_workers=num_workers,
            simulation_batch_size=simulation_batch_size,
            device=device,
            logging_level=logging_level,
            summary_writer=summary_writer,
            show_progress_bars=show_progress_bars,
            show_round_summary=show_round_summary,
        )

        # As detailed in the docstring, `density_estimator` is either a string or
        # a callable. The function creating the neural network is attached to
        # `_build_neural_net`. It will be called in the first round and receive
        # thetas and xs as inputs, so that they can be used for shape inference and
        # potentially for z-scoring.
        check_estimator_arg(density_estimator)
        if isinstance(density_estimator, str):
            self._build_neural_net = utils.posterior_nn(model=density_estimator)
        else:
            self._build_neural_net = density_estimator
        self._posterior = None
        self._sample_with_mcmc = sample_with_mcmc
        self._mcmc_method = mcmc_method
        self._mcmc_parameters = mcmc_parameters
        self._rejection_sampling_parameters = rejection_sampling_parameters

        self._model_bank = []
        self.use_non_atomic_loss = False

        # Extra SNPE-specific fields summary_writer.
        self._summary.update({"rejection_sampling_acceptance_rates": []})  # type:ignore

    def __call__(
        self,
        num_simulations: int,
        proposal: Optional[Any] = None,
        training_batch_size: int = 50,
        learning_rate: float = 5e-4,
        validation_fraction: float = 0.1,
        stop_after_epochs: int = 20,
        max_num_epochs: Optional[int] = None,
        clip_max_norm: Optional[float] = 5.0,
        calibration_kernel: Optional[Callable] = None,
        exclude_invalid_x: bool = True,
        discard_prior_samples: bool = False,
        retrain_from_scratch_each_round: bool = False,
    ) -> DirectPosterior:
        r"""Run SNPE.

        Return posterior $p(\theta|x)$ after inference.

        Args:
            num_simulations: Number of simulator calls.
            proposal: Distribution that the parameters $\theta$ are drawn from.
                `proposal=None` uses the prior. Setting the proposal to a distribution
                targeted on a specific observation, e.g. a posterior $p(\theta|x_o)$
                obtained previously, can lead to less required simulations.
            training_batch_size: Training batch size.
            learning_rate: Learning rate for Adam optimizer.
            validation_fraction: The fraction of data to use for validation.
            stop_after_epochs: The number of epochs to wait for improvement on the
                validation set before terminating training.
            max_num_epochs: Maximum number of epochs to run. If reached, we stop
                training even when the validation loss is still decreasing. If None, we
                train until validation loss increases (see also `stop_after_epochs`).
            clip_max_norm: Value at which to clip the total gradient norm in order to
                prevent exploding gradients. Use None for no clipping.
            calibration_kernel: A function to calibrate the loss with respect to the
                simulations `x`. See Lueckmann, Gonçalves et al., NeurIPS 2017.
            exclude_invalid_x: Whether to exclude simulation outputs `x=NaN` or `x=±∞`
                during training. Expect errors, silent or explicit, when `False`.
            discard_prior_samples: Whether to discard samples simulated in round 1, i.e.
                from the prior. Training may be sped up by ignoring such less targeted
                samples.
            retrain_from_scratch_each_round: Whether to retrain the conditional density
                estimator for the posterior from scratch each round.

        Returns:
            Posterior $p(\theta|x)$ that can be sampled and evaluated.
        """

        # Calibration kernels proposed in Lueckmann, Gonçalves et al., 2017.
        if calibration_kernel is None:
            calibration_kernel = lambda x: ones([len(x)])

        max_num_epochs = 2 ** 31 - 1 if max_num_epochs is None else max_num_epochs

        self._check_proposal(proposal)
        self._round = self._round + 1 if (proposal is not None) else 0

        # If presimulated data was provided from a later round, set the self._round to
        # this value. Otherwise, we would rely on the user to _additionally_ provide the
        # proposal that the presimulated data was sampled from in order for self._round
        # to become larger than 0.
        if self._data_round_index:
            self._round = max(self._round, max(self._data_round_index))

        # Run simulations for the round.
        theta, x = self._run_simulations(proposal, num_simulations)
        self._append_to_data_bank(theta, x, self._round)

        # Load data from most recent round.
        theta, x, _ = self._get_from_data_bank(self._round, exclude_invalid_x, False)

        # First round or if retraining from scratch:
        # Call the `self._build_neural_net` with the rounds' thetas and xs as
        # arguments, which will build the neural network.
        # This is passed into NeuralPosterior, to create a neural posterior which
        # can `sample()` and `log_prob()`. The network is accessible via `.net`.
        if self._posterior is None or retrain_from_scratch_each_round:
            x_shape = x_shape_from_simulation(x)
            self._posterior = DirectPosterior(
                method_family="snpe",
                neural_net=self._build_neural_net(theta, x),
                prior=self._prior,
                x_shape=x_shape,
                sample_with_mcmc=self._sample_with_mcmc,
                mcmc_method=self._mcmc_method,
                mcmc_parameters=self._mcmc_parameters,
                rejection_sampling_parameters=self._rejection_sampling_parameters,
            )
            test_posterior_net_for_multi_d_x(self._posterior.net, theta, x)

        # Copy MCMC init parameters for latest sample init
        if hasattr(proposal, "_mcmc_init_params"):
            self._posterior._mcmc_init_params = proposal._mcmc_init_params

        # Fit posterior using newly aggregated data set.
        self._train(
            proposal=proposal,
            training_batch_size=training_batch_size,
            learning_rate=learning_rate,
            validation_fraction=validation_fraction,
            stop_after_epochs=stop_after_epochs,
            max_num_epochs=cast(int, max_num_epochs),
            clip_max_norm=clip_max_norm,
            calibration_kernel=calibration_kernel,
            exclude_invalid_x=exclude_invalid_x,
            discard_prior_samples=discard_prior_samples,
        )

        # Store models at end of each round.
        self._model_bank.append(deepcopy(self._posterior))
        self._model_bank[-1].net.eval()

        # Making the call to `leakage_correction()` and the update of
        # self._leakage_density_correction_factor explicit here. This is just
        # to make sure this update never gets lost when we e.g. do not log our
        # things to tensorboard anymore. Calling `leakage_correction()` is needed
        # to update the leakage after each round.
        if self._posterior.default_x is None:
            acceptance_rate = torch.tensor(float("nan"))
        else:
            acceptance_rate = self._posterior.leakage_correction(
                x=self._posterior.default_x,
                force_update=True,
                show_progress_bars=self._show_progress_bars,
            )

        # Update tensorboard and summary dict.
        self._summarize(
            round_=self._round,
            x_o=self._posterior.default_x,
            theta_bank=theta,
            x_bank=x,
            posterior_samples_acceptance_rate=acceptance_rate,
        )

        # Update description for progress bar.
        if self._show_round_summary:
            print(self._describe_round(self._round, self._summary))

        self._posterior._num_trained_rounds = self._round + 1
        return deepcopy(self._posterior)

    @abstractmethod
    def _log_prob_proposal_posterior(
        self, theta: Tensor, x: Tensor, masks: Tensor, proposal: Optional[Any]
    ) -> Tensor:
        raise NotImplementedError

    def _train(
        self,
        proposal: Optional[Any],
        training_batch_size: int,
        learning_rate: float,
        validation_fraction: float,
        stop_after_epochs: int,
        max_num_epochs: int,
        clip_max_norm: Optional[float],
        calibration_kernel: Callable,
        exclude_invalid_x: bool,
        discard_prior_samples: bool,
    ) -> None:
        r"""Train the conditional density estimator for the posterior $p(\theta|x)$.

        Update the conditional density estimator weights to maximize the proposal
        posterior using the most recently aggregated bank of $(\theta, x)$ pairs.

        Uses performance on a held-out validation set as a terminating condition (early
        stopping).

        The proposal is only needed for non-atomic SNPE.
        """

        # Starting index for the training set (1 = discard round-0 samples).
        start_idx = int(discard_prior_samples and self._round > 0)

        # For non-atomic loss, we can not reuse samples from previous rounds as of now.
        if self.use_non_atomic_loss:
            start_idx = self._round

        theta, x, prior_masks = self._get_from_data_bank(start_idx, exclude_invalid_x)

        # Select random neural net and validation splits from (theta, x) pairs.
        num_total_examples = len(theta)
        permuted_indices = torch.randperm(num_total_examples)
        num_training_examples = int((1 - validation_fraction) * num_total_examples)
        num_validation_examples = num_total_examples - num_training_examples
        train_indices, val_indices = (
            permuted_indices[:num_training_examples],
            permuted_indices[num_training_examples:],
        )

        # Dataset is shared for training and validation loaders.
        dataset = data.TensorDataset(theta, x, prior_masks)

        # Create neural net and validation loaders using a subset sampler.
        train_loader = data.DataLoader(
            dataset,
            batch_size=min(training_batch_size, num_training_examples),
            drop_last=True,
            sampler=SubsetRandomSampler(train_indices),
        )
        val_loader = data.DataLoader(
            dataset,
            batch_size=min(training_batch_size, num_validation_examples),
            shuffle=False,
            drop_last=True,
            sampler=SubsetRandomSampler(val_indices),
        )

        optimizer = optim.Adam(
            list(self._posterior.net.parameters()), lr=learning_rate,
        )

        epoch, self._val_log_prob = 0, float("-Inf")
        while epoch <= max_num_epochs and not self._converged(epoch, stop_after_epochs):

            # Train for a single epoch.
            self._posterior.net.train()
            for batch in train_loader:
                optimizer.zero_grad()
                theta_batch, x_batch, masks_batch = (
                    batch[0].to(self._device),
                    batch[1].to(self._device),
                    batch[2].to(self._device),
                )

                batch_loss = torch.mean(
                    self._loss(
                        theta_batch, x_batch, masks_batch, proposal, calibration_kernel
                    )
                )
                batch_loss.backward()
                if clip_max_norm is not None:
                    clip_grad_norm_(
                        self._posterior.net.parameters(), max_norm=clip_max_norm,
                    )
                optimizer.step()

            epoch += 1

            # Calculate validation performance.
            self._posterior.net.eval()
            log_prob_sum = 0
            with torch.no_grad():
                for batch in val_loader:
                    theta_batch, x_batch, masks_batch = (
                        batch[0].to(self._device),
                        batch[1].to(self._device),
                        batch[2].to(self._device),
                    )
                    # Take negative loss here to get validation log_prob.
                    batch_log_prob = -self._loss(
                        theta_batch, x_batch, masks_batch, proposal, calibration_kernel
                    )
                    log_prob_sum += batch_log_prob.sum().item()

            self._val_log_prob = log_prob_sum / num_validation_examples

            self._maybe_show_progress(self._show_progress_bars, epoch)

        self._report_convergence_at_end(epoch, stop_after_epochs, max_num_epochs)

        # Update summary.
        self._summary["epochs"].append(epoch)
        self._summary["best_validation_log_probs"].append(self._best_val_log_prob)

    def _loss(
        self,
        theta: Tensor,
        x: Tensor,
        masks: Tensor,
        proposal: Optional[Any],
        calibration_kernel: Callable,
    ) -> Tensor:
        """Return loss with proposal correction (`round_>0`) or without it (`round_=0`).

        The loss is the negative log prob. Irrespective of the round or SNPE method
        (A, B, or C), it can be weighted with a calibration kernel.

        Returns:
            Calibration kernel-weighted negative log prob.
        """

        if self._round == 0:
            # Use posterior log prob (without proposal correction) for first round.
            log_prob = self._posterior.net.log_prob(theta, x)
        else:
            log_prob = self._log_prob_proposal_posterior(theta, x, masks, proposal)

        return -(calibration_kernel(x) * log_prob)
