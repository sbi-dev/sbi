# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Affero General Public License v3, see <https://www.gnu.org/licenses/>.

from __future__ import annotations

from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Callable, Dict, Optional, Tuple, Union, cast
from warnings import warn

import numpy as np
import torch
from torch import Tensor, nn, ones, optim, zeros
from torch.nn.utils import clip_grad_norm_
from torch.utils import data
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.tensorboard import SummaryWriter

import sbi.utils as utils
from sbi.inference import NeuralInference
from sbi.inference.posterior import NeuralPosterior
from sbi.types import OneOrMore, ScalarFloat
from sbi.utils import handle_invalid_x, warn_on_invalid_x, x_shape_from_simulation


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
            mcmc_method: If MCMC sampling is used, specify the method here: either of
                slice_np, slice, hmc, nuts.

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
        if isinstance(density_estimator, str):
            self._build_neural_net = utils.posterior_nn(model=density_estimator)
        else:
            self._build_neural_net = density_estimator
        self._posterior = None
        self._sample_with_mcmc = sample_with_mcmc
        self._mcmc_method = mcmc_method

        self._prior_masks, self._model_bank = [], []

        # Extra SNPE-specific fields summary_writer.
        self._summary.update({"rejection_sampling_acceptance_rates": []})  # type:ignore

    def __call__(
        self,
        num_rounds: int,
        num_simulations_per_round: OneOrMore[int],
        x_o: Optional[Tensor] = None,
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
    ) -> NeuralPosterior:
        r"""Run SNPE.

        Return posterior $p(\theta|x)$ after inference (possibly over several rounds).

        Args:
            num_rounds: Number of rounds to run. Each round consists of a simulation and
                training phase. `num_rounds=1` leads to a posterior $p(\theta|x)$ valid
                for _any_ $x$ ("amortized"), but requires many simulations.
                Alternatively, with `num_rounds>1` the inference returns a posterior
                $p(\theta|x_o)$ focused on a specific observation `x_o`, potentially
                requiring less simulations.
            num_simulations_per_round: Number of simulator calls per round.
            x_o: An observation that is only required when doing inference
                over multiple rounds. After the first round, `x_o` is used to guide the
                sampling so that the simulator is run with parameters that are likely
                for that `x_o`, i.e. they are sampled from the posterior obtained in the
                previous round $p(\theta|x_o)$.
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

        self._warn_if_retrain_from_scratch_snpe(retrain_from_scratch_each_round)

        # Calibration kernels proposed in Lueckmann, Gonçalves et al., 2017.
        if calibration_kernel is None:
            calibration_kernel = lambda x: ones([len(x)])

        max_num_epochs = 2 ** 31 - 1 if max_num_epochs is None else max_num_epochs

        num_sims_per_round = self._ensure_list(num_simulations_per_round, num_rounds)

        for round_, num_sims in enumerate(num_sims_per_round):

            # Run simulations for the round.
            theta, x, prior_mask = self._run_simulations(round_, num_sims)
            x_shape = x_shape_from_simulation(x)

            # First round or if retraining from scratch:
            # Call the `self._build_neural_net` with the rounds' thetas and xs as
            # arguments, which will build the neural network
            # This is passed into NeuralPosterior, to create a neural posterior which
            # can `sample()` and `log_prob()`. The network is accessible via `.net`.
            if round_ == 0 or retrain_from_scratch_each_round:
                self._posterior = NeuralPosterior(
                    method_family="snpe",
                    neural_net=self._build_neural_net(theta, x),
                    prior=self._prior,
                    x_shape=x_shape,
                    sample_with_mcmc=self._sample_with_mcmc,
                    mcmc_method=self._mcmc_method,
                    get_potential_function=PotentialFunctionProvider(),
                )
            self._handle_x_o_wrt_amortization(x_o, x_shape, num_rounds)

            # Check for NaNs in simulations.
            is_valid_x, num_nans, num_infs = handle_invalid_x(x, exclude_invalid_x)
            warn_on_invalid_x(num_nans, num_infs, exclude_invalid_x)

            # XXX Rename bank -> rounds/roundwise.
            self._theta_bank.append(theta[is_valid_x])
            self._x_bank.append(x[is_valid_x])
            self._prior_masks.append(prior_mask[is_valid_x])

            # Fit posterior using newly aggregated data set.
            self._train(
                round_=round_,
                training_batch_size=training_batch_size,
                learning_rate=learning_rate,
                validation_fraction=validation_fraction,
                stop_after_epochs=stop_after_epochs,
                max_num_epochs=cast(int, max_num_epochs),
                clip_max_norm=clip_max_norm,
                calibration_kernel=calibration_kernel,
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
                round_=round_,
                x_o=self._posterior.default_x,
                theta_bank=self._theta_bank,
                x_bank=self._x_bank,
                posterior_samples_acceptance_rate=acceptance_rate,
            )

            # Update description for progress bar.
            if self._show_round_summary:
                print(self._describe_round(round_, self._summary))

        self._posterior._num_trained_rounds = num_rounds
        return self._posterior

    @abstractmethod
    def _log_prob_proposal_posterior(
        self, theta: Tensor, x: Tensor, masks: Tensor, **kwargs
    ) -> Tensor:
        raise NotImplementedError

    def _run_simulations(
        self, round_: int, num_sims: int,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        r"""
        Run the simulations for a given round.

        Args:
            round_: Round number.
            num_sims: Number of desired simulations for the round.

        Returns:
            theta: Parameters used for training.
            x: Simulations used for training.
            prior_mask: Whether each simulation came from a prior parameter sample.
        """

        if round_ == 0:
            theta = self._prior.sample((num_sims,))

            x = self._batched_simulator(theta)
        else:
            theta = self._posterior.sample(
                (num_sims,),
                x=self._posterior.default_x,
                show_progress_bars=self._show_progress_bars,
            )

            x = self._batched_simulator(theta)

        return theta, x, self._mask_sims_from_prior(round_, theta.size(0))

    def _mask_sims_from_prior(self, round_: int, num_simulations: int) -> Tensor:
        """Returns Tensor True where simulated from prior parameters.

        Args:
            round_: Current training round, starting at 0.
            num_simulations: Actually performed simulations. This number can be below
                the one fixed for the round if leakage correction through sampling is
                active and `patience` is not enough to reach it.
        """

        prior_mask_values = ones if round_ == 0 else zeros
        return prior_mask_values((num_simulations, 1), dtype=torch.bool)

    def _train(
        self,
        round_: int,
        training_batch_size: int,
        learning_rate: float,
        validation_fraction: float,
        stop_after_epochs: int,
        max_num_epochs: int,
        clip_max_norm: Optional[float],
        calibration_kernel: Callable,
        discard_prior_samples: bool,
    ) -> None:
        r"""Train the conditional density estimator for the posterior $p(\theta|x)$.

        Update the conditional density estimator weights to maximize the proposal
        posterior using the most recently aggregated bank of $(\theta, x)$ pairs.

        Uses performance on a held-out validation set as a terminating condition (early
        stopping).
        """

        # Starting index for the training set (1 = discard round-0 samples).
        start_idx = int(discard_prior_samples and round_ > 0)
        num_total_examples = sum(len(theta) for theta in self._theta_bank[start_idx:])

        # Select random neural net and validation splits from (theta, x) pairs.
        permuted_indices = torch.randperm(num_total_examples)
        num_training_examples = int((1 - validation_fraction) * num_total_examples)
        num_validation_examples = num_total_examples - num_training_examples
        train_indices, val_indices = (
            permuted_indices[:num_training_examples],
            permuted_indices[num_training_examples:],
        )

        # Dataset is shared for training and validation loaders.
        dataset = data.TensorDataset(
            torch.cat(self._theta_bank[start_idx:]),
            torch.cat(self._x_bank[start_idx:]),
            torch.cat(self._prior_masks[start_idx:]),
        )

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
                        round_, theta_batch, x_batch, masks_batch, calibration_kernel
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
                        round_, theta_batch, x_batch, masks_batch, calibration_kernel
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
        round_: int,
        theta: Tensor,
        x: Tensor,
        masks: Tensor,
        calibration_kernel: Callable,
    ) -> Tensor:
        """Return loss with proposal correction (`round_>0`) or without it (`round_=0`).

        The loss is the negative log prob. Irrespective of the round or SNPE method
        (A, B, or C), it can be weighted with a calibration kernel.

        Returns:
            Calibration kernel-weighted negative log prob.
        """

        if round_ == 0:
            # Use posterior log prob (without proposal correction) for first round.
            log_prob = self._posterior.net.log_prob(theta, x)
        else:
            # Use proposal posterior log prob tailored to snpe version (B, C).
            log_prob = self._log_prob_proposal_posterior(theta, x, masks)

        return -(calibration_kernel(x) * log_prob)

    @staticmethod
    def _warn_if_retrain_from_scratch_snpe(retrain_from_scratch_each_round):
        if retrain_from_scratch_each_round:
            warn(
                "You specified `retrain_from_scratch_each_round=True`. For "
                "SNPE, we have experienced very poor performance in this "
                "scenario and we therefore strongly recommend "
                "`retrain_from_scratch_each_round=False`, see GH #215."
            )


class PotentialFunctionProvider:
    """
    This class is initialized without arguments during the initialization of the
    Posterior class. When called, it specializes to the potential function appropriate
    to the requested `mcmc_method`.

    NOTE: Why use a class?
    ----------------------
    During inference, we use deepcopy to save untrained posteriors in memory. deepcopy
    uses pickle which can't serialize nested functions
    (https://stackoverflow.com/a/12022055).

    It is important to NOT initialize attributes upon instantiation, because we need the
     most current trained posterior neural net.

    Returns:
        Potential function for use by either numpy or pyro sampler
    """

    def __call__(
        self, prior, posterior_nn: nn.Module, x: Tensor, mcmc_method: str
    ) -> Callable:
        """Return potential function.

        Switch on numpy or pyro potential function based on `mcmc_method`.

        """
        self.posterior_nn = posterior_nn
        self.prior = prior
        self.x = x

        if mcmc_method in ("slice", "hmc", "nuts"):
            return self.pyro_potential
        else:
            return self.np_potential

    def np_potential(self, theta: np.ndarray) -> ScalarFloat:
        r"""Return posterior theta log prob. $p(\theta|x)$, $-\infty$ if outside prior."

        Args:
            theta: Parameters $\theta$, batch dimension 1.

        Returns:
            Posterior log probability $\log(p(\theta|x))$.
        """
        theta = torch.as_tensor(theta, dtype=torch.float32)

        is_within_prior = torch.isfinite(self.prior.log_prob(theta))
        if is_within_prior:
            target_log_prob = self.posterior_nn.log_prob(
                inputs=theta.reshape(1, -1), context=self.x.reshape(1, -1),
            )
        else:
            target_log_prob = -float("Inf")

        return target_log_prob

    def pyro_potential(self, theta: Dict[str, Tensor]) -> Tensor:
        r"""Return posterior log prob. of theta $p(\theta|x)$, -inf where outside prior.

        Args:
            theta: Parameters $\theta$ (from pyro sampler).

        Returns:
            Posterior log probability $p(\theta|x)$, masked outside of prior.
        """

        theta = next(iter(theta.values()))
        # Notice opposite sign to numpy.
        log_prob_posterior = -self.posterior_nn.log_prob(inputs=theta, context=self.x)
        log_prob_prior = self.prior.log_prob(theta)

        within_prior = torch.isfinite(log_prob_prior)

        return torch.where(within_prior, log_prob_posterior, log_prob_prior)
