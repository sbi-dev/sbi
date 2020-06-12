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

from sbi.inference import NeuralInference
from sbi.inference.posterior import NeuralPosterior
from sbi.types import OneOrMore, ScalarFloat
import sbi.utils as utils
from sbi.utils import Standardize, handle_invalid_x, warn_on_invalid_x
from sbi.utils.torchutils import get_default_device


class PosteriorEstimator(NeuralInference, ABC):
    def __init__(
        self,
        simulator: Callable,
        prior,
        x_shape: Optional[torch.Size] = None,
        num_workers: int = 1,
        simulation_batch_size: int = 1,
        density_estimator: Union[str, nn.Module] = "maf",
        sample_with_mcmc: bool = False,
        mcmc_method: str = "slice_np",
        device: Union[torch.device, str] = get_default_device(),
        logging_level: Union[int, str] = "WARNING",
        summary_writer: Optional[SummaryWriter] = None,
        show_progress_bars: bool = True,
        show_round_summary: bool = False,
    ):
        """ Base class for Sequential Neural Posterior Estimation methods.

        density_estimator: Either a string or a density estimation neural network
                that can `.log_prob()` and `.sample()`. If it is a string, use a pre-
                configured network of the provided type (one of nsf, maf, mdn, made).

        See docstring of `NeuralInference` class for all other arguments.
        """

        super().__init__(
            simulator=simulator,
            prior=prior,
            x_shape=x_shape,
            num_workers=num_workers,
            simulation_batch_size=simulation_batch_size,
            device=device,
            logging_level=logging_level,
            summary_writer=summary_writer,
            show_progress_bars=show_progress_bars,
            show_round_summary=show_round_summary,
        )

        if isinstance(density_estimator, str):
            density_estimator = utils.posterior_nn(
                model=density_estimator, prior=self._prior, x_o_shape=self._x_shape,
            )

        # Create a neural posterior which can sample(), log_prob().
        self._posterior = NeuralPosterior(
            method_family="snpe",
            neural_net=density_estimator,
            prior=self._prior,
            x_shape=self._x_shape,
            sample_with_mcmc=sample_with_mcmc,
            mcmc_method=mcmc_method,
            get_potential_function=PotentialFunctionProvider(),
        )

        self._prior_masks, self._model_bank = [], []

        # If we're retraining from scratch each round, keep a copy
        # of the original untrained model for reinitialization.
        self._untrained_neural_posterior = deepcopy(self._posterior)

        # Extra SNPE-specific fields summary_writer.
        self._summary.update({"rejection_sampling_acceptance_rates": []})  # type:ignore

    def __call__(
        self,
        num_rounds: int,
        num_simulations_per_round: OneOrMore[int],
        x_o: Optional[Tensor] = None,
        batch_size: int = 50,
        learning_rate: float = 5e-4,
        validation_fraction: float = 0.1,
        stop_after_epochs: int = 20,
        max_num_epochs: Optional[int] = None,
        clip_max_norm: Optional[float] = 5.0,
        calibration_kernel: Optional[Callable] = None,
        exclude_invalid_x: bool = True,
        z_score_x: bool = True,
        z_score_min_std: float = 1e-7,
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
            batch_size: Training batch size.
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
            z_score_x: Whether to z-score simulations `x`.
            z_score_min_std: Minimum value of the standard deviation to use when
                z-scoring `x`. This is typically needed when some simulator outputs are
                constant or nearly so.
            discard_prior_samples: Whether to discard samples simulated in round 1, i.e.
                from the prior. Training may be sped up by ignoring such less targeted
                samples.
            retrain_from_scratch_each_round: Whether to retrain the conditional density
                estimator for the posterior from scratch each round.

        Returns:
            Posterior $p(\theta|x)$ that can be sampled and evaluated.
        """

        self._handle_x_o_wrt_amortization(x_o, num_rounds)
        self._warn_if_retrain_from_scratch_snpe(retrain_from_scratch_each_round)

        # Calibration kernels proposed in Lueckmann, Gonçalves et al., 2017.
        if calibration_kernel is None:
            calibration_kernel = lambda x: ones([len(x)])

        max_num_epochs = 2 ** 31 - 1 if max_num_epochs is None else max_num_epochs

        num_sims_per_round = self._ensure_list(num_simulations_per_round, num_rounds)

        for round_, num_sims in enumerate(num_sims_per_round):

            # Run simulations for the round.
            theta, x, prior_mask = self._run_simulations(round_, num_sims)

            if round_ == 0:
                # What is happening here? By design, we want the neural net to take care
                # of normalizing both input and output, x and theta. But since we don't
                # know the dimensions of these upon instantiation, or in the case of
                # standardization, the mean and std we need to use, we grab the
                # embedding net from the posterior (though presumably we could get it
                # directly here, since what the posterior has is just a reference to the
                # `density_estimator`), we do things with it (such as prepending a
                # standardization step) and then we (destructively!) re-set the
                # attribute in the Posterior to be this now-normalized embedding net.
                # Perhaps the delayed chaining ideas in thinc can help make this a bit
                # more transparent.
                if z_score_x:
                    self._posterior.set_embedding_net(
                        self._prepend_z_score(x, z_score_min_std, exclude_invalid_x)
                    )

                # If we're retraining from scratch each round, keep a copy
                # of the original untrained model for reinitialization.
                self._untrained_neural_posterior = deepcopy(self._posterior)

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
                batch_size=batch_size,
                learning_rate=learning_rate,
                validation_fraction=validation_fraction,
                stop_after_epochs=stop_after_epochs,
                max_num_epochs=cast(int, max_num_epochs),
                clip_max_norm=clip_max_norm,
                calibration_kernel=calibration_kernel,
                discard_prior_samples=discard_prior_samples,
                retrain_from_scratch_each_round=retrain_from_scratch_each_round,
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
            # XXX Make posterior.sample() accept tuples like prior.sample().
            theta = self._posterior.sample(
                (num_sims,),
                x=self._posterior.default_x,
                show_progress_bars=self._show_progress_bars,
            )

            x = self._batched_simulator(theta)

        return theta, x, self._mask_sims_from_prior(round_, theta.size(0))

    def _prepend_z_score(
        self, x: Tensor, z_score_min_std: float, exclude_invalid_x: bool
    ) -> nn.Module:
        """Return embedding net with a standardizing step preprended."""

        embed_nn = self._posterior.net._embedding_net

        # Maybe exclude NaNs and infs from zscoring.
        # No warning on invalid x here because warning will occur in __call__.
        is_valid_x, *_ = handle_invalid_x(x, exclude_invalid_x)
        x_std = torch.std(x[is_valid_x], dim=0)
        x_std[x_std == 0] = z_score_min_std
        preprocess = Standardize(torch.mean(x[is_valid_x], dim=0), x_std)

        # If Sequential has a None component, forward will TypeError.
        return preprocess if embed_nn is None else nn.Sequential(preprocess, embed_nn)

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
        batch_size: int,
        learning_rate: float,
        validation_fraction: float,
        stop_after_epochs: int,
        max_num_epochs: int,
        clip_max_norm: Optional[float],
        calibration_kernel: Callable,
        discard_prior_samples: bool,
        retrain_from_scratch_each_round: bool,
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
            batch_size=min(batch_size, num_training_examples),
            drop_last=True,
            sampler=SubsetRandomSampler(train_indices),
        )
        val_loader = data.DataLoader(
            dataset,
            batch_size=min(batch_size, num_validation_examples),
            shuffle=False,
            drop_last=True,
            sampler=SubsetRandomSampler(val_indices),
        )

        optimizer = optim.Adam(
            list(self._posterior.net.parameters()), lr=learning_rate,
        )

        # If retraining from scratch each round, reset the neural posterior
        # to the untrained copy.
        if retrain_from_scratch_each_round and round_ > 0:
            self._posterior = deepcopy(self._untrained_neural_posterior)
            optimizer = optim.Adam(self._posterior.net.parameters(), lr=learning_rate,)

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
