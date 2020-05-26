from abc import ABC
from copy import deepcopy
from typing import Callable, Optional, Tuple, Dict
import warnings
import logging

import numpy as np
import torch
from torch import Tensor, nn, optim, zeros, ones
from torch.nn.utils import clip_grad_norm_
from torch.utils import data
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.tensorboard import SummaryWriter

from sbi.inference.base import NeuralInference
from sbi.inference.posteriors.sbi_posterior import NeuralPosterior
from sbi.types import ScalarFloat, OneOrMore
import sbi.utils as utils
from sbi.utils import Standardize


class SnpeBase(NeuralInference, ABC):
    def __init__(
        self,
        simulator: Callable,
        prior,
        x_o: Tensor,
        density_estimator: Optional[nn.Module] = None,
        calibration_kernel: Optional[Callable] = None,
        z_score_x: bool = True,
        z_score_min_std: float = 1e-7,
        simulation_batch_size: Optional[int] = 1,
        retrain_from_scratch_each_round: bool = False,
        discard_prior_samples: bool = False,
        device: Optional[torch.device] = None,
        sample_with_mcmc: bool = False,
        mcmc_method: str = "slice_np",
        num_workers: int = 1,
        worker_batch_size: int = 20,
        summary_writer: Optional[SummaryWriter] = None,
        skip_input_checks: bool = False,
        show_progressbar: bool = True,
        show_round_summary: bool = False,
        logging_level: int = logging.WARNING,
    ):
        """ Base class for Sequential Neural Posterior Estimation algorithms.

        Args:
            density_estimator: Neural density estimator.
            calibration_kernel: A function to calibrate the data x.
            z_score_x: Whether to z-score the data features x, default True.
            z_score_min_std: Minimum value of the standard deviation to use when
                standardizing inputs. This is typically needed when some simulator
                outputs are deterministic or nearly so.
            retrain_from_scratch_each_round: Whether to retrain the conditional
                density estimator for the posterior from scratch each round.
            discard_prior_samples: Whether to discard samples simulated in round 1, i.e.
                from the prior. Training may be sped up by ignoring such less targeted
                samples.

        See docstring of `NeuralInference` class for all other arguments.
        """

        super().__init__(
            simulator=simulator,
            prior=prior,
            x_o=x_o,
            simulation_batch_size=simulation_batch_size,
            device=device,
            summary_writer=summary_writer,
            num_workers=num_workers,
            worker_batch_size=worker_batch_size,
            skip_input_checks=skip_input_checks,
            show_progressbar=show_progressbar,
            show_round_summary=show_round_summary,
            logging_level=logging_level,
        )

        if density_estimator is None:
            density_estimator = utils.posterior_nn(
                model="maf",
                prior_mean=self._prior.mean,
                prior_std=self._prior.stddev,
                x_o_shape=self._x_o.shape,
            )

        # Calibration kernels proposed in Lueckmann, Goncalves et al 2017.
        if calibration_kernel is None:
            self.calibration_kernel = lambda x: ones([len(x)])
        else:
            self.calibration_kernel = calibration_kernel

        self._z_score_x, self._z_score_min_std = z_score_x, z_score_min_std
        self._retrain_from_scratch_each_round = retrain_from_scratch_each_round
        self._discard_prior_samples = discard_prior_samples

        # Create a neural posterior which can sample(), log_prob().
        self._posterior = NeuralPosterior(
            algorithm_family="snpe",
            neural_net=density_estimator,
            prior=self._prior,
            x_o=self._x_o,
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
        batch_size: int = 100,
        learning_rate: float = 5e-4,
        validation_fraction: float = 0.1,
        stop_after_epochs: int = 20,
        max_num_epochs: Optional[int] = None,
        clip_max_norm: Optional[float] = 5.0,
    ) -> NeuralPosterior:
        r"""Run SNPE

        Return posterior $p(\theta|x_o)$ after inference over several rounds.

        Args:
            num_rounds: Number of rounds to run.
            num_simulations_per_round: Number of simulator calls per round.
            batch_size: Size of batch to use for training.
            learning_rate: Learning rate for Adam optimizer.
            validation_fraction: The fraction of data to use for validation.
            stop_after_epochs: The number of epochs to wait for improvement on the
                validation set before terminating training.
            max_num_epochs: Maximum number of epochs to run. If max_num_epochs
                is reached, we stop training even if the validation loss is still
                decreasing. If None, we train until validation loss increases (see
                argument stop_after_epochs).
            clip_max_norm: Value at which to clip the total gradient norm in order to
                prevent exploding gradients. Use None for no clipping.

        Returns:
            Posterior $p(\theta|x_o)$ that can be sampled and evaluated.
        """

        max_num_epochs = 2 ** 31 - 1 if max_num_epochs is None else max_num_epochs

        num_sims_per_round = self._ensure_list(num_simulations_per_round, num_rounds)

        for round_, num_sims in enumerate(num_sims_per_round):

            # Run simulations for the round.
            theta, x, prior_mask = self._run_simulations(round_, num_sims)
            # XXX Rename bank -> rounds/roundwise.
            self._theta_bank.append(theta)
            self._x_bank.append(x)
            self._prior_masks.append(prior_mask)

            # Fit posterior using newly aggregated data set.
            self._train(
                round_=round_,
                batch_size=batch_size,
                learning_rate=learning_rate,
                validation_fraction=validation_fraction,
                stop_after_epochs=stop_after_epochs,
                max_num_epochs=max_num_epochs,
                clip_max_norm=clip_max_norm,
            )

            # Store models at end of each round.
            self._model_bank.append(deepcopy(self._posterior))
            self._model_bank[-1].net.eval()

            # making the call to get_leakage_correction() and the update of
            # self._leakage_density_correction_factor explicit here. This is just
            # to make sure this update never gets lost when we e.g. do not log our
            # things to tensorboard anymore. Calling get_leakage_correction() is needed
            # to update the leakage after each round.
            acceptance_rate = self._posterior.get_leakage_correction(
                x=self._x_o, force_update=True, show_progressbar=self._show_progressbar,
            )
            self._summary["rejection_sampling_acceptance_rates"].append(acceptance_rate)

            # Update description for progress bar.
            if self._show_round_summary:
                print(self._describe_round(round_, self._summary))

            # Update tensorboard and summary dict.
            correction = self._posterior.get_leakage_correction(x=self._x_o,)
            self._summarize(
                round_=round_,
                x_o=self._x_o,
                theta_bank=self._theta_bank,
                x_bank=self._x_bank,
                posterior_samples_acceptance_rate=correction,
            )

        self._posterior._num_trained_rounds = num_rounds
        return self._posterior

    def _log_prob_proposal_posterior(
        self, theta: Tensor, x: Tensor, masks: Tensor
    ) -> Tensor:
        """
        Return the log-probability used for the loss.

        Args:
            theta: Parameters θ.
            x: Simulations.
            masks: Binary, indicates whether to use prior samples.

        Returns: log-probability of the proposal posterior.
        """
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

            # why do we return theta just below? When using multiprocessing, the thetas
            # are not handled sequentially anymore. Hence, the x that are returned do
            # not necessarily have the same order as the theta we define above. We
            # therefore return a theta vector with the same ordering as x.
            theta, x = self._batched_simulator(theta)

            # What is happening here? By design, we want the neural net to take care of
            # normalizing both input and output, x and theta. But since we don't know
            # the dimensions of these upon instantiation, or in the case of
            # standardization, the mean and std we need to use, we grab the embedding
            # net from the posterior (though presumably we could get it directly here,
            # since what the posterior has is just a reference to the
            # `density_estimator`), we do things with it (such as prepending a
            # standardization step) and then we (destructively!) re-set the attribute
            # in the Posterior to be this now-normalized embedding net. Perhaps the
            # delayed chaining ideas in thinc can help make this a bit more transparent.
            if self._z_score_x:
                self._posterior.set_embedding_net(self._z_score_embedding(x))

        else:
            # XXX Make posterior.sample() accept tuples like prior.sample().
            theta = self._posterior.sample(
                num_sims, x=self._x_o, show_progressbar=self._show_progressbar
            )

            # why do we return theta just below? When using multiprocessing, the thetas
            # are not handled sequentially anymore. Hence, the x that are returned do
            # not necessarily have the same order as the theta we define above. We
            # therefore return a theta vector with the same ordering as x.
            theta, x = self._batched_simulator(theta)

        return theta, x, self._mask_sims_from_prior(round_, theta.size(0))

    def _z_score_embedding(self, x: Tensor) -> nn.Module:
        """Return embedding net with a standardizing step preprended."""

        # XXX Mouthful, rename self.posterior.nn
        embed_nn = self._posterior.net._embedding_net

        x_std = torch.std(x, dim=0)
        x_std[x_std == 0] = self._z_score_min_std
        preprocess = Standardize(torch.mean(x, dim=0), x_std)

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
        max_num_epochs: Optional[int],
        clip_max_norm: Optional[float],
    ) -> None:
        r"""Train the conditional density estimator for the posterior $p(\theta|x)$.

        Update the conditional density estimator weights to maximize the proposal
        posterior using the most recently aggregated bank of $(\theta, x)$ pairs.

        Uses performance on a held-out validation set as a terminating condition (early
        stopping).
        """

        # Starting index for the training set (1 = discard round-0 samples).
        start_idx = int(self._discard_prior_samples and round_ > 0)
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
        if self._retrain_from_scratch_each_round and round_ > 0:
            self._posterior = deepcopy(self._untrained_neural_posterior)

        epoch, self._val_log_prob = 0, float("-Inf")
        while not self._has_converged(epoch, stop_after_epochs):

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
                    self._loss(round_, theta_batch, x_batch, masks_batch)
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
                        round_, theta_batch, x_batch, masks_batch
                    )
                    log_prob_sum += batch_log_prob.sum().item()

            self._val_log_prob = log_prob_sum / num_validation_examples

            if self._show_progressbar:
                # end="\r" deletes the print statement when a new one appears.
                # https://stackoverflow.com/questions/3419984/
                print("Training neural network. Epochs trained: ", epoch, end="\r")

        if self._show_progressbar and self._has_converged(epoch, stop_after_epochs):
            # network has converged, we print this summary.
            print("Neural network successfully converged after", epoch, "epochs.")
        elif self._show_progressbar and max_num_epochs == epoch:
            # training has stopped because of max_num_epochs argument.
            print("Stopping neural network training after", epoch, "epochs.")

        if max_num_epochs == epoch:
            # warn if training was not stopped by early stopping
            warnings.warn(
                "Maximum number of epochs reached, but network has not yet "
                "fully converged. Consider increasing the value of max_num_epochs."
            )

        # Update summary.
        self._summary["epochs"].append(epoch)
        self._summary["best_validation_log_probs"].append(self._best_val_log_prob)

    def _loss(self, round_idx: int, theta: Tensor, x: Tensor, masks: Tensor) -> Tensor:
        """Return the right calibrated loss, depending on the round.

        The loss is the negative log prob. Irrespective of the round or SNPE method
        (A, B, or C) it is weighted with a calibration kernel.

        Returns:
            Calibration kernel-weighted negative log prob.
        """

        if round_idx == 0:
            # Use posterior log prob (without proposal correction) for first round.
            log_prob = self._posterior.net.log_prob(theta, x)
        else:
            # Use proposal posterior log prob tailored to snpe version (B, C).
            log_prob = self._log_prob_proposal_posterior(theta, x, masks)

        return -(self.calibration_kernel(x) * log_prob)


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
