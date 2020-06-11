from __future__ import annotations

from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Callable, Dict, Optional, Union

import numpy as np
import torch
from torch import Tensor, eye, nn, ones, optim
from torch.nn.utils import clip_grad_norm_
from torch.utils import data
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.tensorboard import SummaryWriter

from sbi.inference.base import NeuralInference
from sbi.inference.posterior import NeuralPosterior
from sbi.types import OneOrMore, ScalarFloat
from sbi.utils import clamp_and_warn
from sbi.utils import handle_invalid_x, warn_on_invalid_x
import sbi.utils as utils
from sbi.utils.torchutils import ensure_theta_batched, ensure_x_batched
from sbi.utils.torchutils import get_default_device


class RatioEstimator(NeuralInference, ABC):
    def __init__(
        self,
        simulator: Callable,
        prior,
        x_shape: Optional[torch.Size] = None,
        num_workers: int = 1,
        simulation_batch_size: int = 1,
        embedding_net: nn.Module = nn.Identity(),
        classifier: Union[str, nn.Module] = "resnet",
        mcmc_method: str = "slice_np",
        device: Union[torch.device, str] = get_default_device(),
        logging_level: Union[int, str] = "warning",
        summary_writer: Optional[SummaryWriter] = None,
        show_progress_bars: bool = True,
        show_round_summary: bool = False,
    ):
        r"""Sequential Neural Ratio Estimation.

        We implement two inference methods in the respective subclasses.

        - SNRE_A / AALR is limited to `num_atoms=2`, but allows for density evaluation
          when training for one round.
        - SNRE_B / SRE can use more than two atoms, potentially boosting performance,
          but allows for posterior evaluation **only up to a normalizing constant**,
          even when training only one round.

        Args:
            classifier: Binary classifier network.
            embedding_net: A trainable network that maps high-dimensional simulation
                outputs $x$ to lower-dimensional feature vectors $f(x)$ to feed the
                classifier.

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

        if isinstance(classifier, str):
            classifier = utils.classifier_nn(
                model=classifier,
                theta_shape=self._prior.sample().shape,
                x_o_shape=self._x_shape,
            )

        # Create posterior object which can sample().
        method_family = self.__class__.__name__.lower()

        self._posterior = NeuralPosterior(
            method_family=method_family,
            neural_net=classifier,
            prior=self._prior,
            x_shape=self._x_shape,
            mcmc_method=mcmc_method,
            get_potential_function=PotentialFunctionProvider(),
        )

        self._posterior.net.train(True)

        if not isinstance(embedding_net, nn.Identity):
            raise NotImplementedError("Embedding net not yet implemented for SNRE.")
        self._embedding_net = embedding_net

        # Ratio-based-specific summary_writer fields.
        self._summary.update({"mcmc_times": []})  # type: ignore

    def __call__(
        self,
        num_rounds: int,
        num_simulations_per_round: OneOrMore[int],
        x_o: Optional[Tensor] = None,
        num_atoms: int = 10,
        batch_size: int = 50,
        learning_rate: float = 5e-4,
        validation_fraction: float = 0.1,
        stop_after_epochs: int = 20,
        max_num_epochs: Optional[int] = None,
        clip_max_norm: Optional[float] = 5.0,
        exclude_invalid_x: bool = True,
        discard_prior_samples: bool = False,
        retrain_from_scratch_each_round: bool = False,
    ) -> NeuralPosterior:
        """Run SNRE.

        Return posterior $p(\theta|x)$ after inference (possibly over several rounds).

        Args:
            num_atoms: Number of atoms to use for classification.
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

        self._handle_x_o_wrt_amortization(x_o, num_rounds)

        # If we're retraining from scratch each round,
        # keep a copy of the original untrained model for reinitialization.
        if retrain_from_scratch_each_round:
            self._untrained_posterior = deepcopy(self._posterior)
            self._untrained_embedding_net = deepcopy(self._embedding_net)

        max_num_epochs = 2 ** 31 - 1 if max_num_epochs is None else max_num_epochs

        num_sims_per_round = self._ensure_list(num_simulations_per_round, num_rounds)

        for round_, num_sims in enumerate(num_sims_per_round):

            # Generate theta from prior in first round, and from most recent posterior
            # estimate in subsequent rounds.
            if round_ == 0:
                theta = self._prior.sample((num_sims,))
            else:
                theta = self._posterior.sample(
                    (num_sims,), show_progress_bars=self._show_progress_bars
                )

            x = self._batched_simulator(theta)

            # Check for NaNs in simulations.
            is_valid_x, num_nans, num_infs = handle_invalid_x(x, exclude_invalid_x)
            warn_on_invalid_x(num_nans, num_infs, exclude_invalid_x)

            # Store (theta, x) pairs.
            self._theta_bank.append(theta[is_valid_x])
            self._x_bank.append(x[is_valid_x])

            # Fit posterior using newly aggregated data set.
            self._train(
                round_=round_,
                num_atoms=num_atoms,
                batch_size=batch_size,
                learning_rate=learning_rate,
                validation_fraction=validation_fraction,
                stop_after_epochs=stop_after_epochs,
                max_num_epochs=max_num_epochs,
                clip_max_norm=clip_max_norm,
                discard_prior_samples=discard_prior_samples,
                retrain_from_scratch_each_round=retrain_from_scratch_each_round,
            )

            # Update description for progress bar.
            if self._show_round_summary:
                print(self._describe_round(round_, self._summary))

            # Update tensorboard and summary dict.
            self._summarize(
                round_=round_,
                x_o=self._posterior.default_x,
                theta_bank=self._theta_bank,
                x_bank=self._x_bank,
            )

        self._posterior._num_trained_rounds = num_rounds

        return self._posterior

    def _train(
        self,
        round_: int,
        num_atoms: int,
        batch_size: int,
        learning_rate: float,
        validation_fraction: float,
        stop_after_epochs: int,
        max_num_epochs: int,
        clip_max_norm: Optional[float],
        discard_prior_samples: bool,
        retrain_from_scratch_each_round: bool,
    ) -> None:
        r"""
        Trains the neural classifier.

        Update the classifier weights by maximizing a Bernoulli likelihood which
        distinguishes between jointly distributed $(\theta, x)$ pairs and randomly
        chosen $(\theta, x)$ pairs.

        Uses performance on a held-out validation set as a terminating condition (early
        stopping).
        """

        # Starting index for the training set (1 = discard round-0 samples).
        start_idx = int(discard_prior_samples and round_ > 0)
        # Get total number of training examples.
        num_examples = sum(len(theta) for theta in self._theta_bank)

        # Select random train and validation splits from (theta, x) pairs.
        permuted_indices = torch.randperm(num_examples)
        num_training_examples = int((1 - validation_fraction) * num_examples)
        num_validation_examples = num_examples - num_training_examples
        train_indices, val_indices = (
            permuted_indices[:num_training_examples],
            permuted_indices[num_training_examples:],
        )

        clipped_batch_size = min(batch_size, num_validation_examples)

        # num_atoms = theta.shape[0]
        clamp_and_warn("num_atoms", num_atoms, min_val=2, max_val=clipped_batch_size)

        # Dataset is shared for training and validation loaders.
        dataset = data.TensorDataset(
            torch.cat(self._theta_bank[start_idx:]), torch.cat(self._x_bank[start_idx:])
        )

        # Create neural net and validation loaders using a subset sampler.

        train_loader = data.DataLoader(
            dataset,
            batch_size=clipped_batch_size,
            drop_last=True,
            sampler=SubsetRandomSampler(train_indices),
        )
        val_loader = data.DataLoader(
            dataset,
            batch_size=clipped_batch_size,
            shuffle=False,
            drop_last=False,
            sampler=SubsetRandomSampler(val_indices),
        )

        optimizer = optim.Adam(
            list(self._posterior.net.parameters())
            + list(self._embedding_net.parameters()),
            lr=learning_rate,
        )

        # If we're retraining from scratch each round, reset the neural posterior
        # to the untrained copy we made at the start.
        if retrain_from_scratch_each_round:
            self._posterior = deepcopy(self._untrained_posterior)
            self._embedding_net = deepcopy(self._untrained_embedding_net)
            optimizer = optim.Adam(
                list(self._posterior.net.parameters())
                + list(self._embedding_net.parameters()),
                lr=learning_rate,
            )

        epoch, self._val_log_prob = 0, float("-Inf")

        while epoch <= max_num_epochs and not self._converged(epoch, stop_after_epochs):

            # Train for a single epoch.
            self._posterior.net.train()
            for theta_batch, x_batch in train_loader:
                optimizer.zero_grad()
                loss = self._loss(theta_batch, x_batch, num_atoms)
                loss.backward()
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
                for theta_batch, x_batch in val_loader:
                    log_prob = self._loss(theta_batch, x_batch, num_atoms)
                    log_prob_sum -= log_prob.sum().item()
                self._val_log_prob = log_prob_sum / num_validation_examples

            self._maybe_show_progress(self._show_progress_bars, epoch)

        self._report_convergence_at_end(epoch, stop_after_epochs, max_num_epochs)

        # Update summary.
        self._summary["epochs"].append(epoch)
        self._summary["best_validation_log_probs"].append(self._best_val_log_prob)

    def _classifier_logits(self, theta: Tensor, x: Tensor, num_atoms: int) -> Tensor:
        """Return logits obtained through classifier forward pass.

        The logits are obtained from atomic sets of (theta,x) pairs.
        """
        batch_size = theta.shape[0]
        repeated_x = utils.repeat_rows(x, num_atoms)

        # Choose `1` or `num_atoms - 1` thetas from the rest of the batch for each x.
        probs = ones(batch_size, batch_size) * (1 - eye(batch_size)) / (batch_size - 1)

        choices = torch.multinomial(probs, num_samples=num_atoms - 1, replacement=False)

        contrasting_theta = theta[choices]

        atomic_theta = torch.cat((theta[:, None, :], contrasting_theta), dim=1).reshape(
            batch_size * num_atoms, -1
        )

        theta_and_x = torch.cat((atomic_theta, repeated_x), dim=1)

        return self._posterior.net(theta_and_x)

    @abstractmethod
    def _loss(self, theta: Tensor, x: Tensor, num_atoms: int) -> Tensor:
        raise NotImplementedError


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
        self, prior, classifier: nn.Module, x: Tensor, mcmc_method: str,
    ) -> Callable:
        r"""Return potential function for posterior $p(\theta|x)$.

        Switch on numpy or pyro potential function based on `mcmc_method`.

        Args:
            prior: Prior distribution that can be evaluated.
            classifier: Binary classifier approximating the likelihood up to a constant.

            x: Conditioning variable for posterior $p(\theta|x)$.
            mcmc_method: One of `slice_np`, `slice`, `hmc` or `nuts`.

        Returns:
            Potential function for sampler.
        """

        self.classifier = classifier
        self.prior = prior
        self.x = x

        if mcmc_method in ("slice", "hmc", "nuts"):
            return self.pyro_potential
        else:
            return self.np_potential

    def np_potential(self, theta: np.array) -> ScalarFloat:
        """Return potential for Numpy slice sampler."

        Args:
            theta: Parameters $\theta$, batch dimension 1.

        Returns:
            Posterior log probability of theta.
        """
        theta = torch.as_tensor(theta, dtype=torch.float32)

        # Theta and x should have shape (1, dim).
        theta = ensure_theta_batched(theta)
        x = ensure_x_batched(self.x)

        log_ratio = self.classifier(torch.cat((theta, x), dim=1).reshape(1, -1))

        # Notice opposite sign to pyro potential.
        return log_ratio + self.prior.log_prob(theta)

    def pyro_potential(self, theta: Dict[str, Tensor]) -> Tensor:
        r"""Return potential for Pyro sampler.

        Args:
            theta: Parameters $\theta$. The tensor's shape will be
             (1, shape_of_single_theta) if running a single chain or just
             (shape_of_single_theta) for multiple chains.

        Returns:
            Potential $-(\log r(x_o, \theta) + \log p(\theta))$.
        """

        theta = next(iter(theta.values()))

        # Theta and x should have shape (1, dim).
        theta = ensure_theta_batched(theta)
        x = ensure_x_batched(self.x)

        log_ratio = self.classifier(torch.cat((theta, x), dim=1).reshape(1, -1))

        return -(log_ratio + self.prior.log_prob(theta))
