# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Affero General Public License v3, see <https://www.gnu.org/licenses/>.


from abc import ABC
from copy import deepcopy
from typing import Any, Callable, Dict, Optional, Union

import numpy as np
import torch
from torch import Tensor, nn, optim
from torch.nn.utils import clip_grad_norm_
from torch.utils import data
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.tensorboard import SummaryWriter

from sbi import utils as utils
from sbi.inference import NeuralInference
from sbi.inference.posteriors.likelihood_based_posterior import LikelihoodBasedPosterior
from sbi.types import ScalarFloat
from sbi.utils import check_estimator_arg, x_shape_from_simulation


class LikelihoodEstimator(NeuralInference, ABC):
    def __init__(
        self,
        simulator: Callable,
        prior,
        num_workers: int = 1,
        simulation_batch_size: int = 1,
        density_estimator: Union[str, Callable] = "maf",
        mcmc_method: str = "slice_np",
        mcmc_parameters: Optional[Dict[str, Any]] = None,
        device: str = "cpu",
        logging_level: Union[int, str] = "WARNING",
        summary_writer: Optional[SummaryWriter] = None,
        show_progress_bars: bool = True,
        show_round_summary: bool = False,
    ):
        r"""Base class for Sequential Neural Likelihood Estimation methods.

        Args:
            density_estimator: If it is a string, use a pre-configured network of the
                provided type (one of nsf, maf, mdn, made). Alternatively, a function
                that builds a custom neural network can be provided. The function will
                be called with the first batch of simulations (theta, x), which can
                thus be used for shape inference and potentially for z-scoring. It
                needs to return a PyTorch `nn.Module` implementing the density
                estimator. The density estimator needs to provide the methods
                `.log_prob` and `.sample()`.
            mcmc_method: Method used for MCMC sampling, one of `slice_np`, `slice`, `hmc`, `nuts`.
                Currently defaults to `slice_np` for a custom numpy implementation of
                slice sampling; select `hmc`, `nuts` or `slice` for Pyro-based sampling.
            mcmc_parameters: Dictionary overriding the default parameters for MCMC.
                The following parameters are supported: `thin` to set the thinning
                factor for the chain, `warmup_steps` to set the initial number of
                samples to discard, `num_chains` for the number of chains, `init_strategy`
                for the initialisation strategy for chains; `prior` will draw init
                locations from prior, whereas `sir` will use Sequential-Importance-
                Resampling using `init_strategy_num_candidates` to find init
                locations.

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
            self._build_neural_net = utils.likelihood_nn(model=density_estimator)
        else:
            self._build_neural_net = density_estimator
        self._posterior = None
        self._mcmc_method = mcmc_method
        self._mcmc_parameters = mcmc_parameters

        # SNLE-specific summary_writer fields.
        self._summary.update({"mcmc_times": []})  # type: ignore

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
        exclude_invalid_x: bool = True,
        discard_prior_samples: bool = False,
        retrain_from_scratch_each_round: bool = False,
    ) -> LikelihoodBasedPosterior:
        r"""Run SNLE.

        Return posterior $p(\theta|x)$ after inference.

        Args:
            exclude_invalid_x: Whether to exclude simulation outputs `x=NaN` or `x=±∞`
                during training. Expect errors, silent or explicit, when `False`.
            discard_prior_samples: Whether to discard samples simulated in round 1, i.e.
                from the prior. Training may be sped up by ignoring such less targeted
                samples.
            retrain_from_scratch_each_round: Whether to retrain the conditional density
                estimator for the posterior from scratch each round.

        Returns:
            Posterior $p(\theta|x_o)$ that can be sampled and evaluated.
        """

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
        # arguments, which will build the neural network
        # This is passed into NeuralPosterior, to create a neural posterior which
        # can `sample()` and `log_prob()`. The network is accessible via `.net`.
        if self._posterior is None or retrain_from_scratch_each_round:
            x_shape = x_shape_from_simulation(x)
            self._posterior = LikelihoodBasedPosterior(
                method_family="snle",
                neural_net=self._build_neural_net(theta, x),
                prior=self._prior,
                x_shape=x_shape,
                mcmc_method=self._mcmc_method,
                mcmc_parameters=self._mcmc_parameters,
                get_potential_function=PotentialFunctionProvider(),
            )

        # Fit neural likelihood to newly aggregated dataset.
        self._train(
            training_batch_size=training_batch_size,
            learning_rate=learning_rate,
            validation_fraction=validation_fraction,
            stop_after_epochs=stop_after_epochs,
            max_num_epochs=max_num_epochs,
            clip_max_norm=clip_max_norm,
            exclude_invalid_x=exclude_invalid_x,
            discard_prior_samples=discard_prior_samples,
        )

        # Update description for progress bar.
        if self._show_round_summary:
            print(self._describe_round(self._round, self._summary))

        # Update TensorBoard and summary dict.
        self._summarize(
            round_=self._round,
            x_o=self._posterior.default_x,
            theta_bank=theta,
            x_bank=x,
        )

        self._posterior._num_trained_rounds = self._round + 1
        return deepcopy(self._posterior)

    def _train(
        self,
        training_batch_size: int,
        learning_rate: float,
        validation_fraction: float,
        stop_after_epochs: int,
        max_num_epochs: int,
        clip_max_norm: Optional[float],
        exclude_invalid_x: bool,
        discard_prior_samples: bool,
    ) -> None:
        r"""
        Train the conditional density estimator for the likelihood.

        Update the conditional density estimator weights to maximize the
        likelihood on the most recently aggregated bank of $(\theta, x)$ pairs.

        Uses performance on a held-out validation set as a terminating condition (early
        stopping).
        """

        # Starting index for the training set (1 = discard round-0 samples).
        start_idx = int(discard_prior_samples and self._round > 0)
        theta, x, _ = self._get_from_data_bank(start_idx, exclude_invalid_x)

        # Get total number of training examples.
        num_examples = len(theta)

        # Select random train and validation splits from (theta, x) pairs.
        permuted_indices = torch.randperm(num_examples)
        num_training_examples = int((1 - validation_fraction) * num_examples)
        num_validation_examples = num_examples - num_training_examples
        train_indices, val_indices = (
            permuted_indices[:num_training_examples],
            permuted_indices[num_training_examples:],
        )

        # Dataset is shared for training and validation loaders.
        dataset = data.TensorDataset(x, theta)

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
            drop_last=False,
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
                theta_batch, x_batch = (
                    batch[0].to(self._device),
                    batch[1].to(self._device),
                )
                log_prob = self._posterior.net.log_prob(theta_batch, context=x_batch)
                loss = -torch.mean(log_prob)
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
                for batch in val_loader:
                    theta_batch, x_batch = (
                        batch[0].to(self._device),
                        batch[1].to(self._device),
                    )
                    log_prob = self._posterior.net.log_prob(
                        theta_batch, context=x_batch
                    )
                    log_prob_sum += log_prob.sum().item()
            self._val_log_prob = log_prob_sum / num_validation_examples

            self._maybe_show_progress(self._show_progress_bars, epoch)

        self._report_convergence_at_end(epoch, stop_after_epochs, max_num_epochs)

        # Update summary.
        self._summary["epochs"].append(epoch)
        self._summary["best_validation_log_probs"].append(self._best_val_log_prob)


class PotentialFunctionProvider:
    """
    This class is initialized without arguments during the initialization of the
     Posterior class. When called, it specializes to the potential function appropriate
     to the requested mcmc_method.

    NOTE: Why use a class?
    ----------------------
    During inference, we use deepcopy to save untrained posteriors in memory. deepcopy
    uses pickle which can't serialize nested functions
    (https://stackoverflow.com/a/12022055).

    It is important to NOT initialize attributes upon instantiation, because we need the
    most current trained posterior neural net.

    Returns:
        Potential function for use by either numpy or pyro sampler.
    """

    def __call__(
        self, prior, likelihood_nn: nn.Module, x: Tensor, mcmc_method: str,
    ) -> Callable:
        r"""Return potential function for posterior $p(\theta|x)$.

        Switch on numpy or pyro potential function based on mcmc_method.

        Args:
            prior: Prior distribution that can be evaluated.
            likelihood_nn: Neural likelihood estimator that can be evaluated.
            x: Conditioning variable for posterior $p(\theta|x)$.
            mcmc_method: One of `slice_np`, `slice`, `hmc` or `nuts`.

        Returns:
            Potential function for sampler.
        """
        self.likelihood_nn = likelihood_nn
        self.prior = prior
        self.x = x

        if mcmc_method in ("slice", "hmc", "nuts"):
            return self.pyro_potential
        else:
            return self.np_potential

    def np_potential(self, theta: np.array) -> ScalarFloat:
        r"""Return posterior log prob. of theta $p(\theta|x)$"

        Args:
            theta: Parameters $\theta$, batch dimension 1.

        Returns:
            Posterior log probability of the theta, $-\infty$ if impossible under prior.
        """
        theta = torch.as_tensor(theta, dtype=torch.float32)
        log_likelihood = self.likelihood_nn.log_prob(
            inputs=self.x.reshape(1, -1), context=theta.reshape(1, -1)
        )

        # Notice opposite sign to pyro potential.
        return log_likelihood + self.prior.log_prob(theta)

    def pyro_potential(self, theta: Dict[str, Tensor]) -> Tensor:
        r"""Return posterior log probability of parameters $p(\theta|x)$.

         Args:
            theta: Parameters $\theta$. The tensor's shape will be
                (1, shape_of_single_theta) if running a single chain or just
                (shape_of_single_theta) for multiple chains.

        Returns:
            The potential $-[\log r(x_o, \theta) + \log p(\theta)]$.
        """

        theta = next(iter(theta.values()))

        log_likelihood = self.likelihood_nn.log_prob(
            inputs=self.x.reshape(1, -1), context=theta.reshape(1, -1)
        )

        return -(log_likelihood + self.prior.log_prob(theta))
