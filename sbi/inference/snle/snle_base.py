# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Affero General Public License v3, see <https://www.gnu.org/licenses/>.


from abc import ABC
from copy import deepcopy
from typing import Any, Callable, Dict, Optional, Union

import torch
from pyknos.nflows import flows
from torch import Tensor, nn, optim
from torch.distributions import Distribution
from torch.nn.utils.clip_grad import clip_grad_norm_
from torch.utils import data
from torch.utils.tensorboard.writer import SummaryWriter

from sbi import utils as utils
from sbi.inference import NeuralInference
from sbi.inference.posteriors import MCMCPosterior, RejectionPosterior, VIPosterior
from sbi.inference.potentials import likelihood_estimator_based_potential
from sbi.utils import (
    check_estimator_arg,
    check_prior,
    handle_invalid_x,
    mask_sims_from_prior,
    nle_nre_apt_msg_on_invalid_x,
    validate_theta_and_x,
    warn_if_zscoring_changes_data,
    x_shape_from_simulation,
)


class LikelihoodEstimator(NeuralInference, ABC):
    def __init__(
        self,
        prior: Optional[Distribution] = None,
        density_estimator: Union[str, Callable] = "maf",
        device: str = "cpu",
        logging_level: Union[int, str] = "WARNING",
        summary_writer: Optional[SummaryWriter] = None,
        show_progress_bars: bool = True,
    ):
        r"""Base class for Sequential Neural Likelihood Estimation methods.

        Args:
            prior: A probability distribution that expresses prior knowledge about the
                parameters, e.g. which ranges are meaningful for them. Any
                object with `.log_prob()`and `.sample()` (for example, a PyTorch
                distribution) can be used.
            density_estimator: If it is a string, use a pre-configured network of the
                provided type (one of nsf, maf, mdn, made). Alternatively, a function
                that builds a custom neural network can be provided. The function will
                be called with the first batch of simulations (theta, x), which can
                thus be used for shape inference and potentially for z-scoring. It
                needs to return a PyTorch `nn.Module` implementing the density
                estimator. The density estimator needs to provide the methods
                `.log_prob` and `.sample()`.

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
            self._build_neural_net = utils.likelihood_nn(model=density_estimator)
        else:
            self._build_neural_net = density_estimator

    def append_simulations(
        self,
        theta: Tensor,
        x: Tensor,
        exclude_invalid_x: bool = False,
        from_round: int = 0,
        data_device: Optional[str] = None,
    ) -> "LikelihoodEstimator":
        r"""Store parameters and simulation outputs to use them for later training.

        Data are stored as entries in lists for each type of variable (parameter/data).

        Stores $\theta$, $x$, prior_masks (indicating if simulations are coming from the
        prior or not) and an index indicating which round the batch of simulations came
        from.

        Args:
            theta: Parameter sets.
            x: Simulation outputs.
            exclude_invalid_x: Whether invalid simulations are discarded during
                training. If `False`, SNLE raises an error when invalid simulations are
                found. If `True`, invalid simulations are discarded and training
                can proceed, but this gives systematically wrong results.
            from_round: Which round the data stemmed from. Round 0 means from the prior.
                With default settings, this is not used at all for `SNLE`. Only when
                the user later on requests `.train(discard_prior_samples=True)`, we
                use these indices to find which training data stemmed from the prior.
            data_device: Where to store the data, default is on the same device where
                the training is happening. If training a large dataset on a GPU with not
                much VRAM can set to 'cpu' to store data on system memory instead.
        Returns:
            NeuralInference object (returned so that this function is chainable).
        """

        is_valid_x, num_nans, num_infs = handle_invalid_x(x, exclude_invalid_x)

        x = x[is_valid_x]
        theta = theta[is_valid_x]

        # Check for problematic z-scoring
        warn_if_zscoring_changes_data(x)
        nle_nre_apt_msg_on_invalid_x(num_nans, num_infs, exclude_invalid_x, "SNLE")

        if data_device is None:
            data_device = self._device
        theta, x = validate_theta_and_x(
            theta, x, data_device=data_device, training_device=self._device
        )

        prior_masks = mask_sims_from_prior(int(from_round), theta.size(0))

        self._theta_roundwise.append(theta)
        self._x_roundwise.append(x)
        self._prior_masks.append(prior_masks)

        self._data_round_index.append(int(from_round))

        return self

    def train(
        self,
        training_batch_size: int = 50,
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
    ) -> flows.Flow:
        r"""Train the density estimator to learn the distribution $p(x|\theta)$.

        Args:
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
        # Load data from most recent round.
        self._round = max(self._data_round_index)
        # Starting index for the training set (1 = discard round-0 samples).
        start_idx = int(discard_prior_samples and self._round > 0)

        train_loader, val_loader = self.get_dataloaders(
            start_idx,
            training_batch_size,
            validation_fraction,
            resume_training,
            dataloader_kwargs=dataloader_kwargs,
        )

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
            self._x_shape = x_shape_from_simulation(x.to("cpu"))
            del theta, x
            assert (
                len(self._x_shape) < 3
            ), "SNLE cannot handle multi-dimensional simulator output."

        self._neural_net.to(self._device)
        if not resume_training:
            self.optimizer = optim.Adam(
                list(self._neural_net.parameters()),
                lr=learning_rate,
            )
            self.epoch, self._val_log_prob = 0, float("-Inf")

        while self.epoch <= max_num_epochs and not self._converged(
            self.epoch, stop_after_epochs
        ):

            # Train for a single epoch.
            self._neural_net.train()
            train_log_probs_sum = 0
            for batch in train_loader:
                self.optimizer.zero_grad()
                theta_batch, x_batch = (
                    batch[0].to(self._device),
                    batch[1].to(self._device),
                )
                # Evaluate on x with theta as context.
                train_losses = self._loss(theta=theta_batch, x=x_batch)
                train_loss = torch.mean(train_losses)
                train_log_probs_sum -= train_losses.sum().item()

                train_loss.backward()
                if clip_max_norm is not None:
                    clip_grad_norm_(
                        self._neural_net.parameters(),
                        max_norm=clip_max_norm,
                    )
                self.optimizer.step()

            self.epoch += 1

            train_log_prob_average = train_log_probs_sum / (
                len(train_loader) * train_loader.batch_size  # type: ignore
            )
            self._summary["training_log_probs"].append(train_log_prob_average)

            # Calculate validation performance.
            self._neural_net.eval()
            val_log_prob_sum = 0
            with torch.no_grad():
                for batch in val_loader:
                    theta_batch, x_batch = (
                        batch[0].to(self._device),
                        batch[1].to(self._device),
                    )
                    # Evaluate on x with theta as context.
                    val_losses = self._loss(theta=theta_batch, x=x_batch)
                    val_log_prob_sum -= val_losses.sum().item()

            # Take mean over all validation samples.
            self._val_log_prob = val_log_prob_sum / (
                len(val_loader) * val_loader.batch_size  # type: ignore
            )
            # Log validation log prob for every epoch.
            self._summary["validation_log_probs"].append(self._val_log_prob)

            self._maybe_show_progress(self._show_progress_bars, self.epoch)

        self._report_convergence_at_end(self.epoch, stop_after_epochs, max_num_epochs)

        # Update summary.
        self._summary["epochs_trained"].append(self.epoch)
        self._summary["best_validation_log_prob"].append(self._best_val_log_prob)

        # Update TensorBoard and summary dict.
        self._summarize(round_=self._round)

        # Update description for progress bar.
        if show_train_summary:
            print(self._describe_round(self._round, self._summary))

        # Avoid keeping the gradients in the resulting network, which can
        # cause memory leakage when benchmarking.
        self._neural_net.zero_grad(set_to_none=True)

        return deepcopy(self._neural_net)

    def build_posterior(
        self,
        density_estimator: Optional[nn.Module] = None,
        prior: Optional[Distribution] = None,
        sample_with: str = "mcmc",
        mcmc_method: str = "slice_np",
        vi_method: str = "rKL",
        mcmc_parameters: Dict[str, Any] = {},
        vi_parameters: Dict[str, Any] = {},
        rejection_sampling_parameters: Dict[str, Any] = {},
    ) -> Union[MCMCPosterior, RejectionPosterior, VIPosterior]:
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
                [`mcmc` | `rejection` | `vi`].
            mcmc_method: Method used for MCMC sampling, one of `slice_np`, `slice`,
                `hmc`, `nuts`. Currently defaults to `slice_np` for a custom numpy
                implementation of slice sampling; select `hmc`, `nuts` or `slice` for
                Pyro-based sampling.
            vi_method: Method used for VI, one of [`rKL`, `fKL`, `IW`, `alpha`]. Note
                some of the methods admit a `mode seeking` property (e.g. rKL) whereas
                some admit a `mass covering` one (e.g fKL).
            mcmc_parameters: Additional kwargs passed to `MCMCPosterior`.
            vi_parameters: Additional kwargs passed to `VIPosterior`.
            rejection_sampling_parameters: Additional kwargs passed to
                `RejectionPosterior`.

        Returns:
            Posterior $p(\theta|x)$  with `.sample()` and `.log_prob()` methods
            (the returned log-probability is unnormalized).
        """
        if prior is None:
            assert (
                self._prior is not None
            ), """You did not pass a prior. You have to pass the prior either at
            initialization `inference = SNLE(prior)` or to `.build_posterior
            (prior=prior)`."""
            prior = self._prior
        else:
            check_prior(prior)

        if density_estimator is None:
            likelihood_estimator = self._neural_net
            # If internal net is used device is defined.
            device = self._device
        else:
            likelihood_estimator = density_estimator
            # Otherwise, infer it from the device of the net parameters.
            device = next(density_estimator.parameters()).device.type

        potential_fn, theta_transform = likelihood_estimator_based_potential(
            likelihood_estimator=likelihood_estimator,
            prior=prior,
            x_o=None,
        )

        if sample_with == "mcmc":
            self._posterior = MCMCPosterior(
                potential_fn=potential_fn,
                theta_transform=theta_transform,
                proposal=prior,
                method=mcmc_method,
                device=device,
                x_shape=self._x_shape,
                **mcmc_parameters,
            )
        elif sample_with == "rejection":
            self._posterior = RejectionPosterior(
                potential_fn=potential_fn,
                proposal=prior,
                device=device,
                x_shape=self._x_shape,
                **rejection_sampling_parameters,
            )
        elif sample_with == "vi":
            self._posterior = VIPosterior(
                potential_fn=potential_fn,
                theta_transform=theta_transform,
                prior=prior,  # type: ignore
                vi_method=vi_method,
                device=device,
                x_shape=self._x_shape,
                **vi_parameters,
            )
        else:
            raise NotImplementedError

        # Store models at end of each round.
        self._model_bank.append(deepcopy(self._posterior))

        return deepcopy(self._posterior)

    def _loss(self, theta: Tensor, x: Tensor) -> Tensor:
        r"""Return loss for SNLE, which is the likelihood of $-\log q(x_i | \theta_i)$.

        Returns:
            Negative log prob.
        """
        return -self._neural_net.log_prob(x, context=theta)
