from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Any, Callable, Dict, Optional, Union

import torch
from torch import Tensor, eye, ones, optim
from torch.nn.utils import clip_grad_norm_
from torch.utils import data
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.tensorboard import SummaryWriter

from sbi import utils as utils
from sbi.inference.base import NeuralInference
from sbi.inference.posteriors.ratio_based_posterior import RatioBasedPosterior
from sbi.types import TorchModule
from sbi.utils import (
    check_estimator_arg,
    clamp_and_warn,
    validate_theta_and_x,
    x_shape_from_simulation,
)
from sbi.utils.sbiutils import mask_sims_from_prior


class RatioEstimator(NeuralInference, ABC):
    def __init__(
        self,
        prior,
        classifier: Union[str, Callable] = "resnet",
        device: str = "cpu",
        logging_level: Union[int, str] = "warning",
        summary_writer: Optional[SummaryWriter] = None,
        show_progress_bars: bool = True,
        **unused_args
    ):
        r"""Sequential Neural Ratio Estimation.

        We implement two inference methods in the respective subclasses.

        - SNRE_A / AALR is limited to `num_atoms=2`, but allows for density evaluation
          when training for one round.
        - SNRE_B / SRE can use more than two atoms, potentially boosting performance,
          but allows for posterior evaluation **only up to a normalizing constant**,
          even when training only one round.

        Args:
            classifier: Classifier trained to approximate likelihood ratios. If it is
                a string, use a pre-configured network of the provided type (one of
                linear, mlp, resnet). Alternatively, a function that builds a custom
                neural network can be provided. The function will be called with the
                first batch of simulations (theta, x), which can thus be used for shape
                inference and potentially for z-scoring. It needs to return a PyTorch
                `nn.Module` implementing the classifier.
            unused_args: Absorbs additional arguments. No entries will be used. If it
                is not empty, we warn. In future versions, when the new interface of
                0.14.0 is more mature, we will remove this argument.

        See docstring of `NeuralInference` class for all other arguments.
        """

        super().__init__(
            prior=prior,
            device=device,
            logging_level=logging_level,
            summary_writer=summary_writer,
            show_progress_bars=show_progress_bars,
            **unused_args
        )

        # As detailed in the docstring, `density_estimator` is either a string or
        # a callable. The function creating the neural network is attached to
        # `_build_neural_net`. It will be called in the first round and receive
        # thetas and xs as inputs, so that they can be used for shape inference and
        # potentially for z-scoring.
        check_estimator_arg(classifier)
        if isinstance(classifier, str):
            self._build_neural_net = utils.classifier_nn(model=classifier)
        else:
            self._build_neural_net = classifier

        # Ratio-based-specific summary_writer fields.
        self._summary.update({"mcmc_times": []})  # type: ignore

    def append_simulations(
        self, theta: Tensor, x: Tensor, from_round: int = 0,
    ) -> "RatioEstimator":
        r"""
        Store parameters and simulation outputs to use them for later training.

        Data are stored as entries in lists for each type of variable (parameter/data).

        Stores $\theta$, $x$, prior_masks (indicating if simulations are coming from the
        prior or not) and an index indicating which round the batch of simulations came
        from.

        Args:
            theta: Parameter sets.
            x: Simulation outputs.
            from_round: Which round the data stemmed from. Round 0 means from the prior.
                With default settings, this is not used at all for `SNRE`. Only when
                the user later on requests `.train(discard_prior_samples=True)`, we
                use these indices to find which training data stemmed from the prior.
        Returns:
            NeuralInference object (returned so that this function is chainable).
        """

        validate_theta_and_x(theta, x)

        self._theta_roundwise.append(theta)
        self._x_roundwise.append(x)
        self._prior_masks.append(mask_sims_from_prior(int(from_round), theta.size(0)))
        self._data_round_index.append(int(from_round))

        return self

    def train(
        self,
        num_atoms: int = 10,
        training_batch_size: int = 50,
        learning_rate: float = 5e-4,
        validation_fraction: float = 0.1,
        stop_after_epochs: int = 20,
        max_num_epochs: Optional[int] = None,
        clip_max_norm: Optional[float] = 5.0,
        exclude_invalid_x: bool = True,
        discard_prior_samples: bool = False,
        retrain_from_scratch_each_round: bool = False,
        show_train_summary: bool = False,
    ) -> RatioBasedPosterior:
        r"""
        Return classifier that approximates the ratio $p(\theta,x)/p(\theta)p(x)$.

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
            Classifier that approximates the ratio $p(\theta,x)/p(\theta)p(x)$.
        """

        max_num_epochs = 2 ** 31 - 1 if max_num_epochs is None else max_num_epochs

        # Load data from most recent round.
        self._round = max(self._data_round_index)
        theta, x, _ = self.get_simulations(self._round, exclude_invalid_x, False)

        # First round or if retraining from scratch:
        # Call the `self._build_neural_net` with the rounds' thetas and xs as
        # arguments, which will build the neural network
        # This is passed into NeuralPosterior, to create a neural posterior which
        # can `sample()` and `log_prob()`. The network is accessible via `.net`.
        if self._neural_net is None or retrain_from_scratch_each_round:
            self._neural_net = self._build_neural_net(theta, x)
            self._x_shape = x_shape_from_simulation(x)
            assert len(self._x_shape) < 3, (
                "For now, SNRE cannot handle multi-dimensional simulator output, see "
                "issue #360."
            )

        # Starting index for the training set (1 = discard round-0 samples).
        start_idx = int(discard_prior_samples and self._round > 0)
        theta, x, _ = self.get_simulations(start_idx, exclude_invalid_x)

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

        clipped_batch_size = min(training_batch_size, num_validation_examples)

        num_atoms = clamp_and_warn(
            "num_atoms", num_atoms, min_val=2, max_val=clipped_batch_size
        )

        # Dataset is shared for training and validation loaders.
        dataset = data.TensorDataset(theta, x)

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

        self._neural_net.to(self._device)
        optimizer = optim.Adam(list(self._neural_net.parameters()), lr=learning_rate,)

        epoch, self._val_log_prob = 0, float("-Inf")

        while epoch <= max_num_epochs and not self._converged(epoch, stop_after_epochs):

            # Train for a single epoch.
            self._neural_net.train()
            for batch in train_loader:
                optimizer.zero_grad()
                theta_batch, x_batch = (
                    batch[0].to(self._device),
                    batch[1].to(self._device),
                )
                loss = self._loss(theta_batch, x_batch, num_atoms)
                loss.backward()
                if clip_max_norm is not None:
                    clip_grad_norm_(
                        self._neural_net.parameters(), max_norm=clip_max_norm,
                    )
                optimizer.step()

            epoch += 1

            # Calculate validation performance.
            self._neural_net.eval()
            log_prob_sum = 0
            with torch.no_grad():
                for batch in val_loader:
                    theta_batch, x_batch = (
                        batch[0].to(self._device),
                        batch[1].to(self._device),
                    )
                    log_prob = self._loss(theta_batch, x_batch, num_atoms)
                    log_prob_sum -= log_prob.sum().item()
                self._val_log_prob = log_prob_sum / num_validation_examples
                # Log validation log prob for every epoch.
                self._summary["validation_log_probs"].append(self._val_log_prob)

            self._maybe_show_progress(self._show_progress_bars, epoch)

        self._report_convergence_at_end(epoch, stop_after_epochs, max_num_epochs)

        # Update summary.
        self._summary["epochs"].append(epoch)
        self._summary["best_validation_log_probs"].append(self._best_val_log_prob)

        # Update TensorBoard and summary dict.
        self._summarize(
            round_=self._round, x_o=None, theta_bank=theta, x_bank=x,
        )

        # Update description for progress bar.
        if show_train_summary:
            print(self._describe_round(self._round, self._summary))

        return deepcopy(self._neural_net)

    def build_posterior(
        self,
        density_estimator: Optional[TorchModule] = None,
        mcmc_method: str = "slice_np",
        mcmc_parameters: Optional[Dict[str, Any]] = None,
    ) -> RatioBasedPosterior:
        r"""
        Build posterior from the neural density estimator.

        SNRE trains a neural network to approximate likelihood ratios, which in turn
        can be used obtain an unnormalized posterior
        $p(\theta|x) \propto p(x|\theta) \cdot p(\theta)$. The posterior returned here
        wraps the trained network such that one can directly evaluate the unnormalized
        posterior log-probability $p(\theta|x) \propto p(x|\theta) \cdot p(\theta)$ and
        draw samples from the posterior with MCMC. Note that, in the case of
        single-round SNRE_A / AALR, it is possible to evaluate the log-probability of
        the **normalized** posterior, but sampling still requires MCMC.

        Args:
            density_estimator: The density estimator that the posterior is based on.
                If `None`, use the latest neural density estimator that was trained.
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

        Returns:
            Posterior $p(\theta|x)$  with `.sample()` and `.log_prob()` methods
            (the returned log-probability is unnormalized).
        """

        if density_estimator is None:
            density_estimator = self._neural_net
            # If internal net is used device is defined.
            device = self._device
        else:
            # Otherwise, infer it from the device of the net parameters.
            device = next(density_estimator.parameters()).device

        self._posterior = RatioBasedPosterior(
            method_family=self.__class__.__name__.lower(),
            neural_net=density_estimator,
            prior=self._prior,
            x_shape=self._x_shape,
            mcmc_method=mcmc_method,
            mcmc_parameters=mcmc_parameters,
            device=device,
        )

        self._posterior._num_trained_rounds = self._round + 1

        # Store models at end of each round.
        self._model_bank.append(deepcopy(self._posterior))
        self._model_bank[-1].net.eval()

        return deepcopy(self._posterior)

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

        return self._neural_net(theta_and_x)

    @abstractmethod
    def _loss(self, theta: Tensor, x: Tensor, num_atoms: int) -> Tensor:
        raise NotImplementedError
