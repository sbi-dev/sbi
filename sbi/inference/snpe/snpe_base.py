# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Affero General Public License v3, see <https://www.gnu.org/licenses/>.


from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Any, Callable, Dict, Optional, Union, cast
from warnings import warn

import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import EarlyStopping, progress
from torch import Tensor, nn, ones, optim
from torch.nn.utils import clip_grad_norm_
from torch.utils import data
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.tensorboard import SummaryWriter

from sbi import utils as utils
from sbi.inference import NeuralInference
from sbi.inference.posteriors.direct_posterior import DirectPosterior
from sbi.types import ScalarFloat
from sbi.utils import check_estimator_arg, x_shape_from_simulation


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
        device: str = "cpu",
        logging_level: Union[int, str] = "WARNING",
        summary_writer: Optional[SummaryWriter] = None,
        show_progress_bars: bool = True,
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
        self._model = None
        self._sample_with_mcmc = sample_with_mcmc
        self._mcmc_method = mcmc_method
        self._mcmc_parameters = mcmc_parameters

        self._model_bank = []
        self.use_non_atomic_loss = False

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
        retrain_from_scratch: bool = False,
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
            retrain_from_scratch: Whether to retrain the conditional density
                estimator for the posterior from scratch each round.

        Returns:
            Posterior $p(\theta|x)$ that can be sampled and evaluated.
        """

        self._warn_if_retrain_from_scratch_snpe(retrain_from_scratch)

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
        if self._model is None or retrain_from_scratch:
            neural_net = self._build_neural_net(theta, x)
        else:
            neural_net = self._model.net

        # Re-initializing the `PosteriorEstimationNet` ensures that the optimizers are
        # reset every round. I think that this makes sense since the loss changes. Also,
        # it is required to set the proposal.
        self._model = PosteriorEstimationNet(
            neural_net,
            prior=self._prior,
            proposal=proposal,
            loss=self._loss,
            lr=learning_rate,
            calibration_kernel=calibration_kernel,
        )

        # Fit posterior using newly aggregated data set.
        self._train(
            training_batch_size=training_batch_size,
            validation_fraction=validation_fraction,
            stop_after_epochs=stop_after_epochs,
            max_num_epochs=cast(int, max_num_epochs),
            clip_max_norm=clip_max_norm,
            exclude_invalid_x=exclude_invalid_x,
            discard_prior_samples=discard_prior_samples,
        )

        x_shape = x_shape_from_simulation(x)

        self._posterior = DirectPosterior(
            method_family="snpe",
            neural_net=self._model.net,
            prior=self._prior,
            x_shape=x_shape,
            sample_with_mcmc=self._sample_with_mcmc,
            mcmc_method=self._mcmc_method,
            mcmc_parameters=self._mcmc_parameters,
            get_potential_function=PotentialFunctionProvider(),
        )

        # Store models at end of each round.
        self._model_bank.append(deepcopy(self._posterior))
        self._model_bank[-1].net.eval()

        # Update tensorboard and summary dict.
        self._summarize(
            round_=self._round, theta_bank=theta,
        )

        self._posterior._num_trained_rounds = self._round + 1
        return deepcopy(self._posterior)

    @abstractmethod
    def _log_prob_proposal_posterior(
        self, theta: Tensor, x: Tensor, masks: Tensor, proposal: Optional[Any]
    ) -> Tensor:
        raise NotImplementedError

    def _train(
        self,
        training_batch_size: int,
        validation_fraction: float,
        stop_after_epochs: int,
        max_num_epochs: int,
        clip_max_norm: Optional[float],
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
            num_workers=4,  # todo should not be set hard.
        )
        val_loader = data.DataLoader(
            dataset,
            batch_size=min(training_batch_size, num_validation_examples),
            shuffle=False,
            drop_last=True,
            sampler=SubsetRandomSampler(val_indices),
            num_workers=4,
        )

        trainer = pl.Trainer(
            logger=self._summary_writer,
            early_stop_callback=EarlyStopping(patience=stop_after_epochs,),
            gradient_clip_val=clip_max_norm,
            max_epochs=max_num_epochs,
            progress_bar_refresh_rate=self._show_progress_bars,
        )
        trainer.fit(self._model, train_loader, val_loader)

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
            log_prob = self._model.net.log_prob(theta, x)
        else:
            log_prob = self._log_prob_proposal_posterior(theta, x, masks, proposal)

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


class PosteriorEstimationNet(pl.LightningModule):
    """
    This is a wrapper class for the neural network defined by pyknos / nflows. It wraps
    the neural network into a pytorch_lightning module.
    """

    def __init__(self, net, prior, proposal, loss, lr, calibration_kernel):
        super().__init__()
        self.net = net
        self._prior = prior
        self.proposal = proposal
        self.loss = loss
        self.lr = lr
        self.calibration_kernel = calibration_kernel

    def configure_optimizers(self):
        optimizer = optim.Adam(list(self.net.parameters()), lr=self.lr)
        return optimizer

    def training_step(self, batch, batch_idx):
        theta, x, masks = batch
        loss = torch.mean(
            self.loss(theta, x, masks, self.proposal, self.calibration_kernel)
        )
        result = pl.TrainResult(loss)
        return result

    def validation_step(self, batch, batch_idx):
        theta, x, masks = batch
        loss = torch.mean(
            self.loss(theta, x, masks, self.proposal, self.calibration_kernel)
        )
        result = pl.EvalResult(checkpoint_on=loss, early_stop_on=loss)
        result.log("val_loss", loss)
        return result


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
