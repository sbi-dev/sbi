# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Affero General Public License v3, see <https://www.gnu.org/licenses/>.

import warnings
from copy import deepcopy
from functools import partial
from typing import Any, Callable, Dict, Optional, Union

import torch
from pyknos.mdn.mdn import MultivariateGaussianMDN
from torch import Tensor
from torch.distributions import MultivariateNormal

import sbi.utils as utils
from sbi.inference.posteriors.direct_posterior import DirectPosterior
from sbi.inference.snpe.snpe_base import PosteriorEstimator
from sbi.types import TensorboardSummaryWriter, TorchModule


class SNPE_A(PosteriorEstimator):
    def __init__(
        self,
        prior: Optional[Any] = None,
        density_estimator: Union[str, Callable] = "mdn_snpe_a",
        num_components: int = 10,
        num_rounds: int = 1,
        device: str = "cpu",
        logging_level: Union[int, str] = "WARNING",
        summary_writer: Optional[TensorboardSummaryWriter] = None,
        show_progress_bars: bool = True,
        **unused_args,
    ):
        r"""SNPE-A [1].

        [1] _Fast epsilon-free Inference of Simulation Models with Bayesian Conditional
            Density Estimation_, Papamakarios et al., NeurIPS 2016,
            https://arxiv.org/abs/1605.06376.

        Args:
            prior: A probability distribution that expresses prior knowledge about the
                parameters, e.g. which ranges are meaningful for them. Any
                object with `.log_prob()`and `.sample()` (for example, a PyTorch
                distribution) can be used.
            density_estimator: If it is a string (only "mdn_snpe_a" is valid), use a
                pre-configured mixture of densities network. Alternatively, a function
                that builds a custom neural network can be provided. The function will
                be called with the first batch of simulations (theta, x), which can
                thus be used for shape inference and potentially for z-scoring. It
                needs to return a PyTorch `nn.Module` implementing the density
                estimator. The density estimator needs to provide the methods
                `.log_prob` and `.sample()`.

            num_components:
                Number of components of the mixture of Gaussians. This number is set to
                1 before running Algorithm 1, and then later set to the specified value
                before running Algorithm 2.
            num_rounds: Total number of training rounds. For all but the last round, Algorithm 1
                from [1] is executed. For last round, Algorithm 2 from [1] is executed once.
                By default, `num_rounds` is set to 1, i.e. only Algorithm 2 is executed once
                without training the proposal prior using Algorithm 1.
            device: torch device on which to compute, e.g. gpu, cpu.
            logging_level: Minimum severity of messages to log. One of the strings
                INFO, WARNING, DEBUG, ERROR and CRITICAL.
            summary_writer: A tensorboard `SummaryWriter` to control, among others, log
                file location (default is `<current working directory>/logs`.)
            show_progress_bars: Whether to show a progressbar during training.
            unused_args: Absorbs additional arguments. No entries will be used. If it
                is not empty, we warn. In future versions, when the new interface of
                0.14.0 is more mature, we will remove this argument.
        """

        # Catch invalid inputs.
        if not isinstance(prior, (utils.BoxUniform, MultivariateNormal)):
            raise TypeError(
                "Only priors of type BoxUniform and MultivariateNormal "
                "are supported for SNPE_A!"
            )
        if not ((density_estimator == "mdn_snpe_a") or callable(density_estimator)):
            raise TypeError(
                "The `density_estimator` passed to SNPE_A needs to be a "
                "callable or the string 'mdn_snpe_a'!"
            )

        self._num_rounds = num_rounds
        self._num_components = num_components

        # WARNING: sneaky trick ahead. We proxy the parent's `train` here,
        # requiring the signature to have `num_atoms`, save it for use below, and
        # continue. It's sneaky because we are using the object (self) as a namespace
        # to pass arguments between functions, and that's implicit state management.
        kwargs = utils.del_entries(
            locals(),
            entries=(
                "self",
                "__class__",
                "unused_args",
                "num_rounds",
                "num_components",
            ),
        )
        super().__init__(**kwargs)

    def train(
        self,
        training_batch_size: int = 50,
        learning_rate: float = 5e-4,
        validation_fraction: float = 0.1,
        stop_after_epochs: int = 20,
        max_num_epochs: Optional[int] = None,
        clip_max_norm: Optional[float] = 5.0,
        calibration_kernel: Optional[Callable] = None,
        exclude_invalid_x: bool = True,
        resume_training: bool = False,
        retrain_from_scratch_each_round: bool = False,
        show_train_summary: bool = False,
        dataloader_kwargs: Optional[Dict] = None,
    ) -> DirectPosterior:
        r"""
        Return density estimator that approximates the distribution $p(\theta|x)$.

        Args:
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
            resume_training: Can be used in case training time is limited, e.g. on a
                cluster. If `True`, the split between train and validation set, the
                optimizer, the number of epochs, and the best validation log-prob will
                be restored from the last time `.train()` was called.
            retrain_from_scratch_each_round: Whether to retrain the conditional density
                estimator for the posterior from scratch each round.
            show_train_summary: Whether to print the number of epochs and validation
                loss and leakage after the training.
            dataloader_kwargs: Additional or updated kwargs to be passed to the training
                and validation dataloaders (like, e.g., a collate_fn)
        Returns:
            Density estimator that approximates the distribution $p(\theta|x)$.
        """
        kwargs = utils.del_entries(locals(), entries=("self", "__class__"))

        # SNPE-A always discards the prior samples.
        kwargs["discard_prior_samples"] = True

        self._round = max(self._data_round_index)

        # In case there is will only be one round, train with Algorithm 2 from [1].
        if self._num_rounds == 1:
            self._build_neural_net = partial(
                self._build_neural_net, num_components=self._num_components
            )

        # Run Algorithm 1 from [1].
        elif self._round + 1 < self._num_rounds:
            # Wrap the function that builds the MDN such that we can make
            # sure that there is only one component when running.
            self._build_neural_net = partial(self._build_neural_net, num_components=1)

        # Run Algorithm 2 from [1].
        elif self._round + 1 == self._num_rounds:
            # Now switch to the specified number of components.
            self._build_neural_net = partial(
                self._build_neural_net, num_components=self._num_components
            )

            # Extend the MDN to the originally desired number of components.
            self._expand_mog()

        else:
            warnings.warn(
                f"Running SNPE-A for more than the specified number of rounds {self._num_rounds} implies running"
                f"Algorithm 2 from [1] multiple times, which can lead to numerical issues. Moreover, the number of "
                f"components in the mixture of Gaussian increases with every round after {self._num_rounds}.",
                UserWarning,
            )

        return super().train(**kwargs)

    def build_posterior(
        self,
        proposal: Optional[
            Union[MultivariateNormal, utils.BoxUniform, DirectPosterior]
        ] = None,
        density_estimator: Optional[TorchModule] = None,
        rejection_sampling_parameters: Optional[Dict[str, Any]] = None,
        sample_with_mcmc: bool = False,
        mcmc_method: str = "slice_np",
        mcmc_parameters: Optional[Dict[str, Any]] = None,
    ) -> DirectPosterior:
        r"""
        Build posterior from the neural density estimator.

        For SNPE, the posterior distribution that is returned here implements the
        following functionality over the raw neural density estimator:

        - correct the calculation of the log probability such that it compensates for
            the leakage.
        - reject samples that lie outside of the prior bounds.
        - alternatively, if leakage is very high (which can happen for multi-round
            SNPE), sample from the posterior with MCMC.

        Args:
            proposal: The proposal prior distribution of the previous round.
                As the density estimator approximates the proposal posterior,
                the proposal prior is used for importance reweighting.
                This allows sampling from the desired posterior during evaluation.
            density_estimator: The density estimator that the posterior is based on.
                If `None`, use the latest neural density estimator that was trained.
            rejection_sampling_parameters: Dictionary overriding the default parameters
                for rejection sampling. The following parameters are supported:
                `max_sampling_batch_size` to set the batch size for drawing new
                samples from the candidate distribution, e.g., the posterior. Larger
                batch size speeds up sampling.
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

        Returns:
            Posterior $p(\theta|x)$  with `.sample()` and `.log_prob()` methods.
        """

        if density_estimator is None:
            density_estimator = deepcopy(
                self._neural_net
            )  # PosteriorEstimator.train() also returns a deepcopy, mimic this here
            # If internal net is used device is defined.
            device = self._device
        else:
            # Otherwise, infer it from the device of the net parameters.
            device = next(density_estimator.parameters()).device

        # Set proposal of the density estimator.
        # This also evokes the z-scoring correction is necessary.
        if proposal is None:
            if self._model_bank:
                density_estimator.set_proposal(self._model_bank[-1].net)
            else:
                density_estimator.set_proposal(self._prior)
        elif isinstance(proposal, (MultivariateNormal, utils.BoxUniform)):
            density_estimator.set_proposal(proposal)
        elif isinstance(proposal, DirectPosterior):
            # Extract the MoGFlow_SNPE_A from the DirectPosterior.
            density_estimator.set_proposal(proposal.net)
        else:
            raise TypeError(
                "So far, only MultivariateNormal, BoxUniform, and DirectPosterior are"
                "supported for the `proposal` arg in SNPE_A.build_posterior()."
            )

        self._posterior = DirectPosterior(
            method_family="snpe",
            neural_net=density_estimator,
            prior=self._prior,
            x_shape=self._x_shape,
            rejection_sampling_parameters=rejection_sampling_parameters,
            sample_with_mcmc=sample_with_mcmc,
            mcmc_method=mcmc_method,
            mcmc_parameters=mcmc_parameters,
            device=device,
        )

        self._posterior._num_trained_rounds = self._round + 1

        # Store models at end of each round.
        self._model_bank.append(deepcopy(self._posterior))
        self._model_bank[-1].net.eval()

        return deepcopy(self._posterior)

    def _log_prob_proposal_posterior(
        self, theta: Tensor, x: Tensor, masks: Tensor, proposal: Optional[Any]
    ) -> Tensor:
        """
        Return the log-probability of the proposal posterior.

        For SNPE-A this is the same as `self._neural_net.log_prob(theta, x)` in
        `_loss()` to be found in `snpe_base.py`.

        Args:
            theta: Batch of parameters θ.
            x: Batch of data.
            masks: Mask that is True for prior samples in the batch in order to train
                them with prior loss.
            proposal: Proposal distribution.

        Returns: Log-probability of the proposal posterior.
        """
        return self._neural_net.log_prob(theta, x)

    def _expand_mog(self, eps: float = 1e-5):
        """
        Replicate a singe Gaussian trained with Algorithm 1 before continuing
        with Algorithm 2. The weights and biases of the associated MDN layers
        are repeated `num_components` times, slightly perturbed to break the
        symmetry such that the gradients in the subsequent training are not
        all identical.

        Args:
            eps: Standard deviation for the random perturbation.
        """
        assert isinstance(self._neural_net._distribution, MultivariateGaussianMDN)

        # Increase the number of components
        self._neural_net._distribution._num_components = self._num_components

        # Expand the 1-dim Gaussian.
        for name, param in self._neural_net.named_parameters():
            if any(
                key in name for key in ["logits", "means", "unconstrained", "upper"]
            ):
                if "bias" in name:
                    param.data = param.data.repeat(self._num_components)
                    param.data.add_(torch.randn_like(param.data) * eps)
                    param.grad = None  # let autograd construct a new gradient
                elif "weight" in name:
                    param.data = param.data.repeat(self._num_components, 1)
                    param.data.add_(torch.randn_like(param.data) * eps)
                    param.grad = None  # let autograd construct a new gradient
