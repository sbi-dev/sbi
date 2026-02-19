# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

from typing import Literal, Optional, Union

import torch
from torch import Tensor, ones
from torch.distributions import Distribution
from torch.utils.tensorboard.writer import SummaryWriter
from typing_extensions import Self

from sbi import utils as utils
from sbi.inference.posteriors import DirectPosterior
from sbi.inference.posteriors.direct_posterior import DirectPosterior
from sbi.inference.trainers._contracts import StartIndexContext
from sbi.inference.trainers.npe.npe_base import PosteriorEstimatorTrainer
from sbi.neural_nets.estimators.base import (
    ConditionalDensityEstimator,
    ConditionalEstimatorBuilder,
)
from sbi.sbi_types import Tracker
from sbi.utils import (
    del_entries,
    handle_invalid_x,
    npe_msg_on_invalid_x,
    validate_theta_and_x,
    warn_if_invalid_for_zscoring,
)
from sbi.utils.sbiutils import ImproperEmpirical, mask_sims_from_prior


class NPE_PFN(PosteriorEstimatorTrainer):
    r"""Neural Posterior Estimation algorithm (NPE-C) as in Greenberg et al. (2019) [1].

    NPE-C (also known as APT - Automatic Posterior Transformation, aka SNPE-C) trains
    a neural network over multiple rounds to directly approximate the posterior for a
    specific observation x_o. In the first round, NPE-C is equivalent to other NPE
    methods and is fully amortized (direct inference for any new observation). After
    the first round, NPE-C automatically selects between two loss variants depending
    on the chosen density estimator: the non-atomic loss (for Mixture of Gaussians)
    which is stable and avoids leakage, or the atomic loss (for flows) which is more
    flexible but may suffer from leakage issues.

    For single-round inference, NPE-A, NPE-B, and NPE-C are equivalent and use
    plain NLL loss.

    [1] *Automatic Posterior Transformation for Likelihood-free Inference*,
        Greenberg et al., ICML 2019, https://arxiv.org/abs/1905.07488.

    Example:
    --------

    ::

        import torch
        from sbi.inference import NPE_C
        from sbi.utils import BoxUniform

        # 1. Setup simulator, prior, and observation
        prior = BoxUniform(low=torch.zeros(3), high=torch.ones(3))
        x_o = torch.randn(1, 3)  # Observed data

        def simulator(theta):
            return theta + torch.randn_like(theta) * 0.1

        # 2. Multi-round inference
        inference = NPE_C(prior=prior)
        proposal = prior

        for round_idx in range(5):
            theta = proposal.sample((100,))
            x = simulator(theta)
            density_estimator = inference.append_simulations(theta, x).train()
            posterior = inference.build_posterior(density_estimator)
            proposal = posterior.set_default_x(x_o)

        # 3. Sample from final posterior
        samples = posterior.sample((1000,), x=x_o)
    """

    def __init__(
        self,
        prior: Optional[Distribution] = None,
        density_estimator: Union[
            Literal["tabpfn"],
            ConditionalEstimatorBuilder[ConditionalDensityEstimator],
        ] = "maf",
        device: str = "cpu",
        logging_level: Union[int, str] = "WARNING",
        summary_writer: Optional[SummaryWriter] = None,
        tracker: Optional[Tracker] = None,
        show_progress_bars: bool = True,
    ):
        r"""Initialize NPE-C.

        Args:
            prior: A probability distribution that expresses prior knowledge about the
                parameters, e.g. which ranges are meaningful for them.
            density_estimator: If it is a string, use a pre-configured network of the
                provided type (one of nsf, maf, mdn, made). Alternatively, a function
                that builds a custom neural network, which adheres to
                `ConditionalEstimatorBuilder` protocol can be provided. The function
                will be called with the first batch of simulations (theta, x), which can
                thus be used for shape inference and potentially for z-scoring. The
                density estimator needs to provide the methods `.log_prob` and
                `.sample()` and must return a `ConditionalDensityEstimator`.
            device: Training device, e.g., "cpu", "cuda" or "cuda:{0, 1, ...}".
            logging_level: Minimum severity of messages to log. One of the strings
                INFO, WARNING, DEBUG, ERROR and CRITICAL.
            summary_writer: Deprecated alias for the TensorBoard summary writer.
                Use ``tracker`` instead.
            tracker: Tracking adapter used to log training metrics. If None, a
                TensorBoard tracker is used with a default log directory.
            show_progress_bars: Whether to show a progressbar during training.
        """

        kwargs = del_entries(locals(), entries=("self", "__class__"))
        super().__init__(**kwargs)

    def append_simulations(
        self,
        theta: Tensor,
        x: Tensor,
        proposal: Optional[DirectPosterior] = None,
        exclude_invalid_x: Optional[bool] = None,
        data_device: Optional[str] = None,
    ) -> Self:
        r"""Store parameters and simulation outputs to use them for later training.

        Data are stored as entries in lists for each type of variable (parameter/data).

        Stores $\theta$, $x$, prior_masks (indicating if simulations are coming from the
        prior or not) and an index indicating which round the batch of simulations came
        from.

        Args:
            theta: Parameter sets.
            x: Simulation outputs.
            proposal: The distribution that the parameters $\theta$ were sampled from.
                Pass `None` if the parameters were sampled from the prior. Multi-round
                training is not yet implemented, so anything other than `None` will
                raise an error.
            exclude_invalid_x: Whether invalid simulations are discarded during
                training. For single-round training, it is fine to discard invalid
                simulations, but for multi-round sequential (atomic) training,
                discarding invalid simulations gives systematically wrong results. If
                `None`, it will be `True` in the first round and `False` in later
                rounds. Note that multi-round training is not yet implemented.
            data_device: Where to store the data, default is on the same device where
                the training is happening. If training a large dataset on a GPU with not
                much VRAM can set to 'cpu' to store data on system memory instead.

        Returns:
            VectorFieldTrainer object (returned so that this function is chainable).
        """
        inference_name = self.__class__.__name__
        assert proposal is None, (
            f"Multi-round {inference_name} is not yet implemented. "
            f"Please use single-round {inference_name}."
        )
        current_round = 0

        if exclude_invalid_x is None:
            exclude_invalid_x = current_round == 0

        if data_device is None:
            data_device = self._device

        theta, x = validate_theta_and_x(
            theta, x, data_device=data_device, training_device=self._device
        )

        is_valid_x, num_nans, num_infs = handle_invalid_x(
            x, exclude_invalid_x=exclude_invalid_x
        )

        x = x[is_valid_x]
        theta = theta[is_valid_x]

        # Check for problematic z-scoring
        warn_if_invalid_for_zscoring(x)

        npe_msg_on_invalid_x(
            num_nans,
            num_infs,
            exclude_invalid_x,
            algorithm=f"Single-round {inference_name}",
        )

        self._data_round_index.append(current_round)
        prior_masks = mask_sims_from_prior(int(current_round > 0), theta.size(0))

        self._theta_roundwise.append(theta)
        self._x_roundwise.append(x)
        self._prior_masks.append(prior_masks)

        self._proposal_roundwise.append(proposal)

        if self._prior is None or isinstance(self._prior, ImproperEmpirical):
            theta_prior = self.get_simulations()[0].to(self._device)
            self._prior = ImproperEmpirical(
                theta_prior, ones(theta_prior.shape[0], device=self._device)
            )

        return self

    def train(
        self,
        retrain_from_scratch: bool = False,
        discard_prior_samples: bool = False,
        force_first_round_loss: bool = False,
    ) -> ConditionalDensityEstimator:
        r"""Return density estimator that approximates the distribution $p(\theta|x)$.

        Args:
            num_atoms: Number of atoms to use for classification.
            training_batch_size: Training batch size.
            learning_rate: Learning rate for Adam optimizer.
            validation_fraction: The fraction of data to use for validation.
            stop_after_epochs: The number of epochs to wait for improvement on the
                validation set before terminating training.
            max_num_epochs: Maximum number of epochs to run. If reached, we stop
                training even when the validation loss is still decreasing. Otherwise,
                we train until validation loss increases (see also
                ``stop_after_epochs``).
            clip_max_norm: Value at which to clip the total gradient norm in order to
                prevent exploding gradients. Use None for no clipping.
            calibration_kernel: A function to calibrate the loss with respect to the
                simulations ``x``. See Lueckmann, Gonçalves et al., NeurIPS 2017.
            resume_training: Can be used in case training time is limited, e.g. on a
                cluster. If ``True``, the split between train and validation set, the
                optimizer, the number of epochs, and the best validation log-prob will
                be restored from the last time ``.train()`` was called.
            force_first_round_loss: If ``True``, train with maximum likelihood,
                i.e., potentially ignoring the correction for using a proposal
                distribution different from the prior.
            discard_prior_samples: Whether to discard samples simulated in round 1, i.e.
                from the prior. Training may be sped up by ignoring such less targeted
                samples.
            use_combined_loss: Whether to train the neural net also on prior samples
                using maximum likelihood in addition to training it on all samples using
                atomic loss. The extra MLE loss helps prevent density leaking with
                bounded priors.
            retrain_from_scratch: Whether to retrain the conditional density
                estimator for the posterior from scratch each round.
            show_train_summary: Whether to print the number of epochs and validation
                loss and leakage after the training.
            dataloader_kwargs: Additional or updated kwargs to be passed to the training
                and validation dataloaders (like, e.g., a collate_fn)

        Returns:
            Density estimator that approximates the distribution $p(\theta|x)$.
        """

        if len(self._data_round_index) == 0:
            raise RuntimeError(
                "No simulations found. You must call .append_simulations() "
                "before calling .train()."
            )

        start_idx = self._get_start_index(
            context=StartIndexContext(
                discard_prior_samples=discard_prior_samples,
                force_first_round_loss=force_first_round_loss,
            )
        )

        # TODO mock these for now
        theta, x, masks = self.get_simulations(0)
        self.train_indices = torch.arange(theta.shape[0])

        # TODO this function is a bit questionable in the context of NPE-PFN
        # The tabpfn builder now already adds the batch used to infer dimensionality to get a working estimator.
        # In initizlize_neural_network, actually all train indices are used, so its the whole dataset.
        # This will however be called only if not yet initizliaed or if retrain from sratch.
        # So its important so set context here (and also more intuitive)
        # Another alternative would be to just set retrain from scracth to True.
        # Some decisions need to be made here
        self._initialize_neural_network(retrain_from_scratch, start_idx)
        print("HALLO")
        self._neural_net.set_context(theta, x)

        return self._neural_net

    def _log_prob_proposal_posterior(
        self,
        theta: Tensor,
        x: Tensor,
        masks: Tensor,
        proposal: DirectPosterior,
    ) -> Tensor:
        """Return the log-probability of the proposal posterior.

        If the proposal is a MoG, the density estimator is a MoG, and the prior is
        either Gaussian or uniform, we use non-atomic loss. Else, use atomic loss (which
        suffers from leakage).

        Args:
            theta: Batch of parameters θ.
            x: Batch of data.
            masks: Mask that is True for prior samples in the batch in order to train
                them with prior loss.
            proposal: Proposal distribution.

        Returns: Log-probability of the proposal posterior.
        """

        # TODO is required by PosteriorEstimatorTrainer

        raise NotImplementedError
