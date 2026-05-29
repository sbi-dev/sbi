# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

from typing import Callable, Dict, Literal, Optional, Union

import torch
from torch.distributions import Distribution, MultivariateNormal, Uniform
from torch.utils.tensorboard.writer import SummaryWriter

from sbi.inference.posteriors.direct_posterior import DirectPosterior
from sbi.inference.trainers.npe.npe_base import (
    PosteriorEstimatorTrainer,
)
from sbi.inference.trainers.npe.npe_loss import (
    AtomicLoss,
    NPELossStrategy,
    NonAtomicGaussianLoss,
)
from sbi.neural_nets.estimators.base import (
    ConditionalDensityEstimator,
    ConditionalEstimatorBuilder,
)
from sbi.neural_nets.estimators.mixture_density_estimator import (
    MixtureDensityEstimator,
)
from sbi.sbi_types import Tracker
from sbi.utils import (
    check_dist_class,
    del_entries,
)
from sbi.utils.torchutils import BoxUniform


class NPE_C(PosteriorEstimatorTrainer):
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
            Literal["nsf", "maf", "mdn", "made"],
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

    def train(
        self,
        num_atoms: int = 10,
        training_batch_size: int = 200,
        learning_rate: float = 5e-4,
        validation_fraction: float = 0.1,
        stop_after_epochs: int = 20,
        max_num_epochs: int = 2**31 - 1,
        clip_max_norm: Optional[float] = 5.0,
        calibration_kernel: Optional[Callable] = None,
        resume_training: bool = False,
        force_first_round_loss: bool = False,
        discard_prior_samples: bool = False,
        use_combined_loss: bool = False,
        retrain_from_scratch: bool = False,
        show_train_summary: bool = False,
        dataloader_kwargs: Optional[Dict] = None,
        loss_strategy: Optional[NPELossStrategy] = None,
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

        kwargs = del_entries(
            locals(),
            entries=("self", "__class__", "num_atoms", "use_combined_loss"),
        )

        self._round = max(self._data_round_index)

        if loss_strategy is not None:
            self._loss_strategy = loss_strategy
        elif self._round > 0:
            # Set the proposal to the last proposal that was passed by the user.
            proposal = self._proposal_roundwise[-1]
            use_non_atomic_loss = (
                isinstance(proposal, DirectPosterior)
                and isinstance(proposal.posterior_estimator, MixtureDensityEstimator)
                and isinstance(self._neural_net, MixtureDensityEstimator)
                and check_dist_class(
                    self._prior, class_to_check=(Uniform, MultivariateNormal)
                )[0]
            )

            algorithm = "non-atomic" if use_non_atomic_loss else "atomic"
            print(f"Using SNPE-C with {algorithm} loss")

            if use_non_atomic_loss:
                # Take care of z-scoring, pre-compute and store prior terms.
                self._set_state_for_mog_proposal()

                # Instantiate Non-Atomic Strategy
                if isinstance(self._maybe_z_scored_prior, MultivariateNormal):
                    prec_m_prod_prior = torch.mv(
                        self._maybe_z_scored_prior.precision_matrix,
                        self._maybe_z_scored_prior.loc,
                    )
                else:
                    prec_m_prod_prior = None

                self._loss_strategy = NonAtomicGaussianLoss(
                    neural_net=self._neural_net,
                    maybe_z_scored_prior=self._maybe_z_scored_prior,
                    prec_m_prod_prior=prec_m_prod_prior,
                    z_score_theta=self.z_score_theta,
                )
            else:
                # Instantiate Atomic Strategy
                self._loss_strategy = AtomicLoss(
                    neural_net=self._neural_net,
                    prior=self._prior,
                    num_atoms=num_atoms,
                    use_combined_loss=use_combined_loss,
                )
        else:
            # Default to None for first round (equivalent to MLE)
            self._loss_strategy = None

        return super().train(**kwargs)

    def _set_state_for_mog_proposal(self) -> None:
        """Set state variables that are used at each training step of non-atomic SNPE-C.

        Three things are computed:
        1) Check if z-scoring was requested.
        2) Define a (potentially standardized) prior.
        3) Compute (Precision * mean) for the prior.
        """
        assert isinstance(self._neural_net, MixtureDensityEstimator)
        self.z_score_theta = self._neural_net.has_input_transform

        self._set_maybe_z_scored_prior()

    def _set_maybe_z_scored_prior(self) -> None:
        r"""Compute and store potentially standardized prior (if z-scoring was done)."""

        if self.z_score_theta:
            # Get z-score parameters from the MixtureDensityEstimator
            assert isinstance(self._neural_net, MixtureDensityEstimator)
            shift = self._neural_net._transform_shift
            scale = self._neural_net._transform_scale

            estim_prior_mean = shift
            estim_prior_std = scale

            # Compute the discrepancy of the true prior mean and std and the mean and
            # std that was empirically estimated from samples.
            almost_zero_mean = (self._prior.mean - estim_prior_mean) / estim_prior_std
            almost_one_std = torch.sqrt(self._prior.variance) / estim_prior_std

            if isinstance(self._prior, MultivariateNormal):
                self._maybe_z_scored_prior = MultivariateNormal(
                    almost_zero_mean, torch.diag(almost_one_std)
                )
            else:
                range_ = torch.sqrt(almost_one_std * 3.0)
                self._maybe_z_scored_prior = BoxUniform(
                    almost_zero_mean - range_, almost_zero_mean + range_
                )
        else:
            self._maybe_z_scored_prior = self._prior


