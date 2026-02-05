# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

from typing import Callable, Dict, Literal, Optional, Union

import torch
from pyknos.mdn.mdn import MultivariateGaussianMDN as mdn
from pyknos.nflows.transforms import CompositeTransform
from torch import Tensor
from torch.distributions import Distribution, MultivariateNormal, Uniform

from sbi.inference.posteriors.direct_posterior import DirectPosterior
from sbi.inference.trainers.npe.npe_base import (
    PosteriorEstimatorTrainer,
)
from sbi.neural_nets.estimators.base import (
    ConditionalDensityEstimator,
    ConditionalEstimatorBuilder,
)
from sbi.sbi_types import TensorBoardSummaryWriter
from sbi.utils import (
    check_dist_class,
    del_entries,
)
from sbi.utils.torchutils import BoxUniform


class NPE_C(PosteriorEstimatorTrainer):
    """Neural Posterior Estimation algorithm (NPE-C) as in Greenberg et al. (2019). [1]

    [1] *Automatic Posterior Transformation for Likelihood-free Inference*,
        Greenberg et al., ICML 2019, https://arxiv.org/abs/1905.07488.

    Like all NPE methods, this method trains a deep neural density estimator to
    directly approximate the posterior. Also like all other NPE methods, in the
    first round, this density estimator is trained with a maximum-likelihood loss.

    For the sequential mode in which the density estimator is trained across rounds,
    this class implements two loss variants of NPE-C: the non-atomic and the atomic
    version. The atomic loss of NPE-C can be used for any density estimator,
    i.e. also for normalizing flows. However, it suffers from leakage issues. On
    the other hand, the non-atomic loss can only be used only if the proposal
    distribution is a mixture of Gaussians, the density estimator is a mixture of
    Gaussians, and the prior is either Gaussian or Uniform. It does not suffer from
    leakage issues. At the beginning of each round, we print whether the non-atomic
    or the atomic version is used.

    In this codebase, we will automatically switch to the non-atomic loss if the
    following criteria are fulfilled:

    - proposal is a `DirectPosterior` with density_estimator `mdn`, as built with
      `sbi.neural_nets.posterior_nn()`.
    - the density estimator is a `mdn`, as built with
      `sbi.neural_nets.posterior_nn()`.
    - `isinstance(prior, MultivariateNormal)` (from `torch.distributions`) or
      ``isinstance(prior, sbi.utils.BoxUniform)``

    Note that custom implementations of any of these densities (or estimators) will
    not trigger the non-atomic loss, and the algorithm will fall back onto using
    the atomic loss.
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
        summary_writer: Optional[TensorBoardSummaryWriter] = None,
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
            summary_writer: A tensorboard ``SummaryWriter`` to control, among others,
                log file location (default is ``<current working directory>/logs``.)
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
        # Load the strategy classes locally to avoid circular imports if any
        from sbi.inference.trainers.npe.npe_c_loss import AtomicLoss, NonAtomicGaussianLoss

        if len(self._data_round_index) == 0:
            raise RuntimeError(
                "No simulations found. You must call .append_simulations() "
                "before calling .train()."
            )
        # WARNING: sneaky trick ahead. We proxy the parent's `train` here,
        # requiring the signature to have `num_atoms`, save it for use below, and
        # continue. It's sneaky because we are using the object (self) as a namespace
        # to pass arguments between functions, and that's implicit state management.
        self._num_atoms = num_atoms
        self._use_combined_loss = use_combined_loss
        kwargs = del_entries(
            locals(),
            entries=("self", "__class__", "num_atoms", "use_combined_loss"),
        )

        self._round = max(self._data_round_index)

        if self._round > 0:
            # Set the proposal to the last proposal that was passed by the user. For
            # atomic SNPE, it does not matter what the proposal is. For non-atomic
            # SNPE, we only use the latest data that was passed, i.e. the one from the
            # last proposal.
            proposal = self._proposal_roundwise[-1]
            self.use_non_atomic_loss = (
                isinstance(proposal, DirectPosterior)
                and isinstance(proposal.posterior_estimator.net._distribution, mdn)
                and isinstance(self._neural_net.net._distribution, mdn)
                and check_dist_class(
                    self._prior, class_to_check=(Uniform, MultivariateNormal)
                )[0]
            )

            algorithm = "non-atomic" if self.use_non_atomic_loss else "atomic"
            print(f"Using SNPE-C with {algorithm} loss")

            if self.use_non_atomic_loss:
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
                    z_score_theta=self.z_score_theta
                )
            else:
                # Instantiate Atomic Strategy
                self._loss_strategy = AtomicLoss(
                    neural_net=self._neural_net,
                    prior=self._prior,
                    num_atoms=self._num_atoms,
                    use_combined_loss=self._use_combined_loss
                )
        else:
             # Default to Atomic for first round or if not specified (though round > 0 check handles this)
             # Actually, in the first round (round 0), we don't have a proposal posterior loss usually, 
             # but if we forced it, we'd default to atomic.
             self._loss_strategy = AtomicLoss(
                neural_net=self._neural_net,
                prior=self._prior,
                num_atoms=self._num_atoms,
                use_combined_loss=self._use_combined_loss
            )

        return super().train(**kwargs)

    def _set_state_for_mog_proposal(self) -> None:
        """Set state variables that are used at each training step of non-atomic SNPE-C.

        Three things are computed:
        1) Check if z-scoring was requested. To do so, we check if the `_transform`
            argument of the net had been a `CompositeTransform`. See pyknos mdn.py.
        2) Define a (potentially standardized) prior. It's standardized if z-scoring
            had been requested.
        3) Compute (Precision * mean) for the prior. This quantity is used at every
            training step if the prior is Gaussian.
        """

        self.z_score_theta = isinstance(
            self._neural_net.net._transform, CompositeTransform
        )

        self._set_maybe_z_scored_prior()

        # NOTE: self.prec_m_prod_prior calculation moved to strategy instantiation

    def _set_maybe_z_scored_prior(self) -> None:
        r"""Compute and store potentially standardized prior (if z-scoring was done).

        The proposal posterior is:
        $pp(\theta|x) = 1/Z * q(\theta|x) * prop(\theta) / p(\theta)$

        Let's denote z-scored theta by `a`: a = (theta - mean) / std
        Then pp'(a|x) = 1/Z_2 * q'(a|x) * prop'(a) / p'(a)$

        The ' indicates that the evaluation occurs in standardized space. The constant
        scaling factor has been absorbed into Z_2.
        From the above equation, we see that we need to evaluate the prior **in
        standardized space**. We build the standardized prior in this function.

        The standardize transform that is applied to the samples theta does not use
        the exact prior mean and std (due to implementation issues). Hence, the z-scored
        prior will not be exactly have mean=0 and std=1.
        """

        if self.z_score_theta:
            scale = self._neural_net.net._transform._transforms[0]._scale
            shift = self._neural_net.net._transform._transforms[0]._shift

            # Following the definintion of the linear transform in
            # `standardizing_transform` in `sbiutils.py`:
            # shift=-mean / std
            # scale=1 / std
            # Solving these equations for mean and std:
            estim_prior_std = 1 / scale
            estim_prior_mean = -shift * estim_prior_std

            # Compute the discrepancy of the true prior mean and std and the mean and
            # std that was empirically estimated from samples.
            # N(theta|m,s) = N((theta-m_e)/s_e|(m-m_e)/s_e, s/s_e)
            # Above: m,s are true prior mean and std. m_e,s_e are estimated prior mean
            # and std (estimated from samples and used to build standardize transform).
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

    def _log_prob_proposal_posterior(
        self,
        theta: Tensor,
        x: Tensor,
        masks: Tensor,
        proposal: DirectPosterior,
    ) -> Tensor:
        """Return the log-probability of the proposal posterior.
        Delegates to the configured loss strategy.
        """
        # Ensure strategy is initialized (it handles checks internally if needed)
        if hasattr(self, "_loss_strategy"):
             return self._loss_strategy(theta, x, masks, proposal)
        
        # Fallback if somehow called without train setup (unlikely)
        raise RuntimeError("Loss strategy not initialized. Call train() first.")

