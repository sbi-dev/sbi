# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Affero General Public License v3, see <https://www.gnu.org/licenses/>.

import warnings
from copy import deepcopy
from functools import partial
from typing import Any, Callable, Dict, Optional, Union

import torch
import torch.nn as nn
from pyknos.mdn.mdn import MultivariateGaussianMDN
from pyknos.nflows import flows
from pyknos.nflows.transforms import CompositeTransform
from torch import Tensor
from torch.distributions import Distribution, MultivariateNormal

import sbi.utils as utils
from sbi.inference.posteriors.direct_posterior import DirectPosterior
from sbi.inference.snpe.snpe_base import PosteriorEstimator
from sbi.types import TensorboardSummaryWriter, TorchModule
from sbi.utils import torchutils


class SNPE_A(PosteriorEstimator):
    def __init__(
        self,
        prior: Optional[Distribution] = None,
        density_estimator: Union[str, Callable] = "mdn_snpe_a",
        num_components: int = 10,
        device: str = "cpu",
        logging_level: Union[int, str] = "WARNING",
        summary_writer: Optional[TensorboardSummaryWriter] = None,
        show_progress_bars: bool = True,
    ):
        r"""SNPE-A [1].

        [1] _Fast epsilon-free Inference of Simulation Models with Bayesian Conditional
            Density Estimation_, Papamakarios et al., NeurIPS 2016,
            https://arxiv.org/abs/1605.06376.

        This class implements SNPE-A. SNPE-A trains across multiple rounds with a
        maximum-likelihood-loss. This will make training converge to the proposal
        posterior instead of the true posterior. To correct for this, SNPE-A applies a
        post-hoc correction after training. This correction has to be performed
        analytically. Thus, SNPE-A is limited to Gaussian distributions for all but the
        last round. In the last round, SNPE-A can use a Mixture of Gaussians.

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
                `.log_prob` and `.sample()`. Note that until the last round only a
                single (multivariate) Gaussian component is used for training (see
                Algorithm 1 in [1]). In the last round, this component is replicated
                `num_components` times, its parameters are perturbed with a very small
                noise, and then the last training round is done with the expanded
                Gaussian mixture as estimator for the proposal posterior.
            num_components: Number of components of the mixture of Gaussians in the
                last round. This overrides the `num_components` value passed to
                `posterior_nn()`.
            device: Training device, e.g., "cpu", "cuda" or "cuda:{0, 1, ...}".
            logging_level: Minimum severity of messages to log. One of the strings
                INFO, WARNING, DEBUG, ERROR and CRITICAL.
            summary_writer: A tensorboard `SummaryWriter` to control, among others, log
                file location (default is `<current working directory>/logs`.)
            show_progress_bars: Whether to show a progressbar during training.
        """

        # Catch invalid inputs.
        if not ((density_estimator == "mdn_snpe_a") or callable(density_estimator)):
            raise TypeError(
                "The `density_estimator` passed to SNPE_A needs to be a "
                "callable or the string 'mdn_snpe_a'!"
            )

        # `num_components` will be used to replicate the Gaussian in the last round.
        self._num_components = num_components
        self._ran_final_round = False

        # WARNING: sneaky trick ahead. We proxy the parent's `train` here,
        # requiring the signature to have `num_atoms`, save it for use below, and
        # continue. It's sneaky because we are using the object (self) as a namespace
        # to pass arguments between functions, and that's implicit state management.
        kwargs = utils.del_entries(
            locals(),
            entries=("self", "__class__", "num_components"),
        )
        super().__init__(**kwargs)

    def train(
        self,
        final_round: bool = False,
        training_batch_size: int = 50,
        learning_rate: float = 5e-4,
        validation_fraction: float = 0.1,
        stop_after_epochs: int = 20,
        max_num_epochs: int = 2**31 - 1,
        clip_max_norm: Optional[float] = 5.0,
        calibration_kernel: Optional[Callable] = None,
        exclude_invalid_x: bool = True,
        resume_training: bool = False,
        force_first_round_loss: bool = False,
        retrain_from_scratch: bool = False,
        show_train_summary: bool = False,
        dataloader_kwargs: Optional[Dict] = None,
        component_perturbation: float = 5e-3,
    ) -> nn.Module:
        r"""Return density estimator that approximates the proposal posterior.

        [1] _Fast epsilon-free Inference of Simulation Models with Bayesian Conditional
            Density Estimation_, Papamakarios et al., NeurIPS 2016,
            https://arxiv.org/abs/1605.06376.

        Training is performed with maximum likelihood on samples from the latest round,
        which leads the algorithm to converge to the proposal posterior.

        Args:
            final_round: Whether we are in the last round of training or not. For all
                but the last round, Algorithm 1 from [1] is executed. In last the
                round, Algorithm 2 from [1] is executed once.
            training_batch_size: Training batch size.
            learning_rate: Learning rate for Adam optimizer.
            validation_fraction: The fraction of data to use for validation.
            stop_after_epochs: The number of epochs to wait for improvement on the
                validation set before terminating training.
            max_num_epochs: Maximum number of epochs to run. If reached, we stop
                training even when the validation loss is still decreasing. Otherwise,
                we train until validation loss increases (see also `stop_after_epochs`).
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
            force_first_round_loss: If `True`, train with maximum likelihood,
                i.e., potentially ignoring the correction for using a proposal
                distribution different from the prior.
            force_first_round_loss: If `True`, train with maximum likelihood,
                regardless of the proposal distribution.
            retrain_from_scratch: Whether to retrain the conditional density
                estimator for the posterior from scratch each round. Not supported for
                SNPE-A.
            show_train_summary: Whether to print the number of epochs and validation
                loss and leakage after the training.
            dataloader_kwargs: Additional or updated kwargs to be passed to the training
                and validation dataloaders (like, e.g., a collate_fn)
            component_perturbation: The standard deviation applied to all weights and
                biases when, in the last round, the Mixture of Gaussians is build from
                a single Gaussian. This value can be problem-specific and also depends
                on the number of mixture components.

        Returns:
            Density estimator that approximates the distribution $p(\theta|x)$.
        """

        assert not retrain_from_scratch, """Retraining from scratch is not supported in SNPE-A yet. The reason for
        this is that, if we reininitialized the density estimator, the z-scoring would
        change, which would break the posthoc correction. This is a pure implementation
        issue."""

        kwargs = utils.del_entries(
            locals(),
            entries=("self", "__class__", "final_round", "component_perturbation"),
        )

        # SNPE-A always discards the prior samples.
        kwargs["discard_prior_samples"] = True

        self._round = max(self._data_round_index)

        if final_round:
            # If there is (will be) only one round, train with Algorithm 2 from [1].
            if self._round == 0:
                self._build_neural_net = partial(
                    self._build_neural_net, num_components=self._num_components
                )
            # Run Algorithm 2 from [1].
            elif not self._ran_final_round:
                # Now switch to the specified number of components. This method will
                # only be used if `retrain_from_scratch=True`. Otherwise,
                # the MDN will be built from replicating the single-component net for
                # `num_component` times (via `_expand_mog()`).
                self._build_neural_net = partial(
                    self._build_neural_net, num_components=self._num_components
                )

                # Extend the MDN to the originally desired number of components.
                self._expand_mog(eps=component_perturbation)
            else:
                warnings.warn(
                    "You have already run SNPE-A with `final_round=True`. Running it"
                    "again with this setting will not allow computing the posthoc"
                    "correction applied in SNPE-A. Thus, you will get an error when "
                    "calling `.build_posterior()` after training.",
                    UserWarning,
                )
        else:
            # Run Algorithm 1 from [1].
            # Wrap the function that builds the MDN such that we can make
            # sure that there is only one component when running.
            self._build_neural_net = partial(self._build_neural_net, num_components=1)

        if final_round:
            self._ran_final_round = True

        return super().train(**kwargs)

    def correct_for_proposal(
        self,
        density_estimator: Optional[TorchModule] = None,
    ) -> "SNPE_A_MDN":
        r"""Build mixture of Gaussians that approximates the posterior.

        Returns a `SNPE_A_MDN` object, which applies the posthoc-correction required in
        SNPE-A.

        Args:
            density_estimator: The density estimator that the posterior is based on.
                If `None`, use the latest neural density estimator that was trained.

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
            device = str(next(density_estimator.parameters()).device)

        # Set proposal of the density estimator.
        # This also evokes the z-scoring correction if necessary.
        if (
            self._proposal_roundwise[-1] is self._prior
            or self._proposal_roundwise[-1] is None
        ):
            proposal = self._prior
            assert isinstance(
                proposal, (MultivariateNormal, utils.BoxUniform)
            ), """Prior must be `torch.distributions.MultivariateNormal` or `sbi.utils.
                BoxUniform`"""
        else:
            assert isinstance(
                self._proposal_roundwise[-1], DirectPosterior
            ), """The proposal you passed to `append_simulations` is neither the prior
                nor a `DirectPosterior`. SNPE-A currently only supports these scenarios.
                """
            proposal = self._proposal_roundwise[-1]

        # Create the SNPE_A_MDN
        wrapped_density_estimator = SNPE_A_MDN(
            flow=density_estimator, proposal=proposal, prior=self._prior, device=device
        )
        return wrapped_density_estimator

    def build_posterior(
        self,
        density_estimator: Optional[TorchModule] = None,
        prior: Optional[Distribution] = None,
    ) -> "DirectPosterior":
        r"""Build posterior from the neural density estimator.

        This method first corrects the estimated density with `correct_for_proposal`
        and then returns a `DirectPosterior`.

        Args:
            density_estimator: The density estimator that the posterior is based on.
                If `None`, use the latest neural density estimator that was trained.
            prior: Prior distribution.

        Returns:
            Posterior $p(\theta|x)$  with `.sample()` and `.log_prob()` methods.
        """
        if prior is None:
            assert (
                self._prior is not None
            ), """You did not pass a prior. You have to pass the prior either at
                initialization `inference = SNPE_A(prior)` or to `.build_posterior
                (prior=prior)`."""
            prior = self._prior

        wrapped_density_estimator = self.correct_for_proposal(
            density_estimator=density_estimator
        )
        self._posterior = DirectPosterior(
            posterior_estimator=wrapped_density_estimator,
            prior=prior,
        )
        return deepcopy(self._posterior)

    def _log_prob_proposal_posterior(
        self,
        theta: Tensor,
        x: Tensor,
        masks: Tensor,
        proposal: Optional[Any],
    ) -> Tensor:
        """Return the log-probability of the proposal posterior.

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


class SNPE_A_MDN(nn.Module):
    """Generates a posthoc-corrected MDN which approximates the posterior.

    This class takes as input the density estimator (abbreviated with `_d` suffix, aka
    the proposal posterior) and the proposal prior (abbreviated with `_pp` suffix) from
    which the simulations were drawn. It uses the algorithm presented in SNPE-A [1] to
    compute the approximate posterior (abbreviated with `_p` suffix) from the two. The
    approximate posterior is a MoG. This class also implements log-prob calculation
    sampling from the approximate posterior. It inherits from `nn.Module` since the
    constructor of `DirectPosterior` expects the argument `neural_net` to be a
    `nn.Module`.

    [1] _Fast epsilon-free Inference of Simulation Models with Bayesian Conditional
        Density Estimation_, Papamakarios et al., NeurIPS 2016,
        https://arxiv.org/abs/1605.06376.
    """

    def __init__(
        self,
        flow: flows.Flow,
        proposal: Union["utils.BoxUniform", "MultivariateNormal", "DirectPosterior"],
        prior: Distribution,
        device: str,
    ):
        """Constructor.

        Args:
            flow: The trained normalizing flow, passed when building the posterior.
            proposal: The proposal distribution.
            prior: The prior distribution.
        """
        # Call nn.Module's constructor.
        super().__init__()

        self._neural_net = flow
        self._prior = prior
        self._device = device

        # Set the proposal using the `default_x`.
        if isinstance(proposal, (utils.BoxUniform, MultivariateNormal)):
            self._apply_correction = False
        else:
            self._apply_correction = True
            logits_pp, m_pp, prec_pp = proposal.posterior_estimator._posthoc_correction(
                proposal.default_x
            )
            self._logits_pp, self._m_pp, self._prec_pp = (
                logits_pp.detach(),
                m_pp.detach(),
                prec_pp.detach(),
            )

        # Take care of z-scoring, pre-compute and store prior terms.
        self._set_state_for_mog_proposal()

    def log_prob(self, inputs: Tensor, context: Tensor) -> Tensor:
        inputs, context = inputs.to(self._device), context.to(self._device)

        if not self._apply_correction:
            return self._neural_net.log_prob(inputs, context)
        else:
            # When we want to compute the approx. posterior, a proposal prior \tilde{p}
            # has already been observed. To analytically calculate the log-prob of the
            # Gaussian, we first need to compute the mixture components.

            # Compute the mixture components of the proposal posterior.
            logits_pp, m_pp, prec_pp = self._posthoc_correction(context)

            # z-score theta if it z-scoring had been requested.
            theta = self._maybe_z_score_theta(inputs)

            # Compute the log_prob of theta under the product.
            log_prob_proposal_posterior = utils.mog_log_prob(
                theta, logits_pp, m_pp, prec_pp
            )
            utils.assert_all_finite(
                log_prob_proposal_posterior, "proposal posterior eval"
            )
            return log_prob_proposal_posterior  # \hat{p} from eq (3) in [1]

    def sample(self, num_samples: int, context: Tensor, batch_size: int = 1) -> Tensor:
        context = context.to(self._device)

        if not self._apply_correction:
            return self._neural_net.sample(num_samples, context, batch_size)
        else:
            # When we want to sample from the approx. posterior, a proposal prior
            # \tilde{p} has already been observed. To analytically calculate the
            # log-prob of the Gaussian, we first need to compute the mixture components.
            return self._sample_approx_posterior_mog(num_samples, context, batch_size)

    def _sample_approx_posterior_mog(
        self, num_samples, x: Tensor, batch_size: int
    ) -> Tensor:
        r"""Sample from the approximate posterior.

        Args:
            num_samples: Desired number of samples.
            x: Conditioning context for posterior $p(\theta|x)$.
            batch_size: Batch size for sampling.

        Returns:
            Samples from the approximate mixture of Gaussians posterior.
        """

        # Compute the mixture components of the posterior.
        logits_p, m_p, prec_p = self._posthoc_correction(x)

        # Compute the precision factors which represent the upper triangular matrix
        # of the cholesky decomposition of the prec_p.
        prec_factors_p = torch.linalg.cholesky(prec_p, upper=True)

        assert logits_p.ndim == 2
        assert m_p.ndim == 3
        assert prec_p.ndim == 4
        assert prec_factors_p.ndim == 4

        # Replicate to use batched sampling from pyknos.
        if batch_size is not None and batch_size > 1:
            logits_p = logits_p.repeat(batch_size, 1)
            m_p = m_p.repeat(batch_size, 1, 1)
            prec_factors_p = prec_factors_p.repeat(batch_size, 1, 1, 1)

        # Get (optionally z-scored) MoG samples.
        theta = MultivariateGaussianMDN.sample_mog(
            num_samples, logits_p, m_p, prec_factors_p
        )

        embedded_context = self._neural_net._embedding_net(x)
        if embedded_context is not None:
            # Merge the context dimension with sample dimension in order to
            # apply the transform.
            theta = torchutils.merge_leading_dims(theta, num_dims=2)
            embedded_context = torchutils.repeat_rows(
                embedded_context, num_reps=num_samples
            )

        theta, _ = self._neural_net._transform.inverse(theta, context=embedded_context)

        if embedded_context is not None:
            # Split the context dimension from sample dimension.
            theta = torchutils.split_leading_dim(theta, shape=[-1, num_samples])

        return theta

    def _posthoc_correction(self, x: Tensor):
        """
        Compute the mixture components of the posterior given the current density
        estimator and the proposal.

        Args:
            x: Conditioning context for posterior.

        Returns:
            Mixture components of the posterior.
        """

        # Evaluate the density estimator.
        encoded_x = self._neural_net._embedding_net(x)
        dist = self._neural_net._distribution  # defined to avoid black formatting.
        logits_d, m_d, prec_d, _, _ = dist.get_mixture_components(encoded_x)
        norm_logits_d = logits_d - torch.logsumexp(logits_d, dim=-1, keepdim=True)

        # The following if case is needed because, in the constructor, we call
        # `_posthoc_correction` regardless of whether the `proposal` itself had a
        # `proposal` or not.
        if not self._apply_correction:
            return norm_logits_d, m_d, prec_d
        else:
            logits_pp, m_pp, prec_pp = self._logits_pp, self._m_pp, self._prec_pp

        # Compute the MoG parameters of the posterior.
        logits_p, m_p, prec_p, cov_p = self._proposal_posterior_transformation(
            logits_pp, m_pp, prec_pp, norm_logits_d, m_d, prec_d
        )
        return logits_p, m_p, prec_p

    def _proposal_posterior_transformation(
        self,
        logits_pp: Tensor,
        means_pp: Tensor,
        precisions_pp: Tensor,
        logits_d: Tensor,
        means_d: Tensor,
        precisions_d: Tensor,
    ):
        r"""Transforms the proposal posterior (the MDN) into the posterior.

        The approximate posterior is:
        $p(\theta|x) = 1/Z * q(\theta|x) * p(\theta) / prop(\theta)$
        In words: posterior = proposal posterior estimate * prior / proposal.

        Since the proposal posterior estimate and the proposal are MoG, and the
        prior is either Gaussian or uniform, we can solve this in closed-form.

        This function implements Appendix C from [1], and is highly similar to
        `SNPE_C._automatic_posterior_transformation()`.

        Args:
            logits_pp: Component weight of each Gaussian of the proposal prior.
            means_pp: Mean of each Gaussian of the proposal prior.
            precisions_pp: Precision matrix of each Gaussian of the proposal prior.
            logits_d: Component weight for each Gaussian of the density estimator.
            means_d: Mean of each Gaussian of the density estimator.
            precisions_d: Precision matrix of each Gaussian of the density estimator.

        Returns: (Component weight, mean, precision matrix, covariance matrix) of each
            Gaussian of the approximate posterior.
        """

        precisions_post, covariances_post = self._precisions_posterior(
            precisions_pp, precisions_d
        )

        means_post = self._means_posterior(
            covariances_post, means_pp, precisions_pp, means_d, precisions_d
        )

        logits_post = SNPE_A_MDN._logits_posterior(
            means_post,
            precisions_post,
            covariances_post,
            logits_pp,
            means_pp,
            precisions_pp,
            logits_d,
            means_d,
            precisions_d,
        )

        return logits_post, means_post, precisions_post, covariances_post

    def _set_state_for_mog_proposal(self) -> None:
        """
        Set state variables of the SNPE_A_MDN instance every time `set_proposal()`
        is called, i.e. every time a posterior is build using
        `SNPE_A.build_posterior()`.

        This function is almost identical to `SNPE_C._set_state_for_mog_proposal()`.

        Three things are computed:
        1) Check if z-scoring was requested. To do so, we check if the `_transform`
            argument of the net had been a `CompositeTransform`. See pyknos mdn.py.
        2) Define a (potentially standardized) prior. It's standardized if z-scoring
            had been requested.
        3) Compute (Precision * mean) for the prior. This quantity is used at every
            training step if the prior is Gaussian.
        """

        self.z_score_theta = isinstance(self._neural_net._transform, CompositeTransform)

        self._set_maybe_z_scored_prior()

        if isinstance(self._maybe_z_scored_prior, MultivariateNormal):
            self.prec_m_prod_prior = torch.mv(
                self._maybe_z_scored_prior.precision_matrix,  # type: ignore
                self._maybe_z_scored_prior.loc,  # type: ignore
            )

    def _set_maybe_z_scored_prior(self) -> None:
        r"""
        Compute and store potentially standardized prior (if z-scoring was requested).

        This function is highly similar to `SNPE_C._set_maybe_z_scored_prior()`.

        The proposal posterior is:
        $p(\theta|x) = 1/Z * q(\theta|x) * prop(\theta) / p(\theta)$

        Let's denote z-scored theta by `a`: a = (theta - mean) / std
        Then $p'(a|x) = 1/Z_2 * q'(a|x) * prop'(a) / p'(a)$

        The ' indicates that the evaluation occurs in standardized space. The constant
        scaling factor has been absorbed into $Z_2$.
        From the above equation, we see that we need to evaluate the prior **in
        standardized space**. We build the standardized prior in this function.

        The standardize transform that is applied to the samples theta does not use
        the exact prior mean and std (due to implementation issues). Hence, the z-scored
        prior will not be exactly have mean=0 and std=1.
        """
        if self.z_score_theta:
            scale = self._neural_net._transform._transforms[0]._scale
            shift = self._neural_net._transform._transforms[0]._shift

            # Following the definition of the linear transform in
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
                self._maybe_z_scored_prior = utils.BoxUniform(
                    almost_zero_mean - range_, almost_zero_mean + range_
                )
        else:
            self._maybe_z_scored_prior = self._prior

    def _maybe_z_score_theta(self, theta: Tensor) -> Tensor:
        """Return potentially standardized theta if z-scoring was requested."""

        if self.z_score_theta:
            theta, _ = self._neural_net._transform(theta)

        return theta

    def _precisions_posterior(self, precisions_pp: Tensor, precisions_d: Tensor):
        r"""Return the precisions and covariances of the MoG posterior.

        As described at the end of Appendix C in [1], it can happen that the
        proposal's precision matrix is not positive definite.

        $S_k^\prime = ( S_k^{-1} - S_0^{-1} )^{-1}$
        (see eq (23) in Appendix C of [1])

        Args:
            precisions_pp: Precision matrices of the proposal prior.
            precisions_d: Precision matrices of the density estimator.

        Returns: (Precisions, Covariances) of the MoG posterior.
        """

        num_comps_p = precisions_pp.shape[1]
        num_comps_d = precisions_d.shape[1]

        # Check if precision matrices are positive definite.
        for batches in precisions_pp:
            for pprior in batches:
                eig_pprior = torch.linalg.eigvalsh(pprior, UPLO="U")
                if not (eig_pprior > 0).all():
                    raise AssertionError(
                        "The precision matrix of the proposal is not positive definite!"
                    )
        for batches in precisions_d:
            for d in batches:
                eig_d = torch.linalg.eigvalsh(d, UPLO="U")
                if not (eig_d > 0).all():
                    raise AssertionError(
                        "The precision matrix of the density estimator is not "
                        "positive definite!"
                    )

        precisions_pp_rep = precisions_pp.repeat_interleave(num_comps_d, dim=1)
        precisions_d_rep = precisions_d.repeat(1, num_comps_p, 1, 1)

        precisions_p = precisions_d_rep - precisions_pp_rep
        if isinstance(self._maybe_z_scored_prior, MultivariateNormal):
            precisions_p += self._maybe_z_scored_prior.precision_matrix

        # Check if precision matrix is positive definite.
        for idx_batch, batches in enumerate(precisions_p):
            for idx_comp, pp in enumerate(batches):
                eig_pp = torch.symeig(pp, eigenvectors=False).eigenvalues
                if not (eig_pp > 0).all():
                    raise AssertionError(
                        "The precision matrix of a posterior is not positive "
                        "definite! This is a known issue for SNPE-A. Either try a "
                        "different parameter setting, e.g. a different number of "
                        "mixture components (when contracting SNPE-A), or a different "
                        "value for the parameter perturbation (when building the "
                        "posterior)."
                    )

        covariances_p = torch.inverse(precisions_p)
        return precisions_p, covariances_p

    def _means_posterior(
        self,
        covariances_p: Tensor,
        means_pp: Tensor,
        precisions_pp: Tensor,
        means_d: Tensor,
        precisions_d: Tensor,
    ):
        r"""Return the means of the MoG posterior.

        $m_k^\prime = S_k^\prime ( S_k^{-1} m_k - S_0^{-1} m_0 )$
        (see eq (24) in Appendix C of [1])

        Args:
            covariances_post: Covariance matrices of the MoG posterior.
            means_pp: Means of the proposal prior.
            precisions_pp: Precision matrices of the proposal prior.
            means_d: Means of the density estimator.
            precisions_d: Precision matrices of the density estimator.

        Returns: Means of the MoG posterior.
        """

        num_comps_pp = precisions_pp.shape[1]
        num_comps_d = precisions_d.shape[1]

        # Compute the products P_k * m_k and P_0 * m_0.
        prec_m_prod_pp = utils.batched_mixture_mv(precisions_pp, means_pp)
        prec_m_prod_d = utils.batched_mixture_mv(precisions_d, means_d)

        # Repeat them to allow for matrix operations: same trick as for the precisions.
        prec_m_prod_pp_rep = prec_m_prod_pp.repeat_interleave(num_comps_d, dim=1)
        prec_m_prod_d_rep = prec_m_prod_d.repeat(1, num_comps_pp, 1)

        # Compute the means P_k^prime * (P_k * m_k - P_0 * m_0).
        summed_cov_m_prod_rep = prec_m_prod_d_rep - prec_m_prod_pp_rep
        if isinstance(self._maybe_z_scored_prior, MultivariateNormal):
            summed_cov_m_prod_rep += self.prec_m_prod_prior

        means_p = utils.batched_mixture_mv(covariances_p, summed_cov_m_prod_rep)
        return means_p

    @staticmethod
    def _logits_posterior(
        means_post: Tensor,
        precisions_post: Tensor,
        covariances_post: Tensor,
        logits_pp: Tensor,
        means_pp: Tensor,
        precisions_pp: Tensor,
        logits_d: Tensor,
        means_d: Tensor,
        precisions_d: Tensor,
    ):
        r"""Return the component weights (i.e. logits) of the MoG posterior.

        $\alpha_k^\prime = \frac{ \alpha_k exp(-0.5 c_k) }{ \sum{j} \alpha_j exp(-0.5
        c_j) } $
        with
        $c_k = logdet(S_k) - logdet(S_0) - logdet(S_k^\prime) +
             + m_k^T P_k m_k - m_0^T P_0 m_0 - m_k^\prime^T P_k^\prime m_k^\prime$
        (see eqs. (25, 26) in Appendix C of [1])

        Args:
            means_post: Means of the posterior.
            precisions_post: Precision matrices of the posterior.
            covariances_post: Covariance matrices of the posterior.
            logits_pp: Component weights (i.e. logits) of the proposal prior.
            means_pp: Means of the proposal prior.
            precisions_pp: Precision matrices of the proposal prior.
            logits_d: Component weights (i.e. logits) of the density estimator.
            means_d: Means of the density estimator.
            precisions_d: Precision matrices of the density estimator.

        Returns: Component weights of the proposal posterior.
        """

        num_comps_pp = precisions_pp.shape[1]
        num_comps_d = precisions_d.shape[1]

        # Compute the ratio of the logits similar to eq (10) in Appendix A.1 of [2]
        logits_pp_rep = logits_pp.repeat_interleave(num_comps_d, dim=1)
        logits_d_rep = logits_d.repeat(1, num_comps_pp)
        logit_factors = logits_d_rep - logits_pp_rep

        # Compute the log-determinants
        logdet_covariances_post = torch.logdet(covariances_post)
        logdet_covariances_pp = -torch.logdet(precisions_pp)
        logdet_covariances_d = -torch.logdet(precisions_d)

        # Repeat the proposal and density estimator terms such that there are LK terms.
        # Same trick as has been used above.
        logdet_covariances_pp_rep = logdet_covariances_pp.repeat_interleave(
            num_comps_d, dim=1
        )
        logdet_covariances_d_rep = logdet_covariances_d.repeat(1, num_comps_pp)

        log_sqrt_det_ratio = 0.5 * (  # similar to eq (14) in Appendix A.1 of [2]
            logdet_covariances_post
            + logdet_covariances_pp_rep
            - logdet_covariances_d_rep
        )

        # Compute for proposal, density estimator, and proposal posterior:
        exponent_pp = utils.batched_mixture_vmv(
            precisions_pp, means_pp  # m_0 in eq (26) in Appendix C of [1]
        )
        exponent_d = utils.batched_mixture_vmv(
            precisions_d, means_d  # m_k in eq (26) in Appendix C of [1]
        )
        exponent_post = utils.batched_mixture_vmv(
            precisions_post, means_post  # m_k^\prime in eq (26) in Appendix C of [1]
        )

        # Extend proposal and density estimator exponents to get LK terms.
        exponent_pp_rep = exponent_pp.repeat_interleave(num_comps_d, dim=1)
        exponent_d_rep = exponent_d.repeat(1, num_comps_pp)
        exponent = -0.5 * (
            exponent_d_rep - exponent_pp_rep - exponent_post  # eq (26) in [1]
        )

        logits_post = logit_factors + log_sqrt_det_ratio + exponent
        return logits_post
