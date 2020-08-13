# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Affero General Public License v3, see <https://www.gnu.org/licenses/>.


from math import pi
from typing import Any, Callable, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
from pyknos.mdn.mdn import MultivariateGaussianMDN as mdn
from pyknos.nflows.transforms import CompositeTransform
from torch import Tensor, eye, ones
from torch.distributions import MultivariateNormal

from sbi import utils as utils
from sbi.inference.posteriors.direct_posterior import DirectPosterior
from sbi.inference.snpe.snpe_base import PosteriorEstimator
from sbi.types import TensorboardSummaryWriter
from sbi.utils import (
    batched_mixture_mv,
    batched_mixture_vmv,
    clamp_and_warn,
    del_entries,
    repeat_rows,
)


class SNPE_C(PosteriorEstimator):
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
        use_combined_loss: bool = False,
        device: str = "cpu",
        logging_level: Union[int, str] = "WARNING",
        summary_writer: Optional[TensorboardSummaryWriter] = None,
        show_progress_bars: bool = True,
        show_round_summary: bool = False,
    ):
        r"""SNPE-C / APT [1].

        [1] _Automatic Posterior Transformation for Likelihood-free Inference_,
            Greenberg et al., ICML 2019, https://arxiv.org/abs/1905.07488.

        This class implements two loss variants of SNPE-C: the non-atomic and the atomic
        version. The atomic loss of SNPE-C can be used for any density estimator,
        i.e. also for normalizing flows. However, it suffers from leakage issues. On
        the other hand, the non-atomic loss can only be used only if the proposal
        distribution is a mixture of Gaussians, the density estimator is a mixture of
        Gaussians, and the prior is either Gaussian or Uniform. It does not suffer from
        leakage issues. At the beginning of each round, we print whether the non-atomic
        or the atomic version is used.

        In this codebase, we will automatically switch to the non-atomic loss if the
        following criteria are fulfilled:
        - proposal has is a `DirectPosterior` with density_estimator `mdn`, as built
            with `utils.sbi.posterior_nn()`.
        - the density estimator is a `mdn`, as built with `utils.sbi.posterior_nn()`.
        - `isinstance(prior, MultivariateNormal)` (from `torch.distributions`) or
            `isinstance(prior, sbi.utils.BoxUniform)`

        Note that custom implementations of any of these densities (or estimators) will
        not trigger the non-atomic loss, and the algorithm will fall back onto using
        the atomic loss.

        Args:
            simulator: A function that takes parameters $\theta$ and maps them to
                simulations, or observations, `x`, $\mathrm{sim}(\theta)\to x$. Any
                regular Python callable (i.e. function or class with `__call__` method)
                can be used.
            prior: A probability distribution that expresses prior knowledge about the
                parameters, e.g. which ranges are meaningful for them. Any
                object with `.log_prob()`and `.sample()` (for example, a PyTorch
                distribution) can be used.
            num_workers: Number of parallel workers to use for simulations.
            simulation_batch_size: Number of parameter sets that the simulator
                maps to data x at once. If None, we simulate all parameter sets at the
                same time. If >= 1, the simulator has to process data of shape
                (simulation_batch_size, parameter_dimension).
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
            use_combined_loss: Whether to train the neural net also on prior samples
                using maximum likelihood in addition to training it on all samples using
                atomic loss. The extra MLE loss helps prevent density leaking with
                bounded priors.
            device: torch device on which to compute, e.g. gpu, cpu.
            logging_level: Minimum severity of messages to log. One of the strings
                INFO, WARNING, DEBUG, ERROR and CRITICAL.
            summary_writer: A tensorboard `SummaryWriter` to control, among others, log
                file location (default is `<current working directory>/logs`.)
            show_progress_bars: Whether to show a progressbar during simulation and
                sampling.
            show_round_summary: Whether to show the validation loss and leakage after
                each round.
        """

        self._use_combined_loss = use_combined_loss

        kwargs = del_entries(
            locals(), entries=("self", "__class__", "use_combined_loss")
        )
        super().__init__(**kwargs)

    def __call__(
        self,
        num_simulations: int,
        proposal: Optional[Any] = None,
        num_atoms: int = 10,
        training_batch_size: int = 50,
        learning_rate: float = 5e-4,
        validation_fraction: float = 0.1,
        stop_after_epochs: int = 20,
        max_num_epochs: Optional[int] = None,
        clip_max_norm: Optional[float] = 5.0,
        calibration_kernel: Optional[Callable] = None,
        exclude_invalid_x: bool = True,
        discard_prior_samples: bool = False,
        retrain_from_scratch_each_round: bool = False,
    ) -> DirectPosterior:
        r"""Run SNPE.

        Return posterior $p(\theta|x)$ after inference.

        Args:
            num_simulations: Number of simulator calls.
            proposal: Distribution that the parameters $\theta$ are drawn from.
                `proposal=None` uses the prior. Setting the proposal to a distribution
                targeted on a specific observation, e.g. a posterior $p(\theta|x_o)$
                obtained previously, can lead to less required simulations.
            num_atoms: Number of atoms to use for classification.
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
            retrain_from_scratch_each_round: Whether to retrain the conditional density
                estimator for the posterior from scratch each round.

        Returns:
            Posterior $p(\theta|x)$ that can be sampled and evaluated.
        """

        # WARNING: sneaky trick ahead. We proxy the parent's `__call__` here,
        # requiring the signature to have `num_atoms`, save it for use below, and
        # continue. It's sneaky because we are using the object (self) as a namespace
        # to pass arguments between functions, and that's implicit state management.

        self._num_atoms = num_atoms
        kwargs = del_entries(locals(), entries=("self", "__class__", "num_atoms"))

        if proposal is not None:
            self.use_non_atomic_loss = (
                isinstance(proposal.net._distribution, mdn)
                and isinstance(self._posterior.net._distribution, mdn)
                and (
                    isinstance(self._prior, utils.BoxUniform)
                    or isinstance(self._prior, MultivariateNormal)
                )
            )

            algorithm = "non-atomic" if self.use_non_atomic_loss else "atomic"
            print(f"Using SNPE-C with {algorithm} loss")

            if self.use_non_atomic_loss:
                # Take care of z-scoring, pre-compute and store prior terms.
                self._set_state_for_mog_proposal()

        return super().__call__(**kwargs)

    def _set_state_for_mog_proposal(self) -> None:
        """
        Set state variables that are used at every training step of non-atomic SNPE-C.

        Three things are computed:
        1) Check if z-scoring was requested. To do so, we check if the `_transform`
            argument of the net had been a `CompositeTransform`. See pyknos mdn.py.
        2) Define a (potentially standardized) prior. It's standardized if z-scoring
            had been requested.
        3) Compute (Precision * mean) for the prior. This quantity is used at every
            training step if the prior is Gaussian.
        """

        self.z_score_theta = isinstance(
            self._posterior.net._transform, CompositeTransform
        )

        self._set_maybe_z_scored_prior()

        if isinstance(self._maybe_z_scored_prior, MultivariateNormal):
            self.prec_m_prod_prior = torch.mv(
                self._maybe_z_scored_prior.precision_matrix,
                self._maybe_z_scored_prior.loc,
            )

    def _set_maybe_z_scored_prior(self) -> None:
        r"""
        Compute and store potentially standardized prior (if z-scoring was requested).

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
            scale = self._posterior.net._transform._transforms[0]._scale
            shift = self._posterior.net._transform._transforms[0]._shift

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
                    almost_zero_mean, torch.diag(almost_one_std),
                )
            else:
                range_ = torch.sqrt(almost_one_std * 3.0)
                self._maybe_z_scored_prior = utils.BoxUniform(
                    almost_zero_mean - range_, almost_zero_mean + range_
                )
        else:
            self._maybe_z_scored_prior = self._prior

    def _log_prob_proposal_posterior(
        self, theta: Tensor, x: Tensor, masks: Tensor, proposal: Optional[Any]
    ) -> Tensor:
        """
        Return the log-probability of the proposal posterior.

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

        if self.use_non_atomic_loss:
            return self._log_prob_proposal_posterior_mog(theta, x, proposal)
        else:
            return self._log_prob_proposal_posterior_atomic(theta, x, masks)

    def _log_prob_proposal_posterior_atomic(
        self, theta: Tensor, x: Tensor, masks: Tensor
    ):
        """
        Return log probability of the proposal posterior for atomic proposals.

        We have two main options when evaluating the proposal posterior.
            (1) Generate atoms from the proposal prior.
            (2) Generate atoms from a more targeted distribution, such as the most
                recent posterior.
        If we choose the latter, it is likely beneficial not to do this in the first
        round, since we would be sampling from a randomly-initialized neural density
        estimator.

        Args:
            theta: Batch of parameters θ.
            x: Batch of data.
            masks: Mask that is True for prior samples in the batch in order to train
                them with prior loss.

        Returns:
            Log-probability of the proposal posterior.
        """

        batch_size = theta.shape[0]

        num_atoms = clamp_and_warn(
            "num_atoms", self._num_atoms, min_val=2, max_val=batch_size
        )

        # Each set of parameter atoms is evaluated using the same x,
        # so we repeat rows of the data x, e.g. [1, 2] -> [1, 1, 2, 2]
        repeated_x = repeat_rows(x, num_atoms)

        # To generate the full set of atoms for a given item in the batch,
        # we sample without replacement num_atoms - 1 times from the rest
        # of the theta in the batch.
        probs = ones(batch_size, batch_size) * (1 - eye(batch_size)) / (batch_size - 1)

        choices = torch.multinomial(probs, num_samples=num_atoms - 1, replacement=False)
        contrasting_theta = theta[choices]

        # We can now create our sets of atoms from the contrasting parameter sets
        # we have generated.
        atomic_theta = torch.cat((theta[:, None, :], contrasting_theta), dim=1).reshape(
            batch_size * num_atoms, -1
        )

        # Evaluate large batch giving (batch_size * num_atoms) log prob posterior evals.
        log_prob_posterior = self._posterior.net.log_prob(atomic_theta, repeated_x)
        self._assert_all_finite(log_prob_posterior, "posterior eval")
        log_prob_posterior = log_prob_posterior.reshape(batch_size, num_atoms)

        # Get (batch_size * num_atoms) log prob prior evals.
        log_prob_prior = self._prior.log_prob(atomic_theta)
        log_prob_prior = log_prob_prior.reshape(batch_size, num_atoms)
        self._assert_all_finite(log_prob_prior, "prior eval")

        # Compute unnormalized proposal posterior.
        unnormalized_log_prob = log_prob_posterior - log_prob_prior

        # Normalize proposal posterior across discrete set of atoms.
        log_prob_proposal_posterior = unnormalized_log_prob[:, 0] - torch.logsumexp(
            unnormalized_log_prob, dim=-1
        )
        self._assert_all_finite(log_prob_proposal_posterior, "proposal posterior eval")

        # XXX This evaluates the posterior on _all_ prior samples
        if self._use_combined_loss:
            log_prob_posterior_non_atomic = self._posterior.net.log_prob(theta, x)
            masks = masks.reshape(-1)
            log_prob_proposal_posterior = (
                masks * log_prob_posterior_non_atomic + log_prob_proposal_posterior
            )

        return log_prob_proposal_posterior

    def _log_prob_proposal_posterior_mog(
        self, theta: Tensor, x: Tensor, proposal: DirectPosterior,
    ) -> Tensor:
        """
        Return log-probability of the proposal posterior for MoG proposal.

        For MoG proposals and MoG density estimators, this can be done in closed form
        and does not require atomic loss (i.e. there will be no leakage issues).

        Notation:

        m are mean vectors.
        prec are precision matrices.
        cov are covariance matrices.

        _p at the end indicates that it is the proposal.
        _d indicates that it is the density estimator.
        _pp indicates the proposal posterior.

        All tensors will have shapes (batch_dim, num_components, ...)

        Args:
            theta: Batch of parameters θ.
            x: Batch of data.
            proposal: Proposal distribution.

        Returns:
            Log-probability of the proposal posterior.
        """

        # Evaluate the proposal. MDNs do not have functionality to run the embedding_net
        # and then get the mixture_components (**without** calling log_prob()). Hence,
        # we call them separately here.
        encoded_x = proposal.net._embedding_net(x)
        dist = proposal.net._distribution  # defined to avoid ugly black formatting.
        logits_p, m_p, prec_p, _, _ = dist.get_mixture_components(encoded_x)
        norm_logits_p = logits_p - torch.logsumexp(logits_p, dim=-1, keepdim=True)

        # Evaluate the density estimator.
        encoded_x = self._posterior.net._embedding_net(x)
        dist = self._posterior.net._distribution  # defined to avoid black formatting.
        logits_d, m_d, prec_d, _, _ = dist.get_mixture_components(encoded_x)
        norm_logits_d = logits_d - torch.logsumexp(logits_d, dim=-1, keepdim=True)

        # z-score theta if it z-scoring had been requested.
        theta = self._maybe_z_score_theta(theta)

        # Compute the MoG parameters of the proposal posterior.
        logits_pp, m_pp, prec_pp, cov_pp = self._automatic_posterior_transformation(
            norm_logits_p, m_p, prec_p, norm_logits_d, m_d, prec_d,
        )

        # Compute the log_prob of theta under the product.
        log_prob_proposal_posterior = _mog_log_prob(theta, logits_pp, m_pp, prec_pp,)
        self._assert_all_finite(log_prob_proposal_posterior, "proposal posterior eval")

        return log_prob_proposal_posterior

    def _automatic_posterior_transformation(
        self,
        logits_p: Tensor,
        means_p: Tensor,
        precisions_p: Tensor,
        logits_d: Tensor,
        means_d: Tensor,
        precisions_d: Tensor,
    ):
        r"""
        Returns the MoG parameters of the proposal posterior.

        The proposal posterior is:
        $pp(\theta|x) = 1/Z * q(\theta|x) * prop(\theta) / p(\theta)$
        In words: proposal posterior = posterior estimate * proposal / prior.

        If the posterior estimate and the proposal are MoG and the prior is either
        Gaussian or uniform, we can solve this in closed-form. The is implemented in
        this function.

        This function implements Appendix A1 from Greenberg et al. 2019.

        We have to build L*K components. How do we do this?
        Example: proposal has two components, density estimator has three components.
        Let's call the two components of the proposal i,j and the three components
        of the density estimator x,y,z. We have to multiply every component of the
        proposal with every component of the density estimator. So, what we do is:
        1) for the proposal, build: i,i,i,j,j,j. Done with torch.repeat_interleave()
        2) for the density estimator, build: x,y,z,x,y,z. Done with torch.repeat()
        3) Multiply them with simple matrix operations.

        Args:
            logits_p: Component weight of each Gaussian of the proposal.
            means_p: Mean of each Gaussian of the proposal.
            precisions_p: Precision matrix of each Gaussian of the proposal.
            logits_d: Component weight for each Gaussian of the density estimator.
            means_d: Mean of each Gaussian of the density estimator.
            precisions_d: Precision matrix of each Gaussian of the density estimator.

        Returns: (Component weight, mean, precision matrix, covariance matrix) of each
            Gaussian of the proposal posterior. Has L*K terms (proposal has L terms,
            density estimator has K terms).
        """

        precisions_pp, covariances_pp = self._precisions_proposal_posterior(
            precisions_p, precisions_d
        )

        means_pp = self._means_proposal_posterior(
            covariances_pp, means_p, precisions_p, means_d, precisions_d,
        )

        logits_pp = self._logits_proposal_posterior(
            means_pp,
            precisions_pp,
            covariances_pp,
            logits_p,
            means_p,
            precisions_p,
            logits_d,
            means_d,
            precisions_d,
        )

        return logits_pp, means_pp, precisions_pp, covariances_pp

    def _precisions_proposal_posterior(
        self, precisions_p: Tensor, precisions_d: Tensor,
    ):
        """
        Return the precisions and covariances of the proposal posterior.

        Args:
            precisions_p: Precision matrices of the proposal distribution.
            precisions_d: Precision matrices of the density estimator.

        Returns: (Precisions, Covariances) of the proposal posterior. L*K terms.
        """

        num_comps_p = precisions_p.shape[1]
        num_comps_d = precisions_d.shape[1]

        precisions_p_rep = precisions_p.repeat_interleave(num_comps_d, dim=1)
        precisions_d_rep = precisions_d.repeat(1, num_comps_p, 1, 1)

        precisions_pp = precisions_p_rep + precisions_d_rep
        if isinstance(self._maybe_z_scored_prior, MultivariateNormal):
            precisions_pp -= self._maybe_z_scored_prior.precision_matrix

        covariances_pp = torch.inverse(precisions_pp)

        return precisions_pp, covariances_pp

    def _means_proposal_posterior(
        self,
        covariances_pp: Tensor,
        means_p: Tensor,
        precisions_p: Tensor,
        means_d: Tensor,
        precisions_d: Tensor,
    ):
        """
        Return the means of the proposal posterior.

        means_pp = C_ix * (P_i * m_i + P_x * m_x - P_o * m_o).

        Args:
            covariances_pp: Covariance matrices of the proposal posterior.
            means_p: Means of the proposal distribution.
            precisions_p: Precision matrices of the proposal distribution.
            means_d: Means of the density estimator.
            precisions_d: Precision matrices of the density estimator.

        Returns: Means of the proposal posterior. L*K terms.
        """

        num_comps_p = precisions_p.shape[1]
        num_comps_d = precisions_d.shape[1]

        # First, compute the product P_i * m_i and P_j * m_j
        prec_m_prod_p = batched_mixture_mv(precisions_p, means_p)
        prec_m_prod_d = batched_mixture_mv(precisions_d, means_d)

        # Repeat them to allow for matrix operations: same trick as for the precisions.
        prec_m_prod_p_rep = prec_m_prod_p.repeat_interleave(num_comps_d, dim=1)
        prec_m_prod_d_rep = prec_m_prod_d.repeat(1, num_comps_p, 1)

        # Means = C_ij * (P_i * m_i + P_x * m_x - P_o * m_o).
        summed_cov_m_prod_rep = prec_m_prod_p_rep + prec_m_prod_d_rep
        if isinstance(self._maybe_z_scored_prior, MultivariateNormal):
            summed_cov_m_prod_rep -= self.prec_m_prod_prior

        means_pp = batched_mixture_mv(covariances_pp, summed_cov_m_prod_rep)

        return means_pp

    @staticmethod
    def _logits_proposal_posterior(
        means_pp: Tensor,
        precisions_pp: Tensor,
        covariances_pp: Tensor,
        logits_p: Tensor,
        means_p: Tensor,
        precisions_p: Tensor,
        logits_d: Tensor,
        means_d: Tensor,
        precisions_d: Tensor,
    ):
        """
        Return the component weights (i.e. logits) of the proposal posterior.

        Args:
            means_pp: Means of the proposal posterior.
            precisions_pp: Precision matrices of the proposal posterior.
            covariances_pp: Covariance matrices of the proposal posterior.
            logits_p: Component weights (i.e. logits) of the proposal distribution.
            means_p: Means of the proposal distribution.
            precisions_p: Precision matrices of the proposal distribution.
            logits_d: Component weights (i.e. logits) of the density estimator.
            means_d: Means of the density estimator.
            precisions_d: Precision matrices of the density estimator.

        Returns: Component weights of the proposal posterior. L*K terms.
        """

        num_comps_p = precisions_p.shape[1]
        num_comps_d = precisions_d.shape[1]

        # Compute log(alpha_i * beta_j)
        logits_p_rep = logits_p.repeat_interleave(num_comps_d, dim=1)
        logits_d_rep = logits_d.repeat(1, num_comps_p)
        logit_factors = logits_p_rep + logits_d_rep

        # Compute sqrt(det()/(det()*det()))
        logdet_covariances_pp = torch.logdet(covariances_pp)
        logdet_covariances_p = -torch.logdet(precisions_p)
        logdet_covariances_d = -torch.logdet(precisions_d)

        # Repeat the proposal and density estimator terms such that there are LK terms.
        # Same trick as has been used above.
        logdet_covariances_p_rep = logdet_covariances_p.repeat_interleave(
            num_comps_d, dim=1
        )
        logdet_covariances_d_rep = logdet_covariances_d.repeat(1, num_comps_p)

        log_sqrt_det_ratio = 0.5 * (
            logdet_covariances_pp
            - (logdet_covariances_p_rep + logdet_covariances_d_rep)
        )

        # Compute for proposal, density estimator, and proposal posterior:
        # mu_i.T * P_i * mu_i
        exponent_p = batched_mixture_vmv(precisions_p, means_p)
        exponent_d = batched_mixture_vmv(precisions_d, means_d)
        exponent_pp = batched_mixture_vmv(precisions_pp, means_pp)

        # Extend proposal and density estimator exponents to get LK terms.
        exponent_p_rep = exponent_p.repeat_interleave(num_comps_d, dim=1)
        exponent_d_rep = exponent_d.repeat(1, num_comps_p)
        exponent = -0.5 * (exponent_p_rep + exponent_d_rep - exponent_pp)

        logits_pp = logit_factors + log_sqrt_det_ratio + exponent

        return logits_pp

    def _maybe_z_score_theta(self, theta: Tensor) -> Tensor:
        """Return potentially standardized theta if z-scoring was requested."""

        if self.z_score_theta:
            theta, _ = self._posterior.net._transform(theta)

        return theta


def _mog_log_prob(
    theta: Tensor, logits_pp: Tensor, means_pp: Tensor, precisions_pp: Tensor,
) -> Tensor:
    r"""
    Returns the log-probability of parameter sets $\theta$ under a mixture of Gaussians.

    Note that the mixture can have different logits, means, covariances for any theta in
    the batch. This is because these values were computed from a batch of $x$ (and the
    $x$ in the batch are not the same).

    This code is similar to the code of mdn.py in pyknos, but it does not use
    log(det(Cov)) = -2*sum(log(diag(L))), L being Cholesky of Precision. Instead, it
    just computes log(det(Cov)). Also, it uses the above-defined helper
    `_batched_vmv()`.

    Args:
        theta: Parameters at which to evaluate the mixture.
        logits_pp: (Unnormalized) mixture components.
        means_pp: Means of all mixture components. Shape
            (batch_dim, num_components, theta_dim).
        precisions_pp: Precisions of all mixtures. Shape
            (batch_dim, num_components, theta_dim, theta_dim).

    Returns: The log-probability.
    """

    _, _, output_dim = means_pp.size()
    theta = theta.view(-1, 1, output_dim)

    # Split up evaluation into parts.
    weights = logits_pp - torch.logsumexp(logits_pp, dim=-1, keepdim=True)
    constant = -(output_dim / 2.0) * torch.log(torch.tensor([2 * pi]))
    log_det = 0.5 * torch.log(torch.det(precisions_pp))
    theta_minus_mean = theta.expand_as(means_pp) - means_pp
    exponent = -0.5 * batched_mixture_vmv(precisions_pp, theta_minus_mean)

    return torch.logsumexp(weights + constant + log_det + exponent, dim=-1)
