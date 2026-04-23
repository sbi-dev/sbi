# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

from typing import Any, List, Optional

try:
    from typing import Protocol
except ImportError:
    from typing_extensions import Protocol

import torch
from torch import Tensor, eye, ones
from torch.distributions import Distribution, MultivariateNormal

import sbi.utils as utils
from sbi.inference.posteriors.direct_posterior import DirectPosterior
from sbi.neural_nets.estimators.base import ConditionalDensityEstimator
from sbi.neural_nets.estimators.mixture_density_estimator import (
    MixtureDensityEstimator,
)
from sbi.neural_nets.estimators.mog import MoG
from sbi.neural_nets.estimators.shape_handling import (
    reshape_to_batch_event,
    reshape_to_sample_batch_event,
)
from sbi.utils.sbiutils import (
    batched_mixture_mv,
    batched_mixture_vmv,
    clamp_and_warn,
)
from sbi.utils.torchutils import assert_all_finite, repeat_rows


class NPELossStrategy(Protocol):
    """Protocol for NPE loss strategies."""

    # Replaces the use_non_atomic_loss flag and the hasattr(self, "_ran_final_round")
    uses_only_latest_round: bool

    def __call__(
        self,
        theta: Tensor,
        x: Tensor,
        masks: Tensor,
        proposal: Optional[Any],
        **kwargs,
    ) -> Tensor:
        """Calculate the log-probability of the proposal posterior.

        Args:
            theta: Batch of parameters θ.
            x: Batch of data.
            masks: Mask that is True for prior samples in the batch.
            proposal: Proposal distribution.

        Returns:
            Log-probability of the proposal posterior.
        """
        ...


class AtomicLoss:
    """Atomic loss for NPE-C (sample-based)."""

    uses_only_latest_round: bool = False

    def __init__(
        self,
        neural_net: ConditionalDensityEstimator,
        prior: Distribution,
        num_atoms: int = 10,
        use_combined_loss: bool = False,
    ):
        self._neural_net = neural_net
        self._prior = prior
        self._num_atoms = num_atoms
        self._use_combined_loss = use_combined_loss

    def __call__(
        self,
        theta: Tensor,
        x: Tensor,
        masks: Tensor,
        proposal: Optional[Any],
        **kwargs,
    ) -> Tensor:
        """Return log probability of the proposal posterior for atomic proposals.

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
            proposal: Proposal distribution.
            **kwargs: Extra arguments.

        Returns:
            Log-probability of the proposal posterior.
        """
        batch_size = theta.shape[0]

        num_atoms = int(
            clamp_and_warn("num_atoms", self._num_atoms, min_val=2, max_val=batch_size)
        )

        # Each set of parameter atoms is evaluated using the same x
        repeated_x = repeat_rows(x, num_atoms)

        # To generate the full set of atoms for a given item in the batch,
        # we sample without replacement num_atoms - 1 times from the rest
        # of the theta in the batch.
        probs = ones(batch_size, batch_size) * (1 - eye(batch_size)) / (batch_size - 1)

        choices = torch.multinomial(probs, num_samples=num_atoms - 1, replacement=False)
        contrasting_theta = theta[choices]

        # We can now create our sets of atoms from the contrasting parameter sets
        atomic_theta = torch.cat((theta[:, None, :], contrasting_theta), dim=1).reshape(
            batch_size * num_atoms, -1
        )

        # Get (batch_size * num_atoms) log prob prior evals.
        log_prob_prior = self._prior.log_prob(atomic_theta)
        log_prob_prior = log_prob_prior.reshape(batch_size, num_atoms)
        assert_all_finite(log_prob_prior, "prior eval")

        # Evaluate large batch giving (batch_size * num_atoms) log prob posterior evals.
        atomic_theta = reshape_to_sample_batch_event(
            atomic_theta, atomic_theta.shape[1:]
        )
        repeated_x = reshape_to_batch_event(
            repeated_x, self._neural_net.condition_shape
        )
        log_prob_posterior = self._neural_net.log_prob(atomic_theta, repeated_x)
        assert_all_finite(log_prob_posterior, "posterior eval")
        log_prob_posterior = log_prob_posterior.reshape(batch_size, num_atoms)

        # Compute unnormalized proposal posterior.
        unnormalized_log_prob = log_prob_posterior - log_prob_prior

        # Normalize proposal posterior across discrete set of atoms.
        log_prob_proposal_posterior = unnormalized_log_prob[:, 0] - torch.logsumexp(
            unnormalized_log_prob, dim=-1
        )
        assert_all_finite(log_prob_proposal_posterior, "proposal posterior eval")

        # combined loss helps prevent density leaking with bounded priors.
        if self._use_combined_loss:
            theta = reshape_to_sample_batch_event(theta, self._neural_net.input_shape)
            x = reshape_to_batch_event(x, self._neural_net.condition_shape)
            log_prob_posterior_non_atomic = self._neural_net.log_prob(theta, x)
            # squeeze to remove sample dimension, which is always one during the loss
            # evaluation of `SNPE_C`.
            log_prob_posterior_non_atomic = log_prob_posterior_non_atomic.squeeze(dim=0)
            masks = masks.reshape(-1)
            log_prob_proposal_posterior = (
                masks * log_prob_posterior_non_atomic + log_prob_proposal_posterior
            )

        return log_prob_proposal_posterior


class NonAtomicGaussianLoss:
    """Non-atomic loss for NPE-C (analytical MoG)."""

    uses_only_latest_round: bool = True

    def __init__(
        self,
        neural_net: MixtureDensityEstimator,
        maybe_z_scored_prior: Distribution,
        prec_m_prod_prior: Optional[Tensor] = None,
        z_score_theta: bool = False,
    ):
        self._neural_net = neural_net
        self._maybe_z_scored_prior = maybe_z_scored_prior
        self.prec_m_prod_prior = prec_m_prod_prior
        self.z_score_theta = z_score_theta

    def __call__(
        self,
        theta: Tensor,
        x: Tensor,
        masks: Tensor,
        proposal: DirectPosterior,
        **kwargs,
    ) -> Tensor:
        """Return log-probability of the proposal posterior for MoG proposal.

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
            masks: Mask that is True for prior samples in the batch.
            proposal: Proposal distribution.
            **kwargs: Extra arguments.

        Returns:
            Log-probability of the proposal posterior.
        """
        # Get the proposal MoG at the default_x
        assert isinstance(proposal.posterior_estimator, MixtureDensityEstimator), (
            "Proposal posterior_estimator must be MixtureDensityEstimator for "
            "non-atomic loss."
        )
        assert proposal.default_x is not None, "Proposal must have default_x set"
        mog_p = proposal.posterior_estimator.get_uncorrected_mog(proposal.default_x)
        norm_logits_p = mog_p.log_weights  # Already normalized
        m_p = mog_p.means
        prec_p = mog_p.precisions

        # Get the density estimator MoG at the training data x
        mog_d = self._neural_net.get_uncorrected_mog(x)
        norm_logits_d = mog_d.log_weights  # Already normalized
        m_d = mog_d.means
        prec_d = mog_d.precisions

        # z-score theta if it z-scoring had been requested.
        if self.z_score_theta:
            theta = self._neural_net._transform_input(theta)

        # Compute the MoG parameters of the proposal posterior.
        (
            logits_pp,
            m_pp,
            prec_pp,
            cov_pp,
        ) = self._automatic_posterior_transformation(
            norm_logits_p, m_p, prec_p, norm_logits_d, m_d, prec_d
        )

        # Create MoG for proposal posterior and compute log_prob
        # We need precision_factors for MoG, compute via Cholesky
        precf_pp = torch.linalg.cholesky(prec_pp, upper=True)
        mog_pp = MoG(
            logits=logits_pp,
            means=m_pp,
            precisions=prec_pp,
            precision_factors=precf_pp,
        )

        # Compute the log_prob of theta under the product.
        log_prob_proposal_posterior = mog_pp.log_prob(theta)
        assert_all_finite(
            log_prob_proposal_posterior,
            """the evaluation of the MoG proposal posterior. This is likely due to a
            numerical instability in the training procedure. Please create an issue on
            Github.""",
        )

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
        r"""Returns the MoG parameters of the proposal posterior.

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
            covariances_pp, means_p, precisions_p, means_d, precisions_d
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
        self, precisions_p: Tensor, precisions_d: Tensor
    ):
        """Return the precisions and covariances of the proposal posterior."""
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
        """Return the means of the proposal posterior."""
        num_comps_p = precisions_p.shape[1]
        num_comps_d = precisions_d.shape[1]

        # First, compute the product P_i * m_i and P_j * m_j
        prec_m_prod_p = batched_mixture_mv(precisions_p, means_p)
        prec_m_prod_d = batched_mixture_mv(precisions_d, means_d)

        # Repeat them to allow for matrix operations
        prec_m_prod_p_rep = prec_m_prod_p.repeat_interleave(num_comps_d, dim=1)
        prec_m_prod_d_rep = prec_m_prod_d.repeat(1, num_comps_p, 1)

        # Means = C_ij * (P_i * m_i + P_x * m_x - P_o * m_o).
        summed_cov_m_prod_rep = prec_m_prod_p_rep + prec_m_prod_d_rep

        if self.prec_m_prod_prior is not None:
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
        """Return the component weights (i.e. logits) of the proposal posterior."""
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

        # Repeat the proposal and density estimator terms
        logdet_covariances_p_rep = logdet_covariances_p.repeat_interleave(
            num_comps_d, dim=1
        )
        logdet_covariances_d_rep = logdet_covariances_d.repeat(1, num_comps_p)

        log_sqrt_det_ratio = 0.5 * (
            logdet_covariances_pp
            - (logdet_covariances_p_rep + logdet_covariances_d_rep)
        )

        # Compute for proposal, density estimator, and proposal posterior:
        exponent_p = batched_mixture_vmv(precisions_p, means_p)
        exponent_d = batched_mixture_vmv(precisions_d, means_d)
        exponent_pp = batched_mixture_vmv(precisions_pp, means_pp)

        # Extend proposal and density estimator exponents
        exponent_p_rep = exponent_p.repeat_interleave(num_comps_d, dim=1)
        exponent_d_rep = exponent_d.repeat(1, num_comps_p)
        exponent = -0.5 * (exponent_p_rep + exponent_d_rep - exponent_pp)

        logits_pp = logit_factors + log_sqrt_det_ratio + exponent

        return logits_pp


class ImportanceWeightedLoss:
    """Importance-weighted loss for NPE-B."""

    uses_only_latest_round: bool = False

    def __init__(
        self,
        neural_net: ConditionalDensityEstimator,
        prior: Distribution,
        round_idx: int,
        theta_roundwise: List[Tensor],
        proposal_roundwise: List[Any],
    ):
        self._neural_net = neural_net
        self._prior = prior
        self._round_idx = round_idx
        self._theta_roundwise = theta_roundwise
        self._proposal_roundwise = proposal_roundwise

    def __call__(
        self,
        theta: Tensor,
        x: Tensor,
        masks: Tensor,
        proposal: Optional[Any],
        **kwargs,
    ) -> Tensor:
        """
        Return importance-weighted log probability (Lueckmann, Goncalves et al., 2017).

        Args:
            theta: Batch of parameters θ.
            x: Batch of data.
            masks: Mask that is True for prior samples in the batch in order to train
                them with prior loss.
            proposal: Proposal distribution.
            **kwargs: Extra arguments.

        Returns:
            Importance-weighted log-probability of the proposal posterior.
        """
        # Evaluate prior
        # we accept prior log prob to be -Inf at theta
        # meaning that theta is out of the prior range (the weight is thus 0)
        utils.assert_not_nan_or_plus_inf(
            self._prior.log_prob(theta), "prior log probs of proposal samples"
        )
        prior = torch.exp(self._prior.log_prob(theta))

        # Evaluate proposal
        # (as theta comes from prior and proposal from previous rounds,
        # the last proposal is actually a mixture of the prior
        # and of all the previous proposals with coefficients representing
        # the proportion of the new theta added at each round)
        prop = torch.zeros(self._round_idx + 1, device=theta.device)
        nb_samples = 0  # total number of theta from all the rounds

        for k in range(self._round_idx + 1):
            nb_samples += self._theta_roundwise[k].size(0)
            # the number of new theta sampled in the round k
            prop[k] = self._theta_roundwise[k].size(0)

        prop /= nb_samples
        log_prop = torch.log(prop).repeat(theta.size(0), 1)

        log_previous_proposals = torch.zeros(
            (theta.size(0), self._round_idx + 1), device=theta.device
        )
        for k, density in enumerate(self._proposal_roundwise):
            # we accept the k th proposal log prob to be -Inf at theta
            # meaning that theta is out of the k th proposal range
            log_previous_proposals[:, k] = density.log_prob(theta)
            utils.assert_not_nan_or_plus_inf(
                log_previous_proposals[:, k], "proposal log probs of proposal samples"
            )

        log_proposal = torch.logsumexp(log_prop + log_previous_proposals, dim=1)
        proposal_weighted = torch.exp(log_proposal)

        # Construct the importance weights and normalize them
        importance_weights = prior / proposal_weighted
        importance_weights /= importance_weights.sum()

        theta = reshape_to_sample_batch_event(theta, theta.shape[1:])
        # Reshape the density estimator log probs
        # from (sample_shape, batch_shape) to (batch_shape)
        posterior_log_probs = self._neural_net.log_prob(theta, x).squeeze(dim=0)

        return importance_weights * posterior_log_probs
