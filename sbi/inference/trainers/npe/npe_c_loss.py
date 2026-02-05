# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

from abc import ABC, abstractmethod
from typing import Any, Tuple

import torch
from pyknos.mdn.mdn import MultivariateGaussianMDN as mdn
from torch import Tensor, eye, ones

from sbi.inference.posteriors.direct_posterior import DirectPosterior
from sbi.neural_nets.estimators.base import ConditionalDensityEstimator
from sbi.utils import (
    batched_mixture_mv,
    batched_mixture_vmv,
    clamp_and_warn,
    repeat_rows,
)
from sbi.utils.sbiutils import mog_log_prob
from sbi.utils.torchutils import assert_all_finite, reshape_to_batch_event, reshape_to_sample_batch_event


class NPELoss(ABC):
    """Abstract base class for NPE-C loss strategies."""

    def __init__(self, neural_net: ConditionalDensityEstimator):
        self._neural_net = neural_net

    @abstractmethod
    def __call__(
        self,
        theta: Tensor,
        x: Tensor,
        masks: Tensor,
        proposal: Any,
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
        raise NotImplementedError


class AtomicLoss(NPELoss):
    """Atomic loss for NPE-C (sample-based)."""

    def __init__(
        self,
        neural_net: ConditionalDensityEstimator,
        prior: Any,
        num_atoms: int = 10,
        use_combined_loss: bool = False,
    ):
        super().__init__(neural_net)
        self._prior = prior
        self._num_atoms = num_atoms
        self._use_combined_loss = use_combined_loss

    def __call__(
        self,
        theta: Tensor,
        x: Tensor,
        masks: Tensor,
        proposal: Any,
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
        """
        batch_size = theta.shape[0]

        num_atoms = int(
            clamp_and_warn("num_atoms", self._num_atoms, min_val=2, max_val=batch_size)
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

        # XXX This evaluates the posterior on _all_ prior samples
        if self._use_combined_loss:
            theta = reshape_to_sample_batch_event(theta, self._neural_net.input_shape)
            x = reshape_to_batch_event(x, self._neural_net.condition_shape)
            log_prob_posterior_non_atomic = self._neural_net.log_prob(theta, x)
            # squeeze to remove sample dimension, which is always one during the loss
            # evaluation of `SNPE_C` (because we have one theta vector per x vector).
            log_prob_posterior_non_atomic = log_prob_posterior_non_atomic.squeeze(dim=0)
            masks = masks.reshape(-1)
            log_prob_proposal_posterior = (
                masks * log_prob_posterior_non_atomic + log_prob_proposal_posterior
            )

        return log_prob_proposal_posterior


class NonAtomicGaussianLoss(NPELoss):
    """Non-atomic loss for NPE-C (analytical MoG)."""

    def __init__(
        self,
        neural_net: ConditionalDensityEstimator,
        maybe_z_scored_prior: Any,
        prec_m_prod_prior: Tensor = None,
        z_score_theta: bool = False,
    ):
        super().__init__(neural_net)
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
        """
        # Evaluate the proposal. MDNs do not have functionality to run the embedding_net
        # and then get the mixture_components (**without** calling log_prob()). Hence,
        # we call them separately here.
        encoded_x = proposal.posterior_estimator.net._embedding_net(proposal.default_x)
        dist = (
            proposal.posterior_estimator.net._distribution
        )  # defined to avoid ugly black formatting.
        logits_p, m_p, prec_p, _, _ = dist.get_mixture_components(encoded_x)
        norm_logits_p = logits_p - torch.logsumexp(logits_p, dim=-1, keepdim=True)

        # Evaluate the density estimator.
        encoded_x = self._neural_net.net._embedding_net(x)
        dist = self._neural_net.net._distribution  # defined to avoid black formatting.
        logits_d, m_d, prec_d, _, _ = dist.get_mixture_components(encoded_x)
        norm_logits_d = logits_d - torch.logsumexp(logits_d, dim=-1, keepdim=True)

        # z-score theta if it z-scoring had been requested.
        theta = self._maybe_z_score_theta(theta)

        # Compute the MoG parameters of the proposal posterior.
        (
            logits_pp,
            m_pp,
            prec_pp,
            cov_pp,
        ) = self._automatic_posterior_transformation(
            norm_logits_p, m_p, prec_p, norm_logits_d, m_d, prec_d
        )

        # Compute the log_prob of theta under the product.
        log_prob_proposal_posterior = mog_log_prob(theta, logits_pp, m_pp, prec_pp)
        assert_all_finite(
            log_prob_proposal_posterior,
            """the evaluation of the MoG proposal posterior. This is likely due to a
            numerical instability in the training procedure. Please create an issue on
            Github.""",
        )

        return log_prob_proposal_posterior

    def _maybe_z_score_theta(self, theta: Tensor) -> Tensor:
        """Return potentially standardized theta if z-scoring was requested."""
        if self.z_score_theta:
            theta, _ = self._neural_net.net._transform(theta)
        return theta

    def _automatic_posterior_transformation(
        self,
        logits_p: Tensor,
        means_p: Tensor,
        precisions_p: Tensor,
        logits_d: Tensor,
        means_d: Tensor,
        precisions_d: Tensor,
    ):
        """Returns the MoG parameters of the proposal posterior."""

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
        # Note: self._maybe_z_scored_prior is an instance attribute in the main class logic,
        # but here we pass it in __init__.
        # We need to check if it's MultivariateNormal for the subtraction.
        # This was handled in the main class by logic.
        if hasattr(self._maybe_z_scored_prior, "precision_matrix"):
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
        # mu_i.T * P_i * mu_i
        exponent_p = batched_mixture_vmv(precisions_p, means_p)
        exponent_d = batched_mixture_vmv(precisions_d, means_d)
        exponent_pp = batched_mixture_vmv(precisions_pp, means_pp)

        # Extend proposal and density estimator exponents
        exponent_p_rep = exponent_p.repeat_interleave(num_comps_d, dim=1)
        exponent_d_rep = exponent_d.repeat(1, num_comps_p)
        exponent = -0.5 * (exponent_p_rep + exponent_d_rep - exponent_pp)

        logits_pp = logit_factors + log_sqrt_det_ratio + exponent

        return logits_pp
