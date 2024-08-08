from __future__ import annotations

from copy import deepcopy
from typing import Any, List, Optional, Tuple

import torch
import torch.nn as nn
from pyknos.mdn.mdn import MultivariateGaussianMDN as mdn
from torch import Tensor
from torch.distributions import MultivariateNormal

from sbi.neural_nets.density_estimators import DensityEstimator
from sbi.utils import assert_all_finite, mog_log_prob
from sbi.utils.conditional_density_utils import condition_mog
from sbi.utils.sbiutils import batched_mixture_mv, batched_mixture_vmv


class MoG:
    def __init__(
        self,
        logits: Optional[Tensor] = None,
        means: Optional[Tensor] = None,
        precisions: Optional[Tensor] = None,
    ):
        self._logits = logits  # (batch_size, num_components)
        self._means = means  # (batch_size, num_components, input_size)
        self._precisions = (
            precisions  # (batch_size, num_components, input_size, input_size)
        )

    @property
    def parameters(self):
        r"""Return the parameters of the mixture of Gaussians."""
        return self._logits, self._means, self._precisions

    @parameters.setter
    def parameters(self, parameters: Tuple[Tensor, Tensor, Tensor]):
        r"""Set the parameters of the mixture of Gaussians."""
        self._logits, self._means, self._precisions = parameters
        self._logits -= self.logsumexplogits

    @property
    def logsumexplogits(self):
        return torch.logsumexp(self._logits, dim=-1, keepdim=True)

    @property
    def batchdim(self):
        return self._logits.shape[0]

    def reset_parameters(self):
        r"""Reset the parameters of the mixture of Gaussians."""
        self.parameters = (None, None, None)

    def condition(self, condition: Tensor, inplace: bool = True) -> Optional[MoG]:
        # return copy of the MDN with conditioned parameters
        cond_logits, cond_means, cond_precs, _ = condition_mog(
            condition, *self.parameters
        )
        # TODO: CHECK IF THIS IS CORRECT OR cond_logits need to be normalized
        if not inplace:
            return MoG(cond_logits, cond_means, cond_precs)
        self.parameters = (cond_logits, cond_means, cond_precs)

    def marginalize(self, mask: List[bool], inplace: bool = True):
        # return copy of the MDN with marginalized parameters
        logits, means, precs = self.parameters
        marg_logits = logits[:, mask]
        marg_means = means[:, mask]
        marg_precs = precs[:, mask][:, :, mask]
        if not inplace:
            return MoG(marg_logits, marg_means, marg_precs)
        self.parameters = (marg_logits, marg_means, marg_precs)

    def sample(self, sample_shape, batch_size=None) -> Tensor:
        logits_p, m_p, prec_p = self.parameters
        prec_factors_p = torch.linalg.cholesky(prec_p, upper=True)

        num_samples = torch.Size(sample_shape).numel()
        if batch_size is None:
            batch_size = 1 if len(sample_shape) == 1 else sample_shape[0]
        logits_p, m_p, prec_factors_p = self.expand_to(
            (logits_p, m_p, prec_factors_p), batch_size
        )

        theta = mdn.sample_mog(num_samples, logits_p, m_p, prec_factors_p)
        return theta.reshape(*sample_shape, -1)

    @staticmethod
    def expand_to(parameters, batch_dim):
        logits, means, precisions = parameters
        logits = logits.repeat(batch_dim, 1)
        means = means.repeat(batch_dim, 1, 1)
        precisions = precisions.repeat(batch_dim, 1, 1, 1)
        return (logits, means, precisions)

    def log_prob(self, input: Tensor) -> Tensor:
        r"""Return the log probability of the input under the mixture of Gaussians."""
        input_batch_dim = 1 if input.ndim == 1 else input.shape[0]
        if input_batch_dim != self.batchdim:
            mog_parameters = self.expand_to(self.parameters, input_batch_dim)
        log_probs = mog_log_prob(input, *mog_parameters)
        assert_all_finite(log_probs, "proposal posterior eval")
        return log_probs


class MixedDensityEstimator(DensityEstimator):
    def __init__(self, mdn: Any, condition_shape=torch.Size) -> None:
        super().__init__(mdn, condition_shape)
        self.mog = MoG()
        self._is_proposal_corrected = False
        self.context = None

    @property
    def embedding_net(self) -> nn.Module:
        r"""Return the embedding network."""
        return self.net._embedding_net

    @property
    def distribution(self):
        r"""Return the distribution of the density estimator."""
        return self.net._distribution

    @property
    def transform(self):
        r"""Return the distribution of the density estimator."""
        return self.net._transform

    def get_mixture_components(self, condition: Optional[Tensor] = None):
        embedded_x = self.embedding_net(condition)
        logits_d, m_d, prec_d = self.distribution.get_mixture_components(embedded_x)[:3]
        return logits_d, m_d, prec_d

    def set_mixture_context(self, context: Tensor, inplace: bool = True):
        mog_parameters = self.get_mixture_components(context)
        self.context = context
        if not inplace:
            mdn = deepcopy(self)
            mdn.mog.parameters = mog_parameters
            return mdn
        self._is_proposal_corrected = False
        self.mog.parameters = mog_parameters

    def correct_mixture(self, proposal: MoG, context: Optional[Tensor] = None):
        context = self.context if context is None else context
        self.set_mixture_context(context, inplace=True)
        self.mog = self._get_corrected_mixture(proposal, self.mog)
        # THIS HAS THE ADVANTAGE THAT IT CAN BE SUPPLIED DIRECTLY WITH MIXTURE PARAMS
        # self.mog.parameters = self._compute_corrected_mixture_parameters(proposal.parameters, self.mog.parameters)
        self._is_proposal_corrected = True

    def sample(
        self,
        sample_shape,
        condition: Optional[Tensor] = None,
        proposal: Optional[MoG] = None,
    ) -> Tensor:
        # mdn sample
        # TODO: add support for sampling the MDN

        # mog sample
        if condition is not None:
            self.set_mixture_context(condition, inplace=True)
        if proposal is not None:
            self.correct_mixture(proposal)
        theta = self.mog.sample(sample_shape)
        return self.transform.inverse(theta)[0]

    def log_prob(
        self,
        input: Tensor,
        condition: Optional[Tensor] = None,
        proposal: Optional[MoG] = None,
    ) -> Tensor:
        # mdn log_prob
        # TODO: add support for evaluating the MDN

        # mog log_prob
        if condition is not None:
            self.set_mixture_context(condition, inplace=True)
        if proposal is not None:
            self.correct_mixture(proposal)
        input, logsumdiag = self.transform(input)
        return self.mog.log_prob(input) + logsumdiag

    def loss(self, input: Tensor, condition: Tensor) -> Tensor:
        r"""Return the loss for training the density estimator.

        Args:
            input: Inputs to evaluate the loss on of shape (batch_size, input_size).
            condition: Conditions of shape (batch_size, *condition_shape).

        Returns:
            Negative log_probability (batch_size,)
        """
        return -self.log_prob(input, condition)

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

        # Check if precision matrix is positive definite.
        for _, batches in enumerate(precisions_p):
            for _, pp in enumerate(batches):
                eig_pp = torch.linalg.eigvalsh(pp, UPLO="U")
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
        prec_m_prod_pp = batched_mixture_mv(precisions_pp, means_pp)
        prec_m_prod_d = batched_mixture_mv(precisions_d, means_d)

        # Repeat them to allow for matrix operations: same trick as for the precisions.
        prec_m_prod_pp_rep = prec_m_prod_pp.repeat_interleave(num_comps_d, dim=1)
        prec_m_prod_d_rep = prec_m_prod_d.repeat(1, num_comps_pp, 1)

        # Compute the means P_k^prime * (P_k * m_k - P_0 * m_0).
        summed_cov_m_prod_rep = prec_m_prod_d_rep - prec_m_prod_pp_rep
        if isinstance(self._maybe_z_scored_prior, MultivariateNormal):
            summed_cov_m_prod_rep += self.prec_m_prod_prior

        means_p = batched_mixture_mv(covariances_p, summed_cov_m_prod_rep)
        return means_p

    def _logits_posterior(
        self,
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
        exponent_pp = batched_mixture_vmv(
            precisions_pp,
            means_pp,  # m_0 in eq (26) in Appendix C of [1]
        )
        exponent_d = batched_mixture_vmv(
            precisions_d,
            means_d,  # m_k in eq (26) in Appendix C of [1]
        )
        exponent_post = batched_mixture_vmv(
            precisions_post,
            means_post,  # m_k^\prime in eq (26) in Appendix C of [1]
        )

        # Extend proposal and density estimator exponents to get LK terms.
        exponent_pp_rep = exponent_pp.repeat_interleave(num_comps_d, dim=1)
        exponent_d_rep = exponent_d.repeat(1, num_comps_pp)
        exponent = -0.5 * (
            exponent_d_rep - exponent_pp_rep - exponent_post  # eq (26) in [1]
        )

        logits_post = logit_factors + log_sqrt_det_ratio + exponent
        return logits_post

    def _get_corrected_mixture(self, mog: MoG, proposal: MoG) -> MoG:
        logits_d, means_d, precs_d = mog.parameters
        logits_pp, means_pp, precs_pp = proposal.parameters

        precisions_post, covariances_post = self._precisions_posterior(
            precs_pp, precs_d
        )
        means_post = self._means_posterior(
            covariances_post, means_pp, precs_pp, means_d, precs_d
        )

        logits_post = self._logits_posterior(
            means_post,
            precisions_post,
            covariances_post,
            logits_pp,
            means_pp,
            precs_pp,
            logits_d,
            means_d,
            precs_d,
        )
        return MoG(logits_post, means_post, precisions_post)
