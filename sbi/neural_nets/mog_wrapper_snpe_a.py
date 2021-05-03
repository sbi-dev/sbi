# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Affero General Public License v3, see <https://www.gnu.org/licenses/>.

from typing import Union
from warnings import warn

import torch
from pyknos.mdn.mdn import MultivariateGaussianMDN
from pyknos.nflows import flows
from pyknos.nflows.transforms import CompositeTransform
from torch import Tensor
from torch.distributions import MultivariateNormal

import sbi.utils as utils
import sbi.utils.sbiutils
from sbi.utils import torchutils


class MoGWrapper_SNPE_A(flows.Flow):
    """
    A wrapper for nflow's `Flow` class to enable a different log prob calculation
    sampling strategy for training and testing, tailored to SNPE-A [1]

    [1] _Fast epsilon-free Inference of Simulation Models with Bayesian Conditional
        Density Estimation_, Papamakarios et al., NeurIPS 2016,
        https://arxiv.org/abs/1605.06376.
    [2] _Automatic Posterior Transformation for Likelihood-free Inference_,
        Greenberg et al., ICML 2019, https://arxiv.org/abs/1905.07488.
    """

    def __init__(
        self,
        transform,
        distribution,
        embedding_net=None,
        allow_precision_correction: bool = False,
    ):
        """Constructor.

        Args:
            transform: A `Transform` object, it transforms data into noise.
            distribution: A `Distribution` object, the base distribution of the flow that
                generates the noise.
            embedding_net: A `nn.Module` which has trainable parameters to encode the
                context (condition). It is trained jointly with the flow.
            allow_precision_correction:
                Add a diagonal with the smallest eigenvalue in every entry in case
                the precision matrix becomes ill-conditioned.
        """
        # Construct the flow.
        super().__init__(transform, distribution, embedding_net)

        self._proposal = None
        self._allow_precision_correction = allow_precision_correction

    @property
    def proposal(
        self,
    ) -> Union["utils.BoxUniform", MultivariateNormal, "MoGWrapper_SNPE_A"]:
        """Get the proposal of the previous round."""
        return self._proposal

    def set_proposal(
        self,
        proposal: Union["utils.BoxUniform", MultivariateNormal, "MoGWrapper_SNPE_A"],
    ):
        """Set the proposal of the previous round."""
        self._proposal = proposal

        # Take care of z-scoring, pre-compute and store prior terms.
        self._set_state_for_mog_proposal()

    def _get_first_prior_from_proposal(
        self,
    ) -> Union["utils.BoxUniform", MultivariateNormal, "MoGWrapper_SNPE_A"]:
        """Iterate a possible chain of proposals."""
        curr_prior = self._proposal

        while curr_prior:
            if isinstance(curr_prior, (utils.BoxUniform, MultivariateNormal)):
                break
            else:
                curr_prior = curr_prior.proposal

        assert curr_prior is not None
        return curr_prior

    def log_prob(self, inputs, context=None):
        if self._proposal is None:
            # Use Flow.lob_prob() if there has been no previous proposal memorized
            # in this instance. This is the case if we are in the training
            # loop, i.e. this MoGWrapper_SNPE_A instance is not an attribute of a
            # DirectPosterior instance.
            return super().log_prob(inputs, context)  # q_phi from eq (3) in [1]

        elif isinstance(self._proposal, (utils.BoxUniform, MultivariateNormal)):
            # No importance re-weighting is needed if the proposal prior is the prior
            return super().log_prob(inputs, context)

        else:
            # When we want to compute the approx. posterior, a proposal prior \tilde{p}
            # has already been observed. To analytically calculate the log-prob of the
            # Gaussian, we first need to compute the mixture components.

            # Compute the mixture components of the proposal posterior.
            logits_pp, m_pp, prec_pp = self._get_mixture_components(context)

            # z-score theta if it z-scoring had been requested.
            theta = self._maybe_z_score_theta(inputs)

            # Compute the log_prob of theta under the product.
            log_prob_proposal_posterior = sbi.utils.sbiutils.mog_log_prob(
                theta,
                logits_pp,
                m_pp,
                prec_pp,
            )
            utils.assert_all_finite(
                log_prob_proposal_posterior, "proposal posterior eval"
            )
            return log_prob_proposal_posterior  # \hat{p} from eq (3) in [1]

    def sample(self, num_samples, context=None, batch_size=None) -> Tensor:
        if self._proposal is None:
            # Use Flow.sample() if there has been no previous proposal memorized
            # in this instance. This is the case if we are in the training
            # loop, i.e. this MoGWrapper_SNPE_A instance is not an attribute of a
            # DirectPosterior instance.
            return super().sample(num_samples, context, batch_size)

        else:
            # No importance re-weighting is needed if the proposal prior is the prior
            if isinstance(self._proposal, (utils.BoxUniform, MultivariateNormal)):
                return super().sample(num_samples, context, batch_size)

            # When we want to sample from the approx. posterior, a proposal prior \tilde{p}
            # has already been observed. To analytically calculate the log-prob of the
            # Gaussian, we first need to compute the mixture components.
            return self._sample_approx_posterior_mog(num_samples, context, batch_size)

    def _sample_approx_posterior_mog(
        self, num_samples, x: Tensor, batch_size: int
    ) -> Tensor:
        r"""
        Sample from the approximate posterior.

        Args:
            num_samples: Desired number of samples.
            x: Conditioning context for posterior $p(\theta|x)$.
            batch_size: Batch size for sampling.

        Returns:
            Samples from the approximate mixture of Gaussians posterior.
        """

        # Compute the mixture components of the proposal posterior.
        logits_pp, m_pp, prec_pp = self._get_mixture_components(x)

        # Compute the precision factors which represent the upper triangular matrix
        # of the cholesky decomposition of the prec_pp.
        prec_factors_pp = torch.cholesky(prec_pp, upper=True)

        assert logits_pp.ndim == 2
        assert m_pp.ndim == 3
        assert prec_pp.ndim == 4
        assert prec_factors_pp.ndim == 4

        # Replicate to use batched sampling from pyknos.
        if batch_size is not None and batch_size > 1:
            logits_pp = logits_pp.repeat(batch_size, 1)
            m_pp = m_pp.repeat(batch_size, 1, 1)
            prec_factors_pp = prec_factors_pp.repeat(batch_size, 1, 1, 1)

        # Get (optionally z-scored) MoG samples.
        theta = MultivariateGaussianMDN.sample_mog(
            num_samples, logits_pp, m_pp, prec_factors_pp
        )

        embedded_context = self._embedding_net(x)
        if embedded_context is not None:
            # Merge the context dimension with sample dimension in order to
            # apply the transform.
            theta = torchutils.merge_leading_dims(theta, num_dims=2)
            embedded_context = torchutils.repeat_rows(
                embedded_context, num_reps=num_samples
            )

        theta, _ = self._transform.inverse(theta, context=embedded_context)

        if embedded_context is not None:
            # Split the context dimension from sample dimension.
            theta = torchutils.split_leading_dim(theta, shape=[-1, num_samples])

        return theta

    def _get_mixture_components(self, x: Tensor):
        """
        Compute the mixture components of the posterior given the current density
        estimator and the proposal.

        Args:
            x: Conditioning context for posterior.

        Returns:
            Mixture components of the posterior.
        """

        # Evaluate the density estimator.
        encoded_x = self._embedding_net(x)
        dist = self._distribution  # defined to avoid black formatting.
        logits_d, m_d, prec_d, _, _ = dist.get_mixture_components(encoded_x)
        norm_logits_d = logits_d - torch.logsumexp(logits_d, dim=-1, keepdim=True)

        if isinstance(self._proposal, (utils.BoxUniform, MultivariateNormal)):
            # Uniform prior is uninformative.
            return norm_logits_d, m_d, prec_d

        else:
            # Recursive ask for the mixture components until the prior is yielded.
            logits_p, m_p, prec_p = self._proposal._get_mixture_components(x)

        # Compute the MoG parameters of the proposal posterior.
        logits_pp, m_pp, prec_pp, cov_pp = self._proposal_posterior_transformation(
            logits_p,
            m_p,
            prec_p,
            norm_logits_d,
            m_d,
            prec_d,
        )
        return logits_pp, m_pp, prec_pp

    def _proposal_posterior_transformation(
        self,
        logits_pprior: Tensor,
        means_pprior: Tensor,
        precisions_pprior: Tensor,
        logits_d: Tensor,
        means_d: Tensor,
        precisions_d: Tensor,
    ):
        r"""
        Transforms the proposal posterior (the MDN) into the posterior.

        The proposal posterior is:
        $p(\theta|x) = 1/Z * q(\theta|x) * p(\theta) / prop(\theta)$
        In words: posterior = proposal posterior estimate * prior / proposal.

        Since the proposal posterior estimate and the proposal are MoG, and the
        prior is either Gaussian or uniform, we can solve this in closed-form.

        This function implements Appendix C from [1], and is highly similar to
        `SNPE_C._automatic_posterior_transformation()`.

        We have to build L*K components. How do we do this?
        Example: proposal has two components, density estimator has three components.
        Let's call the two components of the proposal i,j and the three components
        of the density estimator x,y,z. We have to multiply every component of the
        proposal with every component of the density estimator. So, what we do is:
        1) for the proposal, build: i,i,i,j,j,j. Done with torch.repeat_interleave()
        2) for the density estimator, build: x,y,z,x,y,z. Done with torch.repeat()
        3) Multiply them with simple matrix operations.

        Args:
            logits_pprior: Component weight of each Gaussian of the proposal prior.
            means_pprior: Mean of each Gaussian of the proposal prior.
            precisions_pprior: Precision matrix of each Gaussian of the proposal prior.
            logits_d: Component weight for each Gaussian of the density estimator.
            means_d: Mean of each Gaussian of the density estimator.
            precisions_d: Precision matrix of each Gaussian of the density estimator.

        Returns: (Component weight, mean, precision matrix, covariance matrix) of each
            Gaussian of the proposal posterior. Has L*K terms (proposal has L terms,
            density estimator has K terms).
        """

        precisions_post, covariances_post = self._precisions_posterior(
            precisions_pprior, precisions_d
        )

        means_post = self._means_posterior(
            covariances_post,
            means_pprior,
            precisions_pprior,
            means_d,
            precisions_d,
        )

        logits_post = MoGWrapper_SNPE_A._logits_posterior(
            means_post,
            precisions_post,
            covariances_post,
            logits_pprior,
            means_pprior,
            precisions_pprior,
            logits_d,
            means_d,
            precisions_d,
        )

        return logits_post, means_post, precisions_post, covariances_post

    def _set_state_for_mog_proposal(self) -> None:
        """
        Set state variables of the MoGWrapper_SNPE_A instance evevy time `set_proposal()`
        is called, i.e. every time a posterior is build using `SNPE_A.build_posterior()`.

        This function is almost identical to `SNPE_C._set_state_for_mog_proposal()`.

        Three things are computed:
        1) Check if z-scoring was requested. To do so, we check if the `_transform`
            argument of the net had been a `CompositeTransform`. See pyknos mdn.py.
        2) Define a (potentially standardized) prior. It's standardized if z-scoring
            had been requested.
        3) Compute (Precision * mean) for the prior. This quantity is used at every
            training step if the prior is Gaussian.
        """

        self.z_score_theta = isinstance(self._transform, CompositeTransform)

        self._set_maybe_z_scored_prior()

        if isinstance(self._maybe_z_scored_prior, MultivariateNormal):
            self.prec_m_prod_prior = torch.mv(
                self._maybe_z_scored_prior.precision_matrix,
                self._maybe_z_scored_prior.loc,
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
        prior = self._get_first_prior_from_proposal()

        if self.z_score_theta:
            scale = self._transform._transforms[0]._scale
            shift = self._transform._transforms[0]._shift

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
            almost_zero_mean = (prior.mean - estim_prior_mean) / estim_prior_std
            almost_one_std = torch.sqrt(prior.variance) / estim_prior_std

            if isinstance(prior, MultivariateNormal):
                self._maybe_z_scored_prior = MultivariateNormal(
                    almost_zero_mean, torch.diag(almost_one_std)
                )
            else:
                range_ = torch.sqrt(almost_one_std * 3.0)
                self._maybe_z_scored_prior = utils.BoxUniform(
                    almost_zero_mean - range_, almost_zero_mean + range_
                )
        else:
            self._maybe_z_scored_prior = prior

    def _maybe_z_score_theta(self, theta: Tensor) -> Tensor:
        """Return potentially standardized theta if z-scoring was requested."""

        if self.z_score_theta:
            theta, _ = self._transform(theta)

        return theta

    def _precisions_posterior(self, precisions_pprior: Tensor, precisions_d: Tensor):
        r"""
        Return the precisions and covariances of the MoG posterior.

        As described at the end of Appendix C in [1], it can happen that the
        proposal's precision matrix is not positive definite.

        $S_k^\prime = ( S_k^{-1} - S_0^{-1} )^{-1}$
        (see eq (23) in Appendix C of [1])

        Args:
            precisions_pprior: Precision matrices of the proposal prior.
            precisions_d: Precision matrices of the density estimator.

        Returns: (Precisions, Covariances) of the MoG posterior. L*K terms.
        """

        num_comps_p = precisions_pprior.shape[1]
        num_comps_d = precisions_d.shape[1]

        # Check if precision matrices are positive definite.
        for batches in precisions_pprior:
            for pprior in batches:
                eig_pprior = torch.symeig(pprior, eigenvectors=False).eigenvalues
                assert (
                    eig_pprior > 0
                ).all(), (
                    "The precision matrix of the proposal is not positive definite!"
                )
        for batches in precisions_d:
            for d in batches:
                eig_d = torch.symeig(d, eigenvectors=False).eigenvalues
                assert (
                    eig_d > 0
                ).all(), "The precision matrix of the density estimator is not positive definite!"

        precisions_pprior_rep = precisions_pprior.repeat_interleave(num_comps_d, dim=1)
        precisions_d_rep = precisions_d.repeat(1, num_comps_p, 1, 1)

        precisions_p = precisions_d_rep - precisions_pprior_rep
        if isinstance(self._maybe_z_scored_prior, MultivariateNormal):
            precisions_p += self._maybe_z_scored_prior.precision_matrix

        # Check if precision matrix is positive definite.
        for idx_batch, batches in enumerate(precisions_p):
            for idx_comp, pp in enumerate(batches):
                eig_pp = torch.symeig(pp, eigenvectors=False).eigenvalues
                if not (eig_pp > 0).all():
                    if self._allow_precision_correction:
                        # Shift the eigenvalues to be at minimum 1e-6.
                        precisions_p[idx_batch, idx_comp] = pp - torch.eye(
                            pp.shape[0]
                        ) * (min(eig_pp) - 1e-6)
                        warn(
                            "The precision matrix of a posterior has not been positive "
                            "definite at least once. Added diagonal entries with the "
                            "smallest eigenvalue to 1e-6."
                        )

                    else:
                        # Fail when encountering an ill-conditioned precision matrix.
                        raise AssertionError(
                            "The precision matrix of a posterior is not positive definite! "
                            "This is a known issue for SNPE-A. Either try a different parameter "
                            "setting or pass `allow_precision_correction=True` when constructing "
                            "the `MoGWrapper_SNPE_A` density estimator."
                        )

        covariances_p = torch.inverse(precisions_p)
        return precisions_p, covariances_p

    def _means_posterior(
        self,
        covariances_post: Tensor,
        means_pprior: Tensor,
        precisions_pprior: Tensor,
        means_d: Tensor,
        precisions_d: Tensor,
    ):
        r"""
        Return the means of the MoG posterior.

        $m_k^\prime = S_k^\prime ( S_k^{-1} m_k - S_0^{-1} m_0 )$
        (see eq (24) in Appendix C of [1])

        Args:
            covariances_post: Covariance matrices of the MoG posterior.
            means_pprior: Means of the proposal prior.
            precisions_pprior: Precision matrices of the proposal prior.
            means_d: Means of the density estimator.
            precisions_d: Precision matrices of the density estimator.

        Returns: Means of the MoG posterior. L*K terms.
        """

        num_comps_pprior = precisions_pprior.shape[1]
        num_comps_d = precisions_d.shape[1]

        # Compute the products P_k * m_k and P_0 * m_0.
        prec_m_prod_pprior = utils.batched_mixture_mv(precisions_pprior, means_pprior)
        prec_m_prod_d = utils.batched_mixture_mv(precisions_d, means_d)

        # Repeat them to allow for matrix operations: same trick as for the precisions.
        prec_m_prod_pprior_rep = prec_m_prod_pprior.repeat_interleave(
            num_comps_d, dim=1
        )
        prec_m_prod_d_rep = prec_m_prod_d.repeat(1, num_comps_pprior, 1)

        # Compute the means P_k^prime * (P_k * m_k - P_0 * m_0).
        summed_cov_m_prod_rep = prec_m_prod_d_rep - prec_m_prod_pprior_rep
        if isinstance(self._maybe_z_scored_prior, MultivariateNormal):
            summed_cov_m_prod_rep += self.prec_m_prod_prior

        means_p = utils.batched_mixture_mv(covariances_post, summed_cov_m_prod_rep)
        return means_p

    @staticmethod
    def _logits_posterior(
        means_post: Tensor,
        precisions_post: Tensor,
        covariances_post: Tensor,
        logits_pprior: Tensor,
        means_pprior: Tensor,
        precisions_pprior: Tensor,
        logits_d: Tensor,
        means_d: Tensor,
        precisions_d: Tensor,
    ):
        r"""
        Return the component weights (i.e. logits) of the MoG posterior.

        $\alpha_k^\prime = \frac{ \alpha_k exp(-0.5 c_k) }{ \sum{j} \alpha_j exp(-0.5 c_j) } $
        with
        $c_k = logdet(S_k) - logdet(S_0) - logdet(S_k^\prime) +
             + m_k^T P_k m_k - m_0^T P_0 m_0 - m_k^\prime^T P_k^\prime m_k^\prime$
        (see eqs. (25, 26) in Appendix C of [1])

        Args:
            means_post: Means of the posterior.
            precisions_post: Precision matrices of the posterior.
            covariances_post: Covariance matrices of the posterior.
            logits_pprior: Component weights (i.e. logits) of the proposal prior.
            means_pprior: Means of the proposal prior.
            precisions_pprior: Precision matrices of the proposal prior.
            logits_d: Component weights (i.e. logits) of the density estimator.
            means_d: Means of the density estimator.
            precisions_d: Precision matrices of the density estimator.

        Returns: Component weights of the proposal posterior. L*K terms.
        """

        num_comps_pprior = precisions_pprior.shape[1]
        num_comps_d = precisions_d.shape[1]

        # Compute the ratio of the logits similar to eq (10) in Appendix A.1 of [2]
        logits_pprior_rep = logits_pprior.repeat_interleave(num_comps_d, dim=1)
        logits_d_rep = logits_d.repeat(1, num_comps_pprior)
        logit_factors = logits_d_rep - logits_pprior_rep

        # Compute the log-determinants
        logdet_covariances_post = torch.logdet(covariances_post)
        logdet_covariances_pprior = -torch.logdet(precisions_pprior)
        logdet_covariances_d = -torch.logdet(precisions_d)

        # Repeat the proposal and density estimator terms such that there are LK terms.
        # Same trick as has been used above.
        logdet_covariances_pprior_rep = logdet_covariances_pprior.repeat_interleave(
            num_comps_d, dim=1
        )
        logdet_covariances_d_rep = logdet_covariances_d.repeat(1, num_comps_pprior)

        log_sqrt_det_ratio = 0.5 * (  # similar to eq (14) in Appendix A.1 of [2]
            logdet_covariances_post
            + logdet_covariances_pprior_rep
            - logdet_covariances_d_rep
        )

        # Compute for proposal, density estimator, and proposal posterior:
        exponent_pprior = utils.batched_mixture_vmv(
            precisions_pprior, means_pprior  # m_0 in eq (26) in Appendix C of [1]
        )
        exponent_d = utils.batched_mixture_vmv(
            precisions_d, means_d  # m_k in eq (26) in Appendix C of [1]
        )
        exponent_post = utils.batched_mixture_vmv(
            precisions_post, means_post  # m_k^\prime in eq (26) in Appendix C of [1]
        )

        # Extend proposal and density estimator exponents to get LK terms.
        exponent_prior_rep = exponent_pprior.repeat_interleave(num_comps_d, dim=1)
        exponent_d_rep = exponent_d.repeat(1, num_comps_pprior)
        exponent = -0.5 * (
            exponent_prior_rep - exponent_d_rep - exponent_post  # eq (26) in [1]
        )

        logits_post = logit_factors + log_sqrt_det_ratio + exponent
        return logits_post
