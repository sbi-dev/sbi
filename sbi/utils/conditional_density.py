# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Affero General Public License v3, see <https://www.gnu.org/licenses/>.


from typing import Any, Callable, List, Optional, Tuple, Union
from warnings import warn

import torch
from torch import Tensor

from sbi.utils.torchutils import ensure_theta_batched

from copy import deepcopy

from sbi.inference.posteriors.direct_posterior import DirectPosterior
from tqdm.auto import tqdm

def eval_conditional_density(
    density: Any,
    condition: Tensor,
    limits: Tensor,
    dim1: int,
    dim2: int,
    resolution: int = 50,
    eps_margins1: Union[Tensor, float] = 1e-32,
    eps_margins2: Union[Tensor, float] = 1e-32,
    warn_about_deprecation: bool = True,
) -> Tensor:
    r"""
    Return the unnormalized conditional along `dim1, dim2` given parameters `condition`.

    We compute the unnormalized conditional by evaluating the joint distribution:
        $p(x1 | x2) = p(x1, x2) / p(x2) \propto p(x1, x2)$

    Args:
        density: Probability density function with `.log_prob()` method.
        condition: Parameter set that all dimensions other than dim1 and dim2 will be
            fixed to. Should be of shape (1, dim_theta), i.e. it could e.g. be
            a sample from the posterior distribution. The entries at `dim1` and `dim2`
            will be ignored.
        limits: Bounds within which to evaluate the density. Shape (dim_theta, 2).
        dim1: First dimension along which to evaluate the conditional.
        dim2: Second dimension along which to evaluate the conditional.
        resolution: Resolution of the grid along which the conditional density is
            evaluated.
        eps_margins1: We will evaluate the posterior along `dim1` at
            `limits[0]+eps_margins` until `limits[1]-eps_margins`. This avoids
            evaluations potentially exactly at the prior bounds.
        eps_margins2: We will evaluate the posterior along `dim2` at
            `limits[0]+eps_margins` until `limits[1]-eps_margins`. This avoids
            evaluations potentially exactly at the prior bounds.
        warn_about_deprecation: With sbi v0.15.0, we depracated the import of this
            function from `sbi.utils`. Instead, it should be imported from
            `sbi.analysis`.

    Returns: Conditional probabilities. If `dim1 == dim2`, this will have shape
        (resolution). If `dim1 != dim2`, it will have shape (resolution, resolution).
    """

    if warn_about_deprecation:
        warn(
            "Importing `eval_conditional_density` from `sbi.utils` is deprecated since "
            "sbi v0.15.0. Instead, use "
            "`from sbi.analysis import eval_conditional_density`."
        )

    condition = ensure_theta_batched(condition)

    theta_grid_dim1 = torch.linspace(
        float(limits[dim1, 0] + eps_margins1),
        float(limits[dim1, 1] - eps_margins1),
        resolution,
    )
    theta_grid_dim2 = torch.linspace(
        float(limits[dim2, 0] + eps_margins2),
        float(limits[dim2, 1] - eps_margins2),
        resolution,
    )

    if dim1 == dim2:
        repeated_condition = condition.repeat(resolution, 1)
        repeated_condition[:, dim1] = theta_grid_dim1

        log_probs_on_grid = density.log_prob(repeated_condition)
    else:
        repeated_condition = condition.repeat(resolution ** 2, 1)
        repeated_condition[:, dim1] = theta_grid_dim1.repeat(resolution)
        repeated_condition[:, dim2] = torch.repeat_interleave(
            theta_grid_dim2, resolution
        )

        log_probs_on_grid = density.log_prob(repeated_condition)
        log_probs_on_grid = torch.reshape(log_probs_on_grid, (resolution, resolution))

    # Subtract maximum for numerical stability.
    return torch.exp(log_probs_on_grid - torch.max(log_probs_on_grid))


def conditional_corrcoeff(
    density: Any,
    limits: Tensor,
    condition: Tensor,
    subset: Optional[List[int]] = None,
    resolution: int = 50,
    warn_about_deprecation: bool = True,
) -> Tensor:
    r"""
    Returns the conditional correlation matrix of a distribution.

    To compute the conditional distribution, we condition all but two parameters to
    values from `condition`, and then compute the Pearson correlation
    coefficient $\rho$ between the remaining two parameters under the distribution
    `density`. We do so for any pair of parameters specified in `subset`, thus
    creating a matrix containing conditional correlations between any pair of
    parameters.

    If `condition` is a batch of conditions, this function computes the conditional
    correlation matrix for each one of them and returns the mean.

    Args:
        density: Probability density function with `.log_prob()` function.
        limits: Limits within which to evaluate the `density`.
        condition: Values to condition the `density` on. If a batch of conditions is
            passed, we compute the conditional correlation matrix for each of them and
            return the average conditional correlation matrix.
        subset: Evaluate the conditional distribution only on a subset of dimensions.
            If `None` this function uses all dimensions.
        resolution: Number of grid points on which the conditional distribution is
            evaluated. A higher value increases the accuracy of the estimated
            correlation but also increases the computational cost.
        warn_about_deprecation: With sbi v0.15.0, we depracated the import of this
            function from `sbi.utils`. Instead, it should be imported from
            `sbi.analysis`.

    Returns: Average conditional correlation matrix of shape either `(num_dim, num_dim)`
    or `(len(subset), len(subset))` if `subset` was specified.
    """

    if warn_about_deprecation:
        warn(
            "Importing `conditional_corrcoeff` from `sbi.utils` is deprecated since "
            "sbi v0.15.0. Instead, use "
            "`from sbi.analysis import conditional_corrcoeff`."
        )

    condition = ensure_theta_batched(condition)

    if subset is None:
        subset = range(condition.shape[1])

    correlation_matrices = []
    for cond in condition:
        correlation_matrices.append(
            torch.stack(
                [
                    _compute_corrcoeff(
                        eval_conditional_density(
                            density,
                            cond,
                            limits,
                            dim1=dim1,
                            dim2=dim2,
                            resolution=resolution,
                            warn_about_deprecation=False,
                        ),
                        limits[[dim1, dim2]],
                    )
                    for dim1 in subset
                    for dim2 in subset
                    if dim1 < dim2
                ]
            )
        )

    average_correlations = torch.mean(torch.stack(correlation_matrices), dim=0)

    # `average_correlations` is still a vector containing the upper triangular entries.
    # Below, assemble them into a matrix:
    av_correlation_matrix = torch.zeros((len(subset), len(subset)))
    triu_indices = torch.triu_indices(row=len(subset), col=len(subset), offset=1)
    av_correlation_matrix[triu_indices[0], triu_indices[1]] = average_correlations

    # Make the matrix symmetric by copying upper diagonal to lower diagonal.
    av_correlation_matrix = torch.triu(av_correlation_matrix) + torch.tril(
        av_correlation_matrix.T
    )

    av_correlation_matrix.fill_diagonal_(1.0)
    return av_correlation_matrix


def _compute_corrcoeff(probs: Tensor, limits: Tensor):
    """
    Given a matrix of probabilities `probs`, return the correlation coefficient.

    Args:
        probs: Matrix of (unnormalized) evaluations of a 2D density.
        limits: Limits within which the entries of the matrix are evenly spaced.

    Returns: Pearson correlation coefficient.
    """

    normalized_probs = _normalize_probs(probs, limits)
    covariance = _compute_covariance(normalized_probs, limits)

    marginal_x, marginal_y = _calc_marginals(normalized_probs, limits)
    variance_x = _compute_covariance(marginal_x, limits[0], lambda x: x ** 2)
    variance_y = _compute_covariance(marginal_y, limits[1], lambda x: x ** 2)

    return covariance / torch.sqrt(variance_x * variance_y)


def _compute_covariance(
    probs: Tensor, limits: Tensor, f: Callable = lambda x, y: x * y
) -> Tensor:
    """
    Return the covariance between two RVs from evaluations of their pdf on a grid.

    The function computes the covariance as:
    Cov(X,Y) = E[X*Y] - E[X] * E[Y]

    In the more general case, when using a different function `f`, it returns:
    E[f(X,Y)] - f(E[X], E[Y])

    By using different function `f`, this function can be also deal with more than two
    dimensions, but this has not been tested.

    Lastly, this function can also compute the variance of a 1D distribution. In that
    case, `probs` will be a vector, and f would be: f = lambda x: x**2:
    Var(X,Y) = E[X**2] - E[X]**2

    Args:
        probs: Matrix of evaluations of a 2D density.
        limits: Limits within which the entries of the matrix are evenly spaced.
        f: The operation to be applied to the expected values, usually just the product.

    Returns: Covariance.
    """

    probs = ensure_theta_batched(probs)
    limits = ensure_theta_batched(limits)

    # Compute E[X*Y].
    expected_value_of_joint = _expected_value_f_of_x(probs, limits, f)

    # Compute E[X] * E[Y].
    expected_values_of_marginals = [
        _expected_value_f_of_x(prob.unsqueeze(0), lim.unsqueeze(0))
        for prob, lim in zip(_calc_marginals(probs, limits), limits)
    ]

    return expected_value_of_joint - f(*expected_values_of_marginals)


def _expected_value_f_of_x(
    probs: Tensor, limits: Tensor, f: Callable = lambda x: x
) -> Tensor:
    """
    Return the expected value of a function of random variable(s) E[f(X_i,...,X_k)].

    The expected value is computed from evaluations of the joint density on an evenly
    spaced grid, passed as `probs`.

    This function can not deal with functions `f` that have multiple outputs. They will
    simply be summed over.

    Args:
        probs: Matrix of evaluations of the density.
        limits: Limits within which the entries of the matrix are evenly spaced.
        f: The operation to be applied to the expected values.

    Returns: Expected value.
    """

    probs = ensure_theta_batched(probs)
    limits = ensure_theta_batched(limits)

    x_values_over_which_we_integrate = [
        torch.linspace(lim[0], lim[1], prob.shape[0])
        for lim, prob in zip(torch.flip(limits, [0]), probs)
    ]  # See #403 and #404 for flip().
    grids = list(torch.meshgrid(x_values_over_which_we_integrate))
    expected_val = torch.sum(f(*grids) * probs)

    limits_diff = torch.prod(limits[:, 1] - limits[:, 0])
    expected_val /= probs.numel() / limits_diff.item()

    return expected_val


def _calc_marginals(
    probs: Tensor, limits: Tensor
) -> Union[Tensor, Tuple[Tensor, Tensor]]:
    """
    Given a 2D matrix of probabilities, return the normalized marginal vectors.

    Args:
        probs: Matrix of evaluations of a 2D density.
        limits: Limits within which the entries of the matrix are evenly spaced.
    """

    if probs.shape[0] > 1:
        # Marginalize and normalize if multi-D distribution.
        marginal_x = torch.sum(probs, dim=0)
        marginal_y = torch.sum(probs, dim=1)

        marginal_x = _normalize_probs(marginal_x, limits[0].unsqueeze(0))
        marginal_y = _normalize_probs(marginal_y, limits[1].unsqueeze(0))
        return marginal_x, marginal_y
    else:
        # Only normalize if already a 1D distribution.
        return _normalize_probs(probs, limits)


def _normalize_probs(probs: Tensor, limits: Tensor) -> Tensor:
    """
    Given a matrix or a vector of probabilities, return the normalized matrix or vector.

    Args:
        probs: Matrix / vector of probabilities.
        limits: Limits within which the entries of the matrix / vector are evenly
            spaced. Must have a batch dimension if probs is a vector.

    Returns: Normalized probabilities.
    """
    limits_diff = torch.prod(limits[:, 1] - limits[:, 0])
    return probs * probs.numel() / limits_diff / torch.sum(probs)


class MDNPosterior(DirectPosterior):
    """Wrapper around MDN based DirectPosterior instances.

    Extracts the Gaussian Mixture parameters from the Mixture
    Density Network. Samples from Multivariate Gaussians directly, using
    torch.distributions.multivariate_normal.MultivariateNormal
    rather than going through the MDN.
    Replaces `.sample()` and `.log_prob() methods of the `DirectPosterior`.

    Args:
        MDN_Posterior: `DirectPosterior` instance, i.e. output of
            `inference.build_posterior(density_estimator)`,
            that was trained using a MDN.

    Attributes:
        S: Tensor that holds the covariance matrices of all mixture components.
        m: Tensor that holds the means of all mixture components.
        mc: Tensor that holds mixture coefficients of all mixture components.
        support: An Interval with lower and upper bounds of the support.
    """

    def __init__(self, MDN_Posterior: DirectPosterior):
        if "MultivariateGaussianMDN" in MDN_Posterior.net.__str__():
            # wrap copy of input object into self
            self.__class__ = type(
                "MDNPosterior", (self.__class__, deepcopy(MDN_Posterior).__class__), {}
            )
            self.__dict__ = deepcopy(MDN_Posterior).__dict__

            # MoG parameters
            self.S = None
            self.m = None
            self.mc = None
            self.support = self._prior.support

            self.extract_mixture_components()

        else:
            raise AttributeError("Posterior does not contain a MDN.")

    @staticmethod
    def mulnormpdf(X: Tensor, mu: Tensor, cov: Tensor) -> Tensor:
        """Evaluates the PDF for the multivariate Guassian distribution.

        Args:
            X: torch.tensor with inputs/entries row-wise. Can also be a 1-d array if only a
                single point is evaluated.
            mu: torch.tensor with center/mean, 1d array.
            cov: 2d torch.tensor with covariance matrix.

        Returns:
            prob: Probabilities for entries in `X`.
        """

        # Evaluate pdf at points or point:
        if X.ndim == 1:
            X = torch.atleast_2d(X)
        sigma = torch.atleast_2d(cov)  # So we also can use it for 1-d distributions

        N = mu.shape[0]
        ex1 = torch.inverse(sigma) @ (X - mu).T
        ex = -0.5 * (X - mu).T * ex1
        if ex.ndim == 2:
            ex = torch.sum(ex, axis=0)
        K = 1 / torch.sqrt(
            torch.pow(2 * torch.tensor(3.14159265), N) * torch.det(sigma)
        )
        return K * torch.exp(ex)

    def check_support(self, X: Tensor) -> bool:
        """Takes a set of points X with X.shape[0] being the number of points
        and X.shape[1] the dimensionality of the points and checks, each point
        for its prior support.

        Args:
            X: Contains a set of multidimensional points to check
                against the prior support of the posterior object.

        Returns:
            within_support: Boolean array representing, whether a sample is within the
                prior support or not.
        """

        lbound = self.support.lower_bound
        ubound = self.support.upper_bound

        within_support = torch.logical_and(lbound < X, X < ubound)

        return torch.all(within_support, dim=1)

    def extract_mixture_components(
        self, x: Tensor = None
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Extracts the Mixture of Gaussians (MoG) parameters
        from the MDN at either the default x or input x.

        Args:
            x: x at which to evaluate the MDN in order
                to extract the MoG parameters.
        """
        if x == None:
            encoded_x = self.net._embedding_net(self.default_x)
        else:
            encoded_x = self.net._embedding_net(torch.tensor(x, dtype=torch.float32))
        dist = self.net._distribution
        logits, m, prec, *_ = dist.get_mixture_components(encoded_x)
        norm_logits = logits - torch.logsumexp(logits, dim=-1, keepdim=True)

        scale = self.net._transform._transforms[0]._scale
        shift = self.net._transform._transforms[0]._shift

        self.mc = torch.exp(norm_logits).detach()
        self.m = ((m - shift) / scale).detach()[0]

        L = torch.cholesky(
            prec[0].detach() + torch.eye(self.m.shape[1]) * 1e-6
        )  # sometimes matrices are not pos semi def. dirty fix.
        C = torch.inverse(L)
        self.S = C.transpose(2, 1) @ C
        A_inv = torch.inverse(scale * torch.eye(self.S.shape[1]))
        self.S = A_inv @ self.S @ A_inv.T

        return self.mc, self.m, self.S

    def log_prob(self, X: Tensor, individual=False) -> Tensor:
        """Evaluates the Mixture of Gaussian (MoG)
        probability density function at a value x.

        Args:
            X: Values at which to evaluate the MoG pdf.
            individual: If True the probability density is returned for each cluster component.

        Returns:
            log_prob: Log probabilities at values specified by X.
        """

        pdf = torch.zeros((X.shape[0], self.m.shape[0]))
        for i in range(self.m.shape[0]):
            pdf[:, i] = self.mulnormpdf(X, self.m[i], self.S[i]) * self.mc[0, i]
        if individual:
            return torch.log(pdf)
        else:
            log_factor = torch.log(self.leakage_correction(x=self.default_x))
            return torch.log(torch.sum(pdf, axis=1)) - log_factor

    def sample(self, sample_shape: Tuple[int, int]) -> Tensor:
        """Draw samples from a Mixture of Gaussians (MoG)

        Adpated from code courtesy of @ybernaerts.
        Args:
            sample_shape: The number of samples to draw from the MoG distribution.

        Returns:
            X: A matrix with samples rows, and input dimension columns.
        """

        K, D = self.m.shape  # Determine dimensionality

        num_samples = torch.Size(sample_shape).numel()
        pbar = tqdm(
            total=num_samples,
            desc=f"Drawing {num_samples} posterior samples",
        )

        # Cluster selection
        cs_mc = torch.cumsum(self.mc, 1)
        cs_mc = torch.hstack((torch.tensor([[0]]), cs_mc))
        sel_idx = torch.rand(num_samples)

        # Draw samples
        res = torch.zeros((num_samples, D))
        f1 = sel_idx[:, None] >= cs_mc
        f2 = sel_idx[:, None] < cs_mc
        idxs = f1[:, :-1] * f2[:, 1:]
        ksamples = torch.sum(idxs, axis=0)

        for k, samplesize in enumerate(ksamples):
            # draw initial samples
            chol_factor = torch.cholesky(self.S[k])
            std_normal_sample = torch.randn(D,samplesize)
            drawn_samples = self.m[k] + torch.mm(chol_factor, std_normal_sample).T

            # check if samples are within the support and how many are not
            supported = self.check_support(drawn_samples)
            num_not_supported = torch.count_nonzero(~supported)
            drawn_samples_in_support = drawn_samples[supported]
            if num_not_supported > 0:
                # resample until all samples are within the prior support
                while num_not_supported > 0:
                    # resample
                    std_normal_sample = torch.randn(D,(int(num_not_supported))
                    redrawn_samples = self.m[k] + torch.mm(chol_factor, std_normal_sample).T

                    # reevaluate support
                    supported = self.check_support(redrawn_samples)
                    num_not_supported = torch.count_nonzero(~supported)
                    redrawn_samples_in_support = redrawn_samples[supported]
                    # append the samples
                    drawn_samples_in_support = torch.vstack(
                        [drawn_samples_in_support, redrawn_samples_in_support]
                    )

                    pbar.update(int(sum(ksamples[: k + 1]) - num_not_supported))
            res[idxs[:, k], :] = drawn_samples_in_support
        pbar.close()
        return res.reshape((*sample_shape, -1))

    def conditionalise(self, condition: Tensor) -> ConditionalMDNPosterior:
        """Instantiates a new conditional distribution, which can be evaluated
        and sampled from.

        Args:
            condition: An array of inputs. Inputs set to NaN are not set, and become inputs to
                the resulting distribution. Order is preserved.
        """
        return ConditionalMDNPosterior(self, condition)

    def sample_from_conditional(
        self, condition: Tensor, sample_shape: Tuple[int, int]
    ) -> Tensor:
        """Conditionalises the distribution on the provided condition
        and samples from the the resulting distribution.

        Args:
            condition: An array of inputs. Inputs set to NaN are not set, and become inputs to
                the resulting distribution. Order is preserved.
        sample_shape: The number of samples to draw from the conditional distribution.
        """
        conditional_posterior = ConditionalMDNPosterior(self, condition)
        samples = cond_posteriori.sample(sample_shape)
        return samples


class ConditionalMDNPosterior(MDNPosterior):
    """Wrapperclass for `DirectPosterior` objects that were trained using
    a Mixture Density Network (MDN) and have been conditionalised.
    Replaces `.sample()`, `.sample_conditional()`, `.sample_with_mcmc()` and `.log_prob()`
    methods. Enables the evaluation and sampling of the conditional
    distribution at any arbitrary condition and point.

    Args:
        MDN_Posterior: `DirectPosterior` instance, i.e. output of
            `inference.build_posterior(density_estimator)`,
            that was trained with a MDN.
        condition: A vector that holds the conditioned vector. Entries that contain
            NaNs are not set and become inputs to the resulting distribution,
            i.e. condition = [x1, x2, NaN, NaN] -> p(x3,x4|x1,x2).

    Attributes:
        condition: A Tensor containing the values which the MoG has been conditioned on.
    """

    def __init__(self, MDN_Posterior: DirectPosterior, condition: Tensor):
        self.__class__ = type(
            "ConditionalMDNPosterior",
            (self.__class__, deepcopy(MDN_Posterior).__class__),
            {},
        )
        self.__dict__ = deepcopy(MDN_Posterior).__dict__
        self.condition = condition
        self.__conditionalise(condition)

    def __conditionalise(self, condition: Tensor):
        """Finds the conditional distribution p(X|Y) for a GMM.

        Args:
            condition: An array of inputs. Inputs set to NaN are not set, and become inputs to
                the resulting distribution. Order is preserved.

        Raises:
            ValueError: The chosen condition is not within the prior support.
        """

        # revert to the old GMM parameters first
        self.extract_mixture_components()
        self.support = self._prior.support

        pop = self.condition.isnan().reshape(-1)
        condition_without_NaNs = self.condition.reshape(-1)[~pop]

        # check whether the condition is within the prior bounds
        cond_ubound = self.support.upper_bound[~pop]
        cond_lbound = self.support.lower_bound[~pop]
        within_support = torch.logical_and(
            cond_lbound <= condition_without_NaNs, condition_without_NaNs <= cond_ubound
        )
        if ~torch.all(within_support):
            raise ValueError("The chosen condition is not within the prior support.")

        # adjust the dimensionality of the support
        self.support.upper_bound = self.support.upper_bound[pop]
        self.support.lower_bound = self.support.lower_bound[pop]

        not_set_idx = torch.nonzero(torch.isnan(condition))[
            :, 1
        ]  # indices for not set parameters
        set_idx = torch.nonzero(~torch.isnan(condition))[
            :, 1
        ]  # indices for set parameters
        new_idx = torch.cat(
            (not_set_idx, set_idx)
        )  # indices with not set parameters first and then set parameters
        y = condition[0, set_idx]
        # New centroids and covar matrices
        new_cen = []
        new_ccovs = []
        # Appendix A in C. E. Rasmussen & C. K. I. Williams, Gaussian Processes
        # for Machine Learning, the MIT Press, 2006
        fk = []
        for i in range(self.m.shape[0]):
            # Make a new co-variance matrix with correct ordering
            new_ccov = deepcopy(self.S[i])
            new_ccov = new_ccov[:, new_idx]
            new_ccov = new_ccov[new_idx, :]
            ux = self.m[i, not_set_idx]
            uy = self.m[i, set_idx]
            A = new_ccov[0 : len(not_set_idx), 0 : len(not_set_idx)]
            B = new_ccov[len(not_set_idx) :, len(not_set_idx) :]
            # B = B + 1e-10*torch.eye(B.shape[0]) # prevents B from becoming singular
            C = new_ccov[0 : len(not_set_idx), len(not_set_idx) :]
            cen = ux + C @ torch.inverse(B) @ (y - uy)
            cov = A - C @ torch.inverse(B) @ C.transpose(1, 0)
            new_cen.append(cen)
            new_ccovs.append(cov)
            fk.append(self.mulnormpdf(y, uy, B))  # Used for normalizing the mc
        # Normalize the mixing coef: p(X|Y) = p(Y,X) / p(Y) using the marginal dist.
        fk = torch.tensor(fk)
        new_mc = self.mc * fk
        new_mc = new_mc / torch.sum(new_mc)

        # set new GMM parameters
        self.m = torch.stack(new_cen)
        self.S = torch.stack(new_ccovs)
        self.mc = new_mc

    def sample_with_mcmc(self):
        """Dummy function to overwrite the existing `.sample_with_mcmc()` method."""

        raise DeprecationWarning(
            "MCMC sampling is not yet supported for the conditional MDN."
        )

    def sample_conditional(
        self, n_samples: Tuple[int, int], condition: Tensor = None
    ) -> Tensor:
        """Samples from the condtional distribution. If a condition
        is provided, a new conditional distribution will be calculated.
        If no condition is provided, samples will be drawn from the
        exisiting condition.

        Args:
            n_samples: The number of samples to draw from the conditional distribution.
            condition: An array of inputs. Inputs set to NaN are not set, and become inputs to
                the resulting distribution. Order is preserved.

        Returns:
            samples: Contains samples from the conditional posterior (NxD).
        """

        if condition != None:
            self.__conditionalise(condition)
        samples = self.sample(n_samples)
        return samples
