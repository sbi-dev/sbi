# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Affero General Public License v3, see <https://www.gnu.org/licenses/>.

from copy import deepcopy
from typing import Callable, List, Optional, Tuple, Union

import torch
import torch.distributions.transforms as torch_tf
from pyknos.mdn.mdn import MultivariateGaussianMDN as mdn
from pyknos.nflows.flows import Flow
from torch import Tensor
from torch.distributions import Distribution

from sbi.utils.torchutils import ensure_theta_batched
from sbi.utils.user_input_checks import process_x


def compute_corrcoeff(probs: Tensor, limits: Tensor):
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
    variance_x = _compute_covariance(marginal_x, limits[0], lambda x: x**2)
    variance_y = _compute_covariance(marginal_y, limits[1], lambda x: x**2)

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
        torch.linspace(lim[0].item(), lim[1].item(), prob.shape[0], device=probs.device)
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


def extract_and_transform_mog(
    net: Flow, context: Optional[Tensor] = None
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """Extracts the Mixture of Gaussians (MoG) parameters
    from an MDN based DirectPosterior at either the default x or input x.

    Args:
        posterior: DirectPosterior instance.
        context: Conditioning context for posterior $p(\theta|x)$. If not provided,
            fall back onto `x` passed to `set_default_x()`.

    Returns:
        norm_logits: Normalised log weights of the underyling MoG.
            (batch_size, n_mixtures)
        means_transformed: Recentred and rescaled means of the underlying MoG
            (batch_size, n_mixtures, n_dims)
        precfs_transformed: Rescaled precision factors of the underlying MoG.
            (batch_size, n_mixtures, n_dims, n_dims)
        sumlogdiag: Sum of the log of the diagonal of the precision factors
            of the new conditional distribution. (batch_size, n_mixtures)
    """

    # extract and rescale means, mixture componenets and covariances
    dist = net._distribution
    encoded_x = net._embedding_net(context)

    logits, means, _, sumlogdiag, precfs = dist.get_mixture_components(encoded_x)
    norm_logits = logits - torch.logsumexp(logits, dim=-1, keepdim=True)

    scale = net._transform._transforms[0]._scale
    shift = net._transform._transforms[0]._shift

    means_transformed = (means - shift) / scale

    A = scale * torch.eye(means_transformed.shape[2])
    precfs_transformed = A @ precfs

    sumlogdiag = torch.sum(
        torch.log(torch.diagonal(precfs_transformed, dim1=2, dim2=3)), dim=2
    )

    return norm_logits, means_transformed, precfs_transformed, sumlogdiag


def condition_mog(
    condition: Tensor,
    dims: List[int],
    logits: Tensor,
    means: Tensor,
    precfs: Tensor,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """Finds the conditional distribution p(X|Y) for a MoG.

    Args:
        prior: Prior Distribution. Used to check if condition within support.
        condition: Parameter set that all dimensions not specified in
            `dims_to_sample` will be fixed to. Should contain dim_theta elements,
            i.e. it could e.g. be a sample from the posterior distribution.
            The entries at all `dims_to_sample` will be ignored.
        dims_to_sample: Which dimensions to sample from. The dimensions not
            specified in `dims_to_sample` will be fixed to values given in
            `condition`.
        logits: Log weights of the MoG. (batch_size, n_mixtures)
        means: Means of the MoG. (batch_size, n_mixtures, n_dims)
        precfs: Precision factors of the MoG.
            (batch_size, n_mixtures, n_dims, n_dims)

    Raises:
        ValueError: The chosen condition is not within the prior support.

    Returns:
        logits:  Log weights of the conditioned MoG. (batch_size, n_mixtures)
        means: Means of the conditioned MoG. (batch_size, n_mixtures, n_dims)
        precfs_xx: Precision factors of the MoG.
            (batch_size, n_mixtures, n_dims, n_dims)
        sumlogdiag: Sum of the log of the diagonal of the precision factors
            of the new conditional distribution. (batch_size, n_mixtures)
    """

    n_mixtures, n_dims = means.shape[1:]

    mask = torch.zeros(n_dims, dtype=torch.bool)
    mask[dims] = True

    y = condition[:, ~mask]
    mu_x = means[:, :, mask]
    mu_y = means[:, :, ~mask]

    precfs_xx = precfs[:, :, mask]
    precfs_xx = precfs_xx[:, :, :, mask]
    precs_xx = precfs_xx.transpose(3, 2) @ precfs_xx

    precfs_yy = precfs[:, :, ~mask]
    precfs_yy = precfs_yy[:, :, :, ~mask]
    precs_yy = precfs_yy.transpose(3, 2) @ precfs_yy

    precs = precfs.transpose(3, 2) @ precfs
    precs_xy = precs[:, :, mask]
    precs_xy = precs_xy[:, :, :, ~mask]

    means = mu_x - (
        torch.inverse(precs_xx) @ precs_xy @ (y - mu_y).view(1, n_mixtures, -1, 1)
    ).view(1, n_mixtures, -1)

    diags = torch.diagonal(precfs_yy, dim1=2, dim2=3)
    sumlogdiag_yy = torch.sum(torch.log(diags), dim=2)
    log_prob = mdn.log_prob_mog(y, torch.zeros((1, 1)), mu_y, precs_yy, sumlogdiag_yy)

    # Normalize the mixing coef: p(X|Y) = p(Y,X) / p(Y) using the marginal dist.
    new_mcs = torch.exp(logits + log_prob)
    new_mcs = new_mcs / new_mcs.sum()
    logits = torch.log(new_mcs)

    sumlogdiag = torch.sum(torch.log(torch.diagonal(precfs_xx, dim1=2, dim2=3)), dim=2)
    return logits, means, precfs_xx, sumlogdiag


class ConditionedPotential:
    def __init__(
        self, potential_fn: Callable, condition: Tensor, dims_to_sample: List[int]
    ):
        r"""
        Return conditional posterior log-probability or $-\infty$ if outside prior.

        Args:
            theta: Free parameters $\theta_i$, batch dimension 1.

        Returns:
            Conditional posterior log-probability $\log(p(\theta_i|\theta_j, x))$,
            masked outside of prior.
        """
        self.potential_fn = potential_fn
        self.condition = condition
        self.dims_to_sample = dims_to_sample
        self.device = self.potential_fn.device

    def __call__(self, theta: Tensor, track_gradients: bool = True) -> Tensor:
        r"""
        Returns the conditional potential $\log(p(\theta_i|\theta_j, x))$.

        Args:
            theta: Free parameters $\theta_i$, batch dimension 1.

        Returns:
            Conditional posterior log-probability $\log(p(\theta_i|\theta_j, x))$,
            masked outside of prior.
        """
        theta_ = ensure_theta_batched(torch.as_tensor(theta, dtype=torch.float32))

        # `theta_condition`` will first have all entries of the `condition` and then
        # override the entries that should be sampled with `theta` (see below).
        theta_condition = deepcopy(self.condition)

        # In case `theta` is a batch of theta (e.g. multi-chain MCMC), we have to
        # repeat `theta_condition`` to the same batchsize.
        theta_condition = theta_condition.repeat(theta_.shape[0], 1)
        theta_condition[:, self.dims_to_sample] = theta_

        return self.potential_fn(theta_condition, track_gradients=track_gradients)

    def set_x(self, x_o: Optional[Tensor]):
        """Check the shape of the observed data and, if valid, set it."""
        if x_o is not None:
            x_o = process_x(x_o, allow_iid_x=False).to(self.device)
        self.potential_fn.set_x(x_o)

    @property
    def x_o(self) -> Tensor:
        """Return the observed data at which the potential is evaluated."""
        if self.potential_fn._x_o is not None:
            return self.potential_fn._x_o
        else:
            raise ValueError("No observed data is available.")

    @x_o.setter
    def x_o(self, x_o: Optional[Tensor]) -> None:
        """Check the shape of the observed data and, if valid, set it."""
        self.set_x(x_o)

    def return_x_o(self) -> Optional[Tensor]:
        """Return the observed data at which the potential is evaluated.

        Difference to the `x_o` property is that it will not raise an error if
        `self._x_o` is `None`.
        """
        return self.potential_fn._x_o


class RestrictedPriorForConditional:
    """
    Class to restrict a prior to fewer dimensions as needed for conditional sampling.

    The resulting prior samples only from the free dimensions of the conditional.

    This is needed for the the MCMC initialization functions when conditioning.
    For the prior init, we could post-hoc select the relevant dimensions. But
    for SIR, we want to evaluate the `potential_fn` of the conditional
    posterior, which takes only a subset of the full parameter vector theta
    (only the `dims_to_sample`). This subset is provided by `.sample()` from
    this class.
    """

    def __init__(self, full_prior: Distribution, dims_to_sample: List[int]):
        self.full_prior = full_prior
        self.dims_to_sample = dims_to_sample

    def sample(self, *args, **kwargs):
        """
        Sample only from the relevant dimension. Other dimensions are filled in
        by the `ConditionalPotentialFunctionProvider()` during MCMC.
        """
        return self.full_prior.sample(*args, **kwargs)[:, self.dims_to_sample]

    def log_prob(self, *args, **kwargs):
        r"""
        `log_prob` is same as for the full prior, because we usually evaluate
        the $\theta$ under the full joint once we have added the condition.
        """
        return self.full_prior.log_prob(*args, **kwargs)


class RestrictedTransformForConditional(torch_tf.Transform):
    """
    Class to restrict the transform to fewer dimensions for conditional sampling.

    The resulting transform transforms only the free dimensions of the conditional.
    Notably, the `log_abs_det` is computed given all dimensions. However, the
    `log_abs_det` stemming from the fixed dimensions is a constant and drops out during
    MCMC.

    All methods work in a similar way:
    `full_theta`` will first have all entries of the `condition` and then override the
    entries that should be sampled with `theta`. In case `theta` is a batch of `theta`
    (e.g. multi-chain MCMC), we have to repeat `theta_condition`` to the match the
    batchsize.

    This is needed for the the MCMC initialization functions when conditioning and
    when transforming the samples back into the original theta space after sampling.
    """

    def __init__(
        self,
        transform: torch_tf.Transform,
        condition: Tensor,
        dims_to_sample: List[int],
    ) -> None:
        super().__init__()  # type: ignore
        self.transform = transform
        self.condition = ensure_theta_batched(condition)
        self.dims_to_sample = dims_to_sample

    def __call__(self, theta: Tensor) -> Tensor:
        r"""
        Transform restricted $\theta$.
        """
        full_theta = self.condition.repeat(theta.shape[0], 1)
        full_theta[:, self.dims_to_sample] = theta
        tf_full_theta = self.transform(full_theta)
        return tf_full_theta[:, self.dims_to_sample]  # type: ignore

    def inv(self, theta: Tensor) -> Tensor:
        r"""
        Inverse transform restricted $\theta$.
        """
        full_theta = self.condition.repeat(theta.shape[0], 1)
        full_theta[:, self.dims_to_sample] = theta
        tf_full_theta = self.transform.inv(full_theta)
        return tf_full_theta[:, self.dims_to_sample]  # type: ignore

    def log_abs_det_jacobian(self, theta1: Tensor, theta2: Tensor) -> Tensor:
        """
        Return the `log_abs_det_jacobian` of |dtheta1 / dtheta2|.

        The determinant is summed over all dimensions, not just the `dims_to_sample`
        ones.
        """
        full_theta1 = self.condition.repeat(theta1.shape[0], 1)
        full_theta1[:, self.dims_to_sample] = theta1
        full_theta2 = self.condition.repeat(theta2.shape[0], 1)
        full_theta2[:, self.dims_to_sample] = theta2
        log_abs_det = self.transform.log_abs_det_jacobian(full_theta1, full_theta2)
        return log_abs_det
