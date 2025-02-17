import torch
from torch import Tensor
from torch.distributions import Distribution, Independent, MultivariateNormal, Normal

# Automatic denoising -----------------------------------------------------


def denoise(p: Distribution, m: Tensor, s: Tensor, x_t: Tensor) -> Distribution:
    """Given the prior distribution p(X), scaling factor m, standard deviation of the
    noise s, and observation X_t, return the posterior distribution p(X | X_t = x_t).

    Args:
        p: The prior distribution p(X).
        m: The scaling factor m.
        s: The standard deviation of the noise s.
        x_t: The observation X_t.

    Raises:
        NotImplementedError: If the distribution is not supported.

    Returns:
        The posterior distribution p(X | X_t = x_t).
    """
    if isinstance(p, Independent):
        return denoise_independent(p, m, s, x_t)
    elif isinstance(p, Normal):
        return denoise_gaussian(p, m, s, x_t)
    elif isinstance(p, MultivariateNormal):
        return denoise_multivariate_gaussian(p, m, s, x_t)
    else:
        raise NotImplementedError(f"Automatic denoising for {type(p)} not implemented")


def denoise_independent(
    p: Independent, m: Tensor, s: Tensor, x_t: Tensor
) -> Independent:
    """Denoise an independent distribution.

    Args:
        p: The prior independent distribution.
        m: The scaling factor m.
        s: The standard deviation of the noise s.
        x_t: The observation X_t.

    Returns:
        The posterior independent distribution.
    """
    return Independent(denoise(p.base_dist, m, s, x_t), p.reinterpreted_batch_ndims)


def denoise_gaussian(p: Normal, m: Tensor, s: Tensor, x_t: Tensor) -> Normal:
    """Denoise a Gaussian distribution.

    Args:
        p: The prior Gaussian distribution.
        m: The scaling factor.
        s: The standard deviation of the noise.
        x_t: The observed data.

    Returns:
        The posterior Gaussian distribution.
    """
    mean_0 = p.loc
    std_0 = p.scale

    # Calculate the posterior mean and variance
    posterior_variance = 1 / (1 / std_0**2 + m**2 / s**2)
    posterior_mean = posterior_variance * (mean_0 / std_0**2 + m * x_t / s**2)
    posterior_std = torch.sqrt(posterior_variance)

    return Normal(posterior_mean, posterior_std)


def denoise_multivariate_gaussian(
    p: MultivariateNormal, m: Tensor, s: Tensor, x_t: Tensor
) -> MultivariateNormal:
    mean0 = p.loc
    cov0 = p.covariance_matrix
    n = cov0.size(-1)
    # Support batch dimensions by expanding the identity matrix to match cov0's
    # batch shape
    batch_shape = cov0.shape[:-2]
    id_matrix = torch.eye(n, dtype=cov0.dtype, device=cov0.device)
    id_matrix = id_matrix.expand(*batch_shape, n, n)
    precision_prior = torch.linalg.inv(cov0)
    # Reshape m and s so the operation broadcasts correctly over batch dims
    precision_likelihood = (m**2 / s**2)[..., None, None] * id_matrix
    posterior_cov = torch.linalg.inv(precision_prior + precision_likelihood)
    # unsqueeze mean0 and x_t for proper matrix multiplication
    term1 = torch.matmul(precision_prior, mean0.unsqueeze(-1))
    term2 = (m / s**2)[..., None] * x_t.unsqueeze(-1)
    posterior_mean = torch.matmul(posterior_cov, term1 + term2).squeeze(-1)
    return MultivariateNormal(posterior_mean, covariance_matrix=posterior_cov)


# Automatic marginalization -----------------------------------------------


def marginalize(p: Distribution, m: Tensor, s: Tensor) -> Distribution:
    """Given the prior distribution p(X), scaling factor m, and standard deviation of
    the noise s, return the marginal distribution p(X_t).

    Args:
        p: The prior distribution p(X).
        m: The scaling factor m.
        s: The standard deviation of the noise s.

    Raises:
        NotImplementedError: If the distribution is not supported.

    Returns:
        The marginal distribution p(X_t).
    """
    if isinstance(p, Independent):
        return marginalize_independent(p, m, s)
    elif isinstance(p, Normal):
        return marginalize_gaussian(p, m, s)
    elif isinstance(p, MultivariateNormal):
        return marginalize_multivariate_gaussian(p, m, s)
    else:
        raise NotImplementedError(
            f"Automatic marginalization for {type(p)} not implemented"
        )


def marginalize_independent(p: Independent, m: Tensor, s: Tensor) -> Independent:
    """Marginalize an independent distribution.

    Args:
        p: The prior independent distribution.
        m: The scaling factor m.
        s: The standard deviation of the noise s.

    Returns:
        The marginal independent distribution.
    """
    return Independent(marginalize(p.base_dist, m, s), p.reinterpreted_batch_ndims)


def marginalize_gaussian(p: Normal, m: Tensor, s: Tensor) -> Normal:
    """Marginalize a Gaussian distribution.

    Args:
        p: The prior Gaussian distribution.
        m: The scaling factor.
        s: The standard deviation of the noise.

    Returns:
        The marginal Gaussian distribution.
    """
    mean_0 = p.loc
    std_0 = p.scale

    # Calculate the marginal mean and variance
    marginal_mean = m * mean_0
    marginal_variance = (m * std_0) ** 2 + s**2
    marginal_std = torch.sqrt(marginal_variance)

    return Normal(marginal_mean, marginal_std)


def marginalize_multivariate_gaussian(
    p: MultivariateNormal, m: Tensor, s: Tensor
) -> MultivariateNormal:
    """Marginalize a multivariate Gaussian distribution.

    Given an observation model xₜ = m * x + ε with independent noise ε ~ N(0, s² I),
    where x ~ p, return the marginal distribution p(xₜ) as a MultivariateNormal.

    Args:
        p: The prior multivariate Gaussian distribution.
        m: The scaling factor.
        s: The standard deviation of the noise.
        dim: The dimension of the multivariate output (unused in this implementation).

    Returns:
        The marginal multivariate Gaussian distribution p(xₜ).
    """
    mean_0 = p.loc
    cov_0 = p.covariance_matrix

    marginal_mean = m * mean_0
    marginal_cov = (m**2) * cov_0 + s**2 * torch.eye(
        mean_0.shape[-1], dtype=cov_0.dtype, device=cov_0.device
    )

    return MultivariateNormal(marginal_mean, covariance_matrix=marginal_cov)


# Utility functions --------------------------------------------------------


def mv_diag_or_dense(A_diag_or_dense: Tensor, b: Tensor, batch_dims: int = 0) -> Tensor:
    """Dot product for diagonal or dense matrices.

    Args:
        A_diag_or_dense: Diagonal or dense matrix.
        b: Dense matrix/vector.
        batch_dims: Number of batch dimensions.

    Returns:
        The result of A * b (or A @ b).
    """
    A_dims = A_diag_or_dense.ndim - batch_dims
    if A_dims == 1:
        return A_diag_or_dense * b
    else:
        return torch.einsum('...ij,...j->...i', A_diag_or_dense, b)


def solve_diag_or_dense(
    A_diag_or_dense: Tensor, b: Tensor, batch_dims: int = 0
) -> Tensor:
    """Solve a linear system with a diagonal or dense matrix.

    Args:
        A_diag_or_dense: Diagonal or dense matrix.
        b: Dense matrix/vector.

    Returns:
        The solution to the linear system A x = b.
    """
    A_dim = A_diag_or_dense.ndim - batch_dims
    if A_dim == 1:
        return b / A_diag_or_dense
    else:
        return torch.linalg.solve(A_diag_or_dense, b)


def add_diag_or_dense(
    A_diag_or_dense: Tensor, B_diag_or_dense: Tensor, batch_dims: int = 0
) -> Tensor:
    """Add two diagonal or dense matrices, considering batch dimensions.

    Args:
        A_diag_or_dense: Diagonal or dense matrix.
        B_diag_or_dense: Diagonal or dense matrix.
        batch_dims: Number of batch dimensions.

    Returns:
        The sum of the two matrices.
    """
    A_ndim = A_diag_or_dense.ndim - batch_dims
    B_ndim = B_diag_or_dense.ndim - batch_dims

    if (A_ndim == 1 and B_ndim == 1) or (A_ndim == 2 and B_ndim == 2):
        return A_diag_or_dense + B_diag_or_dense
    elif A_ndim == 2 and B_ndim == 1:
        return A_diag_or_dense + torch.diag_embed(B_diag_or_dense)
    elif A_ndim == 1 and B_ndim == 2:
        return torch.diag_embed(A_diag_or_dense) + B_diag_or_dense
    else:
        raise ValueError("Incompatible dimensions for addition")
