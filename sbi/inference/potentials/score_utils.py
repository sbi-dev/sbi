from torch.distributions import Distribution, Independent, Normal, Uniform
from torch import Tensor

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
        m: The scaling factor m.
        s: The standard deviation of the noise s.
        x_t: The observation X_t.

    Returns:
        The posterior Gaussian distribution.
    """
    mu0 = p.loc
    std0 = p.scale

    # Calculate the posterior mean and variance
    var = 1 / (1 / std0**2 + m**2 / s**2)
    mu = var * (mu0 / std0**2 + m * x_t / s**2)
    std = var**0.5

    return Normal(mu, std)


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
        m: The scaling factor m.
        s: The standard deviation of the noise s.

    Returns:
        The marginal Gaussian distribution.
    """
    mu0 = p.loc
    std0 = p.scale

    # Calculate the marginal mean and variance
    mu = m * mu0
    var = (m * std0) ** 2 + s**2
    std = var**0.5

    return Normal(mu, std)