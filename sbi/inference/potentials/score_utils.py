import torch
from torch import Tensor
from torch.distributions import (
    Distribution,
    Independent,
    MultivariateNormal,
    Normal,
    MixtureSameFamily,
    Categorical,
    Uniform,
    constraints,
)
from sbi.utils.sbiutils import ImproperEmpirical
from sbi.utils.torchutils import BoxUniform

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
        return denoise_empirical(p, m, s, x_t)


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


def denoise_uniform(p: Uniform, m: Tensor, s: Tensor, x_t: Tensor):
    return UniformNormalPosterior(p.low, p.high, m, s, x_t)


def denoise_empirical(
    p: Distribution,
    m: Tensor,
    s: Tensor,
    x_t: Tensor,
    num_particles: int = 5000,
) -> MixtureSameFamily:
    particles = p.sample((num_particles,))  # Sample from prior
    s = torch.clip(s, 1e-1, 1e6)

    # Compute posterior mean and standard deviation
    posterior_mean = (m * particles + s**2 * x_t) / (m**2 + s**2)
    posterior_std = torch.sqrt((s**2) / (m**2 + s**2))
    posterior_std = torch.broadcast_to(posterior_std, posterior_mean.shape)

    # Compute unnormalized log-likelihoods as logits
    logits = -0.5 * torch.sum(
        ((x_t - m * particles) / s) ** 2, axis=-1
    )  # Gaussian likelihood
    logits = logits - logits.logsumexp(dim=-1, keepdims=True)  # Normalize logits

    # print(posterior_mean.shape, posterior_std.shape, logits.shape)
    # Define the mixture components
    components = Independent(Normal(posterior_mean, posterior_std), 1)

    # Create a mixture model
    posterior = MixtureSameFamily(Categorical(logits=logits), components)

    return posterior


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
    elif isinstance(p, (Uniform, BoxUniform)):
        return marginalize_uniform(p, m, s)
    else:
        return marginalize_empirical(p, m, s)


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

def marginalize_mixture(
    p: MixtureSameFamily, m: Tensor, s: Tensor
) -> MixtureSameFamily:
    """Marginalize a MixtureSameFamily distribution.

    Args:
        p: The prior MixtureSameFamily distribution.
        m: The scaling factor m.
        s: The standard deviation of the noise s.

    Returns:
        The marginal MixtureSameFamily distribution.
    """
    return MixtureSameFamily(
        p.mixture_distribution, marginalize(p.component_distribution, m, s)
    )


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
        mean_0.shape[-1],
        dtype=cov_0.dtype,  # type: ignore
        device=cov_0.device,  # type: ignore
    )

    return MultivariateNormal(marginal_mean, covariance_matrix=marginal_cov)


def marginalize_uniform(
    p: Uniform | BoxUniform, m: Tensor, s: Tensor
) -> 'UniformNormalConvolution':
    """Marginalize a uniform distribution.

    Args:
        p: The prior uniform distribution.
        m: The scaling factor.
        s: The standard deviation of the noise.

    Returns:
        The marginal uniform distribution.
    """

    return UniformNormalConvolution(p.low, p.high, m, s)  # type: ignore


def marginalize_empirical(
    p: Distribution, m: Tensor, s: Tensor, num_particles: int = 5000
) -> MixtureSameFamily:
    particles = p.sample((num_particles,))
    logits = torch.zeros(num_particles)
    s = torch.clip(s, 1e-1, 1e6)
    # NOTE: We might want to assume some initial particle variance
    marginal_mean = m[None, ...] * particles
    marginal_std = torch.sqrt(s**2)

    # NOTE: This might not have a nice score at low noise levels (diracs...)
    components = Independent(Normal(marginal_mean, marginal_std), 1)
    p = MixtureSameFamily(Categorical(logits=logits), components)
    # Less exact but more well bahaved score
    # p = Independent(Normal(marginal_mean.mean(0), marginal_std.mean(0)), 1)
    return p

# Special distributions:
class UniformNormalPosterior(Distribution):
    arg_constraints = {
        "low": constraints.real,
        "high": constraints.real,
        "m": constraints.real,
        "s": constraints.positive,
        "x_t": constraints.real,
    }  # type: ignore
    has_rsample: bool = False

    def __init__(self, low, high, m, s, x_t, validate_args=None):
        self.low = low
        self.high = high
        self.m = m
        self.s = s
        self.x_t = x_t
        # The posterior density is proportional to N(x_t; m*x, s) on [low, high].
        # This can be viewed as a truncated normal in x with location x_t/m and scale s/|m|.
        self.normal = Normal(loc=x_t / m, scale=s / abs(m))
        self.a = self.normal.cdf(low)
        self.b = self.normal.cdf(high)
        self.Z = self.b - self.a  # normalization constant
        batch_shape = torch.broadcast_shapes(
            low.shape, high.shape, m.shape, s.shape, x_t.shape
        )
        event_shape = torch.Size()
        super().__init__(batch_shape, event_shape, validate_args=validate_args)

    def sample(self, sample_shape=torch.Size()):
        # Inverse transform sampling from the truncated normal.
        u = torch.rand(sample_shape, device=self.low.device) * self.Z + self.a
        return self.normal.icdf(u)

    def log_prob(self, x):
        # Compute the log-density of x under the truncated normal.
        lp = self.normal.log_prob(x) - torch.log(self.Z)
        # Enforce support: set log-probability to -inf outside [low, high].
        return torch.where(
            (x >= self.low) & (x <= self.high),
            lp,
            -float("inf") * torch.ones_like(lp),
        )


class UniformNormalConvolution(Distribution):
    arg_constraints = {
        "low": constraints.real,
        "high": constraints.real,
        "scale": constraints.real,
        "noise": constraints.positive,
    }  # type: ignore
    support = constraints.real  # type: ignore
    has_rsample = False

    def __init__(self, low, high, scale, noise, validate_args=None):
        self.low = torch.as_tensor(low)
        self.high = torch.as_tensor(high)
        self.scale = torch.as_tensor(scale)
        self.noise = torch.as_tensor(noise)

        # Determine batch_shape from broadcasting low and high (assumes low/high are tensors or scalars)
        batch_shape = self.low.shape

        event_shape = torch.Size()
        super().__init__(batch_shape, event_shape, validate_args=validate_args)

    def sample(self, sample_shape=torch.Size()):
        shape = sample_shape + self.batch_shape  # type: ignore
        # Sample uniformly over [low, high]
        x = torch.rand(shape, device=self.low.device)
        x = x * (self.high - self.low) + self.low
        noise_sample = torch.randn(shape, device=self.noise.device) * self.noise
        return self.scale * x + noise_sample

    def log_prob(self, value):
        # Compute p(value) using the convolution formula:
        # p(value) = 1/(b-a) * [Phi((value - scale*a)/noise) - Phi((value - scale*b)/noise)]
        dist0 = Normal(torch.tensor(0.0, device=value.device), 1.0)
        numerator = dist0.cdf((value - self.scale * self.low) / self.noise) - dist0.cdf(
            (value - self.scale * self.high) / self.noise
        )
        denom = (self.high - self.low).clamp_min(1e-30)
        pdf = numerator / denom
        return torch.log(pdf.clamp_min(1e-30))


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
