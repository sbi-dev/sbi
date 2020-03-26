import torch
import torch.distributions as distributions
import sbi.utils as utils

torch.manual_seed(0)


def test_dkl_one_dim_gauss():
    """
    Test whether for two 1D Gaussians the Monte-Carlo-based D-KL gives similar results
     as the torch implementation.
    """
    dist1 = distributions.Normal(loc=0.0, scale=1.0)
    dist2 = distributions.Normal(loc=1.0, scale=0.5)

    torch_dkl = distributions.kl.kl_divergence(dist1, dist2)
    monte_carlo_dkl = utils.dkl_monte_carlo_estimate(dist1, dist2, num_samples=1000)

    max_dkl_diff = 0.3

    assert torch.abs(torch_dkl - monte_carlo_dkl) < max_dkl_diff, (
        f"Monte Carlo based DKL={monte_carlo_dkl} is too far off from the torch"
        f" implementation={torch_dkl}."
    )


def test_dkl_multi_dim_gauss():
    """
    Test whether for two 2D Gaussians the Monte-Carlo-based D-KL gives similar results
     as the torch implementation.
    """
    dist1 = distributions.MultivariateNormal(torch.zeros(2), torch.eye(2))
    dist2 = distributions.MultivariateNormal(torch.ones(2), 0.5 * torch.eye(2))

    torch_dkl = distributions.kl.kl_divergence(dist1, dist2)
    monte_carlo_dkl = utils.dkl_monte_carlo_estimate(dist1, dist2, num_samples=1000)

    max_dkl_diff = 0.3

    assert torch.abs(torch_dkl - monte_carlo_dkl) < max_dkl_diff, (
        f"Monte Carlo based DKL={monte_carlo_dkl} is too far off from the torch"
        f" implementation={torch_dkl}."
    )
