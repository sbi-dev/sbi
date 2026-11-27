# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

import pytest
import torch

from sbi.analysis import ActiveSubspace, conditional_corrcoeff, conditional_pairplot
from sbi.inference import NPE
from sbi.utils import BoxUniform, get_1d_marginal_peaks_from_kde
from sbi.utils.torchutils import process_device
from tests.test_utils import skip_if_mps_op_unsupported


@pytest.mark.slow
@pytest.mark.parametrize("device", ["cpu", pytest.param("gpu", marks=pytest.mark.gpu)])
def test_analysis_modules(device: str) -> None:
    """Tests sensitivity analysis and conditional posterior utils on GPU and CPU.

    This test performs only API tests. It does not test the accuracy of the modules.

    Args:
        device: Which device to run the inference on.
    """

    num_dim = 3
    device = process_device(device)
    skip_if_mps_op_unsupported(device, "aten::_linalg_eigh.eigenvalues")
    prior = BoxUniform(
        low=-2 * torch.ones(num_dim), high=2 * torch.ones(num_dim), device=device
    )

    def simulator(theta):
        return 1.0 + theta + torch.randn(theta.shape, device=theta.device) * 0.1

    theta = prior.sample((300,)).to(device)
    x = simulator(theta)

    inf = NPE(prior=prior, device=device)
    _ = inf.append_simulations(theta, x).train(max_num_epochs=10)

    observation = torch.zeros(num_dim)
    posterior = inf.build_posterior().set_default_x(observation)

    a = ActiveSubspace(posterior)
    theta = posterior.sample((200,)).to("cpu")
    property_ = simulator(theta)[:, :1]
    _ = a.add_property(theta, property_).train()

    evals, dirs = a.find_directions()
    assert str(evals.device) == device
    assert str(dirs.device) == device

    projected = a.project(theta[:10], 1)
    assert str(projected.device) == device

    # Compute the matrix of correlation coefficients of the slices.
    cond_coeff_mat = conditional_corrcoeff(
        density=posterior,
        condition=posterior.sample((1,)),
        limits=torch.tensor([[-2.0, 2.0], [-2.0, 2.0], [-2, 2.0]]),
    )
    assert str(cond_coeff_mat.device) == device

    _ = conditional_pairplot(
        posterior, condition=posterior.sample((1,)), limits=[[-2, 2], [-2, 2], [-2, 2]]
    )


def test_active_subspace_with_single_parameter() -> None:
    """`find_directions` must keep the parameter axis for a 1-D parameter.

    `torch.squeeze` without a dimension dropped the single parameter axis, so the
    outer product of the gradients raised a RuntimeError for one-parameter models.
    """
    torch.manual_seed(0)
    num_dim = 1
    prior = BoxUniform(low=-2 * torch.ones(num_dim), high=2 * torch.ones(num_dim))

    def simulator(theta):
        return theta + 0.1 * torch.randn_like(theta)

    theta = prior.sample((200,))
    inference = NPE(prior=prior)
    inference.append_simulations(theta, simulator(theta)).train(max_num_epochs=2)
    posterior = inference.build_posterior().set_default_x(torch.zeros(num_dim))

    active_subspace = ActiveSubspace(posterior)
    eigen_values, eigen_vectors = active_subspace.find_directions(
        posterior_log_prob_as_property=True, num_monte_carlo_samples=50
    )

    assert eigen_values.shape == (num_dim,)
    assert eigen_vectors.shape == (num_dim, num_dim)
    assert torch.isfinite(eigen_values).all()
    # The single eigenvector is a unit vector along the only axis.
    assert torch.isclose(eigen_vectors.abs().sum(), torch.tensor(1.0))
    assert active_subspace.project(theta[:10], 1).shape == (10, num_dim)


@pytest.mark.parametrize("num_candidates", (1_000,))
def test_1d_marginals_peaks_from_kde(num_candidates):
    num_samples = 10_000
    num_dim = 4
    samples = torch.ones(num_samples, num_dim) + 0.5 * torch.randn(num_samples, num_dim)

    peaks = get_1d_marginal_peaks_from_kde(samples, num_candidates=num_candidates)
    assert torch.allclose(peaks, torch.ones(num_dim), atol=1e-1)
