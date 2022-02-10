import pytest
import torch

from sbi.analysis import ActiveSubspace, conditional_corrcoeff, conditional_pairplot
from sbi.inference import SNPE
from sbi.utils import BoxUniform


@pytest.mark.slow
@pytest.mark.gpu
@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_analysis_modules(device: str) -> None:
    """Tests sensitivity analysis and conditional posterior utils on GPU and CPU.

    This test performs only API tests. It does not test the accuracy of the modules.

    Args:
        device: Which device to run the inference on.
    """
    num_dim = 3
    prior = BoxUniform(
        low=-2 * torch.ones(num_dim), high=2 * torch.ones(num_dim), device=device
    )

    def simulator(parameter_set):
        return 1.0 + parameter_set + torch.randn(parameter_set.shape) * 0.1

    theta = prior.sample((300,)).to("cpu")
    x = simulator(theta)

    inf = SNPE(prior=prior, device=device)
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
