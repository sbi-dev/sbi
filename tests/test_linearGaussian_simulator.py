import pytest
import sbi.simulators as simulators
import torch


@pytest.mark.parametrize("D, N", [(1, 10000), (5, 100000)])
def test_linearGaussian_simulator(D, N):
    """Test for linear Gaussian simulator. 
    
    Arguments:
        D {int} -- parameter dimension.
        N {int} -- number of samples. 
    """
    simulator = simulators.LinearGaussianSimulator(dim=D)

    true_parameters = torch.zeros(D)
    num_simulations = N
    parameters = true_parameters.repeat(num_simulations).reshape(-1, D)
    observations = simulator(parameters)

    # check out shapes
    assert parameters.shape == torch.Size(
        [N, D]
    ), f"wrong shape of parameters: {parameters.shape} != {torch.Size([N, D])}"
    assert observations.shape == torch.Size([N, D])

    # check mean and std
    assert torch.allclose(
        observations.mean(axis=0), true_parameters, atol=5e-2
    ), f"mean should be zero, is {observations.mean(axis=0)}"
    assert torch.allclose(
        observations.std(axis=0), torch.ones(D), atol=5e-2
    ), f"std should be one, is {observations.std(axis=0)}"
