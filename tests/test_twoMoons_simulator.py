import pytest
import sbi.simulators as simulators
import torch


@pytest.mark.parametrize("N", [1, 100])
def test_twomoon_simulator(N):
    """Test twomoon simulator. 
    
    Just checking parameters and observations shapes. 

    Arguments:
        N {int} -- batch size.
    """

    simulator = simulators.TwoMoonsSimulator()
    true_parameters = torch.tensor([0, 0])
    parameters = true_parameters.repeat(N).reshape(-1, 2)
    observations = simulator(parameters)

    assert parameters.shape == torch.Size([N, 2])
    assert observations.shape == torch.Size([N, 2])
