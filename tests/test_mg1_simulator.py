import pytest
import sbi.simulators as simulators
import torch


@pytest.mark.parametrize("N", [1, 100])
def test_mg_simulator(N):
    """Test mg1 simulator. 
    
    Just checking parameters and observations shapes. 

    Arguments:
        N {int} -- batch size.
    """

    simulator = simulators.MG1Simulator()
    true_parameters = torch.tensor([1.0, 5.0, 0.2])
    parameters = true_parameters.repeat(N).reshape(-1, 3)
    observations = simulator(parameters)

    assert parameters.shape == torch.Size([N, 3])
    assert observations.shape == torch.Size([N, 5])
