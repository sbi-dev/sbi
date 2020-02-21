import pytest
import torch
import torch.distributions as dist
from sbi.simulators.linear_gaussian import LinearGaussianSimulator
from sbi.simulators.nonlinear_gaussian import NonlinearGaussianSimulator
from sbi.simulators.simutils import get_simulator_dimensions, get_simulator_name


@pytest.mark.parametrize("name", [None, "mysimulator"])
def test_get_simualtor_name(name):

    # get simualtor
    def mysimulator(params):
        return params

    assert get_simulator_name(mysimulator, name) == "mysimulator"


def test_get_simualtor_dimensions():

    # test linear gaussian
    dim = 3
    dim_in, dim_out = get_simulator_dimensions(
        LinearGaussianSimulator(dim=dim),
        lambda num_samples: dist.Uniform(
            low=torch.tensor(dim * [-2.0]), high=torch.tensor(dim * [2.0])
        ).sample((num_samples,)),
    )
    assert dim_in == dim, "input dim inferred incorrectly"
    assert dim_out == dim, "output dim inferred incorrectly"

    # test non linear gaussian
    dim_in, dim_out = get_simulator_dimensions(
        NonlinearGaussianSimulator(),
        lambda num_samples: dist.Uniform(
            low=torch.tensor(5 * [-2.0]), high=torch.tensor(5 * [2.0])
        ).sample((num_samples,)),
    )
    assert dim_in == 5, "input dim inferred incorrectly"
    assert dim_out == 8, "output dim inferred incorrectly"
