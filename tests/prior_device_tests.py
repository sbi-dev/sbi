import pytest
import torch

from sbi.utils.torchutils import BoxUniform
from sbi.utils.user_input_checks_utils import PytorchReturnTypeWrapper


@pytest.mark.gpu
@pytest.mark.parametrize("device", ["cpu", "mps"])
def test_BoxUniform(device: str):
    low = torch.tensor([0.0])
    high = torch.tensor([1.0])
    prior = BoxUniform(low, high)
    prior.sample((1,))
    assert prior.device == "cpu"

    prior.to(device)
    assert prior.device == device
    assert prior.low.device.type == device
    assert prior.high.device.type == device
    prior.sample((1,))


@pytest.mark.gpu
@pytest.mark.parametrize("device", ["cpu", "mps"])
def test_ReturnTypeWrapper(device: str):
    prior = torch.distributions.Normal(loc=0.0, scale=1.0)
    prior = PytorchReturnTypeWrapper(prior)

    prior.to(device)
    assert prior.device == device
