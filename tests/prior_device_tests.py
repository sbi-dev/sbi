import pytest
import torch

from sbi.utils.torchutils import BoxUniform, process_device
from sbi.utils.user_input_checks_utils import PytorchReturnTypeWrapper


@pytest.mark.gpu
@pytest.mark.parametrize("device", ["cpu", "mps", "gpu"])
def test_BoxUniform(device: str):
    device = process_device(device)
    low = torch.tensor([0.0])
    high = torch.tensor([1.0])
    prior = BoxUniform(low, high)
    prior.sample((1,))
    assert prior.device == "cpu"

    prior.to(device)
    assert prior.device == device
    assert prior.low.device.type == device.strip(":0")
    assert prior.high.device.type == device.strip(":0")
    prior.sample((1,))


@pytest.mark.gpu
@pytest.mark.parametrize("device", ["cpu", "mps", "gpu"])
def test_ReturnTypeWrapper(device: str):
    device = process_device(device)
    prior = torch.distributions.Normal(loc=0.0, scale=1.0)
    prior = PytorchReturnTypeWrapper(prior)

    prior.to(device)
    assert prior.device == device
