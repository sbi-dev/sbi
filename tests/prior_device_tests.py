import pytest
import torch

from sbi.utils.torchutils import BoxUniform


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
