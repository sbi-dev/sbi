import pytest
import torch
from torch.distributions import Beta, Binomial, Gamma, MultivariateNormal, Normal

from sbi.utils.torchutils import BoxUniform, process_device
from sbi.utils.user_input_checks_utils import (
    MultipleIndependent,
    PytorchReturnTypeWrapper,
)


@pytest.mark.gpu
@pytest.mark.parametrize("device", ["cpu", "gpu"])
def test_BoxUniform(device: str):
    device = process_device(device)
    low = torch.tensor([0.0])
    high = torch.tensor([1.0])
    prior = BoxUniform(low, high)
    sample = prior.sample((1,))
    assert prior.device == "cpu"
    assert sample.device.type == "cpu"
    log_probs = prior.log_prob(sample)
    assert log_probs.device.type == "cpu"

    prior.to(device)
    assert prior.device == device
    assert prior.low.device.type == device.strip(":0")
    assert prior.high.device.type == device.strip(":0")

    sample_device = prior.sample((100,))
    assert sample_device.device.type == device.strip(":0")
    log_probs = prior.log_prob(sample_device)
    assert log_probs.device.type == device.strip(":0")


@pytest.mark.gpu
@pytest.mark.parametrize("device", ["cpu", "gpu"])
@pytest.mark.parametrize(
    "prior",
    [
        Normal(loc=0.0, scale=1.0),
        Binomial(total_count=10, probs=torch.tensor([0.5])),
        MultivariateNormal(torch.tensor([0.1, 0.0]), covariance_matrix=torch.eye(2)),
    ],
)
def test_PytorchReturnTypeWrapper(device: str, prior: torch.distributions):
    device = process_device(device)
    prior = PytorchReturnTypeWrapper(prior)

    prior.to(device)
    assert prior.device == device

    sample_device = prior.sample((100,))
    assert sample_device.device.type == device.strip(":0")
    log_probs = prior.log_prob(sample_device)
    assert log_probs.device.type == device.strip(":0")


@pytest.mark.gpu
@pytest.mark.parametrize("device", ["cpu", "gpu"])
def test_MultipleIndependent(device: str):
    device = process_device(device)
    dists = [
        Gamma(torch.tensor([1.0]), torch.tensor([0.5])),
        Beta(torch.tensor([2.0]), torch.tensor([2.0])),
        BoxUniform(torch.zeros(1), torch.ones(1)),
        Normal(torch.tensor([0.0]), torch.tensor([0.5])),
        Binomial(torch.tensor([10]), torch.tensor([0.5])),
    ]

    prior = MultipleIndependent(dists)
    prior.to(device)
    assert prior.device == device

    sample_device = prior.sample((100,))
    assert sample_device.device.type == device.strip(":0")
    log_probs = prior.log_prob(sample_device)
    assert log_probs.device.type == device.strip(":0")
