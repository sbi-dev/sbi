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
    """Test moving BoxUniform prior between devices."""
    device = process_device(device)
    low = torch.tensor([0.0])
    high = torch.tensor([1.0])
    prior = BoxUniform(low, high)
    sample = prior.sample((1,))
    assert prior.device == "cpu", "Prior is not initially in cpu."
    assert sample.device.type == "cpu", "sample is not initially in cpu."
    log_probs = prior.log_prob(sample)
    assert log_probs.device.type == "cpu", "Log probs are not initially in cpu."

    prior.to(device)
    assert prior.device == device, f"Prior was not moved to {device}."
    assert prior.low.device.type == device.strip(":0"), (
        f"BoxUniform low tensor is not in {device}."
    )
    assert prior.high.device.type == device.strip(":0"), (
        f"BoxUniform high tensor is not in {device}."
    )

    sample_device = prior.sample((100,))
    assert sample_device.device.type == device.strip(":0"), (
        f"sample tensor is not in {device}."
    )
    log_probs = prior.log_prob(sample_device)
    assert log_probs.device.type == device.strip(":0"), (
        f"log_prob tensor is not in {device}."
    )


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
    """Test moving PytorchReturnTypeWrapper objects between devices.

    Asserts that samples, prior, and log_probs are in device.
    """
    device = process_device(device)
    prior = PytorchReturnTypeWrapper(prior)

    prior.to(device)
    assert prior.device == device, f"Prior was not correctly moved to {device}."

    sample_device = prior.sample((100,))
    assert sample_device.device.type == device.strip(":0"), (
        f"sample was not correctly moved to {device}."
    )
    log_probs = prior.log_prob(sample_device)
    assert log_probs.device.type == device.strip(":0"), (
        f"log_prob was not correctly moved to {device}."
    )


@pytest.mark.gpu
@pytest.mark.parametrize("device", ["cpu", "gpu"])
def test_MultipleIndependent(device: str):
    """Test moving MultipleIndependent objects between devices.

    Asserts that samples, prior, and log_probs are in device.
    Uses Gamma, Beta, Normal and Binomial, from
    torch.distributions and BoxUniform form sbi.
    """
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
    assert prior.device == device, f"Prior was not correctly moved to {device}."

    sample_device = prior.sample((100,))
    assert sample_device.device.type == device.strip(":0"), (
        f"sample was not correctly moved to {device}."
    )
    log_probs = prior.log_prob(sample_device)
    assert log_probs.device.type == device.strip(":0"), (
        f"log_prob was not correctly moved to {device}."
    )
