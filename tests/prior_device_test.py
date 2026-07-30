# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

import pytest
import torch
from torch.distributions import Beta, Binomial, Gamma, MultivariateNormal, Normal

from sbi.utils import RestrictedPrior
from sbi.utils.torchutils import BoxUniform, process_device
from sbi.utils.user_input_checks_utils import (
    MultipleIndependent,
    PytorchReturnTypeWrapper,
)
from tests.test_utils import skip_if_mps_op_unsupported


@pytest.mark.parametrize("device", ["cpu", pytest.param("gpu", marks=pytest.mark.gpu)])
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
    assert prior.low.device.type == device.split(":")[0], (
        f"BoxUniform low tensor is not in {device}."
    )
    assert prior.high.device.type == device.split(":")[0], (
        f"BoxUniform high tensor is not in {device}."
    )

    sample_device = prior.sample((100,))
    assert sample_device.device.type == device.split(":")[0], (
        f"sample tensor is not in {device}."
    )
    log_probs = prior.log_prob(sample_device)
    assert log_probs.device.type == device.split(":")[0], (
        f"log_prob tensor is not in {device}."
    )


@pytest.mark.parametrize("device", ["cpu", pytest.param("gpu", marks=pytest.mark.gpu)])
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
    if isinstance(prior, Binomial):
        skip_if_mps_op_unsupported(device, "aten::binomial")
    prior = PytorchReturnTypeWrapper(prior)

    prior.to(device)
    assert prior.device == device, f"Prior was not correctly moved to {device}."

    sample_device = prior.sample((100,))
    assert sample_device.device.type == device.split(":")[0], (
        f"sample was not correctly moved to {device}."
    )
    log_probs = prior.log_prob(sample_device)
    assert log_probs.device.type == device.split(":")[0], (
        f"log_prob was not correctly moved to {device}."
    )


@pytest.mark.parametrize("device", ["cpu", pytest.param("gpu", marks=pytest.mark.gpu)])
def test_MultipleIndependent(device: str):
    """Test moving MultipleIndependent objects between devices.

    Asserts that samples, prior, and log_probs are in device.
    Uses Gamma, Beta, Normal and Binomial, from
    torch.distributions and BoxUniform form sbi.
    """
    device = process_device(device)
    skip_if_mps_op_unsupported(device, "aten::binomial")
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
    assert sample_device.device.type == device.split(":")[0], (
        f"sample was not correctly moved to {device}."
    )
    log_probs = prior.log_prob(sample_device)
    assert log_probs.device.type == device.split(":")[0], (
        f"log_prob was not correctly moved to {device}."
    )


@pytest.mark.parametrize(
    "prior",
    [
        BoxUniform(torch.zeros(2), torch.ones(2)),
        PytorchReturnTypeWrapper(MultivariateNormal(torch.zeros(2), torch.eye(2))),
        MultipleIndependent([
            Gamma(torch.tensor([1.0]), torch.tensor([0.5])),
            Beta(torch.tensor([2.0]), torch.tensor([2.0])),
        ]),
    ],
)
def test_prior_to_stores_resolved_device(prior):
    """Test that `.to()` stores the resolved device, whatever the spelling."""
    prior.to("cpu:0")
    assert prior.device == "cpu"


def test_restricted_prior_stores_resolved_device():
    """Test that RestrictedPrior resolves the device it is constructed with."""
    prior = BoxUniform(torch.zeros(2), torch.ones(2))

    def accept_all(theta):
        return torch.ones(theta.shape[0], dtype=torch.bool)

    restricted = RestrictedPrior(prior, accept_all, device="cpu:0")
    assert restricted._device == "cpu"
