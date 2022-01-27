from turtle import pd
import pytest
import torch

from torch.distributions import Uniform, MultivariateNormal, LogNormal
from torch.distributions.transforms import (
    AffineTransform,
    ComposeTransform,
    IndependentTransform,
)

from sbi.utils import BoxUniform, mcmc_transform, process_prior
from tests.user_input_checks_test import UserNumpyUniform


@pytest.mark.parametrize(
    "prior, target_transform",
    (
        (Uniform(-torch.ones(1), torch.ones(1)), ComposeTransform),
        (BoxUniform(-torch.ones(2), torch.ones(2)), ComposeTransform),
        (UserNumpyUniform(torch.zeros(2), torch.ones(2)), ComposeTransform),
        (MultivariateNormal(torch.zeros(2), torch.eye(2)), AffineTransform),
        (LogNormal(loc=torch.zeros(1), scale=torch.ones(1)), AffineTransform),
    ),
)
def test_transforms(prior, target_transform):

    if isinstance(prior, UserNumpyUniform):
        prior, *_ = process_prior(
            prior,
            dict(lower_bound=torch.zeros(2), upper_bound=torch.ones(2)),
        )

    transform = mcmc_transform(prior)
    core_transform = transform._inv

    if isinstance(core_transform, IndependentTransform):
        core_transform = core_transform.base_transform

    assert isinstance(core_transform, target_transform)

    samples = prior.sample((2,))
    transformed_samples = transform(samples)
    assert torch.allclose(samples, transform.inv(transformed_samples))
