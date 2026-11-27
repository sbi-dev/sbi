# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

import sys

import pytest
import torch
from torch import eye, ones, zeros
from torch.distributions import (
    Categorical,
    Cauchy,
    Exponential,
    Independent,
    LogNormal,
    MixtureSameFamily,
    MultivariateNormal,
    Normal,
    TransformedDistribution,
    Uniform,
    constraints,
)
from torch.distributions.transforms import (
    AffineTransform,
    ExpTransform,
    IndependentTransform,
    SigmoidTransform,
)

from sbi.utils import BoxUniform, MultipleIndependent, mcmc_transform, process_prior
from sbi.utils.sbiutils import prior_support_is_bounded
from sbi.utils.torchutils import process_device
from tests.user_input_checks_test import UserNumpyUniform


@pytest.mark.parametrize(
    "prior, target_transform",
    (
        (Uniform(-torch.ones(1), torch.ones(1)), SigmoidTransform),
        (BoxUniform(-torch.ones(2), torch.ones(2)), SigmoidTransform),
        (UserNumpyUniform(torch.zeros(2), torch.ones(2)), SigmoidTransform),
        (MultivariateNormal(torch.zeros(2), torch.eye(2)), AffineTransform),
        (LogNormal(loc=torch.zeros(1), scale=torch.ones(1)), ExpTransform),
        (
            MixtureSameFamily(
                Categorical(torch.ones(2)),
                MultivariateNormal(torch.zeros(2, 2), eye(2).expand(2, 2, 2)),
            ),
            AffineTransform,
        ),
        (
            MixtureSameFamily(
                Categorical(torch.ones(2)),
                Independent(LogNormal(zeros(2, 2), ones(2, 2)), 1),
            ),
            ExpTransform,
        ),
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

    if hasattr(core_transform, "parts"):
        transform_to_inspect = core_transform.parts[0]
    else:
        transform_to_inspect = core_transform

    assert isinstance(transform_to_inspect, target_transform)

    samples = prior.sample((2,))
    transformed_samples = transform(samples)
    assert torch.allclose(samples, transform.inv(transformed_samples))


@pytest.mark.parametrize(
    "prior, enable_transform",
    (
        (BoxUniform(zeros(5), ones(5)), True),
        (BoxUniform(zeros(1), ones(1)), True),
        (BoxUniform(zeros(5), ones(5)), False),
        (MultivariateNormal(zeros(5), eye(5)), True),
        (Exponential(rate=ones(1)), True),
        (LogNormal(zeros(1), ones(1)), True),
        (
            MultipleIndependent([
                Exponential(rate=ones(1)),
                BoxUniform(zeros(5), ones(5)),
            ]),
            True,
        ),
        (
            MultipleIndependent([
                Normal(zeros(1), ones(1)),
                MultivariateNormal(zeros(2), eye(2)),
            ]),
            True,
        ),
        (
            # One mixture per parameter, in float64, which `process_prior` wraps.
            Independent(
                MixtureSameFamily(
                    Categorical(ones(2, 2).double()),
                    LogNormal(zeros(2, 2).double(), ones(2, 2).double()),
                ),
                1,
            ),
            True,
        ),
        (
            MultipleIndependent([
                MixtureSameFamily(
                    Categorical(ones(2)),
                    MultivariateNormal(zeros(2, 2), eye(2).expand(2, 2, 2)),
                ),
                BoxUniform(zeros(1), ones(1)),
            ]),
            True,
        ),
    ),
)
def test_mcmc_transform(prior, enable_transform):
    """
    Test whether the transform for MCMC returns the log_abs_det in the correct shape.
    """

    num_samples = 1000
    prior, _, _ = process_prior(prior)
    tf = mcmc_transform(prior, enable_transform=enable_transform)

    samples = prior.sample((num_samples,))
    unconstrained_samples = tf(samples)
    samples_original = tf.inv(unconstrained_samples)

    log_abs_det = tf.log_abs_det_jacobian(samples_original, unconstrained_samples)
    assert log_abs_det.shape == torch.Size([num_samples])


@pytest.mark.parametrize(
    "parts, bounded",
    (
        ([Normal(zeros(1), ones(1)), Normal(zeros(1), ones(1))], False),
        ([MultivariateNormal(zeros(2), eye(2)), Normal(zeros(1), ones(1))], False),
        ([Normal(zeros(1), ones(1)), Exponential(rate=ones(1))], True),
        ([Normal(zeros(1), ones(1)), BoxUniform(zeros(2), ones(2))], True),
    ),
)
def test_prior_support_is_bounded_multiple_independent(parts, bounded):
    """A `MultipleIndependent` prior is unbounded only if all its parts are."""
    assert prior_support_is_bounded(MultipleIndependent(parts)) is bounded


def test_mcmc_transform_z_scores_unbounded_multiple_independent():
    """An unbounded `MultipleIndependent` prior is z-scored like the same prior
    given as one distribution."""
    scale = torch.tensor([100.0, 10.0])
    prior = MultipleIndependent([
        Normal(zeros(1), scale[:1]),
        Normal(zeros(1), scale[1:]),
    ])
    reference = Independent(Normal(zeros(2), scale), 1)
    theta = torch.tensor([[100.0, -20.0]])

    assert torch.allclose(mcmc_transform(prior)(theta), torch.tensor([[1.0, -2.0]]))
    assert torch.allclose(
        mcmc_transform(prior)(theta), mcmc_transform(reference)(theta)
    )


@pytest.mark.gpu
@pytest.mark.parametrize("prior_on_gpu", (True, False))
def test_mcmc_transform_z_scores_across_devices(prior_on_gpu):
    """The z-scoring transform is built on the prior's device and moved to
    `device` afterwards."""
    gpu = process_device("gpu")
    prior_device, device = (gpu, "cpu") if prior_on_gpu else ("cpu", gpu)
    scale = torch.tensor([100.0, 10.0], device=prior_device)
    prior = MultipleIndependent(
        [Normal(zeros(1, device=prior_device), s.reshape(1)) for s in scale],
        device=prior_device,
    )
    theta = torch.tensor([[100.0, -20.0]], device=device)

    assert torch.allclose(
        mcmc_transform(prior, device=device)(theta).cpu(), torch.tensor([[1.0, -2.0]])
    )


@pytest.mark.parametrize(
    "prior",
    (
        Independent(Cauchy(zeros(2), ones(2)), 1),
        MultipleIndependent([Cauchy(zeros(1), ones(1)), Normal(zeros(1), ones(1))]),
        MultipleIndependent([
            Normal(zeros(1), ones(1)),
            Normal(zeros(1), zeros(1), validate_args=False),
        ]),
    ),
)
def test_mcmc_transform_skips_z_scoring_without_usable_moments(prior):
    """A prior without a finite mean and a finite, nonzero std is left unscaled."""
    theta = torch.tensor([[100.0, -20.0]])
    with pytest.warns(UserWarning, match="not z-scored"):
        transform = mcmc_transform(prior)
    assert torch.allclose(transform(theta), theta)


@pytest.mark.skipif(
    not hasattr(constraints, "MixtureSameFamilyConstraint"),
    reason="Torch < 2.8 returns the component support for a mixture.",
)
def test_mcmc_transform_rejects_mixture_with_per_component_bounds():
    """Torch's mixture support is wrong when the bounds depend on the component."""
    prior = MixtureSameFamily(
        Categorical(ones(2)),
        Independent(Uniform(zeros(2, 2), torch.tensor([[1.0, 1.0], [3.0, 3.0]])), 1),
    )
    with pytest.raises(ValueError, match="depends on the parameters"):
        mcmc_transform(prior)


@pytest.mark.xfail(
    reason="Known issue in PyTorch https://github.com/pytorch/pytorch/issues/20682",
    strict=True,
    raises=AssertionError,
)
def test_transformed_dist_support():
    """
    Test whether the support of the transformed distribution is correct.
    This is a known pytorch issue
    SBI issue: https://github.com/sbi-dev/sbi/issues/738
    PyTorch issue: https://github.com/pytorch/pytorch/issues/20682
    """
    base = Uniform(zeros(1), ones(1))
    dist = TransformedDistribution(base, AffineTransform(zeros(1), ones(1)))
    assert not dist.support.check(100 * ones(1)).item(), (
        "Support check failed with python version={} and torch version={}".format(
            sys.version, torch.__version__
        )
    )
