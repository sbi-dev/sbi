from __future__ import annotations

import numpy as np
import pytest
import torch

from tests.test_utils import check_c2st

from sbi.vi.mixture_of_flows import (
    MixtureDiagGaussians,
    MixtureFullGaussians,
    MixtureAffineAutoregressive,
    MixtureSplineAutoregressive,
    MixtureSameTransform,
)

_MODELS = [
    MixtureDiagGaussians,
    MixtureFullGaussians,
    MixtureAffineAutoregressive,
    MixtureSplineAutoregressive,
]


def kolmogoriv_smirnov_test_uniform(samples):
    """ Tests if samples are uniformly distributed. """
    N = samples.shape[0]
    d = samples.shape[-1]
    # Cdf of uniform is linear
    x = torch.linspace(0, 1, 1000)
    # empirical cdf
    F_n = torch.stack([1 / N * (torch.sum(samples < x_i, 0)) for x_i in x])
    # Deviation
    d = torch.abs(F_n.T - x)
    T = np.sqrt(N) * d.max()
    return T < 2


@pytest.mark.parametrize("model", _MODELS)
def test_standardization(model):
    p = model(3, 2)
    samples = p.sample((10000,))
    eps = p.standardize(samples)
    assert kolmogoriv_smirnov_test_uniform(eps)
    assert not kolmogoriv_smirnov_test_uniform(samples)


@pytest.mark.parametrize(
    "num_components,event_dim", [(2, 3), (3, 2), (3, 3), (4, 3), (3, 4), (4, 4)]
)
def test_mean_var_diag_gauss(num_components, event_dim):
    loc = torch.ones(num_components, event_dim)
    scale = torch.ones(num_components, event_dim)

    p = MixtureDiagGaussians(num_components, event_dim, loc=loc, scale=scale)
    assert torch.isclose(p.mean, torch.ones(event_dim)).all()
    assert torch.isclose(p.variance, torch.ones(event_dim)).all()


@pytest.mark.parametrize(
    "num_components,event_dim", [(2, 3), (3, 2), (3, 3), (4, 3), (3, 4), (4, 4)]
)
def test_mean_var_full_gauss(num_components, event_dim):
    loc = torch.ones(num_components, event_dim)
    scale = torch.eye(event_dim)
    scale = scale.reshape((1, event_dim, event_dim))
    scale = scale.repeat(num_components, 1, 1)

    p = MixtureFullGaussians(num_components, event_dim, loc=loc, scale_tril=scale)
    assert torch.isclose(p.mean, torch.ones(event_dim)).all()
    assert torch.isclose(p.variance, torch.ones(event_dim)).all()


@pytest.mark.parametrize("model", _MODELS)
def test_parameters(model):
    p = model(3, 2)
    paras = list(p.parameters())
    assert paras[0].shape[0] == 3
    for p in paras:
        assert isinstance(p, torch.Tensor)


@pytest.mark.parametrize("model", _MODELS)
def test_cache_correct(model):
    p = model(3, 2)
    s = p.sample()
    logp1 = p.log_prob(s)
    for t in p.transforms:
        t.clear_cache()
    logp2 = p.log_prob(s)
    assert (logp1 == logp2).all()


@pytest.mark.parametrize(
    "num_components,event_dim", [(2, 3), (3, 2), (3, 3), (4, 3), (3, 4), (4, 4)]
)
def test_sample_and_logprob_shapes(num_components, event_dim):

    BATCH_SIZE = 10

    for m in _MODELS:
        p = m(num_components, event_dim, check_rsample=True)
        sample = p.sample()
        batch_sample = p.sample((BATCH_SIZE,))
        rsample = p.rsample()
        batch_rsample = p.rsample((BATCH_SIZE,))
        assert sample.shape[0] == event_dim
        assert (
            batch_sample.shape[0] == BATCH_SIZE and batch_sample.shape[1] == event_dim
        )
        assert rsample.shape[1] == event_dim
        assert (
            batch_rsample.shape[0] == BATCH_SIZE and batch_sample.shape[1] == event_dim
        )

        logprob = p.log_prob(sample)
        batched_logprob = p.log_prob(batch_sample)
        assert logprob.shape[0] == 1
        assert batched_logprob.shape[0] == BATCH_SIZE
