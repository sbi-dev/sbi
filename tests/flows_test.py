from __future__ import annotations

import numpy as np
import pytest
import torch

from sbi.vi.flows import build_flow, TYPES
from torch.distributions.constraints import real, positive, unit_interval, interval


@pytest.mark.parametrize("type", TYPES)
def test_build_flow(type):
    for d in range(3, 10):
        p = build_flow(d, type=type, num_flows=1)
        sample = p.sample((100,))
        log_prob = p.log_prob(sample)
        assert int(sample.shape[-1]) == d
        assert not torch.isnan(log_prob).all()


@pytest.mark.parametrize("type", TYPES)
def test_parameters(type):
    for d in range(3, 10):
        p = build_flow(d, type=type)
        for p in p.parameters():
            assert p.requires_grad
            assert isinstance(p, torch.Tensor)


@pytest.mark.parametrize(
    "support", (real, unit_interval, interval(-1, 2), interval(-5, 5), positive)
)
def test_support(support):
    dim = 2
    for type in TYPES:
        p = build_flow(dim, support=support, type=type)
        samples = p.sample((1000,))
        samples = samples[torch.isfinite(samples)]
        assert support.check(samples).sum() > 990

