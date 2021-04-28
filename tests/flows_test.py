from __future__ import annotations

import numpy as np
import pytest
import torch

from sbi.vi.flows import build_flow, TYPES
from torch.distributions.constraints import real, simplex, unit_interval, interval


@pytest.mark.parametrize("type", TYPES)
def test_build_flow(type):
    for d in range(2, 5):
        p = build_flow(d, type=type)
        sample = p.sample()
        log_prob = p.log_prob(sample)
        assert int(sample.shape[-1]) == d
        assert not torch.isnan(log_prob).all()


@pytest.mark.parametrize("type", TYPES)
def test_parameters(type):
    for d in range(2, 5):
        p = build_flow(d, type=type)
        for p in p.parameters():
            assert p.requires_grad
            assert isinstance(p, torch.Tensor)


@pytest.mark.parametrize("support", (real, simplex, unit_interval, interval(-1, 2)))
def test_support(support):
    dim = 2
    for type in TYPES:
        p = build_flow(dim, support=support, type=type)
        samples = p.sample((1000,))
        assert support.check(samples).all()

