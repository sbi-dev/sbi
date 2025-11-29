# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

from unittest.mock import patch

import pytest
import torch

from sbi.inference.posteriors.rejection_posterior import RejectionPosterior
from sbi.samplers.rejection import accept_reject_sample, rejection_sample


class DummyProposal:
    def __call__(self, shape, **kwargs):
        return torch.randn(shape[0], 1)

    def sample(self, shape, **kwargs):
        return torch.randn(shape[0], 1)

    def log_prob(self, x, **kwargs):
        return -0.5 * x.pow(2).sum(dim=-1)


def always_reject_fn(x):
    return torch.zeros(x.shape[0], dtype=torch.bool)


def test_accept_reject_sample_timeout():
    proposal = DummyProposal()
    with pytest.raises(RuntimeError, match="rejection sampling exceeded"):
        accept_reject_sample(
            proposal=proposal,
            accept_reject_fn=always_reject_fn,
            num_samples=5,
            max_sampling_time=0.01,
        )


def test_rejection_sample_timeout():
    proposal = DummyProposal()

    def dummy_potential_fn(x):
        return torch.full((x.shape[0],), -1e6)

    with pytest.raises(RuntimeError, match="rejection sampling exceeded"):
        rejection_sample(
            potential_fn=dummy_potential_fn,
            proposal=proposal,
            num_samples=5,
            max_sampling_time=0.01,
            m=1e12,
        )


def test_reject_outside_prior_support_behavior():
    class UniformProposal:
        def sample(self, shape, **kwargs):
            return torch.rand(shape[0], 1)

        def __call__(self, shape, **kwargs):
            return self.sample(shape, **kwargs)

        def log_prob(self, x, **kwargs):
            inside = (x >= 0.0) & (x <= 1.0)
            return torch.where(
                inside.squeeze(-1),
                torch.zeros(x.shape[0]),
                torch.full((x.shape[0],), -float("inf")),
            )

    class BadProposal:
        def sample(self, shape, **kwargs):
            return torch.full((shape[0], 1), 5.0)

        def __call__(self, shape, **kwargs):
            return self.sample(shape, **kwargs)

        def log_prob(self, x, **kwargs):
            return -0.5 * x.pow(2).sum(dim=-1)

    class SimplePotential:
        device = "cpu"

        def __call__(self, theta, x_o=None):
            return torch.zeros(theta.shape[0])

        def set_x(self, x):
            pass

        def to(self, device):
            self.device = device
            return self

    bounded_posterior = RejectionPosterior(
        potential_fn=SimplePotential(),
        proposal=UniformProposal(),
    )
    bounded_posterior.set_default_x(torch.zeros(1))

    with patch(
        "sbi.inference.posteriors.rejection_posterior.rejection_sample"
    ) as patched:
        patched.return_value = (torch.rand(5, 1), torch.tensor(1.0))
        samples = bounded_posterior.sample((5,), reject_outside_prior=True)

    assert torch.all((samples >= 0.0) & (samples <= 1.0))

    unbounded_posterior = RejectionPosterior(
        potential_fn=SimplePotential(),
        proposal=BadProposal(),
    )
    unbounded_posterior.set_default_x(torch.zeros(1))

    with patch(
        "sbi.inference.posteriors.rejection_posterior.rejection_sample"
    ) as patched:
        patched.return_value = (torch.full((5, 1), 5.0), torch.tensor(1.0))
        no_reject = unbounded_posterior.sample((5,), reject_outside_prior=False)

    assert torch.all(no_reject == 5.0)
