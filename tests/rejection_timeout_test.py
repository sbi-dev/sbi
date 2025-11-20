# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

import pytest
import torch

from sbi.inference.posteriors.rejection_posterior import RejectionPosterior
from sbi.samplers.rejection import accept_reject_sample, rejection_sample


class DummyProposal:
    """Callable proposal returning Gaussian samples and log-prob."""

    def __call__(self, shape, **kwargs):
        return torch.randn(shape[0], 1)

    def sample(self, shape, **kwargs):
        return torch.randn(shape[0], 1)

    def log_prob(self, x, **kwargs):
        return -0.5 * x.pow(2).sum(dim=-1)


def always_reject_fn(x):
    return torch.zeros(x.shape[0], dtype=torch.bool)


@pytest.mark.slow
def test_accept_reject_sample_timeout():
    proposal = DummyProposal()

    with pytest.raises(RuntimeError, match="rejection sampling exceeded"):
        accept_reject_sample(
            proposal=proposal,
            accept_reject_fn=always_reject_fn,
            num_samples=5,
            max_sampling_time=0.01,
        )


@pytest.mark.slow
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


@pytest.mark.slow
def test_rejection_posterior_timeout():
    prior = torch.distributions.MultivariateNormal(torch.zeros(1), torch.eye(1))

    class DummyPotential:
        """
        Minimal compliant potential implementing CustomPotential interface.
        Forces rejection by returning very low log-probability.
        """

        device = "cpu"

        def __call__(
            self, theta: torch.Tensor, x_o: torch.Tensor = None
        ) -> torch.Tensor:
            return torch.full((theta.shape[0],), -1e6)

        def set_x(self, x):
            pass

        def to(self, device):
            self.device = device
            return self

    posterior = RejectionPosterior(
        potential_fn=DummyPotential(),
        proposal=prior,
    )

    posterior.set_default_x(torch.zeros(1))

    with pytest.raises(RuntimeError, match="max_sampling_time"):
        posterior.sample((5,), max_sampling_time=0.01)
