# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

import logging
import warnings
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from torch.distributions import Independent, Normal, Uniform

from sbi.inference import DirectPosterior
from sbi.inference.posteriors.rejection_posterior import RejectionPosterior
from sbi.inference.potentials.base_potential import CustomPotentialWrapper
from sbi.neural_nets import posterior_nn
from sbi.samplers.rejection import accept_reject_sample, rejection_sample
from sbi.utils import BoxUniform, RestrictedPrior
from sbi.utils.sbiutils import warn_if_outside_prior_support


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


@pytest.mark.parametrize(
    "accept, suggest_options, blames, bypass",
    (
        (True, True, False, False),
        (False, True, True, True),
        (False, False, True, False),
    ),
)
def test_accept_reject_sample_timeout_blames_only_low_acceptance(
    accept, suggest_options, blames, bypass
):
    """The timeout error blames low acceptance only after a low-acceptance batch.

    A fake clock lets each batch take one second, so the deadline of half a second
    always runs out during the first batch.
    """
    clock = [0.0]

    def slow_proposal(shape, **kwargs):
        clock[0] += 1.0
        return torch.randn(shape[0], 1)

    fake_time = SimpleNamespace(time=lambda: clock[0])
    with (
        patch("sbi.samplers.rejection.rejection.time", fake_time),
        pytest.raises(RuntimeError, match="rejection sampling exceeded") as error,
    ):
        accept_reject_sample(
            proposal=slow_proposal,
            accept_reject_fn=lambda x: torch.full((x.shape[0],), accept),
            num_samples=20,
            max_sampling_batch_size=10,
            max_sampling_time=0.5,
            suggest_sampling_options=suggest_options,
        )
    assert ("low acceptance" in str(error.value)) == blames
    assert ("reject_outside_prior" in str(error.value)) == bypass


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

        def bind(self, x_o, x_is_iid=True):
            return self

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


def test_accept_reject_sample_partial_return():
    """Test that return_partial_on_timeout returns collected samples."""

    def accept_rare_fn(x):
        # Accept only 1% of samples to ensure we don't finish
        return torch.rand(x.shape[0]) < 0.01

    proposal = DummyProposal()
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        samples, acceptance = accept_reject_sample(
            proposal=proposal,
            accept_reject_fn=accept_rare_fn,
            num_samples=10000,  # Request many samples
            max_sampling_time=0.001,  # Very short timeout
            return_partial_on_timeout=True,
        )
        # Should have some samples (not all 10000)
        assert samples.shape[0] > 0
        assert samples.shape[0] < 10000
        # Same event shape as a complete run, without an extra x_o dim.
        assert samples.shape[1:] == (1,)
        # Should have issued a warning
        assert len(w) == 1
        assert "partial results" in str(w[0].message).lower()


def _accept_reject(num_samples):
    proposal = Uniform(-torch.ones(1), torch.ones(1))
    samples, _ = accept_reject_sample(
        proposal=proposal.sample,
        accept_reject_fn=lambda theta: theta[:, 0] > 0.99,  # About 0.5 %.
        num_samples=num_samples,
    )
    return samples


def _rejection(num_samples):
    proposal = Independent(Normal(torch.zeros(1), torch.ones(1)), 1)
    # The target is the proposal, so `m` alone sets the acceptance: 1/200 = 0.5 %.
    samples, _ = rejection_sample(
        proposal.log_prob, proposal, num_samples=num_samples, m=200.0
    )
    return samples


@pytest.mark.parametrize(
    "sample", (_accept_reject, _rejection), ids=("accept_reject", "rejection")
)
@pytest.mark.parametrize(
    ("num_samples", "expect_warning"),
    (
        # The second batch finishes sampling with samples to spare, so none remain
        # to warn about.
        (10, False),
        (1000, True),
    ),
)
def test_low_acceptance_warning_only_while_samples_remain(
    caplog, sample, num_samples, expect_warning
):
    torch.manual_seed(0)

    with caplog.at_level(logging.WARNING):
        samples = sample(num_samples)

    assert samples.shape == (num_samples, 1)
    assert ("may take a long time" in caplog.text) is expect_warning


def test_zero_samples_returns_empty_tensor_with_requested_shape():
    samples, rates = accept_reject_sample(
        proposal=DummyProposal(),
        accept_reject_fn=always_reject_fn,
        num_samples=0,
    )
    assert samples.shape == (0, 1)
    assert rates.shape == (1,)

    def batched_proposal(shape, **kwargs):
        return torch.randn(shape[0], 2, 3)

    samples, rates = accept_reject_sample(
        proposal=batched_proposal,
        accept_reject_fn=lambda theta: torch.ones(theta.shape[0], 2, dtype=torch.bool),
        num_samples=0,
        num_xos=2,
    )
    assert samples.shape == (0, 2, 3)
    assert rates.shape == (2,)

    samples, _ = accept_reject_sample(
        proposal=lambda shape, **kwargs: torch.randn(shape[0], 1, 3),
        accept_reject_fn=lambda theta: torch.ones(theta.shape[0], 1, dtype=torch.bool),
        num_samples=0,
        num_xos=1,
    )
    assert samples.shape == (0, 1, 3)

    proposal = Uniform(-torch.ones(1), torch.ones(1))

    def accept_all(theta):
        return torch.ones(theta.shape[0], dtype=torch.bool)

    samples, _ = accept_reject_sample(
        proposal=proposal.sample,
        accept_reject_fn=accept_all,
        num_samples=0,
    )
    positive, _ = accept_reject_sample(
        proposal=proposal.sample,
        accept_reject_fn=accept_all,
        num_samples=1,
        max_sampling_batch_size=1,
    )
    assert samples.shape == (0, 1)
    assert samples.shape[1:] == positive.shape[1:]

    samples, _ = rejection_sample(
        potential_fn=lambda theta: torch.zeros(theta.shape[0]),
        proposal=proposal,
        num_samples=0,
    )
    assert samples.shape == (0, 1)

    batched = Independent(Normal(torch.zeros(2, 3), torch.ones(2, 3)), 1)
    samples, _ = accept_reject_sample(
        proposal=batched.sample,
        accept_reject_fn=lambda theta: torch.ones(theta.shape[0], 2, dtype=torch.bool),
        num_samples=0,
        num_xos=2,
    )
    assert samples.shape == (0, 2, 3)

    class MetaProposal:
        def __call__(self, shape, **kwargs):
            return torch.empty(shape[0], 1, device="meta")

    samples, rates = accept_reject_sample(
        proposal=MetaProposal(),
        accept_reject_fn=always_reject_fn,
        num_samples=0,
    )
    assert samples.device.type == rates.device.type == "meta"

    prior = BoxUniform(torch.zeros(2), torch.ones(2))
    theta = prior.sample((100,))
    x = torch.randn(100, 2)
    estimator = posterior_nn("maf", hidden_features=8, num_transforms=1)(theta, x)
    posterior = DirectPosterior(estimator, prior).set_default_x(torch.zeros(2))

    assert posterior.sample((0,)).shape == (0, 2)
    assert posterior.sample_batched((0,), x=torch.zeros(3, 2)).shape == (0, 3, 2)
    restricted = RestrictedPrior(
        prior, lambda theta: torch.ones(theta.shape[0], dtype=torch.bool)
    )
    assert restricted.sample((0,)).shape == (0, 2)
    composed = RestrictedPrior(
        restricted, lambda theta: torch.ones(theta.shape[0], dtype=torch.bool)
    )
    assert composed.sample((0,)).shape == (0, 2)
    with pytest.raises(ValueError, match="batch dimension"):
        posterior.sample((5,), x=torch.empty(0, 2))
    with pytest.raises(ValueError, match="batch dimension"):
        posterior.sample_batched((5,), x=torch.empty(0, 2))


def test_zero_samples_return_undefined_acceptance_rate():
    """An empty request reports `nan` instead of an acceptance rate."""
    proposal = Uniform(-torch.ones(1), torch.ones(1))

    samples, acceptance_rate = rejection_sample(
        potential_fn=lambda theta: torch.zeros(theta.shape[0]),
        proposal=proposal,
        num_samples=0,
    )
    assert samples.shape == (0, 1)
    assert acceptance_rate.shape == ()
    assert torch.isnan(acceptance_rate)

    condition = torch.zeros(2, 1)
    samples, acceptance_rates = accept_reject_sample(
        proposal=lambda shape, **kwargs: torch.randn(shape[0], *condition.shape),
        accept_reject_fn=lambda theta: torch.ones(theta.shape[0], 2, dtype=torch.bool),
        num_samples=0,
        proposal_sampling_kwargs={"condition": condition},
    )
    assert samples.shape == (0, 2, 1)
    assert acceptance_rates.shape == (2,)
    assert torch.isnan(acceptance_rates).all()


def test_zero_samples_raise_when_shape_cannot_be_inferred():
    class FailingProposal:
        def __call__(self, shape, **kwargs):
            raise RuntimeError("empty draws are not supported")

    with pytest.raises(ValueError, match="Cannot determine the shape and device"):
        accept_reject_sample(
            proposal=FailingProposal(),
            accept_reject_fn=always_reject_fn,
            num_samples=0,
        )


def test_zero_rejection_posterior_does_not_probe_proposal():
    prior = BoxUniform(torch.zeros(2), torch.ones(2))
    potential = CustomPotentialWrapper(
        lambda theta, x_o: torch.zeros(theta.shape[0]), prior=prior
    )

    class FailingProposal:
        def sample(self, shape, **kwargs):
            raise AssertionError("zero-sample request must not probe the proposal")

    posterior = RejectionPosterior(potential_fn=potential, proposal=FailingProposal())
    assert posterior.sample((0,), x=torch.zeros(1)).shape == (0, 2)


def test_rejection_sample_stops_without_progress():
    with pytest.raises(RuntimeError, match="No samples were accepted after"):
        accept_reject_sample(
            proposal=DummyProposal(),
            accept_reject_fn=always_reject_fn,
            num_samples=1,
            max_sampling_batch_size=1,
        )


def test_warn_if_outside_prior_support():
    """Test the warning utility for samples outside prior support."""
    prior = Uniform(torch.zeros(2), torch.ones(2))

    # Samples inside prior - no warning expected
    samples_inside = torch.rand(100, 2)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        warn_if_outside_prior_support(prior, samples_inside)
        assert len(w) == 0

    # Samples mostly outside prior - warning expected
    samples_outside = torch.rand(100, 2) + 2.0  # All outside [0, 1]
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        warn_if_outside_prior_support(prior, samples_outside)
        assert len(w) == 1
        assert "outside the prior support" in str(w[0].message)

    # Samples just below threshold (5%) - no warning expected
    samples_mixed = torch.rand(100, 2)
    samples_mixed[:4] = 2.0  # 4% outside
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        warn_if_outside_prior_support(prior, samples_mixed, threshold=0.05)
        assert len(w) == 0

    # Samples just above threshold - warning expected
    samples_mixed = torch.rand(100, 2)
    samples_mixed[:6] = 2.0  # 6% outside
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        warn_if_outside_prior_support(prior, samples_mixed, threshold=0.05)
        assert len(w) == 1
