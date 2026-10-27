# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

import pytest
import torch
from torch.testing import assert_close

from sbi.inference import DirectPosterior, EnsemblePosterior
from sbi.neural_nets import posterior_nn
from sbi.utils import BoxUniform
from sbi.utils.torchutils import process_device


def make_posteriors(device):
    """Build small real estimators on CPU before moving each complete posterior."""
    posteriors = []
    for _ in range(2):
        # Moving a posterior mutates its prior, so do not share it during setup.
        prior = BoxUniform(-torch.ones(2), torch.ones(2))
        theta = prior.sample((32,))
        x = theta + 0.1 * torch.randn_like(theta)
        estimator = posterior_nn("mdn", hidden_features=8, num_components=2)(theta, x)
        posterior = DirectPosterior(estimator, prior)
        posterior.to(device)
        posteriors.append(posterior)
    return posteriors


@pytest.mark.parametrize("device", ["cpu", pytest.param("gpu", marks=pytest.mark.gpu)])
@pytest.mark.parametrize("spelling", ["inferred", "alias", "canonical", "torch_device"])
def test_ensemble_constructor_device_and_reweighting(device, spelling):
    """Resolve devices before moving weights and preserve bound weight snapshots."""
    resolved = process_device(device)
    selected = {
        "inferred": None,
        "alias": device,
        "canonical": resolved,
        "torch_device": torch.device(resolved),
    }[spelling]
    posteriors = make_posteriors(resolved)
    ensemble = EnsemblePosterior(posteriors, weights=[3.0, 1.0], device=selected)
    theta = torch.tensor([[0.0, 0.0], [0.25, -0.25]], device=resolved)
    x = torch.zeros(1, 2, device=resolved)
    ensemble.set_default_x(x)
    old_value = ensemble.potential(theta).detach().clone()
    working = ensemble.potential_fn
    bound = working.bind(x)

    for weights in ([1.0, 3.0], torch.tensor([2.0, 1.0]), None):
        # potential(theta) binds a new working potential; only the setter must
        # retain the current one.
        working = ensemble.potential_fn
        ensemble.weights = weights
        assert ensemble.potential_fn is working
        assert working._weights is ensemble.weights
        assert_close(bound(theta), old_value)
        assert_close(working.x_o, x)
        expected = torch.logsumexp(
            torch.stack([
                p.log_prob(theta, x=x, norm_posterior=False) for p in posteriors
            ])
            + ensemble.weights.log().reshape(-1, 1),
            dim=0,
        )
        assert ensemble.weights.device == torch.device(resolved)
        assert working.device == resolved
        assert_close(ensemble.potential(theta), expected)
        assert_close(ensemble.log_prob(theta, norm_posterior=False), expected)


@pytest.mark.gpu
@pytest.mark.parametrize(
    "component_device, selected_device", [("cpu", "gpu"), ("gpu", "cpu")]
)
def test_ensemble_constructor_rejects_device_mismatch(
    component_device, selected_device
):
    """Reject mismatches without moving caller-owned components implicitly."""
    source = process_device(component_device)
    target = process_device(selected_device)
    posteriors = make_posteriors(source)
    with pytest.raises(ValueError, match="ensemble.to"):
        EnsemblePosterior(posteriors, device=target)
    assert all(p._device == source for p in posteriors)
    assert all(
        next(p.posterior_estimator.parameters()).device == torch.device(source)
        for p in posteriors
    )

    ensemble = EnsemblePosterior(posteriors)
    ensemble.to(selected_device)
    theta = torch.zeros(2, 2, device=target)
    x = torch.zeros(1, 2, device=target)
    ensemble.set_default_x(x)
    assert ensemble.potential(theta).device == torch.device(target)
    assert ensemble.log_prob(theta, norm_posterior=False).device == torch.device(target)


@pytest.mark.gpu
@pytest.mark.parametrize("device", [None, "cpu", "gpu"])
def test_ensemble_constructor_rejects_mixed_component_devices(device):
    """Explicit device selection must not bypass the component device check."""
    posteriors = make_posteriors("cpu")
    posteriors[1].to("gpu")
    with pytest.raises(AssertionError, match="all posteriors are on the same device"):
        EnsemblePosterior(posteriors, device=device)
