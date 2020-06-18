import torch
import pyro

from sbi.utils.pyroutils import get_transforms


def test_unbounded_transform():
    prior_dim = 10
    prior_params = {
        "low": -1.0 * torch.ones((prior_dim,)),
        "high": +1.0 * torch.ones((prior_dim,)),
    }
    prior_dist = pyro.distributions.Uniform(**prior_params).to_event(1)

    def prior(num_samples=1):
        return pyro.sample("theta", prior_dist.expand_by([num_samples]))

    transforms = get_transforms(prior)

    to_unbounded = transforms["theta"]
    to_bounded = transforms["theta"].inv

    assert to_unbounded(prior(1000)).max() > 1.0
    assert to_bounded(to_unbounded(prior(1000))).max() < 1.0
