# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

import pickle

import pytest
import torch

from sbi import utils as utils
from sbi.inference import FMPE, NLE, NPE, NPSE, NRE
from sbi.inference.posteriors.posterior_parameters import (
    DirectPosteriorParameters,
    MCMCPosteriorParameters,
    RejectionPosteriorParameters,
    VIPosteriorParameters,
    VectorFieldPosteriorParameters,
)
from sbi.inference.posteriors.vi_posterior import VIPosterior


@pytest.mark.parametrize(
    "inference_method, posterior_parameters",
    (
        (NPE, DirectPosteriorParameters),
        (NPSE, VectorFieldPosteriorParameters),
        (FMPE, VectorFieldPosteriorParameters),
        pytest.param(NLE, MCMCPosteriorParameters, marks=pytest.mark.mcmc),
        pytest.param(NRE, MCMCPosteriorParameters, marks=pytest.mark.mcmc),
        pytest.param(NRE, VIPosteriorParameters, marks=pytest.mark.mcmc),
        (NRE, RejectionPosteriorParameters),
    ),
)
def test_picklability(
    inference_method,
    posterior_parameters,
    tmp_path,
    mcmc_params_fast: MCMCPosteriorParameters,
):
    num_dim = 2
    prior = utils.BoxUniform(low=-2 * torch.ones(num_dim), high=2 * torch.ones(num_dim))
    x_o = torch.zeros(1, num_dim)

    theta = prior.sample((500,))
    x = theta + 1.0 + torch.randn_like(theta) * 0.1

    inference = inference_method(prior=prior)
    _ = inference.append_simulations(theta, x).train(max_num_epochs=1)
    if posterior_parameters is MCMCPosteriorParameters:
        posterior = inference.build_posterior(
            posterior_parameters=mcmc_params_fast
        ).set_default_x(x_o)
    else:
        posterior = inference.build_posterior(
            posterior_parameters=posterior_parameters()
        ).set_default_x(x_o)
    # After sample and log_prob, the posterior should still be picklable
    if isinstance(posterior, VIPosterior):
        posterior.train(max_num_iters=10)
    _ = posterior.sample((1,))
    _ = posterior.potential(torch.zeros(1, num_dim))

    with open(f"{tmp_path}/saved_posterior.pickle", "wb") as handle:
        pickle.dump(posterior, handle)
    with open(f"{tmp_path}/saved_posterior.pickle", "rb") as handle:
        _ = pickle.load(handle)

    with open(f"{tmp_path}/saved_inference.pickle", "wb") as handle:
        pickle.dump(inference, handle)
    with open(f"{tmp_path}/saved_inference.pickle", "rb") as handle:
        _ = pickle.load(handle)


def test_mdn_transform_to_unconstrained_picklable():
    """An MDN using transform_to_unconstrained survives a pickle round-trip.

    Regression: sbi stores the prior transform as an inverse transform, whose data
    lives behind the `_inv` link that torch's Transform.__getstate__ nulls on
    pickling. Naively pickling therefore discards the transform's tensors and yields
    an object that raises "_inv must not be None" on first use.
    """
    from sbi.neural_nets.net_builders.mdn import build_mdn
    from sbi.utils import BoxUniform

    prior = BoxUniform(-2 * torch.ones(2), 2 * torch.ones(2))
    bx, by = prior.sample((256,)), torch.randn(256, 3)
    est = build_mdn(bx, by, z_score_x="transform_to_unconstrained", x_dist=prior)

    theta, cond = prior.sample((5,)).unsqueeze(1), torch.randn(1, 3)
    expected = est.log_prob(theta, cond)

    reloaded = pickle.loads(pickle.dumps(est))
    actual = reloaded.log_prob(theta, cond)

    assert torch.allclose(expected, actual)


def test_zuko_transform_to_unconstrained_picklable():
    """A Zuko flow using transform_to_unconstrained survives a pickle round-trip.

    Same root cause as the MDN case, but the transform is wrapped in a
    CallableTransform inside the flow, and is additionally referenced by
    ZukoFlow._prior_transform — both must come back consistent.
    """
    from sbi.neural_nets.net_builders.flow import build_zuko_maf
    from sbi.utils import BoxUniform

    prior = BoxUniform(-2 * torch.ones(2), 2 * torch.ones(2))
    bx, by = prior.sample((256,)), torch.randn(256, 3)
    est = build_zuko_maf(bx, by, z_score_x="transform_to_unconstrained", x_dist=prior)

    theta, cond = prior.sample((5,)).unsqueeze(1), torch.randn(1, 3)
    expected = est.log_prob(theta, cond)

    reloaded = pickle.loads(pickle.dumps(est))
    actual = reloaded.log_prob(theta, cond)

    assert torch.allclose(expected, actual)
