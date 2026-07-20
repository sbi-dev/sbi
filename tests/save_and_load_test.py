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


@pytest.mark.parametrize("builder_name", ["mdn", "zuko"])
def test_unconstraining_transform_survives_pickle_and_dtype_cast(builder_name):
    """Transformed MDN and Zuko estimators remain usable after serialization."""
    from sbi.neural_nets.net_builders.flow import build_zuko_maf
    from sbi.neural_nets.net_builders.mdn import build_mdn
    from sbi.utils import BoxUniform

    builder = build_mdn if builder_name == "mdn" else build_zuko_maf
    prior = BoxUniform(-2 * torch.ones(2), 2 * torch.ones(2))
    bx, by = prior.sample((256,)), torch.randn(256, 3)
    estimator = builder(bx, by, z_score_x="transform_to_unconstrained", x_dist=prior)

    theta, condition = prior.sample((5,)).unsqueeze(1), torch.randn(1, 3)
    expected = estimator.log_prob(theta, condition)
    reloaded = pickle.loads(pickle.dumps(estimator))
    assert torch.allclose(reloaded.log_prob(theta, condition), expected)

    reloaded.double()
    actual = reloaded.log_prob(theta.double(), condition.double())
    assert actual.dtype == torch.float64
    assert torch.isfinite(actual).all()
