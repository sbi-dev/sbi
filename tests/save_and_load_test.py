# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

import pickle

import pytest
import torch

from sbi import utils as utils
from sbi.inference import NLE, NPE, NRE
from sbi.inference.posteriors.vi_posterior import VIPosterior


@pytest.mark.parametrize(
    "inference_method, sampling_method",
    (
        (NPE, "direct"),
        pytest.param(NLE, "mcmc", marks=pytest.mark.mcmc),
        pytest.param(NRE, "mcmc", marks=pytest.mark.mcmc),
        pytest.param(NRE, "vi", marks=pytest.mark.mcmc),
        (NRE, "rejection"),
    ),
)
def test_picklability(
    inference_method, sampling_method: str, tmp_path, mcmc_params_fast
):
    num_dim = 2
    prior = utils.BoxUniform(low=-2 * torch.ones(num_dim), high=2 * torch.ones(num_dim))
    x_o = torch.zeros(1, num_dim)

    theta = prior.sample((500,))
    x = theta + 1.0 + torch.randn_like(theta) * 0.1

    inference = inference_method(prior=prior)
    _ = inference.append_simulations(theta, x).train(max_num_epochs=1)
    posterior = inference.build_posterior(
        sample_with=sampling_method, mcmc_parameters=mcmc_params_fast
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
