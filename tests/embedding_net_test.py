# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Affero General Public License v3, see <https://www.gnu.org/licenses/>.

from __future__ import annotations

import pytest
import torch
from torch import eye, ones, zeros

from sbi import utils as utils
from sbi.inference import SNLE, SNPE, SNRE
from sbi.neural_nets.embedding_nets import FCEmbedding, PermutationInvariantEmbedding
from sbi.simulators.linear_gaussian import linear_gaussian
from sbi.utils import classifier_nn, likelihood_nn, posterior_nn


@pytest.mark.parametrize("method", ["SNPE", "SNLE", "SNRE"])
@pytest.mark.parametrize("num_dim", [1, 2])
@pytest.mark.parametrize("embedding_net", ["mlp"])
def test_embedding_net_api(method, num_dim: int, embedding_net: str):
    """Tests the API when using a preconfigured embedding net."""

    x_o = zeros(1, num_dim)

    # likelihood_mean will be likelihood_shift+theta
    likelihood_shift = -1.0 * ones(num_dim)
    likelihood_cov = 0.3 * eye(num_dim)

    prior = utils.BoxUniform(-2.0 * ones(num_dim), 2.0 * ones(num_dim))

    theta = prior.sample((1000,))
    x = linear_gaussian(theta, likelihood_shift, likelihood_cov)

    if embedding_net == "mlp":
        embedding = FCEmbedding(input_dim=num_dim)
    else:
        raise NameError(f"{embedding_net} not supported.")

    if method == "SNPE":
        density_estimator = posterior_nn("maf", embedding_net=embedding)
        inference = SNPE(
            prior, density_estimator=density_estimator, show_progress_bars=False
        )
    elif method == "SNLE":
        density_estimator = likelihood_nn("maf", embedding_net=embedding)
        inference = SNLE(
            prior, density_estimator=density_estimator, show_progress_bars=False
        )
    elif method == "SNRE":
        classifier = classifier_nn("resnet", embedding_net_x=embedding)
        inference = SNRE(prior, classifier=classifier, show_progress_bars=False)
    else:
        raise NameError

    _ = inference.append_simulations(theta, x).train(max_num_epochs=5)
    posterior = inference.build_posterior().set_default_x(x_o)

    s = posterior.sample((1,))
    _ = posterior.potential(s)


@pytest.mark.parametrize("num_trials", [1, 10])
@pytest.mark.parametrize("num_dim", [1, 2])
def test_iid_embedding_api(num_trials, num_dim):

    prior = utils.BoxUniform(-2.0 * ones(num_dim), 2.0 * ones(num_dim))

    num_thetas = 1000
    theta = prior.sample((num_thetas,))

    # simulate iid x.
    iid_theta = theta.reshape(num_thetas, 1, num_dim).repeat(1, num_trials, 1)
    x = torch.randn_like(iid_theta) + iid_theta
    x_o = zeros(1, num_trials, num_dim)

    output_dim = 5
    single_trial_net = FCEmbedding(input_dim=num_dim, output_dim=output_dim)
    embedding_net = PermutationInvariantEmbedding(
        single_trial_net,
        latent_dim=output_dim,
    )

    density_estimator = posterior_nn("maf", embedding_net=embedding_net)
    inference = SNPE(prior, density_estimator=density_estimator)

    _ = inference.append_simulations(theta, x).train(max_num_epochs=5)
    posterior = inference.build_posterior().set_default_x(x_o)

    s = posterior.sample((1,))
    _ = posterior.potential(s)
