from __future__ import annotations

import pytest
import torch
from torch import eye, ones, zeros
from torch.distributions import MultivariateNormal

from sbi import utils
from sbi.inference import SNLE, SNPE, SNRE
from sbi.neural_nets import classifier_nn, likelihood_nn, posterior_nn
from sbi.neural_nets.embedding_nets import (
    CNNEmbedding,
    FCEmbedding,
    PermutationInvariantEmbedding,
)
from sbi.simulators.linear_gaussian import (
    linear_gaussian,
    true_posterior_linear_gaussian_mvn_prior,
)

from .test_utils import check_c2st


@pytest.mark.mcmc
@pytest.mark.parametrize("method", ["SNPE", "SNLE", "SNRE"])
@pytest.mark.parametrize("num_dim", [1, 2])
@pytest.mark.parametrize("embedding_net", ["mlp"])
def test_embedding_net_api(
    method, num_dim: int, embedding_net: str, mcmc_params_fast: dict
):
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

    _ = inference.append_simulations(theta, x).train(max_num_epochs=2)
    posterior = inference.build_posterior(
        mcmc_method="slice_np_vectorized",
        mcmc_parameters=mcmc_params_fast,
    ).set_default_x(x_o)

    s = posterior.sample((1,))
    _ = posterior.potential(s)


@pytest.mark.parametrize("num_trials", [1, 2])
@pytest.mark.parametrize("num_dim", [1, 2])
def test_embedding_api_with_multiple_trials(num_trials, num_dim):
    """Tests the API when using iid trial-based data."""
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
        trial_net_output_dim=output_dim,
    )

    density_estimator = posterior_nn("maf", embedding_net=embedding_net)
    inference = SNPE(prior, density_estimator=density_estimator)

    _ = inference.append_simulations(theta, x).train(max_num_epochs=5)

    posterior = inference.build_posterior().set_default_x(x_o)

    s = posterior.sample((1,))
    _ = posterior.potential(s)


@pytest.mark.parametrize("input_shape", [(32,), (32, 32), (32, 64)])
@pytest.mark.parametrize("num_channels", (1, 2, 3))
def test_1d_and_2d_cnn_embedding_net(input_shape, num_channels):
    estimator_provider = posterior_nn(
        "mdn",
        embedding_net=CNNEmbedding(
            input_shape, in_channels=num_channels, output_dim=20
        ),
    )

    num_dim = input_shape[0]

    def simulator2d(theta):
        x = MultivariateNormal(
            loc=theta, covariance_matrix=0.5 * torch.eye(num_dim)
        ).sample()
        return x.unsqueeze(2).repeat(1, 1, input_shape[1])

    def simulator1d(theta):
        return torch.rand_like(theta) + theta

    if len(input_shape) == 1:
        simulator = simulator1d
        xo = torch.ones(1, num_channels, *input_shape).squeeze(1)
    else:
        simulator = simulator2d
        xo = torch.ones(1, num_channels, *input_shape).squeeze(1)

    prior = MultivariateNormal(torch.zeros(num_dim), torch.eye(num_dim))

    num_simulations = 1000
    theta = prior.sample(torch.Size((num_simulations,)))
    x = simulator(theta)
    if num_channels > 1:
        x = x.unsqueeze(1).repeat(
            1, num_channels, *[1 for _ in range(len(input_shape))]
        )

    trainer = SNPE(prior=prior, density_estimator=estimator_provider)
    trainer.append_simulations(theta, x).train(max_num_epochs=2)
    posterior = trainer.build_posterior().set_default_x(xo)

    s = posterior.sample((10,))
    posterior.potential(s)


@pytest.mark.slow
def test_npe_with_with_iid_embedding_varying_num_trials(trial_factor=50):
    """Test inference accuracy with embeddings for varying number of trials.

    Test c2st accuracy and permutation invariance for up to 20 trials.
    """
    num_dim = 2
    max_num_trials = 20
    prior = torch.distributions.MultivariateNormal(
        torch.zeros(num_dim), torch.eye(num_dim)
    )

    # Scale number of training samples with num_trials.
    num_thetas = 5000 + trial_factor * max_num_trials

    theta = prior.sample(sample_shape=torch.Size((num_thetas,)))
    num_trials = torch.randint(1, max_num_trials, size=(num_thetas,))

    # simulate iid x, pad smaller number of trials with nans.
    x = ones(num_thetas, max_num_trials, 2) * float("nan")

    for i in range(num_thetas):
        th = theta[i].repeat(num_trials[i], 1)
        x[i, : num_trials[i]] = torch.randn_like(th) + th

    # build embedding net
    output_dim = 5
    single_trial_net = FCEmbedding(input_dim=num_dim, output_dim=output_dim)
    embedding_net = PermutationInvariantEmbedding(
        single_trial_net,
        trial_net_output_dim=output_dim,
        output_dim=output_dim,
        aggregation_fn="sum",
    )

    # test embedding net
    assert embedding_net(x[:3]).shape == (3, output_dim)

    density_estimator = posterior_nn(
        model="mdn",
        embedding_net=embedding_net,
        z_score_x="none",  # turn off z-scoring because of NaN encodings.
        z_score_theta="independent",
    )
    inference = SNPE(prior, density_estimator=density_estimator)

    # do not exclude invalid x, as we padded with nans.
    _ = inference.append_simulations(theta, x, exclude_invalid_x=False).train(
        training_batch_size=100
    )
    posterior = inference.build_posterior()

    num_samples = 1000
    # test different number of trials
    num_test_trials = torch.linspace(1, max_num_trials, 5, dtype=torch.int)
    for num_trials in num_test_trials:
        # x_o must have the same number of trials as x, thus we pad with nans.
        x_o = ones(1, max_num_trials, num_dim) * float("nan")
        x_o[:, :num_trials] = 0.0

        # get reference samples from true posterior
        reference_samples = true_posterior_linear_gaussian_mvn_prior(
            x_o[0, :num_trials, :],  # omit nans
            likelihood_shift=torch.zeros(num_dim),
            likelihood_cov=torch.eye(num_dim),
            prior_cov=prior.covariance_matrix,
            prior_mean=prior.loc,
        ).sample((num_samples,))

        # test inference accuracy and permutation invariance
        num_repeats = 2
        for _ in range(num_repeats):
            trial_permutet_x_o = x_o[:, torch.randperm(x_o.shape[1]), :]
            samples = posterior.sample((num_samples,), x=trial_permutet_x_o)
            check_c2st(
                samples, reference_samples, alg=f"iid-NPE with {num_trials} trials"
            )
