# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

import pytest
import torch
from torch import Tensor
from torch.distributions import Bernoulli, Beta, Normal

from sbi.inference import MNPE, MCMCPosterior
from sbi.inference.posteriors.rejection_posterior import RejectionPosterior
from sbi.inference.posteriors.vi_posterior import VIPosterior
from sbi.neural_nets import posterior_nn
from sbi.neural_nets.embedding_nets import FCEmbedding
from sbi.utils import mcmc_transform
from sbi.utils.torchutils import process_device
from sbi.utils.user_input_checks_utils import MultipleIndependent
from tests.test_utils import check_c2st


# toy simulator for continuous data with mixed parameters
def mixed_param_simulator(theta: Tensor) -> Tensor:
    """Simulator for continuous data with mixed parameters.

    Args:
        theta: Parameters with mixed types - continuous and discrete.
               First component is continuous (Beta distributed)
               Second component is discrete (0 or 1)

    Returns:
        x: Continuous observation drawn from a mixture of two normals.
    """
    # Extract parameters
    mixing_weight, component = theta[:, 0], theta[:, 1]

    # Generate data from mixture of normals
    # component 0: N(0,1), component 1: N(3,1)
    means = torch.where(component > 0.5, torch.tensor(3.0), torch.tensor(0.0))

    # Use mixing weight to determine variance
    stds = 0.5 + mixing_weight

    return Normal(means, stds).sample().reshape(-1, 1)


@pytest.mark.mcmc
@pytest.mark.gpu
@pytest.mark.parametrize("device", ("cpu", "gpu"))
def test_mnpe_on_device(
    device,
    mcmc_params_fast: dict,
    num_simulations: int = 100,
    mcmc_method: str = "slice_np",
):
    """Test MNPE API on device."""

    device = process_device(device)

    # Generate data with mixed parameter types
    theta = torch.cat(
        (
            torch.rand(num_simulations, 1),  # continuous
            torch.randint(0, 2, (num_simulations, 1)),  # discrete
        ),
        dim=1,
    ).to(device)
    x = mixed_param_simulator(theta).to(device)

    # Train and infer
    prior = MultipleIndependent([
        Beta(torch.ones(1), torch.ones(1)),  # continuous
        Bernoulli(probs=0.5 * torch.ones(1)),  # discrete (Bernoulli)
    ]).to(device)

    trainer = MNPE(prior=prior, device=device)
    trainer.append_simulations(theta, x).train(max_num_epochs=1)

    # Test sampling on device
    posterior = trainer.build_posterior()
    posterior.sample(
        (1,),
        x=x[0],
        show_progress_bars=False,
        mcmc_method=mcmc_method,
        **mcmc_params_fast,
    )


@pytest.mark.parametrize(
    "sampler", (pytest.param("mcmc", marks=pytest.mark.mcmc), "rejection", "vi")
)
@pytest.mark.parametrize("flow_model", ("mdn", "maf", "nsf", "zuko_nsf"))
@pytest.mark.parametrize("z_score_theta", ("independent", "none"))
def test_mnpe_api(flow_model: str, sampler, mcmc_params_fast: dict, z_score_theta: str):
    """Test MNPE API."""

    # Generate data
    num_simulations = 100
    theta = torch.cat(
        (
            torch.rand(num_simulations, 1),  # continuous
            torch.randint(0, 2, (num_simulations, 1)),  # discrete
        ),
        dim=1,
    )
    x = mixed_param_simulator(theta)

    # Train and infer
    prior = MultipleIndependent([
        Beta(torch.ones(1), torch.ones(1)),  # continuous
        Bernoulli(probs=0.5 * torch.ones(1)),  # discrete (Bernoulli)
    ])

    x_o = x[0]
    # Build estimator manually
    theta_embedding = FCEmbedding(2, 2)  # simple embedding net
    density_estimator = posterior_nn(
        model="mnpe",
        flow_model=flow_model,
        z_score_theta=z_score_theta,
        embedding_net=theta_embedding,
    )
    trainer = MNPE(density_estimator=density_estimator)
    trainer.append_simulations(theta, x).train(max_num_epochs=5)

    # Test different samplers
    posterior = trainer.build_posterior(prior=prior, sample_with=sampler)
    posterior.set_default_x(x_o)
    if isinstance(posterior, VIPosterior):
        posterior.train().sample((1,))
    elif isinstance(posterior, RejectionPosterior):
        posterior.sample((1,))
    else:
        posterior.sample(
            (1,),
            init_strategy="proposal",
            method="slice_np_vectorized",
            **mcmc_params_fast,
        )


@pytest.mark.slow
@pytest.mark.parametrize(
    "sampler", (pytest.param("mcmc", marks=pytest.mark.mcmc), "rejection", "vi")
)
@pytest.mark.parametrize("flow_model", ("nsf", "zuko_nsf"))
def test_mnpe_accuracy_with_different_samplers(
    flow_model: str, sampler, mcmc_params_accurate: dict
):
    """Test MNPE c2st accuracy for different samplers."""

    num_simulations = 3000
    num_samples = 500

    prior = MultipleIndependent([
        Beta(torch.ones(1), torch.ones(1)),  # continuous
        Bernoulli(probs=0.5 * torch.ones(1)),  # discrete (Bernoulli)
    ])

    theta = prior.sample((num_simulations,))
    x = mixed_param_simulator(theta)

    # MNPE
    density_estimator = posterior_nn(model="mnpe", flow_model=flow_model)
    trainer = MNPE(prior, density_estimator=density_estimator)
    trainer.append_simulations(theta, x).train(training_batch_size=200)
    posterior = trainer.build_posterior()

    theta_o = prior.sample((1,))
    x_o = mixed_param_simulator(theta_o)

    mcmc_kwargs = dict(
        method="slice_np_vectorized", init_strategy="proposal", **mcmc_params_accurate
    )

    # true posterior samples
    transform = mcmc_transform(prior)
    true_posterior_samples = MCMCPosterior(
        potential_fn=posterior._potential_fn,
        theta_transform=transform,
        proposal=prior,
        **mcmc_kwargs,
    ).sample((num_samples,), x=x_o)

    # Get posterior samples
    posterior = trainer.build_posterior(prior=prior, sample_with=sampler)
    posterior.set_default_x(x_o)
    if sampler == "vi":
        posterior.train()

    mnpe_posterior_samples = posterior.sample(
        sample_shape=(num_samples,),
        show_progress_bars=True,
        **mcmc_kwargs if sampler == "mcmc" else {},
    )

    check_c2st(
        mnpe_posterior_samples,
        true_posterior_samples,
        alg=f"MNPE with {sampler}",
    )


@pytest.mark.parametrize(
    "sampler", (pytest.param("mcmc", marks=pytest.mark.mcmc), "rejection", "vi")
)
def test_sample_batched(sampler, mcmc_params_fast: dict):
    """Test that MNPE posterior sampling works with batched x using sample_batched()."""
    num_simulations = 100
    batch_size = 5
    num_samples = 10

    # Generate training data
    theta = torch.cat(
        (
            torch.rand(num_simulations, 1),  # continuous
            torch.randint(0, 2, (num_simulations, 1)),  # discrete
        ),
        dim=1,
    )
    x = mixed_param_simulator(theta)

    # Set up prior
    prior = MultipleIndependent([
        Beta(torch.ones(1), torch.ones(1)),  # continuous
        Bernoulli(probs=0.5 * torch.ones(1)),  # discrete
    ])

    # Train
    trainer = MNPE(prior=prior)
    trainer.append_simulations(theta, x).train(max_num_epochs=5)

    # Generate batch of observations
    theta_o = prior.sample((batch_size,))
    x_o = mixed_param_simulator(theta_o)

    # Build posterior with different samplers
    posterior = trainer.build_posterior(sample_with=sampler)

    # Sample using sample_batched
    if isinstance(posterior, MCMCPosterior):
        samples = posterior.sample_batched(
            (num_samples,),
            x=x_o,
            show_progress_bars=False,
            init_strategy="proposal",
            method="slice_np_vectorized",
            **mcmc_params_fast,
        )
    else:
        samples = posterior.sample_batched(
            (num_samples,), x=x_o, show_progress_bars=False
        )

    # Check output shape
    assert samples.shape == (batch_size, num_samples, 2)
