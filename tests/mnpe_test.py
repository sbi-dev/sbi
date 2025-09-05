# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

import pytest
import torch
from torch import nn
from torch.distributions import Bernoulli, Normal

from sbi.inference import MNPE
from sbi.inference.posteriors.rejection_posterior import RejectionPosterior
from sbi.inference.posteriors.vi_posterior import VIPosterior
from sbi.neural_nets import posterior_nn
from sbi.neural_nets.embedding_nets import FCEmbedding
from sbi.utils import BoxUniform
from sbi.utils.metrics import check_c2st
from sbi.utils.torchutils import process_device
from sbi.utils.user_input_checks_utils import MultipleIndependent


# toy simulator for continuous data with mixed parameters
def mixed_simulator(theta, sigma2=0.1):
    """Simulator for continuous data with mixed parameters.
    Returns x~N(mu_z, tau^2)"""
    device = theta.device

    if isinstance(sigma2, float):
        sigma2 = torch.tensor(sigma2, device=device)

    mu = theta[:, :2]
    z = theta[:, -1]
    # Select mu based on z
    mu_z = torch.where(z == 0, mu[:, 0], mu[:, 1])
    # Sample from N(mu_z, sigma2)
    x = Normal(loc=mu_z, scale=torch.sqrt(sigma2)).rsample()
    return x.reshape(-1, 1).to(device)


@pytest.mark.gpu
@pytest.mark.parametrize("device", [pytest.param("gpu")])
def test_mnpe_on_device(
    device,
    num_simulations: int = 100,
):
    """Test MNPE API on device (CPU/GPU)."""
    num_samples = 10
    device = process_device(device)

    prior = MultipleIndependent(
        [
            BoxUniform(low=-torch.ones(2), high=torch.ones(2)),  # continuous
            Bernoulli(probs=0.8 * torch.ones(1)),  # discrete (Bernoulli)
        ],
        device=device,
    )
    theta = prior.sample((num_simulations,))
    x = mixed_simulator(theta).to(device)

    theta_true = prior.sample((1,))
    x_o = mixed_simulator(theta_true).to(device)

    trainer = MNPE(prior=prior, device=device)
    trainer.append_simulations(theta, x).train(max_num_epochs=1)

    # Test sampling on device
    posterior = trainer.build_posterior()
    samples = posterior.sample(
        sample_shape=(num_samples,),
        x=x_o,
        show_progress_bars=False,
    )
    assert samples.shape == (num_samples, 3)


def test_batched_sampling(num_simulations: int = 100):
    """Test MNPE API with batched sampling."""
    batch_size = 5
    num_samples = 10

    # Generate data with mixed parameter types
    theta_true = torch.cat(
        (
            torch.rand(batch_size, 2),
            torch.bernoulli(0.8 * torch.ones(batch_size, 1)),
        ),
        dim=1,
    )
    x_o = mixed_simulator(theta_true)  # This will return batch_size observations

    prior = MultipleIndependent(
        [
            BoxUniform(low=-torch.ones(2), high=torch.ones(2)),  # continuous
            Bernoulli(probs=0.8 * torch.ones(1)),  # discrete (Bernoulli)
        ],
    )
    theta = prior.sample((num_simulations,))
    x = mixed_simulator(theta)

    trainer = MNPE(prior=prior)
    trainer.append_simulations(theta, x).train(max_num_epochs=1)

    # Test batched sampling
    posterior = trainer.build_posterior()
    samples = posterior.sample_batched(
        sample_shape=(num_samples,),
        x=x_o,
        show_progress_bars=False,
    )
    print(samples.shape)
    assert samples.shape == (num_samples, batch_size, 3)


@pytest.mark.parametrize("flow_model", ("mdn", "nsf", "zuko_nsf"))
@pytest.mark.parametrize("z_score_x", ("independent", "none"))
@pytest.mark.parametrize("embedding_net", (torch.nn.Identity(), FCEmbedding(1, 1)))
def test_mnpe_api(flow_model: str, z_score_x: str, embedding_net: nn.Module):
    """Test MNPE API."""

    # Generate data
    num_simulations = 100
    theta_true = torch.tensor([[-0.5, 0.5, 1.0]])
    x_o = mixed_simulator(theta_true)

    # Train and infer
    prior = MultipleIndependent([
        BoxUniform(-torch.ones(2), torch.ones(2)),  # continuous
        Bernoulli(probs=0.8 * torch.ones(1)),  # discrete (Bernoulli)
    ])
    theta = prior.sample((num_simulations,))
    x = mixed_simulator(theta)

    # Build estimator manually
    # x_embedding = FCEmbedding(1, 1)  # simple embedding net, 1 continuous parameter
    density_estimator = posterior_nn(
        model="mnpe",
        flow_model=flow_model,
        z_score_x=z_score_x,
        embedding_net=embedding_net,
        log_transform_x=False,
    )
    trainer = MNPE(density_estimator=density_estimator)
    trainer.append_simulations(theta, x).train(max_num_epochs=1)

    # Test different samplers
    posterior = trainer.build_posterior(prior=prior)
    posterior.set_default_x(x_o)
    if isinstance(posterior, VIPosterior):
        posterior.train().sample((1,))
    elif isinstance(posterior, RejectionPosterior):
        posterior.sample((1,))
    else:
        posterior.sample((1,))


def reference_posterior_mog(tau2, sigma2, mu_0, mu_1, x_o, q, num_posterior_samples):
    """Reference posterior for a mixture of two Gaussians.
    Args:
        tau2: prior variance of mu0, mu1
        sigma2: variance of x
        mu_0: prior mean of mu0
        mu_1: prior mean of mu1
        x_o: observation
        q: probability of z=1
        num_posterior_samples: number of samples from the posterior
    """
    prior_mu0 = Normal(loc=mu_0, scale=torch.sqrt(tau2))
    prior_mu1 = Normal(loc=mu_1, scale=torch.sqrt(tau2))

    s2 = tau2 + sigma2

    ###### Posterior of z ######
    normal_0 = Normal(loc=mu_0, scale=torch.sqrt(s2))
    normal_1 = Normal(loc=mu_1, scale=torch.sqrt(s2))
    likelihood_0 = normal_0.log_prob(x_o).exp()  # p(x|z=0)
    likelihood_1 = normal_1.log_prob(x_o).exp()  # p(x|z=1)
    numerator = q * likelihood_1
    denominator = numerator + (1 - q) * likelihood_0
    bernoulli_probs = numerator / denominator
    posterior_z_ref = Bernoulli(probs=bernoulli_probs)
    posterior_z_ref_samples = posterior_z_ref.sample((num_posterior_samples,))
    posterior_z_ref_samples = posterior_z_ref_samples.reshape(-1, 1)

    ###### Posterior of mu ######
    # posterior std
    tau2_ref = (1 / tau2 + 1 / sigma2).pow(-1)
    # posterior mean
    mu_0_ref = tau2_ref * (mu_0 / tau2 + x_o / sigma2)
    mu_1_ref = tau2_ref * (mu_1 / tau2 + x_o / sigma2)
    # posterior distribution
    posterior_mu0_ref = Normal(loc=mu_0_ref, scale=torch.sqrt(tau2_ref))
    posterior_mu1_ref = Normal(loc=mu_1_ref, scale=torch.sqrt(tau2_ref))

    # Create binary mask from z samples (True where z=0, False where z=1)
    z_mask = (posterior_z_ref_samples == 0).flatten()

    # Create empty tensor for results
    posterior_ref_samples = torch.zeros(num_posterior_samples, 3)

    # Sample and populate directly without broadcasting issues
    posterior_ref_samples[:, 2] = posterior_z_ref_samples.flatten()

    # For mu0 column: use posterior_mu0_ref when z=0, prior_mu0 when z=1
    posterior_ref_samples[z_mask, 0] = posterior_mu0_ref.sample((
        z_mask.sum(),
    )).flatten()
    posterior_ref_samples[~z_mask, 0] = prior_mu0.sample(((~z_mask).sum(),)).flatten()

    # For mu1 column: use prior_mu1 when z=0, posterior_mu1_ref when z=1
    posterior_ref_samples[z_mask, 1] = prior_mu1.sample((z_mask.sum(),)).flatten()
    posterior_ref_samples[~z_mask, 1] = posterior_mu1_ref.sample((
        (~z_mask).sum(),
    )).flatten()

    return posterior_ref_samples


@pytest.mark.slow
def test_mnpe_accuracy():
    """Test MNPE accuracy."""
    num_simulations = 1000
    num_posterior_samples = 1000

    # Prior
    mu_0 = torch.tensor([-1.0])
    mu_1 = torch.tensor([1.0])
    tau2 = torch.tensor([0.1])
    prior_mu0 = Normal(loc=mu_0, scale=torch.sqrt(tau2))
    prior_mu1 = Normal(loc=mu_1, scale=torch.sqrt(tau2))

    q = torch.tensor(0.8)
    prior_z = Bernoulli(probs=q * torch.ones(1))

    prior = MultipleIndependent(
        [prior_mu0, prior_mu1, prior_z],
    )

    # Simulator
    sigma2 = torch.tensor(0.1)

    # observation
    theta_o = prior.sample((1,))
    x_o = mixed_simulator(theta_o, sigma2)

    # Reference posterior
    posterior_ref_samples = reference_posterior_mog(
        tau2, sigma2, mu_0, mu_1, x_o, q, num_posterior_samples
    )

    # MNPE
    theta = prior.sample((num_simulations,))
    x = mixed_simulator(theta, sigma2)

    trainer = MNPE()
    trainer.append_simulations(theta, x).train()

    posterior_mnpe = trainer.build_posterior(prior=prior)
    posterior_mnpe.set_default_x(x_o)

    posterior_mnpe_samples = posterior_mnpe.sample((num_posterior_samples,))

    # Check C2ST
    check_c2st(
        posterior_mnpe_samples,
        posterior_ref_samples,
        alg="MNPE",
    )
