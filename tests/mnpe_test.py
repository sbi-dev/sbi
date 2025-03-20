# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

import pytest
import torch
from torch import Tensor
from torch.distributions import Bernoulli, Normal

from sbi.inference import MNPE
from sbi.inference.posteriors.rejection_posterior import RejectionPosterior
from sbi.inference.posteriors.vi_posterior import VIPosterior
from sbi.neural_nets import posterior_nn
from sbi.neural_nets.embedding_nets import FCEmbedding
from sbi.utils import BoxUniform
from sbi.utils.torchutils import process_device
from sbi.utils.user_input_checks_utils import MultipleIndependent
from tests.test_utils import check_c2st


# toy simulator for continuous data with mixed parameters
def mixed_param_simulator(theta: Tensor) -> Tensor:
    """Simulator for continuous data with mixed parameters.

    Args:
        theta: Parameters with mixed types - continuous and discrete.
    Returns:
        x: Continuous observation.
    """
    device = theta.device

    # Extract parameters
    a, b = theta[:, 0], theta[:, 1]
    noise = 0.05 * torch.randn(a.shape, device=device).reshape(-1, 1)
    return (a + 2 * b).reshape(-1, 1) + noise


@pytest.mark.gpu
@pytest.mark.parametrize("device", ("cpu", "gpu"))
def test_mnpe_on_device(
    device,
    num_simulations: int = 100,
):
    """Test MNPE API on device (CPU/GPU)."""
    num_samples = 10
    device = process_device(device)

    # Generate data with mixed parameter types
    theta_true = torch.tensor([[0.5, 1.0]], device=device)
    x_o = mixed_param_simulator(theta_true).to(device)

    prior = MultipleIndependent(
        [
            BoxUniform(low=torch.zeros(1), high=torch.ones(1)),  # continuous
            Bernoulli(probs=0.8 * torch.ones(1)),  # discrete (Bernoulli)
        ],
        device=device,
    )
    theta = prior.sample((num_simulations,))
    x = mixed_param_simulator(theta).to(device)

    trainer = MNPE(prior=prior, device=device)
    trainer.append_simulations(theta, x).train(max_num_epochs=3)

    # Test sampling on device
    posterior = trainer.build_posterior()
    samples = posterior.sample(
        sample_shape=(num_samples,),
        x=x_o,
        show_progress_bars=False,
    )
    assert samples.shape == (num_samples, 2)


def test_batched_sampling(num_simulations: int = 100):
    """Test MNPE API with batched sampling."""
    batch_size = 5
    num_samples = 10

    # Generate data with mixed parameter types
    theta_true = torch.cat(
        (
            torch.rand(batch_size, 1),
            torch.ones(batch_size, 1),
        ),
        dim=1,
    )
    x_o = mixed_param_simulator(theta_true)  # This will return batch_size observations

    prior = MultipleIndependent(
        [
            BoxUniform(low=torch.zeros(1), high=torch.ones(1)),  # continuous
            Bernoulli(probs=0.8 * torch.ones(1)),  # discrete (Bernoulli)
        ],
    )
    theta = prior.sample((num_simulations,))
    x = mixed_param_simulator(theta)

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
    assert samples.shape == (num_samples, batch_size, 2)


@pytest.mark.parametrize("flow_model", ("mdn", "maf", "nsf", "zuko_nsf"))
@pytest.mark.parametrize("z_score_theta", ("independent", "none"))
@pytest.mark.parametrize("use_embed_net", (True, False))
def test_mnpe_api(flow_model: str, z_score_theta: str, use_embed_net: bool):
    """Test MNPE API."""

    # Generate data
    num_simulations = 100
    theta_true = torch.tensor([[0.5, 1.0]])
    x_o = mixed_param_simulator(theta_true)

    # Train and infer
    prior = MultipleIndependent([
        BoxUniform(torch.zeros(1), torch.ones(1)),  # continuous
        Bernoulli(probs=0.8 * torch.ones(1)),  # discrete (Bernoulli)
    ])
    theta = prior.sample((num_simulations,))
    x = mixed_param_simulator(theta)

    # Build estimator manually
    x_embedding = FCEmbedding(1, 1)  # simple embedding net, 1 continuous parameter
    density_estimator = posterior_nn(
        model="mnpe",
        flow_model=flow_model,
        z_score_theta=z_score_theta,
        embedding_net=x_embedding if use_embed_net else torch.nn.Identity(),
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


class MixedSimulator(torch.nn.Module):
    def __init__(self, sigma2):
        super().__init__()
        self.sigma2 = sigma2

    def forward(self, theta):
        # return x~N(mu_z, tau^2)
        mu = theta[:, :2]
        z = theta[:, -1]
        # Select mu based on z
        mu_z = torch.where(z == 0, mu[:, 0], mu[:, 1])
        # Sample from N(mu_z, sigma2)
        x = Normal(loc=mu_z, scale=torch.sqrt(self.sigma2)).rsample()
        return x.reshape(-1, 1)


@pytest.mark.slow
@pytest.mark.parametrize("flow_model", ("nsf", "zuko_nsf"))
def test_mnpe_accuracy(
    flow_model: str,
):
    """Test MNPE accuracy."""
    num_simulations = 1000
    num_posterior_samples = 1000

    z_score_theta = "none"
    use_embed_net = False

    # Prio
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
    simulator = MixedSimulator(sigma2=sigma2)

    # observation
    theta_o = prior.sample((1,))
    x_o = simulator(theta_o)

    ############################################################
    # MNPE
    num_simulations = 4000
    num_posterior_samples = 1_000

    theta = prior.sample((num_simulations,))
    x = simulator(theta)

    # Build estimator manually
    x_embedding = FCEmbedding(1, 1)  # simple embedding net, 1 continuous parameter
    density_estimator = posterior_nn(
        model="mnpe",
        flow_model=flow_model,
        z_score_theta=z_score_theta,
        embedding_net=x_embedding if use_embed_net else torch.nn.Identity(),
        log_transform_x=False,
    )
    trainer = MNPE(density_estimator=density_estimator)
    trainer.append_simulations(theta, x).train()

    # Test different samplers
    posterior_mnpe = trainer.build_posterior(prior=prior)
    posterior_mnpe.set_default_x(x_o)

    posterior_mnpe_samples = posterior_mnpe.sample((num_posterior_samples,))

    ############################################################
    # True posterior
    ###### Posterior of z ######
    s2 = tau2 + sigma2

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

    # Check C2ST
    check_c2st(
        posterior_mnpe_samples,
        posterior_ref_samples,
        alg="MNPE",
    )
