# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

import pytest
import torch
from torch import Tensor
from torch.distributions import Bernoulli

from sbi.inference import MNPE
from sbi.inference.posteriors.rejection_posterior import RejectionPosterior
from sbi.inference.posteriors.vi_posterior import VIPosterior
from sbi.neural_nets import posterior_nn
from sbi.neural_nets.embedding_nets import FCEmbedding
from sbi.utils import BoxUniform
from sbi.utils.torchutils import process_device
from sbi.utils.user_input_checks_utils import MultipleIndependent


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
    trainer.append_simulations(theta, x).train(max_num_epochs=5)

    # Test different samplers
    posterior = trainer.build_posterior(prior=prior)
    posterior.set_default_x(x_o)
    if isinstance(posterior, VIPosterior):
        posterior.train().sample((1,))
    elif isinstance(posterior, RejectionPosterior):
        posterior.sample((1,))
    else:
        posterior.sample((1,))
