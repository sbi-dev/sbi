# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

import pytest
import torch
from pyro.distributions import InverseGamma
from torch.distributions import Beta, Binomial, Categorical, Gamma

from sbi.inference import MNLE, MCMCPosterior
from sbi.inference.posteriors.rejection_posterior import RejectionPosterior
from sbi.inference.posteriors.vi_posterior import VIPosterior
from sbi.inference.potentials.base_potential import BasePotential
from sbi.inference.potentials.likelihood_based_potential import (
    MixedLikelihoodBasedPotential,
)
from sbi.neural_nets import likelihood_nn
from sbi.neural_nets.embedding_nets import FCEmbedding
from sbi.utils import BoxUniform, mcmc_transform
from sbi.utils.conditional_density_utils import ConditionedPotential
from sbi.utils.torchutils import atleast_2d, process_device
from sbi.utils.user_input_checks_utils import MultipleIndependent
from tests.test_utils import check_c2st


# toy simulator for mixed data
def mixed_simulator(theta, stimulus_condition=2.0):
    """Simulator for mixed data."""
    # Extract parameters
    beta, ps = theta[:, :1], theta[:, 1:]

    # Sample choices and rts independently.
    choices = Binomial(probs=ps).sample()
    rts = InverseGamma(
        concentration=stimulus_condition * torch.ones_like(beta), rate=beta
    ).sample()

    return torch.cat((rts, choices), dim=1)


@pytest.mark.mcmc
@pytest.mark.gpu
@pytest.mark.parametrize("device", ("cpu", "gpu"))
def test_mnle_on_device(
    device,
    mcmc_params_fast: dict,
    num_simulations: int = 100,
    mcmc_method: str = "slice_np",
):
    """Test MNLE API on device."""

    device = process_device(device)

    # Generate mixed data.
    theta = torch.rand(num_simulations, 2)
    x = torch.cat(
        (
            torch.rand(num_simulations, 1),
            torch.randint(0, 2, (num_simulations, 1)),
        ),
        dim=1,
    ).to(device)

    # Train and infer.
    prior = BoxUniform(torch.zeros(2), torch.ones(2), device=device)
    trainer = MNLE(prior=prior, device=device)
    trainer.append_simulations(theta, x).train(max_num_epochs=1)

    # Test sampling on device.
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
@pytest.mark.parametrize("flow_model", ("mdn", "maf", "nsf", "zuko_nsf", "zuko_bpf"))
@pytest.mark.parametrize("z_score_theta", ("independent", "none"))
def test_mnle_api(flow_model: str, sampler, mcmc_params_fast: dict, z_score_theta: str):
    """Test MNLE API."""
    # Generate mixed data.
    num_simulations = 100
    theta = torch.rand(num_simulations, 2)
    x = torch.cat(
        (
            torch.rand(num_simulations, 1),
            torch.randint(0, 2, (num_simulations, 1)),
        ),
        dim=1,
    )

    # Train and infer.
    prior = BoxUniform(torch.zeros(2), torch.ones(2))
    x_o = x[0]
    # Build estimator manually.
    theta_embedding = FCEmbedding(2, 2)  # simple embedding net
    density_estimator = likelihood_nn(
        model="mnle",
        flow_model=flow_model,
        z_score_theta=z_score_theta,
        embedding_net=theta_embedding,
    )
    trainer = MNLE(density_estimator=density_estimator)
    trainer.append_simulations(theta, x).train(max_num_epochs=5)

    # Test different samplers.
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
@pytest.mark.parametrize("num_trials", [5, 10])
@pytest.mark.parametrize("flow_model", ("nsf", "zuko_nsf"))
def test_mnle_accuracy_with_different_samplers_and_trials(
    flow_model: str, sampler, num_trials: int, mcmc_params_accurate: dict
):
    """Test MNLE c2st accuracy for different samplers and number of trials."""

    num_simulations = 3200
    num_samples = 500

    prior = MultipleIndependent(
        [
            Gamma(torch.tensor([1.0]), torch.tensor([0.5])),
            Beta(torch.tensor([2.0]), torch.tensor([2.0])),
        ],
        validate_args=False,
    )

    theta = prior.sample((num_simulations,))
    x = mixed_simulator(theta, stimulus_condition=1.0)

    # MNLE
    density_estimator = likelihood_nn(model="mnle", flow_model=flow_model)
    trainer = MNLE(prior, density_estimator=density_estimator)
    trainer.append_simulations(theta, x).train(training_batch_size=200)
    posterior = trainer.build_posterior()

    theta_o = prior.sample((1,))
    x_o = mixed_simulator(theta_o.repeat(num_trials, 1))

    mcmc_kwargs = dict(
        method="slice_np_vectorized", init_strategy="proposal", **mcmc_params_accurate
    )

    # True posterior samples
    transform = mcmc_transform(prior)
    true_posterior_samples = MCMCPosterior(
        BinomialGammaPotential(prior, atleast_2d(x_o)),
        theta_transform=transform,
        proposal=prior,
        **mcmc_kwargs,
    ).sample((num_samples,), show_progress_bars=False)

    posterior = trainer.build_posterior(prior=prior, sample_with=sampler)
    posterior.set_default_x(x_o)
    if sampler == "vi":
        posterior.train()

    mnle_posterior_samples = posterior.sample(
        sample_shape=(num_samples,),
        show_progress_bars=True,
        **mcmc_kwargs if sampler == "mcmc" else {},
    )

    check_c2st(
        mnle_posterior_samples,
        true_posterior_samples,
        alg=f"MNLE with {sampler}",
    )


class BinomialGammaPotential(BasePotential):
    def __init__(self, prior, x_o, concentration_scaling=1.0, device="cpu"):
        super().__init__(prior, x_o, device)
        self.concentration_scaling = concentration_scaling

    def __call__(self, theta, track_gradients: bool = True):
        theta = atleast_2d(theta)

        with torch.set_grad_enabled(track_gradients):
            iid_ll = self.iid_likelihood(theta)

        return iid_ll + self.prior.log_prob(theta)

    def iid_likelihood(self, theta):
        batch_size = theta.shape[0]
        num_trials = self.x_o.shape[0]
        theta = theta.reshape(batch_size, 1, -1)
        beta, rho = theta[:, :, :1], theta[:, :, 1:]
        # vectorized
        logprob_choices = Binomial(probs=rho).log_prob(
            self.x_o[:, 1:].reshape(1, num_trials, -1)
        )

        logprob_rts = InverseGamma(
            concentration=self.concentration_scaling * torch.ones_like(beta),
            rate=beta,
        ).log_prob(self.x_o[:, :1].reshape(1, num_trials, -1))

        joint_likelihood = (logprob_choices + logprob_rts).squeeze()

        assert joint_likelihood.shape == torch.Size([theta.shape[0], self.x_o.shape[0]])
        return joint_likelihood.sum(1)


@pytest.mark.slow
@pytest.mark.mcmc
def test_mnle_with_experimental_conditions(mcmc_params_accurate: dict):
    """Test MNLE c2st accuracy when conditioned on a subset of the parameters, e.g.,
    experimental conditions.

    MNLE is trained a on simulator with 3D parameter space. After training, the
    categorical parameter is set to a fixed value (conditioned posterior), and the
    accuracy of the conditioned posterior is tested against the true posterior.
    """
    num_simulations = 6000
    num_samples = 500

    def sim_wrapper(theta):
        # simulate with experiment conditions
        return mixed_simulator(theta[:, :2], theta[:, 2:] + 1)

    proposal = MultipleIndependent(
        [
            Gamma(torch.tensor([1.0]), torch.tensor([0.5])),
            Beta(torch.tensor([2.0]), torch.tensor([2.0])),
            Categorical(probs=torch.ones(1, 3)),
        ],
        validate_args=False,
    )

    theta = proposal.sample((num_simulations,))
    x = sim_wrapper(theta)
    assert x.shape == (num_simulations, 2)

    num_trials = 10
    theta_o = proposal.sample((1,))
    theta_o[0, 2] = 2.0  # set condition to 2 as in original simulator.
    x_o = sim_wrapper(theta_o.repeat(num_trials, 1))

    mcmc_kwargs = dict(
        method="slice_np_vectorized", init_strategy="proposal", **mcmc_params_accurate
    )

    # MNLE
    trainer = MNLE(proposal)
    estimator = trainer.append_simulations(theta, x).train(training_batch_size=1000)

    potential_fn = MixedLikelihoodBasedPotential(estimator, proposal, x_o)

    conditioned_potential_fn = ConditionedPotential(
        potential_fn, condition=theta_o, dims_to_sample=[0, 1]
    )

    # True posterior samples
    prior = MultipleIndependent(
        [
            Gamma(torch.tensor([1.0]), torch.tensor([0.5])),
            Beta(torch.tensor([2.0]), torch.tensor([2.0])),
        ],
        validate_args=False,
    )
    prior_transform = mcmc_transform(prior)
    true_posterior_samples = MCMCPosterior(
        BinomialGammaPotential(
            prior,
            atleast_2d(x_o),
            concentration_scaling=float(theta_o[0, 2])
            + 1.0,  # add one because the sim_wrapper adds one (see above)
        ),
        theta_transform=prior_transform,
        proposal=prior,
        **mcmc_kwargs,
    ).sample((num_samples,), x=x_o)

    cond_samples = MCMCPosterior(
        potential_fn=conditioned_potential_fn,
        theta_transform=prior_transform,
        proposal=prior,
        **mcmc_kwargs,
    ).sample((num_samples,), x=x_o)

    check_c2st(
        cond_samples,
        true_posterior_samples,
        alg=f"MNLE trained with {num_simulations}",
    )
