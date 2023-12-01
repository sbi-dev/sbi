# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Affero General Public License v3, see <https://www.gnu.org/licenses/>.

import pytest
import torch
from numpy import isin
from pyro.distributions import InverseGamma
from torch.distributions import Beta, Binomial, Categorical, Gamma

from sbi.inference import MNLE, MCMCPosterior, likelihood_estimator_based_potential
from sbi.inference.posteriors.rejection_posterior import RejectionPosterior
from sbi.inference.posteriors.vi_posterior import VIPosterior
from sbi.inference.potentials.base_potential import BasePotential
from sbi.inference.potentials.likelihood_based_potential import (
    MixedLikelihoodBasedPotential,
)
from sbi.utils import BoxUniform, likelihood_nn, mcmc_transform
from sbi.utils.conditional_density_utils import ConditionedPotential
from sbi.utils.torchutils import atleast_2d
from sbi.utils.user_input_checks_utils import MultipleIndependent
from tests.test_utils import check_c2st


# toy simulator for mixed data
def mixed_simulator(theta, stimulus_condition=2.0):
    # Extract parameters
    beta, ps = theta[:, :1], theta[:, 1:]

    # Sample choices and rts independently.
    choices = Binomial(probs=ps).sample()
    rts = InverseGamma(
        concentration=stimulus_condition * torch.ones_like(beta), rate=beta
    ).sample()

    return torch.cat((rts, choices), dim=1)


mcmc_kwargs = dict(
    num_chains=10,
    warmup_steps=100,
    method="slice_np_vectorized",
    init_strategy="proposal",
)


@pytest.mark.gpu
@pytest.mark.parametrize("device", ("cpu", "cuda"))
def test_mnle_on_device(device):
    # Generate mixed data.
    num_simulations = 100
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
    posterior.sample((1,), x=x[0], show_progress_bars=False, mcmc_method="nuts")


@pytest.mark.parametrize("sampler", ("mcmc", "rejection", "vi"))
def test_mnle_api(sampler):
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
    density_estimator = likelihood_nn(model="mnle")
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
            num_chains=2,
            warmup_steps=1,
            method="slice_np_vectorized",
            init_strategy="proposal",
            thin=1,
        )


@pytest.mark.slow
@pytest.mark.parametrize("sampler", ("mcmc", "rejection", "vi"))
def test_mnle_accuracy(sampler):
    def mixed_simulator(theta):
        # Extract parameters
        beta, ps = theta[:, :1], theta[:, 1:]

        # Sample choices and rts independently.
        choices = Binomial(probs=ps).sample()
        rts = InverseGamma(concentration=1 * torch.ones_like(beta), rate=beta).sample()

        return torch.cat((rts, choices), dim=1)

    prior = MultipleIndependent(
        [
            Gamma(torch.tensor([1.0]), torch.tensor([0.5])),
            Beta(torch.tensor([2.0]), torch.tensor([2.0])),
        ],
        validate_args=False,
    )

    num_simulations = 2000
    num_samples = 1000
    theta = prior.sample((num_simulations,))
    x = mixed_simulator(theta)

    # MNLE
    trainer = MNLE(prior)
    trainer.append_simulations(theta, x).train()
    posterior = trainer.build_posterior()

    for num_trials in [10]:
        theta_o = prior.sample((1,))
        x_o = mixed_simulator(theta_o.repeat(num_trials, 1))

        # True posterior samples
        transform = mcmc_transform(prior)
        true_posterior_samples = MCMCPosterior(
            PotentialFunctionProvider(prior, atleast_2d(x_o)),
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


class PotentialFunctionProvider(BasePotential):
    """Returns potential function for reference posterior of a mixed likelihood."""

    allow_iid_x = True  # type: ignore

    def __init__(self, prior, x_o, concentration_scaling=1.0, device="cpu"):
        super().__init__(prior, x_o, device)

        self.concentration_scaling = concentration_scaling

    def __call__(self, theta, track_gradients: bool = True):
        theta = atleast_2d(theta)

        with torch.set_grad_enabled(track_gradients):
            iid_ll = self.iid_likelihood(theta)

        return iid_ll + self.prior.log_prob(theta)

    def iid_likelihood(self, theta: torch.Tensor) -> torch.Tensor:
        """Returns the likelihood summed over a batch of i.i.d. data."""

        lp_choices = torch.stack(
            [
                Binomial(probs=th.reshape(1, -1)).log_prob(self.x_o[:, 1:])
                for th in theta[:, 1:]
            ],
            dim=1,
        )

        lp_rts = torch.stack(
            [
                InverseGamma(
                    concentration=self.concentration_scaling * torch.ones_like(beta_i),
                    rate=beta_i,
                ).log_prob(self.x_o[:, :1])
                for beta_i in theta[:, :1]
            ],
            dim=1,
        )

        joint_likelihood = (lp_choices + lp_rts).reshape(
            self.x_o.shape[0], theta.shape[0]
        )

        return joint_likelihood.sum(0)


@pytest.mark.slow
def test_mnle_with_experiment_conditions():
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

    num_simulations = 10000
    num_samples = 1000
    theta = proposal.sample((num_simulations,))
    x = sim_wrapper(theta)
    assert x.shape == (num_simulations, 2)

    num_trials = 10
    theta_o = proposal.sample((1,))
    theta_o[0, 2] = 2.0  # set condition to 2 as in original simulator.
    x_o = sim_wrapper(theta_o.repeat(num_trials, 1))

    # MNLE
    trainer = MNLE(proposal)
    estimator = trainer.append_simulations(theta, x).train()

    potential_fn = MixedLikelihoodBasedPotential(estimator, proposal, x_o)

    conditioned_potential_fn = ConditionedPotential(
        potential_fn, condition=theta_o, dims_to_sample=[0, 1], allow_iid_x=True
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
        PotentialFunctionProvider(
            prior,
            atleast_2d(x_o),
            concentration_scaling=float(theta_o[0, 2])
            + 1.0,  # add one because the sim_wrapper adds one (see above)
        ),
        theta_transform=prior_transform,
        proposal=prior,
        **mcmc_kwargs,
    ).sample((num_samples,), x=x_o)

    mcmc_posterior = MCMCPosterior(
        potential_fn=conditioned_potential_fn,
        theta_transform=prior_transform,
        proposal=prior,
        **mcmc_kwargs,
    )
    cond_samples = mcmc_posterior.sample((num_samples,), x=x_o)

    check_c2st(
        cond_samples,
        true_posterior_samples,
        alg="MNLE with experiment conditions",
    )
