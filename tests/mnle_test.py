# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Affero General Public License v3, see <https://www.gnu.org/licenses/>.

import pytest
import torch
from pyro.distributions import InverseGamma
from torch.distributions import Beta, Binomial, Gamma

from sbi.inference import MNLE, MCMCPosterior, likelihood_estimator_based_potential
from sbi.inference.potentials.base_potential import BasePotential
from sbi.utils import BoxUniform, likelihood_nn, mcmc_transform
from sbi.utils.torchutils import atleast_2d
from sbi.utils.user_input_checks_utils import MultipleIndependent
from tests.test_utils import check_c2st


@pytest.mark.gpu
@pytest.mark.parametrize("device", ("cpu", "cuda"))
def test_mnle_on_device(device):

    # Generate mixed data.
    num_simulations = 100
    theta = torch.rand(num_simulations, 2)
    x = torch.cat(
        (torch.rand(num_simulations, 1), torch.randint(0, 2, (num_simulations, 1))),
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
        (torch.rand(num_simulations, 1), torch.randint(0, 2, (num_simulations, 1))),
        dim=1,
    )

    # Train and infer.
    prior = BoxUniform(torch.zeros(2), torch.ones(2))
    x_o = x[0]
    # Build estimator manually.
    density_estimator = likelihood_nn(model="mnle", **dict(tail_bound=2.0))
    trainer = MNLE(density_estimator=density_estimator)
    mnle = trainer.append_simulations(theta, x).train(max_num_epochs=1)

    # Test different samplers.
    posterior = trainer.build_posterior(prior=prior, sample_with=sampler)
    posterior.set_default_x(x_o)
    if sampler == "vi":
        posterior.train()
    posterior.sample((1,), show_progress_bars=False)

    # MNLE should work with the default potential as well.
    potential_fn, parameter_transform = likelihood_estimator_based_potential(
        mnle, prior, x_o
    )
    posterior = MCMCPosterior(
        potential_fn,
        proposal=prior,
        theta_transform=parameter_transform,
        init_strategy="proposal",
    )
    posterior.sample((1,), show_progress_bars=False)


@pytest.mark.slow
@pytest.mark.parametrize(
    "sampler",
    (
        "mcmc",
        "rejection",
        # "vi",  # Failing because of transformed space dimension mismatch.
    ),
)
def test_mnle_accuracy(sampler):
    def mixed_simulator(theta):
        # Extract parameters
        beta, ps = theta[:, :1], theta[:, 1:]

        # Sample choices and rts independently.
        choices = Binomial(probs=ps).sample()
        rts = InverseGamma(concentration=2 * torch.ones_like(beta), rate=beta).sample()

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

    mcmc_kwargs = dict(
        num_chains=10,
        warmup_steps=100,
        method="slice_np_vectorized",
        init_strategy="proposal",
    )

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
            show_progress_bars=False,
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

    def __init__(self, prior, x_o, device="cpu"):
        super().__init__(prior, x_o, device)

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
                    concentration=2 * torch.ones_like(beta_i), rate=beta_i
                ).log_prob(self.x_o[:, :1])
                for beta_i in theta[:, :1]
            ],
            dim=1,
        )

        joint_likelihood = (lp_choices + lp_rts).reshape(
            self.x_o.shape[0], theta.shape[0]
        )

        return joint_likelihood.sum(0)
