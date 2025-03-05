# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

from typing import Union

import pytest
import torch
from pyro.distributions import InverseGamma
from torch import Tensor
from torch.distributions import Beta, Binomial, Distribution, Gamma

from sbi.inference import MNLE, MCMCPosterior
from sbi.inference.posteriors.rejection_posterior import RejectionPosterior
from sbi.inference.posteriors.vi_posterior import VIPosterior
from sbi.inference.potentials.base_potential import BasePotential
from sbi.inference.potentials.likelihood_based_potential import (
    _log_likelihood_over_iid_trials_and_local_theta,
    likelihood_estimator_based_potential,
)
from sbi.neural_nets import likelihood_nn
from sbi.neural_nets.embedding_nets import FCEmbedding
from sbi.utils import BoxUniform, mcmc_transform
from sbi.utils.torchutils import atleast_2d, process_device
from sbi.utils.user_input_checks_utils import MultipleIndependent
from tests.test_utils import check_c2st


# toy simulator for mixed data
def mixed_simulator(theta: Tensor, stimulus_condition: Union[Tensor, float] = 2.0):
    """Simulator for mixed data."""
    # Extract parameters
    beta, ps = theta[:, :1], theta[:, 1:]

    # Sample choices and rts independently.
    choices = Binomial(probs=ps).sample()
    rts = InverseGamma(
        concentration=stimulus_condition * torch.ones_like(beta), rate=beta
    ).sample()

    return torch.cat((rts, choices), dim=1)


def mixed_simulator_with_conditions(
    theta_and_condition: Tensor, last_idx_parameters: int = 3
) -> Tensor:
    """Simulator for mixed data with experimental conditions."""
    # simulate with experiment conditions
    theta = theta_and_condition[:, :last_idx_parameters]
    condition = theta_and_condition[:, last_idx_parameters:]
    return mixed_simulator(theta, condition)


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

    num_simulations = 4000
    num_samples = 1000

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
    density_estimator = likelihood_nn(
        model="mnle", flow_model=flow_model, log_transform_x=True
    )
    trainer = MNLE(prior, density_estimator=density_estimator)
    trainer.append_simulations(theta, x).train(training_batch_size=200)

    theta_o = prior.sample((1,))
    x_o = mixed_simulator(theta_o.repeat(num_trials, 1))

    # True posterior samples
    transform = mcmc_transform(prior)
    true_posterior_samples = MCMCPosterior(
        BinomialGammaPotential(prior, atleast_2d(x_o)),
        theta_transform=transform,
        proposal=prior,
        **mcmc_params_accurate,
    ).sample((num_samples,), show_progress_bars=False)

    posterior = trainer.build_posterior(
        prior=prior, sample_with=sampler, mcmc_parameters=mcmc_params_accurate
    )
    posterior.set_default_x(x_o)
    if sampler == "vi":
        posterior.train()

    mnle_posterior_samples = posterior.sample(
        sample_shape=(num_samples,),
        show_progress_bars=True,
        **mcmc_params_accurate if sampler == "mcmc" else {},
    )

    check_c2st(
        mnle_posterior_samples,
        true_posterior_samples,
        alg=f"MNLE with {flow_model} and {sampler}",
    )


class BinomialGammaPotential(BasePotential):
    """Binomial-Gamma potential for mixed data."""

    def __init__(
        self,
        prior: Distribution,
        x_o: Tensor,
        concentration_scaling: Union[Tensor, float] = 1.0,
        device="cpu",
    ):
        super().__init__(prior, x_o, device)

        # concentration_scaling needs to be a float or match the batch size
        if isinstance(concentration_scaling, Tensor):
            num_trials = x_o.shape[0]
            assert concentration_scaling.shape[0] == num_trials

            # Reshape to match convention (batch_size, num_trials, *event_shape)
            concentration_scaling = concentration_scaling.reshape(1, num_trials, -1)

        self.concentration_scaling = concentration_scaling

    def __call__(self, theta: Tensor, track_gradients: bool = True) -> Tensor:
        theta = atleast_2d(theta)

        with torch.set_grad_enabled(track_gradients):
            iid_ll = self.iid_likelihood(theta)

        return iid_ll + self.prior.log_prob(theta)

    def iid_likelihood(self, theta: Tensor) -> Tensor:
        batch_size = theta.shape[0]
        num_trials = self.x_o.shape[0]
        theta = theta.reshape(batch_size, 1, -1)
        # We assume the InverseGamma to be in the first position of theta.
        # And potentially multiple Binomials in the rest.
        beta, rhos = theta[:, :, :1], theta[:, :, 1:]

        # evaluate vectorized across batch of thetas.
        logprob_choices = (
            torch.stack(
                [
                    Binomial(probs=rhos[:, :, rho_idx]).log_prob(
                        self.x_o[:, 1 + rho_idx]
                    )
                    for rho_idx in range(rhos.shape[-1])
                ],
            )
            .transpose(0, 1)
            .transpose(1, 2)
        )

        logprob_rts = InverseGamma(
            concentration=self.concentration_scaling * torch.ones_like(beta), rate=beta
        ).log_prob(self.x_o[:, :1].reshape(1, num_trials, -1))

        # sum across parameter dimensions.
        joint_likelihood = torch.sum(logprob_choices, dim=-1) + logprob_rts.squeeze()

        assert joint_likelihood.shape == torch.Size([theta.shape[0], self.x_o.shape[0]])
        assert joint_likelihood.isfinite().all()
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
    num_simulations = 10000
    num_samples = 1000
    mcmc_params_accurate["num_chains"] = 100

    proposal = MultipleIndependent(
        [
            Gamma(torch.tensor([1.0]), torch.tensor([0.5])),
            Beta(torch.tensor([2.0]), torch.tensor([2.0])),
            BoxUniform(torch.tensor([0.5]), torch.tensor([1.5])),
        ],
        validate_args=False,
    )

    theta = proposal.sample((num_simulations,))
    last_idx_parameters = 2
    x = mixed_simulator_with_conditions(theta, last_idx_parameters)
    assert x.shape == (num_simulations, last_idx_parameters)

    num_trials = 10
    theta_and_condition = proposal.sample((num_trials,))
    # use only a single parameter (iid trials)
    theta_o = theta_and_condition[:1, :last_idx_parameters].repeat(num_trials, 1)
    # but different conditions
    condition_o = theta_and_condition[:, last_idx_parameters:]
    theta_and_conditions_o = torch.cat((theta_o, condition_o), dim=1)

    x_o = mixed_simulator_with_conditions(theta_and_conditions_o, last_idx_parameters)

    mcmc_kwargs = dict(init_strategy="proposal", **mcmc_params_accurate)

    # Get True posterior.
    prior = MultipleIndependent(
        [
            Gamma(torch.tensor([1.0]), torch.tensor([1.0])),
            Beta(torch.tensor([2.0]), torch.tensor([2.0])),
        ],
        validate_args=False,
    )
    prior_transform = mcmc_transform(prior)
    true_posterior_samples = MCMCPosterior(
        BinomialGammaPotential(
            prior, atleast_2d(x_o), concentration_scaling=condition_o
        ),
        theta_transform=prior_transform,
        proposal=prior,
        **mcmc_kwargs,
    ).sample((num_samples,), x=x_o)

    # MNLE
    estimator_fun = likelihood_nn(model="mnle", log_transform_x=True)
    trainer = MNLE(proposal, estimator_fun)
    estimator = trainer.append_simulations(theta, x).train()

    potential_fn, _ = likelihood_estimator_based_potential(estimator, proposal, x_o)
    conditioned_potential_fn = potential_fn.condition_on_theta(
        condition_o, dims_global_theta=[_ for _ in range(last_idx_parameters)]
    )

    # test theta with sample shape.
    conditioned_potential_fn(prior.sample((10,)).unsqueeze(0))

    cond_samples = MCMCPosterior(
        potential_fn=conditioned_potential_fn,
        theta_transform=prior_transform,
        proposal=prior,
        **mcmc_kwargs,
    ).sample((num_samples,), x=x_o)

    check_c2st(
        cond_samples,
        true_posterior_samples,
        alg=f"MNLE trained with {num_simulations} simulations",
    )


@pytest.mark.parametrize("num_thetas", [1, 10])
@pytest.mark.parametrize("num_trials", [1, 5])
@pytest.mark.parametrize(
    "num_xs",
    [
        1,
        pytest.param(
            2,
            marks=pytest.mark.xfail(
                reason="Batched x not supported for iid trials.",
                raises=NotImplementedError,
            ),
        ),
    ],
)
@pytest.mark.parametrize(
    "num_conditions",
    [
        1,
        pytest.param(
            2,
            marks=pytest.mark.xfail(
                reason="Batched theta_condition is not supported",
            ),
        ),
    ],
)
def test_log_likelihood_over_local_iid_theta(
    num_thetas, num_trials, num_xs, num_conditions
):
    """Test log likelihood over iid conditions using MNLE.

    Args:
        num_thetas: batch of theta to condition on.
        num_trials: number of i.i.d. trials in x
        num_xs: batch of x, e.g., different subjects in a study.
        num_conditions: number of batches of conditions, e.g., different conditions
            for each x (not implemented yet).
    """

    # train mnle on mixed data
    trainer = MNLE()
    proposal = MultipleIndependent(
        [
            Gamma(torch.tensor([1.0]), torch.tensor([0.5])),
            Beta(torch.tensor([2.0]), torch.tensor([2.0])),
            Beta(
                torch.tensor([2.0]), torch.tensor([2.0])
            ),  # tests discrete dims > 1 works
            BoxUniform(torch.tensor([0.0]), torch.tensor([1.0])),
        ],
        validate_args=False,
    )

    num_simulations = 100
    idx_of_cond = 3
    theta = proposal.sample((num_simulations,))
    x = mixed_simulator_with_conditions(theta, idx_of_cond)
    estimator = trainer.append_simulations(theta, x).train(max_num_epochs=1)

    # condition on multiple conditions
    theta_o = proposal.sample((num_xs,))[:, :idx_of_cond]

    x_o = torch.zeros(num_trials, num_xs, idx_of_cond)
    condition_o = proposal.sample((
        num_conditions,
        num_trials,
    ))[:, idx_of_cond:].reshape(num_trials, 1)
    for i in range(num_xs):
        # simulate with same iid theta but different conditions
        x_o[:, i, :] = mixed_simulator(theta_o[i].repeat(num_trials, 1), condition_o)

    # batched conditioning
    theta = proposal.sample((num_thetas,))[:, :idx_of_cond]
    # x_o has shape (iid, batch, *event)
    # condition_o has shape (iid, num_conditions)
    ll_batched = _log_likelihood_over_iid_trials_and_local_theta(
        x_o, theta, condition_o, estimator
    )

    # looped conditioning
    ll_single = []
    for i in range(num_trials):
        theta_and_condition = torch.cat(
            (theta, condition_o[i].repeat(num_thetas, 1)), dim=1
        )
        x_i = x_o[i].reshape(num_xs, 1, -1).repeat(1, num_thetas, 1)
        ll_single.append(estimator.log_prob(input=x_i, condition=theta_and_condition))
    ll_single = (
        torch.stack(ll_single).sum(0).squeeze(0)
    )  # sum over trials, squeeze x batch.

    assert ll_batched.shape == torch.Size([num_thetas])
    assert ll_batched.shape == ll_single.shape
    assert torch.allclose(ll_batched, ll_single, atol=1e-5)
