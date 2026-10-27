# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

"""`posterior.to()` moves a posterior in place and keeps its state."""

from __future__ import annotations

import pytest
import torch
from torch.distributions import MultivariateNormal

from sbi.inference import FMPE, NPE, NPE_A, MCMCPosterior
from sbi.inference.posteriors.posterior_parameters import (
    ImportanceSamplingPosteriorParameters,
    MCMCPosteriorParameters,
    RejectionPosteriorParameters,
    VIPosteriorParameters,
    VectorFieldPosteriorParameters,
)
from sbi.utils import BoxUniform

NUM_DIM = 2
MCMC_FAST = MCMCPosteriorParameters(
    num_chains=1, thin=1, warmup_steps=1, init_strategy="proposal"
)


def _prior(name: str):
    if name == "box":
        return BoxUniform(-2 * torch.ones(NUM_DIM), 2 * torch.ones(NUM_DIM))
    return MultivariateNormal(torch.zeros(NUM_DIM), torch.eye(NUM_DIM))


def _simulate(prior, num_simulations: int = 100):
    theta = prior.sample((num_simulations,))
    return theta, theta + 0.1 * torch.randn_like(theta)


@pytest.fixture(scope="module", params=["box", "mvn"])
def trained(request):
    """NPE and FMPE trained for one epoch on a `BoxUniform` or a torch prior."""
    prior = _prior(request.param)
    theta, x = _simulate(prior)
    npe = NPE(prior, show_progress_bars=False)
    npe.append_simulations(theta, x).train(max_num_epochs=1)
    fmpe = FMPE(prior, show_progress_bars=False)
    fmpe.append_simulations(theta, x).train(max_num_epochs=1)
    return request.param, npe, fmpe


def _npe_a_round_two(prior_name: str):
    prior = _prior(prior_name)
    trainer = NPE_A(prior, num_components=2, show_progress_bars=False)
    theta, x = _simulate(prior, 300)
    trainer.append_simulations(theta, x).train(max_num_epochs=2)
    proposal = trainer.build_posterior().set_default_x(torch.zeros(1, NUM_DIM))
    theta = proposal.sample((300,), show_progress_bars=False)
    x = theta + 0.1 * torch.randn_like(theta)
    trainer.append_simulations(theta, x, proposal=proposal).train(max_num_epochs=2)
    return trainer.build_posterior()


def _build(kind: str, npe, fmpe, prior_name: str):
    if kind == "direct":
        return npe.build_posterior()
    if kind == "rejection":
        return npe.build_posterior(
            posterior_parameters=RejectionPosteriorParameters(
                num_samples_to_find_max=100, num_iter_to_find_max=5
            )
        )
    if kind == "importance":
        return npe.build_posterior(
            posterior_parameters=ImportanceSamplingPosteriorParameters()
        )
    if kind == "mcmc":
        return npe.build_posterior(posterior_parameters=MCMC_FAST)
    if kind == "mcmc_posterior_proposal":
        # A previous-round posterior as proposal, as in multi-round NLE or NRE.
        proposal = npe.build_posterior().set_default_x(torch.zeros(1, NUM_DIM))
        return MCMCPosterior(
            npe.build_posterior().potential_fn,
            proposal=proposal,
            num_chains=1,
            thin=1,
            warmup_steps=1,
            init_strategy="proposal",
        )
    if kind == "vi":
        posterior = npe.build_posterior(
            posterior_parameters=VIPosteriorParameters(q="gaussian")
        )
        posterior.set_default_x(torch.zeros(1, NUM_DIM))
        posterior.train(
            max_num_iters=5,
            check_for_convergence=False,
            quality_control=False,
            show_progress_bar=False,
        )
        return posterior
    if kind == "vector_field":
        return fmpe.build_posterior(
            posterior_parameters=VectorFieldPosteriorParameters(iid_method="fnpe")
        )
    if kind == "npe_a":
        return _npe_a_round_two(prior_name)
    raise ValueError(kind)


@pytest.mark.parametrize(
    "kind",
    [
        "direct",
        "rejection",
        "importance",
        "mcmc",
        "mcmc_posterior_proposal",
        "vi",
        "vector_field",
        "npe_a",
    ],
)
def test_to_keeps_posterior_state(trained, kind: str):
    """`.to()` returns the posterior and keeps default x, MAP, transform and prior."""
    prior_name, npe, fmpe = trained
    if kind == "npe_a" and prior_name == "box":
        pytest.skip("Two-round NPE-A with a BoxUniform hits the known PD error.")
    posterior = _build(kind, npe, fmpe, prior_name)
    x_o = torch.zeros(1, NUM_DIM)
    posterior.set_default_x(x_o)
    map_kwargs = dict(num_iter=2, num_to_optimize=1, num_init_samples=1)
    map_ = posterior.map(init_method=torch.zeros(1, NUM_DIM), **map_kwargs)
    description = str(posterior)
    transform = posterior.theta_transform
    iid_method = getattr(posterior.potential_fn, "iid_method", None)
    shares_prior = {
        attr: getattr(posterior, attr) is posterior.potential_fn.prior
        for attr in ("prior", "_prior", "proposal")
        if hasattr(posterior, attr)
    }

    assert posterior.to("cpu") is posterior

    assert torch.equal(posterior.default_x, x_o)
    # A dropped cache would run gradient ascent again from the new init.
    new_init = torch.ones(1, NUM_DIM)
    assert torch.equal(posterior.map(init_method=new_init, **map_kwargs), map_)
    assert str(posterior) == description
    assert posterior.theta_transform is transform
    assert getattr(posterior.potential_fn, "iid_method", None) == iid_method
    for attr, shared in shares_prior.items():
        assert (getattr(posterior, attr) is posterior.potential_fn.prior) == shared
    samples = posterior.sample((2,), show_progress_bars=False)
    assert posterior.potential(samples).shape == (2,)
