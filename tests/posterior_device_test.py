# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

"""`posterior.to()` moves a posterior in place and keeps its state."""

from __future__ import annotations

import pytest
import torch
import torch.distributions.transforms as torch_tf
from torch.distributions import MultivariateNormal

from sbi.inference import FMPE, NPE, NPE_A, EnsemblePosterior, MCMCPosterior
from sbi.inference.posteriors.posterior_parameters import (
    ImportanceSamplingPosteriorParameters,
    MCMCPosteriorParameters,
    RejectionPosteriorParameters,
    VIPosteriorParameters,
    VectorFieldPosteriorParameters,
)
from sbi.utils import BoxUniform, RestrictionEstimator
from sbi.utils.torchutils import process_device

NUM_DIM = 2
POSTERIOR_KINDS = [
    "direct",
    "rejection",
    "importance",
    "mcmc",
    "mcmc_posterior_proposal",
    "mcmc_restricted_proposal",
    "vi",
    "vector_field",
    "npe_a",
]
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


def _train(prior_name: str):
    """NPE and FMPE trained for one epoch on a `BoxUniform` or a torch prior."""
    prior = _prior(prior_name)
    theta, x = _simulate(prior)
    npe = NPE(prior, show_progress_bars=False)
    npe.append_simulations(theta, x).train(max_num_epochs=1)
    fmpe = FMPE(prior, show_progress_bars=False)
    fmpe.append_simulations(theta, x).train(max_num_epochs=1)
    return prior_name, npe, fmpe


@pytest.fixture(scope="module", params=["box", "mvn"])
def trained(request):
    return _train(request.param)


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
    if kind == "mcmc_restricted_proposal":
        prior = npe._prior
        theta = prior.sample((200,))
        x = theta.clone()
        x[theta[:, 0] > 1.0] = float("nan")
        restriction = RestrictionEstimator(prior=prior)
        restriction.append_simulations(theta, x).train(max_num_epochs=2)
        return MCMCPosterior(
            npe.build_posterior().potential_fn,
            proposal=restriction.restrict_prior(),
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


@pytest.mark.parametrize("kind", POSTERIOR_KINDS)
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


@pytest.mark.parametrize("theta_transform", ["default", "user"])
def test_ensemble_to_keeps_default_x_and_transform(trained, theta_transform: str):
    """`.to()` keeps the ensemble's default x and the transform it was built with."""
    _, npe, _ = trained
    user_transform = (
        torch_tf.IndependentTransform(
            torch_tf.AffineTransform(torch.zeros(NUM_DIM), torch.ones(NUM_DIM)), 1
        )
        if theta_transform == "user"
        else None
    )
    ensemble = EnsemblePosterior(
        [npe.build_posterior(), npe.build_posterior()],
        weights=[3.0, 1.0],
        theta_transform=user_transform,
    )
    x_o = torch.zeros(1, NUM_DIM)
    ensemble.set_default_x(x_o)
    transform = ensemble.theta_transform
    if user_transform is not None:
        assert transform is user_transform

    assert ensemble.to("cpu") is ensemble

    assert torch.equal(ensemble.default_x, x_o)
    assert ensemble.theta_transform is transform
    theta = torch.zeros(3, NUM_DIM)
    assert ensemble.potential(theta).shape == (3,)
    map_ = ensemble.map(init_method=theta, num_iter=2, num_to_optimize=1)
    assert map_.shape == (1, NUM_DIM)


@pytest.mark.gpu
@pytest.mark.parametrize("prior_name", ["box", "mvn"])
@pytest.mark.parametrize("kind", POSTERIOR_KINDS)
def test_to_gpu_and_back(prior_name: str, kind: str):
    """After `.to(gpu)` and `.to("cpu")`, sample, potential and MAP run there."""
    if kind == "npe_a" and prior_name == "box":
        pytest.skip("Two-round NPE-A with a BoxUniform hits the known PD error.")
    _, npe, fmpe = _train(prior_name)
    posterior = _build(kind, npe, fmpe, prior_name)
    posterior.set_default_x(torch.zeros(1, NUM_DIM))

    for device in (process_device("gpu"), "cpu"):
        posterior.to(device)
        device_type = torch.device(device).type
        samples = posterior.sample((2,), show_progress_bars=False)
        assert samples.device.type == device_type
        assert posterior.potential(samples).device.type == device_type
        map_ = posterior.map(
            init_method=samples[:1], num_iter=2, num_to_optimize=1, force_update=True
        )
        assert map_.device.type == device_type
