# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

"""`posterior.to()` moves a posterior in place and keeps its state."""

from __future__ import annotations

import functools
from dataclasses import asdict

import pytest
import torch
import torch.distributions.transforms as torch_tf
from torch.distributions import MultivariateNormal

from sbi.inference import FMPE, NPE, NPE_A, EnsemblePosterior, MCMCPosterior
from sbi.inference.posteriors.posterior_parameters import (
    ImportanceSamplingPosteriorParameters,
    MCMCPosteriorParameters,
    VIPosteriorParameters,
    VectorFieldPosteriorParameters,
)
from sbi.utils import BoxUniform
from sbi.utils.torchutils import process_device

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


def _train(prior_name: str):
    """NPE and FMPE trained for one epoch on a `BoxUniform` or a torch prior."""
    prior = _prior(prior_name)
    theta, x = _simulate(prior)
    npe = NPE(prior, show_progress_bars=False)
    npe.append_simulations(theta, x).train(max_num_epochs=1)
    fmpe = FMPE(prior, show_progress_bars=False)
    fmpe.append_simulations(theta, x).train(max_num_epochs=1)
    return npe, fmpe


@pytest.fixture(scope="module")
def trained_on_cpu():
    """`_train`, once per prior, for tests that keep the estimators on the CPU."""
    return functools.cache(_train)


def _npe_a_round_two(prior_name: str):
    prior = _prior(prior_name)
    trainer = NPE_A(prior, num_components=2, show_progress_bars=False)
    theta, x = _simulate(prior, 300)
    trainer.append_simulations(theta, x).train(max_num_epochs=2, final_round=False)
    proposal = trainer.build_posterior().set_default_x(torch.zeros(1, NUM_DIM))
    theta = proposal.sample((300,), show_progress_bars=False)
    x = theta + 0.1 * torch.randn_like(theta)
    trainer.append_simulations(theta, x, proposal=proposal).train(
        max_num_epochs=2, final_round=True
    )
    return trainer.build_posterior()


def _build(kind: str, npe, fmpe, prior_name: str):
    if kind == "direct":
        return npe.build_posterior()
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
            npe.build_posterior().potential_fn, proposal=proposal, **asdict(MCMC_FAST)
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


@pytest.mark.parametrize("device", ["cpu", pytest.param("gpu", marks=pytest.mark.gpu)])
@pytest.mark.parametrize("prior_name", ["box", "mvn"])
@pytest.mark.parametrize(
    "kind",
    [
        "direct",
        "importance",
        "mcmc",
        "mcmc_posterior_proposal",
        "vi",
        "vector_field",
        "npe_a",
    ],
)
def test_to_moves_posterior_and_keeps_state(
    trained_on_cpu, kind: str, prior_name: str, device: str
):
    """`.to()` returns the posterior and keeps default x, MAP, transform and prior;
    sample, potential and MAP then run on the device."""
    if kind == "npe_a" and prior_name == "box":
        pytest.skip("Two-round NPE-A with a BoxUniform hits the known PD error.")
    # `.to()` moves the estimators in place, so a GPU case trains its own.
    npe, fmpe = trained_on_cpu(prior_name) if device == "cpu" else _train(prior_name)
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
    device = process_device(device)

    assert posterior.to(device) is posterior

    assert torch.equal(posterior.default_x, x_o.to(device))
    # A dropped cache would run gradient ascent again from the new init.
    new_init = torch.ones(1, NUM_DIM, device=device)
    cached_map = posterior.map(init_method=new_init, **map_kwargs)
    assert torch.equal(cached_map, map_.to(device))
    assert str(posterior) == description
    assert posterior.theta_transform is transform
    assert getattr(posterior.potential_fn, "iid_method", None) == iid_method
    for attr, shared in shares_prior.items():
        assert (getattr(posterior, attr) is posterior.potential_fn.prior) == shared
    device_type = torch.device(device).type
    samples = posterior.sample((2,), show_progress_bars=False)
    assert samples.device.type == device_type
    assert posterior.potential(samples).device.type == device_type
    new_map = posterior.map(init_method=samples[:1], force_update=True, **map_kwargs)
    assert new_map.device.type == device_type
    if device != "cpu":
        posterior.to("cpu")
        assert posterior.sample((2,), show_progress_bars=False).device.type == "cpu"


def test_ensemble_to_keeps_default_x_and_transform(trained_on_cpu):
    """`.to()` keeps the ensemble's default x and the transform it was built with."""
    npe, _ = trained_on_cpu("box")
    transform = torch_tf.IndependentTransform(
        torch_tf.AffineTransform(torch.zeros(NUM_DIM), torch.ones(NUM_DIM)), 1
    )
    ensemble = EnsemblePosterior(
        [npe.build_posterior(), npe.build_posterior()],
        weights=[3.0, 1.0],
        theta_transform=transform,
    )
    x_o = torch.zeros(1, NUM_DIM)
    ensemble.set_default_x(x_o)

    assert ensemble.to("cpu") is ensemble

    assert torch.equal(ensemble.default_x, x_o)
    assert ensemble.theta_transform is transform
    theta = torch.zeros(3, NUM_DIM)
    assert ensemble.potential(theta).shape == (3,)
    map_ = ensemble.map(init_method=theta, num_iter=2, num_to_optimize=1)
    assert map_.shape == (1, NUM_DIM)


def test_to_keeps_ode_options_of_bound_vector_field_potential(trained_on_cpu):
    """A bound vector field potential keeps the ODE options given to `bind()`."""
    _, fmpe = trained_on_cpu("box")
    potential = fmpe.build_posterior().potential_fn.bind(
        torch.zeros(1, NUM_DIM), atol=1e-3, rtol=1e-3, exact=False
    )
    theta = torch.zeros(3, NUM_DIM)
    torch.manual_seed(0)
    before = potential(theta)

    potential.to("cpu")

    torch.manual_seed(0)
    assert torch.equal(potential(theta), before)
