# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

"""`posterior.to()` moves a posterior in place and keeps its state."""

from __future__ import annotations

import functools
from dataclasses import asdict

import pytest
import torch
import torch.distributions.transforms as torch_tf
from torch.distributions import (
    Distribution,
    Independent,
    MultivariateNormal,
    Normal,
    constraints,
)

from sbi.inference import FMPE, NPE, EnsemblePosterior, MCMCPosterior, VIPosterior
from sbi.inference.posteriors.posterior_parameters import (
    ImportanceSamplingPosteriorParameters,
    MCMCPosteriorParameters,
    RejectionPosteriorParameters,
    VIPosteriorParameters,
    VectorFieldPosteriorParameters,
)
from sbi.utils import BoxUniform
from sbi.utils.torchutils import process_device
from sbi.utils.user_input_checks_utils import move_distribution_to_device

NUM_DIM = 2
MCMC_FAST = MCMCPosteriorParameters(
    num_chains=1, thin=1, warmup_steps=1, init_strategy="proposal"
)
# `log_prob()` arguments that fill the posterior's cache of a normalization estimate.
CACHED_LOG_PROB_KWARGS = {
    "direct": dict(leakage_correction_params=dict(num_rejection_samples=100)),
    "importance": dict(normalization_constant_params=dict(num_samples=100)),
    "vector_field": dict(leakage_correction_params=dict(num_rejection_samples=100)),
}


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


def _build(kind: str, npe, fmpe):
    if kind == "direct":
        return npe.build_posterior()
    if kind == "rejection":
        return npe.build_posterior(posterior_parameters=RejectionPosteriorParameters())
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
    raise ValueError(kind)


@pytest.mark.parametrize("device", ["cpu", pytest.param("gpu", marks=pytest.mark.gpu)])
@pytest.mark.parametrize("prior_name", ["box", "mvn"])
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
    ],
)
def test_to_moves_posterior_and_keeps_state(
    trained_on_cpu, kind: str, prior_name: str, device: str
):
    """`.to()` returns the posterior and keeps default x, MAP, transform and prior;
    sample, potential and MAP then run on the device."""
    # `.to()` moves the estimators in place, so a GPU case trains its own.
    npe, fmpe = trained_on_cpu(prior_name) if device == "cpu" else _train(prior_name)
    posterior = _build(kind, npe, fmpe)
    x_o = torch.zeros(1, NUM_DIM)
    posterior.set_default_x(x_o)
    map_kwargs = dict(num_iter=2, num_to_optimize=1, num_init_samples=1)
    map_ = posterior.map(init_method=torch.zeros(1, NUM_DIM), **map_kwargs)
    theta = torch.zeros(1, NUM_DIM)
    log_prob_kwargs = CACHED_LOG_PROB_KWARGS.get(kind)
    if log_prob_kwargs is not None:
        log_prob = posterior.log_prob(theta, **log_prob_kwargs)
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
    if log_prob_kwargs is not None:
        # A new estimate from 100 samples would differ.
        moved_log_prob = posterior.log_prob(theta.to(device), **log_prob_kwargs)
        assert torch.allclose(moved_log_prob.cpu(), log_prob, atol=1e-4)
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


class _CustomNormal(Distribution):
    """A custom prior that declares no parameters in `arg_constraints`."""

    arg_constraints = {}
    support = constraints.real_vector

    def __init__(self, loc: float | torch.Tensor = 0.0, scale: float = 1.0):
        self.loc, self.scale = loc, scale
        super().__init__(event_shape=torch.Size([NUM_DIM]), validate_args=False)

    def log_prob(self, value):
        return Normal(self.loc, self.scale).log_prob(value).sum(-1)


@pytest.mark.parametrize("loc", [4.0, torch.full((NUM_DIM,), 4.0)])
def test_move_keeps_custom_prior_without_declared_parameters(loc):
    """Such a prior moves in place and is not rebuilt with its default arguments."""
    prior = _CustomNormal(loc=loc, scale=2.0)

    assert move_distribution_to_device(prior, "meta") is prior


@pytest.mark.parametrize("device", ["cpu", pytest.param("gpu", marks=pytest.mark.gpu)])
def test_vi_posterior_with_user_q_retrains_after_move(trained_on_cpu, device: str):
    """A VI posterior with a user `q` retrains from scratch after `.to()`, and moving a
    posterior built from it leaves the source's initial distribution in place."""
    npe, _ = trained_on_cpu("box")
    loc = torch.zeros(NUM_DIM, requires_grad=True)
    scale = torch.ones(NUM_DIM, requires_grad=True)
    q = Independent(Normal(loc, scale), 1)
    train_kwargs = dict(
        max_num_iters=2,
        check_for_convergence=False,
        quality_control=False,
        show_progress_bar=False,
    )
    potential_fn = npe.build_posterior().potential_fn
    prior = potential_fn.prior
    x_o = torch.zeros(1, NUM_DIM)
    source = VIPosterior(potential_fn, prior, q=q, parameters=[loc, scale])
    source.set_default_x(x_o).train(**train_kwargs)
    copy = VIPosterior(potential_fn, prior, q=source).set_default_x(x_o)
    device = process_device(device)

    copy.to(device)

    source_snapshot = source._q_spec.init_snapshot
    assert source_snapshot is not copy._q_spec.init_snapshot
    assert all(p.device.type == "cpu" for p in source_snapshot.parameters())
    copy.train(retrain_from_scratch=True, **train_kwargs)
    samples = copy.sample((2,), show_progress_bars=False)
    assert samples.device.type == torch.device(device).type
