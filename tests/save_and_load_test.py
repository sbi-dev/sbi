# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

import io
import pickle

import pytest
import torch
from torch.distributions import MultivariateNormal

from sbi import utils as utils
from sbi.inference import FMPE, NLE, NPE, NPSE, NRE
from sbi.inference.posteriors.ensemble_posterior import EnsemblePosterior
from sbi.inference.posteriors.mcmc_posterior import MCMCPosterior
from sbi.inference.posteriors.posterior_parameters import (
    DirectPosteriorParameters,
    ImportanceSamplingPosteriorParameters,
    MCMCPosteriorParameters,
    RejectionPosteriorParameters,
    VIPosteriorParameters,
    VectorFieldPosteriorParameters,
)
from sbi.inference.posteriors.vi_posterior import VIPosterior


def _assert_survives_pickling(posterior, num_dim: int) -> None:
    """Saving must leave the posterior unchanged and reload an equivalent one.

    Compares seeded samples rather than their shape: a posterior that lost a weight, or
    a `VIPosterior` that silently retrained, still returns correctly shaped samples.
    """
    torch.manual_seed(0)
    expected = posterior.sample((3,))

    attributes = {name: id(value) for name, value in vars(posterior).items()}
    reloaded = pickle.loads(pickle.dumps(posterior))
    assert {name: id(value) for name, value in vars(posterior).items()} == attributes, (
        "pickling replaced attributes on the posterior it serialized"
    )

    torch.manual_seed(0)
    samples = reloaded.sample((3,))
    assert samples.shape == (3, num_dim)
    if not isinstance(posterior, MCMCPosterior):
        # MCMC drops its sampler on purpose (#1291) and so redraws a fresh chain.
        assert torch.allclose(samples, expected)


@pytest.mark.parametrize(
    "inference_method, posterior_parameters",
    (
        (NPE, DirectPosteriorParameters),
        (NPSE, VectorFieldPosteriorParameters),
        (FMPE, VectorFieldPosteriorParameters),
        pytest.param(NLE, MCMCPosteriorParameters, marks=pytest.mark.mcmc),
        pytest.param(NRE, MCMCPosteriorParameters, marks=pytest.mark.mcmc),
        pytest.param(NRE, VIPosteriorParameters, marks=pytest.mark.mcmc),
        (NRE, RejectionPosteriorParameters),
        (NRE, ImportanceSamplingPosteriorParameters),
    ),
)
def test_picklability(
    inference_method,
    posterior_parameters,
    tmp_path,
    mcmc_params_fast: MCMCPosteriorParameters,
):
    num_dim = 2
    prior = utils.BoxUniform(low=-2 * torch.ones(num_dim), high=2 * torch.ones(num_dim))
    x_o = torch.zeros(1, num_dim)

    theta = prior.sample((500,))
    x = theta + 1.0 + torch.randn_like(theta) * 0.1

    inference = inference_method(prior=prior)
    _ = inference.append_simulations(theta, x).train(max_num_epochs=1)
    if posterior_parameters is MCMCPosteriorParameters:
        posterior = inference.build_posterior(
            posterior_parameters=mcmc_params_fast
        ).set_default_x(x_o)
    else:
        posterior = inference.build_posterior(
            posterior_parameters=posterior_parameters()
        ).set_default_x(x_o)
    # After sample and log_prob, the posterior should still be picklable
    if isinstance(posterior, VIPosterior):
        posterior.train(max_num_iters=10)
    _ = posterior.sample((1,))
    _ = posterior.potential(torch.zeros(1, num_dim))

    with open(f"{tmp_path}/saved_posterior.pickle", "wb") as handle:
        pickle.dump(posterior, handle)
    with open(f"{tmp_path}/saved_posterior.pickle", "rb") as handle:
        loaded_posterior = pickle.load(handle)

    # A corrupted `theta_transform` unpickles fine and only fails on use (#1952).
    assert loaded_posterior.sample((1,)).shape == (1, num_dim)
    _assert_survives_pickling(posterior, num_dim)

    with open(f"{tmp_path}/saved_inference.pickle", "wb") as handle:
        pickle.dump(inference, handle)
    with open(f"{tmp_path}/saved_inference.pickle", "rb") as handle:
        _ = pickle.load(handle)


def _build_transformed_estimator(builder_name: str):
    """Build an `"mdn"` or `"zuko"` estimator with an unconstraining input transform."""
    from sbi.neural_nets.net_builders.flow import build_zuko_maf
    from sbi.neural_nets.net_builders.mdn import build_mdn
    from sbi.utils import BoxUniform

    builder = build_mdn if builder_name == "mdn" else build_zuko_maf
    prior = BoxUniform(-2 * torch.ones(2), 2 * torch.ones(2))
    batch_x, batch_y = prior.sample((256,)), torch.randn(256, 3)
    estimator = builder(
        batch_x, batch_y, z_score_x="transform_to_unconstrained", x_dist=prior
    )
    return estimator, prior


@pytest.mark.parametrize("builder_name", ["mdn", "zuko"])
def test_unconstraining_transform_survives_pickle(builder_name):
    """Pickling preserves the module-owned transform and leaves log-probs unchanged."""
    estimator, prior = _build_transformed_estimator(builder_name)

    theta, condition = prior.sample((5,)).unsqueeze(1), torch.randn(1, 3)
    expected = estimator.log_prob(theta, condition)

    reloaded = pickle.loads(pickle.dumps(estimator))
    assert torch.allclose(reloaded.log_prob(theta, condition), expected)


@pytest.mark.parametrize("builder_name", ["mdn", "zuko"])
def test_unconstraining_transform_follows_dtype_cast(builder_name):
    """`.double()` reaches the transform tensors, not only the weights.

    Guards `_apply_to_transform`: rebuilding from the prior would leave the transform
    in float32 and silently desync it from the weights.
    """
    from sbi.utils.sbiutils import CallableTransform, _apply_to_transform

    estimator, prior = _build_transformed_estimator(builder_name)

    estimator.double()

    # Both backends own the transform through a CallableTransform submodule.
    wrappers = [m for m in estimator.modules() if isinstance(m, CallableTransform)]
    assert len(wrappers) == 1, f"expected one transform wrapper, found {len(wrappers)}"
    transform_dtypes = []

    def record_dtype(tensor):
        transform_dtypes.append(tensor.dtype)
        return tensor

    _apply_to_transform(wrappers[0].transform, record_dtype)
    assert transform_dtypes, "expected the transform to hold tensors"
    assert all(dtype == torch.float64 for dtype in transform_dtypes)

    theta, condition = prior.sample((5,)).unsqueeze(1), torch.randn(1, 3)
    log_probs = estimator.log_prob(theta.double(), condition.double())
    assert log_probs.dtype == torch.float64
    assert torch.isfinite(log_probs).all()


def _box_uniform():
    return utils.BoxUniform(-2 * torch.ones(2), 2 * torch.ones(2))


def _mvn():
    return MultivariateNormal(torch.zeros(2), torch.eye(2))


def _build_posterior(kind: str, prior, num_dim: int = 2):
    """A small trained posterior on CPU for `torch.load` round-trips."""
    if kind == "ensemble":
        posteriors = [_build_posterior("direct", prior) for _ in range(2)]
        return EnsemblePosterior(posteriors).set_default_x(torch.zeros(1, num_dim))
    theta = prior.sample((200,))
    x = theta + 0.1 * torch.randn_like(theta)
    trainer_cls = FMPE if kind == "vector_field" else NPE
    trainer = trainer_cls(prior=prior, show_progress_bars=False)
    trainer.append_simulations(theta, x).train(max_num_epochs=1)
    return trainer.build_posterior().set_default_x(torch.zeros(1, num_dim))


def _torch_load_roundtrip(posterior, map_location: str):
    buffer = io.BytesIO()
    torch.save(posterior, buffer)
    buffer.seek(0)
    return torch.load(buffer, weights_only=False, map_location=map_location)


@pytest.mark.parametrize(
    "kind, make_prior",
    [
        pytest.param("direct", _box_uniform, id="direct-box_uniform"),
        pytest.param("direct", _mvn, id="direct-mvn"),
        pytest.param("vector_field", _mvn, id="vector_field-mvn"),
        pytest.param("ensemble", _mvn, id="ensemble-mvn"),
    ],
)
def test_torch_load_map_location_reconciles_claimed_device(kind, make_prior):
    """A posterior whose device strings disagree with its tensors is reconciled on load.

    Claiming `cuda:0` on a CPU posterior simulates one that was saved on a GPU and
    loaded with `map_location="cpu"`, so the reconciliation runs on CPU-only CI. The
    plain torch prior has no `.to()`, which every potential must handle without raising.
    """
    posterior = _build_posterior(kind, make_prior())
    posterior._device = posterior.device = posterior.potential_fn.device = "cuda:0"
    if hasattr(posterior.prior, "device"):
        posterior.prior.device = "cuda:0"

    loaded = _torch_load_roundtrip(posterior, map_location="cpu")

    assert loaded._device == "cpu", f"_device is {loaded._device!r}"
    assert loaded.device == "cpu", f"device is {loaded.device!r}"
    assert loaded.potential_fn.device == "cpu", (
        f"potential_fn.device is {loaded.potential_fn.device!r}"
    )
    assert getattr(loaded.prior, "device", "cpu") == "cpu"
    assert loaded.prior is loaded.potential_fn.prior, "prior is no longer shared"
    samples = loaded.sample((10,))
    assert samples.device.type == "cpu", f"samples on {samples.device}"
    potential = loaded.potential(samples)
    assert potential.device.type == "cpu", f"potential on {potential.device}"


def test_torch_load_map_location_same_device_is_passthrough():
    """Loading a CPU posterior with `map_location="cpu"` leaves its state unchanged."""
    posterior = _build_posterior("direct", _box_uniform())

    loaded = _torch_load_roundtrip(posterior, map_location="cpu")

    assert loaded._device == "cpu"
    assert loaded.device == "cpu"
    assert loaded.sample((10,)).shape == (10, 2)
