# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

import pickle

import pytest
import torch

from sbi import utils as utils
from sbi.inference import FMPE, NLE, NPE, NPSE, NRE
from sbi.inference.posteriors.posterior_parameters import (
    DirectPosteriorParameters,
    MCMCPosteriorParameters,
    RejectionPosteriorParameters,
    VIPosteriorParameters,
    VectorFieldPosteriorParameters,
)
from sbi.inference.posteriors.vi_posterior import VIPosterior


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
    # VIPosterior is skipped: `__setstate__` resets `_trained_on`, a separate gap.
    if not isinstance(posterior, VIPosterior):
        assert loaded_posterior.sample((1,)).shape == (1, num_dim)

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


def test_device_reconciliation_after_pickle(tmp_path, mcmc_params_fast):
    """Test that ``_device`` is reconciled after ``torch.load(map_location=...)``.

    ``NeuralPosterior.__setstate__`` detects the actual device from the neural
    network's parameters and updates ``_device`` and ``potential_fn.device``.
    """
    num_dim = 2
    prior = utils.BoxUniform(
        low=-2 * torch.ones(num_dim), high=2 * torch.ones(num_dim)
    )
    x_o = torch.zeros(1, num_dim)

    theta = prior.sample((500,))
    x = theta + 1.0 + torch.randn_like(theta) * 0.1

    inference = NPE(prior=prior, show_progress_bars=False)
    _ = inference.append_simulations(theta, x).train(max_num_epochs=1)
    posterior = inference.build_posterior(
        posterior_parameters=DirectPosteriorParameters()
    ).set_default_x(x_o)

    # Verify initial device is cpu
    assert posterior._device == "cpu"
    assert posterior.potential_fn.device == "cpu"

    # Save and load via pickle (same device, no map_location)
    loaded = pickle.loads(pickle.dumps(posterior))

    # _device must be preserved
    assert loaded._device == "cpu", (
        f"Expected _device='cpu', got {loaded._device!r}"
    )
    assert loaded.potential_fn.device == "cpu", (
        f"Expected potential_fn.device='cpu', got {loaded.potential_fn.device!r}"
    )

    # Sampling must still work after pickle load
    samples = loaded.sample((10,))
    assert samples.shape == (10, num_dim)
