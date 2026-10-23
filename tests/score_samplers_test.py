# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

from __future__ import annotations

from typing import Tuple

import pytest
import torch
from torch import Tensor
from torch.distributions import Gamma, Independent, MultivariateNormal, Normal, Uniform

from sbi.inference.potentials.vector_field_adaptor import ensure_lam_positive_definite
from sbi.inference.potentials.vector_field_potential import (
    vector_field_estimator_based_potential,
)
from sbi.neural_nets import posterior_score_nn
from sbi.samplers.score import Diffuser
from sbi.utils import BoxUniform, MultipleIndependent
from sbi.utils.vector_field_utils import (
    VectorFieldNet,
    add_diag_or_dense,
    solve_diag_or_dense,
)


def build_random_priors(num_dim: int):
    # Diag normal prior
    prior1 = Independent(Normal(torch.zeros(num_dim), torch.ones(num_dim)), 1)
    # Uniform prior
    prior2 = Independent(Uniform(torch.zeros(num_dim), torch.ones(num_dim)), 1)
    prior2_2 = BoxUniform(torch.zeros(num_dim), torch.ones(num_dim))
    # Multivariate normal prior
    prior3 = MultivariateNormal(torch.zeros(num_dim), torch.eye(num_dim))
    # Gamma prior - analytical not implemented but should fall back to general case
    prior4 = Independent(Gamma(torch.ones(num_dim), torch.ones(num_dim)), 1)
    # Multiple independent prior
    if num_dim == 1:
        prior5 = Independent(Normal(torch.zeros(1), torch.ones(1)), 1)
    else:
        prior5 = MultipleIndependent([
            Normal(torch.zeros(1), torch.ones(1)) for _ in range(num_dim)
        ])

    priors = [prior1, prior2, prior2_2, prior3, prior4, prior5]

    return priors


@pytest.mark.parametrize("sde_type", ["vp", "ve", "subvp"])
@pytest.mark.parametrize(
    "iid_method",
    [
        "fnpe",
        "gauss",
        "auto_gauss",
        "jac_gauss",
    ],
)
@pytest.mark.parametrize("num_dim", [1, 2, 3])
def test_score_fn_iid_on_different_priors(sde_type, iid_method, num_dim):
    """Test the the iid methods work with the most common priors that are used in
    practice (or are implemented in this library).

    This is mostly becuase "gauss" based methods do need to perform analytical
    integration over the prior (which is automatically done in this library). It ensures
    that it doesn't lead to any errors, but does not test the correctness of the
    integration!
    """
    mean0 = torch.zeros(num_dim)
    std0 = torch.ones(num_dim)
    score_fn = _build_gaussian_score_estimator(sde_type, (num_dim,), mean0, std0)

    priors = build_random_priors(num_dim)
    x_o_iid = torch.ones((5, 1))
    score_fn = score_fn.bind(x_o_iid, x_is_iid=True, iid_method=iid_method)
    inputs = torch.ones((1, 1, num_dim))
    time = torch.ones(1)
    for prior in priors:
        score_fn.prior = prior
        output = score_fn.gradient(inputs, time=time)

        assert output.shape == (1, 1, num_dim), (
            f"Expected shape {(1, 1, num_dim)}, got {output.shape}"
        )
        assert torch.isfinite(output).all(), "Output contains non-finite values"


@pytest.mark.parametrize("sde_type", ["vp", "ve", "subvp"])
@pytest.mark.parametrize(
    "guidance_method",
    [
        ("affine_classifier_free", {"likelihood_scale": 0.1}),
        ("affine_classifier_free", {"likelihood_scale": 10.0, "prior_scale": 1.0}),
        (
            "affine_classifier_free",
            {
                "likelihood_scale": 0.1,
                "prior_scale": 1.0,
                "prior_shift": 1.0,
                "likelihood_shift": 1.0,
            },
        ),
        ("universal", {"guidance_fn": lambda x, t, y, z: x}),
        (
            "universal",
            {
                "guidance_fn": lambda x, t, y, z: x + 1.0,
                "guidance_fn_score": lambda x, t, y, z: x,
            },
        ),
        ("interval", {"lower_bound": 0.0, "upper_bound": 1.0}),
        ("interval", {"lower_bound": None, "upper_bound": 1.0}),
        ("interval", {"lower_bound": -1.0, "upper_bound": None}),
    ],
)
@pytest.mark.parametrize("num_dim", [1, 2, 3])
def test_score_fn_guidance_general(sde_type, guidance_method, num_dim):
    """ """
    mean0 = torch.zeros(num_dim)
    std0 = torch.ones(num_dim)
    score_fn = _build_gaussian_score_estimator(sde_type, (num_dim,), mean0, std0)
    x_o = torch.ones((1, 1))
    guidance_name, guidance_params = guidance_method
    score_fn = score_fn.bind(
        x_o, guidance_method=guidance_name, guidance_params=guidance_params
    )
    inputs = torch.ones((1, 1, num_dim))
    time = torch.ones(1)

    priors = build_random_priors(num_dim)
    for prior in priors:
        score_fn.prior = prior
        output = score_fn.gradient(inputs, time=time)

    assert output.shape == (1, 1, num_dim), (
        f"Expected shape {(1, 1, num_dim)}, got {output.shape}"
    )
    assert torch.isfinite(output).all(), "Output contains non-finite values"


@pytest.mark.parametrize("sde_type", ["vp", "ve"])
@pytest.mark.parametrize("iid_method", ["fnpe", "gauss"])
@pytest.mark.parametrize(
    "guidance_method",
    [
        ("affine_classifier_free", {"likelihood_scale": 0.7}),
        ("interval", {"lower_bound": 0.0, "upper_bound": 1.0}),
    ],
)
def test_score_fn_combined_guidance_and_iid(sde_type, iid_method, guidance_method):
    """Test that guidance can be combined with IID score accumulation."""
    num_dim = 2
    mean0 = torch.zeros(num_dim)
    std0 = torch.ones(num_dim)
    score_fn = _build_gaussian_score_estimator(sde_type, (num_dim,), mean0, std0)

    x_o_iid = torch.ones((5, 1))
    guidance_name, guidance_params = guidance_method
    score_fn = score_fn.bind(
        x_o_iid,
        x_is_iid=True,
        iid_method=iid_method,
        guidance_method=guidance_name,
        guidance_params=guidance_params,
    )

    inputs = torch.ones((1, 1, num_dim))
    time = torch.ones(1)

    for prior in build_random_priors(num_dim):
        score_fn.prior = prior
        output = score_fn.gradient(inputs, time=time)
        assert output.shape == (1, 1, num_dim), (
            f"Expected shape {(1, 1, num_dim)}, got {output.shape}"
        )
        assert torch.isfinite(output).all(), "Output contains non-finite values"


def test_ensure_lam_positive_definite_survives_float32_cancellation():
    """`Lam` must stay invertible when the precision sum cancels to rank one.

    At a magnitude of 1e7, float32 cannot represent an absolute nugget of 0.01.
    """
    num_trials, magnitude = 5, 1e7
    ones = torch.ones(2)
    prior_precision = torch.eye(2).expand(1, 1, 2, 2)
    # Chosen such that (1 - N) * prior + sum(posterior) = magnitude * ones ones^T.
    posterior_precision = (
        (magnitude * torch.outer(ones, ones) + (num_trials - 1) * torch.eye(2))
        / num_trials
    ).expand(1, num_trials, 2, 2)

    prior_precision, posterior_precision = ensure_lam_positive_definite(
        prior_precision, posterior_precision, num_trials, precision_nugget=0.01
    )
    lam = add_diag_or_dense(
        (1 - num_trials) * prior_precision,
        posterior_precision.sum(1, keepdim=True),
        batch_dims=2,
    ).squeeze(1)

    eigenvalues = torch.linalg.eigvalsh(lam)
    assert (eigenvalues[:, 0] / eigenvalues[:, -1] > 5e-6).all()
    assert torch.isfinite(solve_diag_or_dense(lam, torch.ones(1, 2), 1)).all()


def test_ensure_lam_positive_definite_keeps_well_conditioned_tall_data():
    """An isotropic `Lam` from many trials must not be regularized.

    The summed precisions are huge while `Lam` is not. Values are exact in float32.
    """
    num_trials, prior_precision_value = 2048, 2.0**20
    prior_precision = prior_precision_value * torch.eye(2).expand(1, 1, 2, 2)
    posterior_precision = (prior_precision_value + 1) * torch.eye(2)
    posterior_precision = posterior_precision.expand(1, num_trials, 2, 2)

    _, corrected = ensure_lam_positive_definite(
        prior_precision, posterior_precision, num_trials, precision_nugget=0.01
    )

    assert torch.allclose(corrected, posterior_precision, rtol=1e-6)


@pytest.mark.parametrize("sde_type", ["vp", "ve", "subvp"])
@pytest.mark.parametrize("predictor", ("euler_maruyama",))
@pytest.mark.parametrize("corrector", (None, "gibbs", "langevin"))
@pytest.mark.parametrize("input_event_shape", ((1,), (4,)))
@pytest.mark.parametrize("mu", (-1.0, 0.0, 1.0))
@pytest.mark.parametrize("std", (1.0, 0.1))
def test_gaussian_score_sampling(
    sde_type, predictor, corrector, input_event_shape, mu, std
):
    mean0 = mu * torch.ones(input_event_shape)
    std0 = std * torch.ones(input_event_shape)

    score_fn = _build_gaussian_score_estimator(sde_type, input_event_shape, mean0, std0)

    sampler = Diffuser(score_fn, predictor, corrector)

    t_min = score_fn.vector_field_estimator.t_min
    t_max = score_fn.vector_field_estimator.t_max
    ts = torch.linspace(t_max, t_min, 500)
    samples = sampler.run(1_000, ts)

    mean_est = samples.mean(0)
    std_est = samples.std(0)

    assert torch.allclose(mean_est, mean0, atol=1e-1)
    assert torch.allclose(std_est, std0, atol=1e-1)


def _build_gaussian_score_estimator(
    sde_type: str,
    input_event_shape: Tuple[int],
    mean0: Tensor,
    std0: Tensor,
):
    """Helper function for all tests that deal with shapes of density estimators."""

    # Use discrete thetas such that categorical density esitmators can also use them.
    building_thetas = (
        torch.randn((1000, *input_event_shape), dtype=torch.float32) * std0 + mean0
    )
    building_xs = torch.ones((1000, 1))

    # Note the precondition predicts a correct Gaussian score by default if the neural
    # net predicts 0!
    class DummyNet(VectorFieldNet):
        def __init__(self):
            super().__init__()
            self.dummy_param_for_device_detection = torch.nn.Linear(1, 1)

        def forward(self, input, condition, time):
            return torch.zeros_like(input)

    score_estimator = posterior_score_nn(
        sde_type=sde_type,
        model=DummyNet(),
        embedding_net=torch.nn.Identity(),
    )(building_thetas, building_xs)

    score_fn, _ = vector_field_estimator_based_potential(
        score_estimator, prior=None, x_o=torch.ones((1,))
    )

    return score_fn
