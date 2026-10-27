# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

from __future__ import annotations

from typing import Tuple

import pytest
import torch
from torch import Tensor
from torch.distributions import Gamma, Independent, MultivariateNormal, Normal, Uniform

from sbi.inference.posteriors import VectorFieldPosterior
from sbi.inference.potentials.vector_field_adaptor import (
    AffineClassifierFreeGuidance,
    AffineClassifierFreeGuidanceConfig,
    FactorizedNPEScoreFunction,
    GaussCorrectedScoreFn,
    JacCorrectedScoreFn,
    PriorGuide,
    PriorGuideConfig,
    UniversalGuidance,
    UniversalGuidanceConfig,
    ensure_lam_positive_definite,
)
from sbi.inference.potentials.vector_field_potential import (
    vector_field_estimator_based_potential,
)
from sbi.neural_nets import posterior_flow_nn, posterior_score_nn
from sbi.samplers.score import Diffuser
from sbi.simulators.linear_gaussian import true_posterior_linear_gaussian_mvn_prior
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


@pytest.mark.parametrize("estimator_type", ["ve", "fmpe"])
@pytest.mark.parametrize("guidance", ["affine_classifier_free", "universal"])
def test_neutral_guidance_returns_the_score(estimator_type, guidance):
    """Guidance that changes nothing returns the estimator's score.

    For flow matching, the estimator's `forward` returns the velocity, not the score.
    """
    num_dim = 2
    prior = MultivariateNormal(torch.zeros(num_dim), torch.eye(num_dim))
    theta = prior.sample((100,))
    x = theta + 0.1 * torch.randn_like(theta)
    if estimator_type == "fmpe":
        estimator = posterior_flow_nn()(theta, x)
    else:
        estimator = posterior_score_nn(sde_type=estimator_type)(theta, x)
    # The zero-initialized output layer would make the net output zero.
    for parameter in estimator.net.parameters():
        torch.nn.init.normal_(parameter, std=0.1)

    if guidance == "affine_classifier_free":
        guided = AffineClassifierFreeGuidance(
            estimator, prior, AffineClassifierFreeGuidanceConfig()
        )
    else:
        guided = UniversalGuidance(
            estimator,
            prior,
            UniversalGuidanceConfig(
                guidance_fn=lambda input, condition, m, std: input.sum(-1) * 0.0,
                guidance_fn_score=lambda input, condition, m, std: input * 0.0,
            ),
        )

    inputs, conditions = torch.randn(4, num_dim), torch.randn(4, num_dim)
    time = torch.tensor([0.5])
    with torch.no_grad():
        torch.testing.assert_close(
            guided.score(inputs, conditions, time),
            estimator.score(inputs, conditions, time),
        )


@pytest.mark.parametrize("covariance_type", ["diag", "full"])
def test_prior_guide_score_does_not_depend_on_the_batch_layout(covariance_type):
    """The sampler passes inputs of shape (samples, 1, dim) to the guidance."""
    train_prior = MultivariateNormal(torch.zeros(2), torch.eye(2))
    test_prior = MultivariateNormal(torch.ones(2), 0.5 * torch.eye(2))
    theta = train_prior.sample((100,))
    estimator = posterior_score_nn(sde_type="ve")(theta, theta)
    guidance = PriorGuide(
        estimator,
        train_prior,
        PriorGuideConfig(
            train_prior=train_prior,
            test_prior=test_prior,
            K=2,
            covariance_type=covariance_type,
            num_steps=50,
        ),
    )
    inputs, conditions = torch.randn(5, 2), torch.randn(5, 2)
    time = torch.tensor([0.5])

    with torch.no_grad():
        torch.testing.assert_close(
            guidance(inputs[:, None], conditions[:, None], time)[:, 0],
            guidance(inputs, conditions, time),
        )


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


@pytest.mark.parametrize(
    "num_trials, magnitude, prior_scale",
    (
        pytest.param(5, 1e7, 1.0, id="large-lam"),
        pytest.param(2048, 2.0**20, 2.0**20, id="many-trials"),
    ),
)
def test_ensure_lam_positive_definite_survives_float32_cancellation(
    num_trials, magnitude, prior_scale
):
    """`Lam` must stay invertible when the precision sum cancels to rank one.

    An absolute nugget of 0.01 is lost in float32, at the magnitude of `Lam` for
    `large-lam` and at the magnitude of each posterior precision for `many-trials`.
    """
    ones = torch.ones(2)
    prior_precision = prior_scale * torch.eye(2).expand(1, 1, 2, 2)
    # Chosen such that (1 - N) * prior + sum(posterior) = magnitude * ones ones^T.
    posterior_precision = (
        (magnitude * torch.outer(ones, ones) + (num_trials - 1) * prior_precision)
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


def test_ensure_lam_positive_definite_survives_float32_cancellation_diagonal():
    """A diagonal `Lam` that cancels to zero must become positive.

    At a precision of 1e7, float32 cannot represent an absolute nugget of 0.01.
    """
    num_trials = 4
    prior_precision = torch.full((1, 1, 2), 2.0**24)
    # Chosen such that (1 - N) * prior + sum(posterior) = 0.
    posterior_precision = torch.full((1, num_trials, 2), 3 * 2.0**22)

    _, posterior_precision = ensure_lam_positive_definite(
        prior_precision, posterior_precision, num_trials, precision_nugget=0.01
    )
    lam = (1 - num_trials) * prior_precision + posterior_precision.sum(1, keepdim=True)

    assert (lam > 0).all()


def test_ensure_lam_positive_definite_keeps_well_conditioned_tall_data():
    """The guard changes a well-conditioned `Lam` by about one rounding unit per trial.

    The summed precisions are huge, `Lam` is not. Values are exact in float32.
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
@pytest.mark.parametrize("iid_method", ["gauss", "jac_gauss"])
def test_iid_score_is_exact_for_linear_gaussian(sde_type, iid_method):
    """`gauss` and `jac_gauss` must reproduce the exact iid posterior score.

    With exact single-trial scores, a Gaussian prior and a Gaussian likelihood, both
    methods (`gauss` given the true posterior precision) must reproduce the score of
    the diffused posterior given all trials, at every diffusion time. Runs in float64
    so that the comparison is exact up to rounding on any CPU.
    """
    num_dim, num_trials = 2, 5
    dtype = torch.float64
    prior_mean = torch.zeros(num_dim, dtype=dtype)
    prior_cov = torch.eye(num_dim, dtype=dtype)
    likelihood_shift = -torch.ones(num_dim, dtype=dtype)
    likelihood_cov = 0.5 * torch.eye(num_dim, dtype=dtype)
    prior = MultivariateNormal(prior_mean, prior_cov)
    x_o = torch.randn(num_trials, num_dim, dtype=dtype)

    # Single-trial posterior N(A (x - shift) + b, S) of the linear Gaussian model.
    prior_precision = torch.inverse(prior_cov)
    likelihood_precision = torch.inverse(likelihood_cov)
    posterior_cov = torch.inverse(prior_precision + likelihood_precision)
    A = posterior_cov @ likelihood_precision
    b = posterior_cov @ prior_precision @ prior_mean

    def diffused_gaussian_score(theta, mean, cov, m, s):
        precision = torch.inverse(m**2 * cov + s**2 * torch.eye(num_dim, dtype=dtype))
        return -(theta - m * mean) @ precision

    estimator = posterior_score_nn(
        sde_type=sde_type, model=_ZeroNet(), embedding_net=torch.nn.Identity()
    )(prior.sample((100,)), torch.randn(100, num_dim, dtype=dtype)).to(dtype)

    def exact_single_trial_score(input: Tensor, condition: Tensor, t: Tensor):
        mean = (condition - likelihood_shift) @ A.T + b
        m, s = estimator.mean_t_fn(t), estimator.std_fn(t)
        return diffused_gaussian_score(input, mean, posterior_cov, m, s)

    estimator.score = exact_single_trial_score

    if iid_method == "gauss":
        score_fn = GaussCorrectedScoreFn(
            estimator, prior, posterior_precision=torch.inverse(posterior_cov)
        )
    else:
        score_fn = JacCorrectedScoreFn(estimator, prior, ensure_lam_psd=False)

    posterior = true_posterior_linear_gaussian_mvn_prior(
        x_o, likelihood_shift, likelihood_cov, prior_mean, prior_cov
    )
    for t in [estimator.t_min, 0.01, 0.1, 0.5, 0.9, estimator.t_max]:
        time = torch.tensor([t], dtype=dtype)
        m, s = estimator.mean_t_fn(time), estimator.std_fn(time)
        theta = m * posterior.mean + s * torch.randn(64, 1, num_dim, dtype=dtype)

        expected = diffused_gaussian_score(
            theta, posterior.mean, posterior.covariance_matrix, m, s
        )
        assert torch.allclose(score_fn(theta, x_o, time), expected, rtol=1e-8), t


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


def test_fnpe_batched_observations_keep_base_distribution():
    """fnpe only contracts the sampler when it composes iid observations, not for a
    batch of independent observations as in `sample_batched`."""
    score_fn = _build_gaussian_score_estimator("vp", (2,), torch.ones(2), torch.ones(2))
    score_fn = score_fn.bind(torch.ones((100, 1)), x_is_iid=False, iid_method="fnpe")

    assert score_fn.init_std_scale == 1.0


def test_fnpe_divergence_error_names_cause():
    """Diverged fnpe sampling for many iid observations raises an error that names
    the cause and the fixes."""
    estimator = _build_gaussian_score_estimator(
        "vp", (2,), torch.ones(2), torch.ones(2)
    ).vector_field_estimator
    prior = Independent(Normal(torch.zeros(2), 10 * torch.ones(2)), 1)
    posterior = VectorFieldPosterior(estimator, prior, iid_method="fnpe")

    with pytest.raises(RuntimeError, match=r"No finite sample.*\(1000 here\)"):
        posterior.sample((100,), x=torch.ones((1000, 1)), steps=100)


@pytest.mark.parametrize(
    "diverged, value",
    (
        pytest.param((slice(None), 0), float("nan"), id="one-observation-all-nan"),
        pytest.param((..., 0), float("inf"), id="every-sample-partly-inf"),
    ),
)
def test_sample_batched_raises_when_samples_diverge(monkeypatch, diverged, value):
    """Sampling raises when one observation has no finite sample, also if the
    other observations are fine or each sample is only partly non-finite."""
    estimator = _build_gaussian_score_estimator(
        "vp", (2,), torch.ones(2), torch.ones(2)
    ).vector_field_estimator
    prior = Independent(Normal(torch.zeros(2), torch.ones(2)), 1)
    posterior = VectorFieldPosterior(estimator, prior)
    run = Diffuser.run

    def run_with_diverged_entries(self, *args, **kwargs):
        samples = run(self, *args, **kwargs)
        samples[diverged] = value
        return samples

    monkeypatch.setattr(Diffuser, "run", run_with_diverged_entries)

    with pytest.raises(RuntimeError, match="No finite sample"):
        posterior.sample_batched(
            (10,), x=torch.ones((3, 1)), steps=5, reject_outside_prior=False
        )


def test_fnpe_uses_custom_prior_score_weight():
    """A custom prior score weight replaces the default one."""
    estimator = _build_gaussian_score_estimator(
        "vp", (2,), torch.ones(2), torch.ones(2)
    ).vector_field_estimator
    prior = Independent(Normal(torch.zeros(2), torch.ones(2)), 1)
    args = (torch.randn(4, 1, 2), torch.ones(5, 1), torch.tensor([0.5]))

    default = FactorizedNPEScoreFunction(estimator, prior)
    no_prior = FactorizedNPEScoreFunction(
        estimator, prior, prior_score_weight=torch.zeros_like
    )

    assert not torch.allclose(no_prior(*args), default(*args))


class _ZeroNet(VectorFieldNet):
    """Vector field net that predicts zero everywhere."""

    def __init__(self):
        super().__init__()
        self.dummy_param_for_device_detection = torch.nn.Linear(1, 1)

    def forward(self, input, condition, time):
        return torch.zeros_like(input)


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
    score_estimator = posterior_score_nn(
        sde_type=sde_type,
        model=_ZeroNet(),
        embedding_net=torch.nn.Identity(),
    )(building_thetas, building_xs)

    score_fn, _ = vector_field_estimator_based_potential(
        score_estimator, prior=None, x_o=torch.ones((1,))
    )

    return score_fn
