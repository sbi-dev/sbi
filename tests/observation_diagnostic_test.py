# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

from __future__ import annotations

import pickle
import warnings

import pytest
import torch
import torch.nn as nn

from sbi.diagnostics.observation import (
    ObservationDiagnostic,
    ReferenceObservationStats,
    diagnose_observation,
)
from sbi.inference import NPE
from sbi.inference.posteriors.ensemble_posterior import EnsemblePosterior
from sbi.neural_nets import posterior_nn
from sbi.neural_nets.embedding_nets import (
    FCEmbedding,
    PermutationInvariantEmbedding,
)
from sbi.utils import BoxUniform
from sbi.utils.sbiutils import seed_all_backends

seed = 2026


@pytest.fixture(autouse=True)
def set_seed():
    seed_all_backends(seed)


def test_diagnostic_dataclass_and_dict_access():
    """Verify ObservationDiagnostic structure, attribute and dict-like access."""
    diag = ObservationDiagnostic(
        is_in_distribution=True,
        empirical_percentile=0.42,
        mahalanobis_distance=1.23,
        warning=None,
        method="mahalanobis",
        threshold=0.01,
        space="observation",
        details={"n_reference": 1000},
    )

    # Attribute access
    assert diag.is_in_distribution is True
    assert diag.empirical_percentile == 0.42
    assert diag.mahalanobis_distance == 1.23
    assert diag.warning is None
    assert diag.method == "mahalanobis"
    assert diag.threshold == 0.01
    assert diag.space == "observation"
    assert diag.details["n_reference"] == 1000

    # Dict-like access
    assert diag["is_in_distribution"] is True
    assert diag["empirical_percentile"] == 0.42
    assert diag["mahalanobis_distance"] == 1.23
    assert diag["warning"] is None
    assert diag.get("method") == "mahalanobis"
    assert diag.get("nonexistent", "fallback") == "fallback"

    d = diag.to_dict()
    assert isinstance(d, dict)
    assert d["is_in_distribution"] is True
    assert "empirical_percentile" in d
    assert "mahalanobis_distance" in d

    # Iteration / keys / items
    keys = list(diag.keys())
    assert "is_in_distribution" in keys
    assert "warning" in keys

    # String representation
    s = str(diag)
    assert "is_in_distribution=True" in s
    assert "empirical_percentile=0.4200" in s


def test_reference_stats_computation_and_device():
    """Verify ReferenceObservationStats creation, memory size, and device transfer."""
    x = torch.randn(500, 4)
    stats = ReferenceObservationStats.from_simulations(x)

    assert stats.n_simulations == 500
    assert stats.event_shape == torch.Size([4])
    assert torch.allclose(stats.mean, x.mean(dim=0), atol=1e-5)
    assert len(stats.reference_distances) == 500
    # Distances must be sorted
    assert torch.all(stats.reference_distances[:-1] <= stats.reference_distances[1:])

    # Device transfer
    stats_cpu = stats.to("cpu")
    assert stats_cpu.mean.device.type == "cpu"
    assert stats_cpu.reference_distances.device.type == "cpu"


def test_typical_and_ood_observations():
    """Verify that typical observation is accepted and extreme outlier flagged."""
    # Reference: 1000 samples from standard Gaussian in 3D
    x_ref = torch.randn(1000, 3)
    stats = ReferenceObservationStats.from_simulations(x_ref)

    # In-distribution observation: close to mean
    x_in = torch.zeros(3)
    diag_in = stats.diagnose(x_in, threshold=0.01)

    assert diag_in.is_in_distribution is True
    assert diag_in.warning is None
    assert diag_in.empirical_percentile is not None
    assert diag_in.empirical_percentile > 0.05
    assert diag_in.mahalanobis_distance is not None
    assert diag_in.mahalanobis_distance < 3.0

    # Out-of-distribution observation: far outlier (10 standard deviations away)
    x_out = torch.tensor([10.0, 10.0, 10.0])
    diag_out = stats.diagnose(x_out, threshold=0.01)

    assert diag_out.is_in_distribution is False
    assert diag_out.warning is not None
    assert "extreme tails" in diag_out.warning
    assert diag_out.empirical_percentile == 0.0
    assert diag_out.mahalanobis_distance is not None
    assert diag_out.mahalanobis_distance > 10.0


def test_statistical_correctness_known_distribution():
    """Verify empirical percentiles on 2D standard Gaussian against chi-square theory.

    For standard 2D Gaussian, d_M^2 ~ Chi^2(2) = Exponential(1/2).
    Survival function: P(d_M >= r) = exp(-r^2 / 2).
    """
    n_sims = 5000
    x_ref = torch.randn(n_sims, 2)
    stats = ReferenceObservationStats.from_simulations(x_ref)

    # Check median: r = sqrt(2 * ln(2)) ~= 1.1774 has theoretical survival prob 0.50
    r_median = float((2.0 * torch.log(torch.tensor(2.0))).sqrt())
    x_med = torch.tensor([r_median, 0.0])
    diag_med = stats.diagnose(x_med, threshold=0.01)
    assert diag_med.empirical_percentile is not None
    assert abs(diag_med.empirical_percentile - 0.50) < 0.05

    # Check 10th percentile: r = sqrt(2 * ln(10)) ~= 2.146 has theoretical survival 0.10
    r_10 = float((2.0 * torch.log(torch.tensor(10.0))).sqrt())
    x_10 = torch.tensor([r_10, 0.0])
    diag_10 = stats.diagnose(x_10, threshold=0.01)
    assert diag_10.empirical_percentile is not None
    assert abs(diag_10.empirical_percentile - 0.10) < 0.03


def test_opt_in_warning_set_default_x():
    """Verify opt-in warning behavior in posterior.set_default_x()."""
    prior = BoxUniform(low=-2.0 * torch.ones(2), high=2.0 * torch.ones(2))
    theta = prior.sample((100,))
    x = theta + 0.1 * torch.randn_like(theta)

    trainer = NPE(prior=prior)
    trainer.append_simulations(theta, x).train(max_num_epochs=2)
    posterior = trainer.build_posterior()

    x_in = torch.zeros(2)
    x_ood = torch.tensor([50.0, 50.0])

    # 1. warn_if_ood=False (default): no warning even for OOD
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        posterior.set_default_x(x_ood, warn_if_ood=False)
    ood_warnings = [
        w
        for w in record
        if issubclass(w.category, UserWarning) and "extreme tails" in str(w.message)
    ]
    assert len(ood_warnings) == 0

    # 2. warn_if_ood=True for in-distribution: no warning
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        posterior.set_default_x(x_in, warn_if_ood=True)
    ood_warnings = [
        w
        for w in record
        if issubclass(w.category, UserWarning) and "extreme tails" in str(w.message)
    ]
    assert len(ood_warnings) == 0

    # 3. warn_if_ood=True for OOD: emits UserWarning
    with pytest.warns(UserWarning, match="falls in the extreme tails"):
        posterior.set_default_x(x_ood, warn_if_ood=True)

    # 4. Method chaining works
    post = posterior.set_default_x(x_in)
    assert post is posterior
    assert torch.allclose(posterior.default_x, x_in)


def test_numerical_robustness_singular_and_d_gt_n():
    """Verify spectral fallback when covariance is singular or D > N."""
    # Case 1: Collinear features (feature 1 is exactly 2 * feature 0)
    x_base = torch.randn(100, 1)
    x_singular = torch.cat([x_base, 2.0 * x_base], dim=1)
    stats_singular = ReferenceObservationStats.from_simulations(x_singular)

    assert stats_singular.cholesky_factor is None
    assert stats_singular.eigenvectors is not None

    # Point consistent with subspace
    x_consistent = torch.tensor([1.0, 2.0])
    diag_c = stats_singular.diagnose(x_consistent, threshold=0.01)
    assert diag_c.is_in_distribution is True

    # Point orthogonal to subspace (violates collinearity)
    x_orthogonal = torch.tensor([1.0, -2.0])
    diag_ortho = stats_singular.diagnose(x_orthogonal, threshold=0.01)
    # Detected via subspace residual
    assert diag_ortho.details.get("out_of_subspace", False) is True

    # Case 2: D > N (10 simulations in 20 dimensions)
    x_highdim = torch.randn(10, 20)
    stats_highdim = ReferenceObservationStats.from_simulations(x_highdim)
    assert stats_highdim.eigenvectors is not None
    diag_hd = stats_highdim.diagnose(torch.zeros(20))
    assert diag_hd.is_in_distribution is True


def test_constant_features():
    """Verify strict OOD handling when simulator produces constant features."""
    x = torch.randn(200, 3)
    # Feature index 1 is constant across all simulations
    x[:, 1] = 42.0

    stats = ReferenceObservationStats.from_simulations(x)
    assert 1 in stats.constant_indices.tolist()

    # Observation with matching constant feature
    x_match = torch.tensor([0.0, 42.0, 0.0])
    diag_match = stats.diagnose(x_match)
    assert diag_match.is_in_distribution is True
    assert diag_match.warning is None

    # Observation with deviating constant feature
    x_violate = torch.tensor([0.0, 42.5, 0.0])
    diag_violate = stats.diagnose(x_violate)
    assert diag_violate.is_in_distribution is False
    assert diag_violate.empirical_percentile == 0.0
    assert diag_violate.mahalanobis_distance == float("inf")
    assert "violates one or more constant features" in diag_violate.warning


def test_invalid_inputs_and_edge_cases():
    """Verify input validation for NaNs, Infs, shape mismatches, and N < 2."""
    x_ref = torch.randn(50, 2)
    stats = ReferenceObservationStats.from_simulations(x_ref)

    # 1. NaN in observation
    with pytest.raises(ValueError, match="NaN"):
        stats.diagnose(torch.tensor([float("nan"), 1.0]))

    # 2. Inf in observation
    with pytest.raises(ValueError, match="Inf"):
        stats.diagnose(torch.tensor([float("inf"), 1.0]))

    # 3. Dimension mismatch
    with pytest.raises(ValueError, match="Dimension mismatch"):
        stats.diagnose(torch.tensor([1.0, 2.0, 3.0]))

    # 4. Insufficient reference data (N = 1)
    x_small = torch.randn(1, 2)
    diag_small = diagnose_observation(x_o=torch.zeros(2), x_ref=x_small)
    assert diag_small.is_in_distribution is None
    assert diag_small.empirical_percentile is None
    assert "insufficient" in diag_small.warning.lower()


def test_batched_observations():
    """Verify batched observation diagnostics."""
    x_ref = torch.randn(500, 2)
    stats = ReferenceObservationStats.from_simulations(x_ref)

    # Batch of 3 observations: one central, one medium, one extreme
    x_batch = torch.tensor([
        [0.0, 0.0],
        [2.0, 0.0],
        [20.0, 20.0],
    ])
    diag = stats.diagnose(x_batch, threshold=0.01)

    assert diag.is_in_distribution is False  # Batch has at least one OOD
    assert diag.warning is not None
    assert "batch" in diag.warning.lower()
    assert "batch_percentiles" in diag.details
    assert len(diag.details["batch_percentiles"]) == 3
    assert diag.details["batch_percentiles"][0] > 0.05
    assert diag.details["batch_percentiles"][2] == 0.0


def test_multidimensional_and_image_observations():
    """Verify multi-dimensional event shapes (e.g. image-like observations)."""
    # 200 samples of 1x4x4 images
    x_ref = torch.randn(200, 1, 4, 4)
    stats = ReferenceObservationStats.from_simulations(x_ref)

    assert stats.event_shape == torch.Size([1, 4, 4])

    x_o = torch.zeros(1, 4, 4)
    diag = stats.diagnose(x_o)
    assert diag.is_in_distribution is True
    assert diag.mahalanobis_distance is not None


def test_embedding_network_space():
    """Verify diagnostic evaluation in learned embedding space."""
    x_ref = torch.randn(200, 6)
    embedding_net = nn.Sequential(
        nn.Linear(6, 3),
        nn.Tanh(),
    )

    # Diagnose in embedded space
    diag = diagnose_observation(
        x_o=torch.zeros(6),
        x_ref=x_ref,
        embedding_net=embedding_net,
        space="embedded",
    )
    assert diag.space == "embedded"
    assert diag.is_in_distribution is True
    assert diag.details["embedding_dim"] == 3


def test_ensemble_posterior_diagnostics():
    """Verify EnsemblePosterior diagnostic delegation and warning deduplication."""
    prior = BoxUniform(low=-2.0 * torch.ones(2), high=2.0 * torch.ones(2))
    theta = prior.sample((100,))
    x = theta + 0.1 * torch.randn_like(theta)

    trainer1 = NPE(prior=prior)
    trainer1.append_simulations(theta, x).train(max_num_epochs=1)
    post1 = trainer1.build_posterior()

    trainer2 = NPE(prior=prior)
    trainer2.append_simulations(theta, x).train(max_num_epochs=1)
    post2 = trainer2.build_posterior()

    ensemble = EnsemblePosterior(posteriors=[post1, post2])

    # Diagnose delegates to component posteriors
    diag = ensemble.diagnose_observation(torch.zeros(2))
    assert diag.is_in_distribution is True

    # Check warning is issued at most ONCE (not twice for 2 ensemble members)
    x_ood = torch.tensor([50.0, 50.0])
    with pytest.warns(UserWarning) as record:
        ensemble.set_default_x(x_ood, warn_if_ood=True)

    ood_warnings = [
        w
        for w in record
        if issubclass(w.category, UserWarning) and "extreme tails" in str(w.message)
    ]
    assert len(ood_warnings) == 1


def test_posterior_save_load_preserves_diagnostics():
    """Verify that reference statistics survive serialization (pickle)."""
    prior = BoxUniform(low=-2.0 * torch.ones(2), high=2.0 * torch.ones(2))
    theta = prior.sample((100,))
    x = theta + 0.1 * torch.randn_like(theta)

    trainer = NPE(prior=prior)
    trainer.append_simulations(theta, x).train(max_num_epochs=1)
    posterior = trainer.build_posterior()

    diag_before = posterior.diagnose_observation(torch.zeros(2))

    # Pickle dump and reload
    serialized = pickle.dumps(posterior)
    reloaded = pickle.loads(serialized)

    diag_after = reloaded.diagnose_observation(torch.zeros(2))

    assert diag_before.is_in_distribution == diag_after.is_in_distribution
    assert (
        abs(diag_before.mahalanobis_distance - diag_after.mahalanobis_distance) < 1e-5
    )
    assert (
        abs(diag_before.empirical_percentile - diag_after.empirical_percentile) < 1e-5
    )


def test_unattached_reference_data_behavior():
    """Verify that a posterior without reference data reports indeterminate cleanly."""
    prior = BoxUniform(low=-2.0 * torch.ones(2), high=2.0 * torch.ones(2))
    theta = prior.sample((50,))
    x = theta + 0.1 * torch.randn_like(theta)

    trainer = NPE(prior=prior)
    trainer.append_simulations(theta, x).train(max_num_epochs=1)
    posterior = trainer.build_posterior()

    # Manually clear reference stats to simulate legacy / unattached posterior
    posterior._reference_statistics = None
    posterior._reference_simulations = None

    diag = posterior.diagnose_observation(torch.zeros(2))
    assert diag.is_in_distribution is None
    assert diag.empirical_percentile is None
    assert diag.mahalanobis_distance is None
    assert diag.warning is not None
    assert "No reference simulations" in diag.warning


def test_posterior_types_compatibility():
    """Verify diagnostic on MCMC, Rejection, and VI posteriors."""
    prior = BoxUniform(low=-2.0 * torch.ones(2), high=2.0 * torch.ones(2))
    theta = prior.sample((60,))
    x = theta + 0.1 * torch.randn_like(theta)

    trainer = NPE(prior=prior)
    trainer.append_simulations(theta, x).train(max_num_epochs=1)

    # Rejection
    post_rej = trainer.build_posterior(sample_with="rejection")
    diag_rej = post_rej.diagnose_observation(torch.zeros(2))
    assert diag_rej.is_in_distribution is True

    # MCMC
    post_mcmc = trainer.build_posterior(sample_with="mcmc")
    diag_mcmc = post_mcmc.diagnose_observation(torch.zeros(2))
    assert diag_mcmc.is_in_distribution is True

    # VI
    post_vi = trainer.build_posterior(sample_with="vi")
    diag_vi = post_vi.diagnose_observation(torch.zeros(2))
    assert diag_vi.is_in_distribution is True


def test_deterministic_reproducibility():
    """Verify identical results across repeated calls for same inputs."""
    x_ref = torch.randn(300, 3)
    stats = ReferenceObservationStats.from_simulations(x_ref)

    x_query = torch.tensor([1.2, -0.5, 0.8])
    diag1 = stats.diagnose(x_query)
    diag2 = stats.diagnose(x_query)

    assert diag1.is_in_distribution == diag2.is_in_distribution
    assert diag1.empirical_percentile == diag2.empirical_percentile
    assert diag1.mahalanobis_distance == diag2.mahalanobis_distance
    assert diag1.warning == diag2.warning


def test_tied_scores_and_small_sample():
    """Verify handling of tied scores and fractional ranking."""
    # Several identical points to produce exact ties
    x_ref = torch.cat(
        [
            torch.zeros(10, 2),
            torch.ones(10, 2),
        ],
        dim=0,
    )

    stats = ReferenceObservationStats.from_simulations(x_ref)
    diag = stats.diagnose(torch.zeros(2))
    assert diag.empirical_percentile is not None
    # 0.5 * (10 ties) / 20 points = at least 0.25
    assert diag.empirical_percentile >= 0.25


def test_exact_leave_one_out_formula():
    """Verify that the analytical leave-one-out scaling matches brute force."""
    torch.manual_seed(42)
    n, d = 30, 3
    x = torch.randn(n, d)

    # Brute-force leave-one-out distances
    brute_loo_dists = []
    for i in range(n):
        mask = torch.ones(n, dtype=torch.bool)
        mask[i] = False
        x_minus_i = x[mask]
        mean_minus_i = x_minus_i.mean(dim=0)
        cov_minus_i = ((x_minus_i - mean_minus_i).t() @ (x_minus_i - mean_minus_i)) / (
            n - 2
        )
        inv_cov_minus_i = torch.linalg.inv(cov_minus_i)
        diff = x[i] - mean_minus_i
        dist_sq = diff @ inv_cov_minus_i @ diff
        brute_loo_dists.append(torch.sqrt(dist_sq).item())

    sorted_brute_loo = torch.tensor(sorted(brute_loo_dists))

    # ReferenceObservationStats LOO calculation
    stats = ReferenceObservationStats.from_simulations(x)
    ref_dists = stats.reference_distances

    assert torch.allclose(ref_dists, sorted_brute_loo, rtol=1e-4, atol=1e-4)


def test_multiround_sequential_inference():
    """Verify reference simulation tracking across multiple sequential rounds."""
    prior = BoxUniform(low=-2.0 * torch.ones(2), high=2.0 * torch.ones(2))
    trainer = NPE(prior=prior)

    # Round 1
    theta1 = prior.sample((40,))
    x1 = theta1 + 0.1 * torch.randn_like(theta1)
    trainer.append_simulations(theta1, x1).train(max_num_epochs=1)

    # Round 2
    theta2 = prior.sample((60,))
    x2 = theta2 + 0.1 * torch.randn_like(theta2)
    trainer.append_simulations(theta2, x2).train(
        max_num_epochs=1, force_first_round_loss=True
    )

    posterior = trainer.build_posterior()

    assert posterior.has_reference_statistics
    assert posterior.reference_statistics.n_simulations == 100

    diag = posterior.diagnose_observation(torch.zeros(2))
    assert diag.is_in_distribution is True
    assert diag.details["n_reference"] == 100


def test_nan_padded_trials_ood_diagnostic_ensemble_and_direct():
    """Verify OOD diagnostic with NaN-padded trials for direct and ensemble
    posteriors.
    """
    num_dim, max_trials, num_sims = 2, 3, 30
    prior = BoxUniform(torch.zeros(num_dim), torch.ones(num_dim))
    theta = prior.sample((num_sims,))
    num_trials = torch.randint(1, max_trials + 1, size=(num_sims,))
    x = torch.full((num_sims, max_trials, num_dim), float("nan"))
    for i in range(num_sims):
        x[i, : num_trials[i]] = theta[i] + 0.1 * torch.randn(num_trials[i], num_dim)

    embedding_net = PermutationInvariantEmbedding(
        trial_net=FCEmbedding(input_dim=num_dim, output_dim=4),
        trial_net_output_dim=4,
        output_dim=4,
    )
    density_estimator = posterior_nn(
        model="mdn", embedding_net=embedding_net, z_score_x="none"
    )
    inference = NPE(
        prior=prior, density_estimator=density_estimator, show_progress_bars=False
    )
    inference.append_simulations(theta, x, exclude_invalid_x=False).train(
        max_num_epochs=1, training_batch_size=30
    )

    # 1. Non-ensemble (direct) case
    posterior = inference.build_posterior()
    assert posterior.has_reference_statistics
    assert posterior.reference_statistics is not None
    assert getattr(posterior.reference_statistics, "is_embedded", False) is True
    assert posterior.reference_statistics.n_simulations == num_sims

    # Valid observation with 1 real trial and 2 NaN-padded trials
    x_valid = torch.full((1, max_trials, num_dim), float("nan"))
    x_valid[0, :1] = 0.5
    diag = posterior.diagnose_observation(x_valid)
    assert diag.is_in_distribution is not None
    assert diag.space == "embedded"
    assert diag.details["space"] == "embedded"
    assert diag.details["n_reference"] == num_sims
    assert "embedding_dim" in diag.details

    # 2. Ensemble case
    posterior2 = inference.build_posterior()
    ensemble = EnsemblePosterior([posterior, posterior2])
    assert ensemble.has_reference_statistics
    diag_ens = ensemble.diagnose_observation(x_valid)
    assert diag_ens.is_in_distribution is not None
    assert diag_ens.space == "embedded"
    assert diag_ens.details["space"] == "embedded"
    assert diag_ens.details["n_reference"] == num_sims


def test_nan_padded_trials_invalid_padding_rejected():
    """Verify that invalid NaN padding and Inf values are strictly rejected."""
    num_dim, max_trials = 2, 3
    embedding_net = PermutationInvariantEmbedding(
        trial_net=FCEmbedding(input_dim=num_dim, output_dim=4),
        trial_net_output_dim=4,
        output_dim=4,
    )
    x_sims = torch.full((20, max_trials, num_dim), float("nan"))
    for i in range(20):
        x_sims[i, :2] = torch.randn(2, num_dim)

    stats = ReferenceObservationStats.from_simulations(
        x_sims, embedding_net=embedding_net
    )
    assert stats.is_embedded is True

    # Partially NaN trial (a trial has some NaN coordinates and some finite)
    x_partial_nan = torch.full((1, max_trials, num_dim), float("nan"))
    x_partial_nan[0, 0, 0] = 0.5
    with pytest.raises(ValueError, match="NaN may only mark padded trials"):
        diagnose_observation(x_partial_nan, reference_stats=stats)

    with pytest.raises(ValueError, match="NaN may only mark padded trials"):
        stats.diagnose(x_partial_nan)

    # Fully padded (no real trials at all)
    x_all_nan = torch.full((1, max_trials, num_dim), float("nan"))
    with pytest.raises(ValueError, match="NaN may only mark padded trials"):
        diagnose_observation(x_all_nan, reference_stats=stats)

    with pytest.raises(ValueError, match="NaN may only mark padded trials"):
        stats.diagnose(x_all_nan)

    # Inf in observation
    x_inf = torch.full((1, max_trials, num_dim), float("nan"))
    x_inf[0, 0] = float("inf")
    with pytest.raises(ValueError, match="NaN/Inf"):
        diagnose_observation(x_inf, reference_stats=stats)

    with pytest.raises(ValueError, match="NaN/Inf"):
        stats.diagnose(x_inf)


def test_finite_reference_simulations_preserve_standard_behavior():
    """Verify that standard finite simulations retain standard observation space
    behavior.
    """
    x_finite = torch.randn(50, 4)
    stats = ReferenceObservationStats.from_simulations(x_finite)
    assert not getattr(stats, "is_embedded", False)
    assert stats.n_simulations == 50
    assert stats.event_shape == torch.Size([4])

    x_obs = torch.randn(4)
    diag = stats.diagnose(x_obs)
    assert diag.space == "observation"
    assert diag.details.get("space") != "embedded"

    # NaN in observation without NaN-tolerant net must be rejected
    x_nan = torch.tensor([1.0, float("nan"), 0.5, 0.2])
    with pytest.raises(ValueError):
        stats.diagnose(x_nan)

    with pytest.raises(ValueError):
        diagnose_observation(x_nan, reference_stats=stats)


def test_parameterless_embedding_net():
    """Verify that diagnose_observation works with parameterless embedding net."""
    identity_net = nn.Identity()
    x_ref = torch.randn(50, 4)
    x_o = torch.randn(4)
    diag = diagnose_observation(
        x_o, x_ref=x_ref, space="embedded", embedding_net=identity_net
    )
    assert diag.is_in_distribution is not None
    assert diag.space == "embedded"


def test_posterior_diagnose_embedded_space_with_dimension_reduction():
    """Verify that posterior.diagnose_observation(space='embedded') computes
    embedded reference statistics and does not fail with dimension mismatch.
    """
    prior = BoxUniform(low=-2.0 * torch.ones(2), high=2.0 * torch.ones(2))
    theta = prior.sample((80,))
    x = torch.randn(80, 6)

    # Embedding network that reduces dimension 6 -> 3
    emb = nn.Sequential(nn.Linear(6, 3), nn.Tanh())
    de = posterior_nn("maf", embedding_net=emb)
    trainer = NPE(prior=prior, density_estimator=de)
    trainer.append_simulations(theta, x).train(max_num_epochs=1)
    posterior = trainer.build_posterior()

    # Diagnosing with space="embedded" should project simulations and observation
    # into dimension 3 and diagnose successfully without dimension mismatch.
    x_o = torch.randn(6)
    diag = posterior.diagnose_observation(x_o, space="embedded")
    assert diag.space == "embedded"
    assert diag.is_in_distribution is not None
    assert diag.details["embedding_dim"] == 3


def test_lazy_reference_statistics_construction():
    """Verify that build_posterior() does not eagerly construct reference statistics,
    and constructs them on demand when diagnose_observation is called.
    """
    prior = BoxUniform(low=-2.0 * torch.ones(2), high=2.0 * torch.ones(2))
    theta = prior.sample((40,))
    x = torch.randn(40, 4)

    trainer = NPE(prior=prior)
    trainer.append_simulations(theta, x).train(max_num_epochs=1)
    posterior = trainer.build_posterior()

    # Before calling diagnose, _reference_statistics must be None (lazy evaluation)
    assert posterior._reference_statistics is None
    assert posterior.has_reference_statistics is True

    # Upon calling diagnose_observation, reference statistics are built and cached
    diag = posterior.diagnose_observation(torch.zeros(4))
    assert diag.is_in_distribution is not None
    assert posterior._reference_statistics is not None


def test_likelihood_and_ratio_trainers_do_not_attach_theta_embedding():
    """Verify that NLE and NRE do not pass theta embedding net to observation stats."""
    from sbi.inference import NLE, NRE
    from sbi.neural_nets import likelihood_nn

    prior = BoxUniform(low=-2.0 * torch.ones(2), high=2.0 * torch.ones(2))
    theta = prior.sample((40,))
    x = torch.randn(40, 4)

    # NLE: likelihood models p(x|theta), so embedding_net is on theta (dim 2)
    theta_emb = nn.Sequential(nn.Linear(2, 5), nn.Tanh())
    le = likelihood_nn("maf", embedding_net=theta_emb)
    trainer_nle = NLE(prior=prior, density_estimator=le)
    trainer_nle.append_simulations(theta, x).train(max_num_epochs=1)
    post_nle = trainer_nle.build_posterior()

    # Posterior's observation reference data must NOT use theta embedding net
    assert post_nle._reference_embedding_net is None
    diag = post_nle.diagnose_observation(torch.zeros(4))
    assert diag.is_in_distribution is not None
    assert diag.space == "observation"

    # NRE: ratio estimator
    trainer_nre = NRE(prior=prior)
    trainer_nre.append_simulations(theta, x).train(max_num_epochs=1)
    post_nre = trainer_nre.build_posterior()
    diag_nre = post_nre.diagnose_observation(torch.zeros(4))
    assert diag_nre.is_in_distribution is not None


def test_low_variance_dimension_preservation():
    """Verify that features with wide dynamic range of variances (e.g. 1.0 and 1e-8)
    are preserved via standardized decomposition, and anomalies are reliably detected.
    """
    torch.manual_seed(42)
    n = 5000
    f0 = torch.randn(n, 1)
    f1 = 1e-4 * torch.randn(n, 1)  # var = 1e-8
    x_sims = torch.cat([f0, f1], dim=1)

    stats = ReferenceObservationStats.from_simulations(x_sims)
    # Correlation-based Cholesky should succeed and not be discarded
    assert stats.cholesky_factor is not None

    # Centered at (1, 0.01): 1-sigma on feature 0 and 100-sigma on feature 1
    x_ood = torch.tensor([[1.0, 0.01]])
    diag = stats.diagnose(x_ood)
    assert diag.is_in_distribution is False
    assert diag.empirical_percentile == 0.0
    assert diag.mahalanobis_distance > 50.0
