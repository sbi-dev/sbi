# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

"""Smoke tests for the azula sampler benchmark adapter classes."""

import pytest
import torch

from sbi.inference import NPSE

from .mini_sbibm import get_task

try:
    from azula.sample import DDIMSampler, DDPMSampler, EulerSampler, HeunSampler

    AZULA_AVAILABLE = True
except ImportError:
    AZULA_AVAILABLE = False


# Import adapter classes from the benchmark script.
if AZULA_AVAILABLE:
    import importlib
    from pathlib import Path

    _bench_dir = Path(__file__).resolve().parent.parent / "benchmarks"
    _spec = importlib.util.spec_from_file_location(
        "azula_sampler_benchmark",
        _bench_dir / "azula_sampler_benchmark.py",
    )
    _mod = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)
    SBISchedule = _mod.SBISchedule
    SBIDenoiser = _mod.SBIDenoiser
    sample_with_azula = _mod.sample_with_azula
    validate_schedule_alignment = _mod.validate_schedule_alignment


@pytest.mark.skipif(not AZULA_AVAILABLE, reason="azula not installed")
@pytest.mark.slow
class TestAzulaBenchmarkAdapters:
    """Smoke tests verifying the azula adapter classes produce valid output."""

    @pytest.fixture(scope="class")
    def trained_estimator(self):
        """Train a minimal NPSE for testing (few sims, few epochs)."""
        torch.manual_seed(0)
        task = get_task("two_moons")
        prior = task.get_prior()
        theta, x = task.get_data(500)

        inference = NPSE(prior, sde_type="vp")
        inference.append_simulations(theta, x)
        score_estimator = inference.train(max_num_epochs=5, show_train_summary=False)
        posterior = inference.build_posterior(score_estimator, sample_with="sde")
        x_o = task.get_observation(1)
        return score_estimator, posterior, x_o

    def test_schedule_alignment(self, trained_estimator):
        """SBISchedule must match sbi's internal noise schedule exactly."""
        score_estimator, _, _ = trained_estimator
        # Should not raise.
        validate_schedule_alignment(score_estimator)

    def test_denoiser_output_shape(self, trained_estimator):
        """SBIDenoiser must return correctly shaped DiracPosterior."""
        score_estimator, _, x_o = trained_estimator
        denoiser = SBIDenoiser(score_estimator, x_o)

        batch_size = 16
        theta_t = torch.randn(batch_size, *score_estimator.input_shape)
        t = torch.tensor(0.5)

        result = denoiser(theta_t, t)
        assert result.mean.shape == (batch_size, *score_estimator.input_shape)
        assert torch.isfinite(result.mean).all()

    @pytest.mark.parametrize(
        "sampler_cls", [DDIMSampler, DDPMSampler, EulerSampler, HeunSampler]
    )
    def test_azula_sampler_produces_finite_samples(
        self, trained_estimator, sampler_cls
    ):
        """Each azula sampler must produce finite, correctly shaped samples."""
        score_estimator, _, x_o = trained_estimator
        denoiser = SBIDenoiser(score_estimator, x_o)

        num_samples, num_steps = 32, 5
        kwargs = {"eta": 0.0} if sampler_cls is DDIMSampler else {}
        samples = sample_with_azula(
            denoiser, sampler_cls, num_samples, num_steps, kwargs
        )

        assert samples.shape == (num_samples, *score_estimator.input_shape)
        assert torch.isfinite(samples).all()
