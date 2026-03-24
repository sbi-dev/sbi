# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

"""Benchmark comparing azula diffusion samplers against sbi's default Euler-Maruyama.

This script trains NPSE on a mini-sbibm task (e.g., two_moons) and then compares
sampling performance between sbi's built-in predictor/corrector sampler and azula's
DDIM, DDPM, and Heun samplers via a score-to-denoiser adapter (SBIDenoiser).

The adapter uses Tweedie's formula to convert sbi's score function output to azula's
expected denoiser output. A custom SBISchedule handles the noise schedule mismatch
between sbi's VP-SDE parameterization (beta_min/beta_max) and azula's VPSchedule
(alpha_min), which use incompatible formulas for alpha(t).

Usage:
    python benchmarks/azula_sampler_benchmark.py
    python benchmarks/azula_sampler_benchmark.py --task two_moons --sde-types ve vp
    python benchmarks/azula_sampler_benchmark.py --steps 50 100 200 --seeds 0 1 2

Related to: https://github.com/sbi-dev/sbi/issues/1468
"""

import argparse
import sys
import time
from typing import Any, Optional

import numpy as np
import torch
from torch import Tensor

try:
    from azula.denoise import Denoiser, DiracPosterior
    from azula.noise import Schedule, VPSchedule
    from azula.sample import DDIMSampler, DDPMSampler, EulerSampler, HeunSampler
except ImportError as err:
    raise ImportError(
        "azula is required for this benchmark. Install with: pip install azula"
    ) from err

from sbi.inference import NPSE
from sbi.neural_nets.estimators.score_estimator import ConditionalScoreEstimator
from sbi.utils.metrics import c2st, unbiased_mmd_squared

# Append tests/ to path so mini_sbibm is importable.
_tests_dir = str(__import__("pathlib").Path(__file__).resolve().parent.parent / "tests")
sys.path.insert(0, _tests_dir)
from mini_sbibm import get_task  # noqa: E402

# ---------------------------------------------------------------------------
# Adapter classes: bridging sbi's score interface to azula's denoiser protocol
# ---------------------------------------------------------------------------


class SBISchedule(Schedule):
    """Noise schedule adapter that delegates to sbi's score estimator internals.

    Azula's samplers call ``denoiser.schedule(t)`` at every step to retrieve the
    signal scale alpha_t and noise scale sigma_t. Rather than using azula's built-in
    VPSchedule — which computes ``alpha(t) = exp(t^2 * log(alpha_min))`` — this adapter
    delegates to sbi's ``mean_t_fn(t)`` and ``std_fn(t)``, ensuring the schedule
    exactly matches what the score estimator was trained with.

    This is critical for VP-SDE, where sbi uses
    ``alpha(t) = exp(-0.25 * t^2 * (beta_max - beta_min) - 0.5 * t * beta_min)``
    which is mathematically incompatible with azula's one-parameter formula.

    Note:
        We use ``mean_t_fn`` (the raw signal scaling factor) rather than
        ``approx_marginal_mean`` (which multiplies by ``mean_0``). After
        standardization ``mean_0 ~ 0``, so ``approx_marginal_mean`` would collapse
        to zero, breaking the Tweedie formula's division by alpha_t.
    """

    def __init__(self, score_estimator: ConditionalScoreEstimator) -> None:
        self.estimator = score_estimator
        self._input_ndim = len(score_estimator.input_shape)

    def __call__(self, t: Tensor) -> tuple[Tensor, Tensor]:
        alpha_t = self.estimator.mean_t_fn(t)
        sigma_t = self.estimator.std_fn(t)
        # sbi's mean_t_fn/std_fn append len(input_shape) trailing singleton dims
        # for broadcasting with input tensors. Squeeze them for azula's protocol,
        # which expects (alpha, sigma) to have the same shape as t.
        for _ in range(self._input_ndim):
            alpha_t = alpha_t.squeeze(-1)
            sigma_t = sigma_t.squeeze(-1)
        return alpha_t, sigma_t


class SBIDenoiser(Denoiser):
    """Adapter converting sbi's conditional score estimator to azula's denoiser.

    sbi's score estimator outputs the score function:
        s(theta_t, x, t) = nabla_theta log p_t(theta_t | x)

    Azula's samplers expect a denoiser returning a predicted clean sample:
        D(theta_t, t) -> DiracPosterior(mean=theta_hat_0)

    The conversion uses Tweedie's formula for the perturbation kernel
    p(theta_t | theta_0) = N(alpha_t * theta_0, sigma_t^2 * I):

        theta_hat_0 = (theta_t + sigma_t^2 * score(theta_t, x, t)) / alpha_t

    Args:
        score_estimator: Trained sbi ConditionalScoreEstimator.
        x_o: Conditioning observation, shape ``(x_dim,)`` or ``(1, x_dim)``.
        schedule: Noise schedule for azula's samplers. If None, an SBISchedule
            wrapping the score estimator is created automatically.
    """

    def __init__(
        self,
        score_estimator: ConditionalScoreEstimator,
        x_o: Tensor,
        schedule: Optional[Schedule] = None,
    ) -> None:
        super().__init__()
        self.score_estimator = score_estimator
        self.x_o = x_o if x_o.ndim >= 2 else x_o.unsqueeze(0)
        self.schedule: Schedule = schedule or SBISchedule(score_estimator)

    def forward(self, theta_t: Tensor, t: Tensor, **kwargs: Any) -> DiracPosterior:
        """Predict clean theta_0 from noisy theta_t at diffusion time t.

        Args:
            theta_t: Noisy parameter samples, shape ``(B, *event_shape)``.
            t: Diffusion time, shape ``()`` or ``(B,)`` (azula convention).

        Returns:
            DiracPosterior with mean = predicted clean sample theta_hat_0.
        """
        alpha_t, sigma_t = self.schedule(t)

        # Expand scalar schedule values for broadcasting with (B, D) input.
        while alpha_t.ndim < theta_t.ndim:
            alpha_t = alpha_t.unsqueeze(-1)
            sigma_t = sigma_t.unsqueeze(-1)

        # Expand conditioning observation to batch size.
        x_expanded = self.x_o.expand(theta_t.shape[0], -1)

        # Evaluate sbi's conditional score. Scalar t broadcasts to batch internally.
        score = self.score_estimator.score(theta_t, x_expanded, t)

        # Tweedie's formula: theta_hat_0 = (theta_t + sigma_t^2 * score) / alpha_t
        theta_0_hat = (theta_t + sigma_t**2 * score) / alpha_t

        return DiracPosterior(mean=theta_0_hat)


# ---------------------------------------------------------------------------
# Sampling helpers
# ---------------------------------------------------------------------------


def sample_with_azula(
    denoiser: SBIDenoiser,
    sampler_cls: type,
    num_samples: int,
    num_steps: int,
    sampler_kwargs: Optional[dict] = None,
) -> Tensor:
    """Draw posterior samples using an azula sampler through the SBIDenoiser adapter.

    Args:
        denoiser: SBIDenoiser wrapping a trained score estimator.
        sampler_cls: Azula sampler class (e.g., DDIMSampler).
        num_samples: Number of samples to generate.
        num_steps: Number of discretization steps.
        sampler_kwargs: Extra keyword arguments for the sampler constructor.

    Returns:
        Tensor of posterior samples, shape ``(num_samples, theta_dim)``.
    """
    est = denoiser.score_estimator
    device = est._mean_base.device

    sampler = sampler_cls(
        denoiser,
        start=float(est.t_max),
        stop=float(est.t_min),
        steps=num_steps,
        silent=True,
        **(sampler_kwargs or {}),
    )

    # Initialize from the base distribution, matching sbi's Diffuser.initialize().
    # mean_base and std_base are computed at t_max during estimator construction.
    init_mean = est.mean_base
    init_std = est.std_base
    eps = torch.randn(num_samples, *est.input_shape, device=device)
    mean, std, eps = torch.broadcast_tensors(init_mean, init_std, eps)
    theta_T = mean + std * eps

    return sampler(theta_T)


def sample_with_sbi(
    posterior: Any,
    x_o: Tensor,
    num_samples: int,
    num_steps: int,
) -> Tensor:
    """Draw posterior samples using sbi's default Euler-Maruyama sampler.

    Args:
        posterior: sbi VectorFieldPosterior object.
        x_o: Conditioning observation.
        num_samples: Number of samples to generate.
        num_steps: Number of diffusion steps.

    Returns:
        Tensor of posterior samples, shape ``(num_samples, theta_dim)``.
    """
    # Disable prior rejection to match azula sampling (which bypasses it).
    # This ensures a fair comparison of sampler quality alone.
    return posterior.sample(
        (num_samples,),
        x=x_o,
        steps=num_steps,
        show_progress_bars=False,
        reject_outside_prior=False,
    ).squeeze(1)  # Remove batch dim from single observation


# ---------------------------------------------------------------------------
# Schedule validation
# ---------------------------------------------------------------------------


def validate_schedule_alignment(
    score_estimator: ConditionalScoreEstimator,
) -> None:
    """Numerically verify that SBISchedule matches sbi's internal noise schedule.

    Also demonstrates the mismatch with azula's built-in VPSchedule for VP-SDE,
    which uses an incompatible formula: alpha(t) = exp(t^2 * log(alpha_min)).

    Args:
        score_estimator: Trained sbi score estimator.
    """
    sbi_schedule = SBISchedule(score_estimator)
    t_min, t_max = score_estimator.t_min, score_estimator.t_max
    times = torch.linspace(t_min, t_max, 50, device=score_estimator._mean_base.device)

    max_alpha_diff = 0.0
    max_sigma_diff = 0.0

    for t_val in times:
        alpha_sbi, sigma_sbi = sbi_schedule(t_val)
        alpha_direct = score_estimator.mean_t_fn(t_val)
        sigma_direct = score_estimator.std_fn(t_val)
        # Squeeze the trailing dims added by mean_t_fn/std_fn.
        for _ in range(len(score_estimator.input_shape)):
            alpha_direct = alpha_direct.squeeze(-1)
            sigma_direct = sigma_direct.squeeze(-1)

        max_alpha_diff = max(max_alpha_diff, (alpha_sbi - alpha_direct).abs().item())
        max_sigma_diff = max(max_sigma_diff, (sigma_sbi - sigma_direct).abs().item())

    assert max_alpha_diff < 1e-7, f"Alpha mismatch: {max_alpha_diff}"
    assert max_sigma_diff < 1e-7, f"Sigma mismatch: {max_sigma_diff}"
    max_diff = max(max_alpha_diff, max_sigma_diff)
    print(f"  SBISchedule <-> sbi internals: max diff = {max_diff:.2e}")

    # For VP-SDE, show the mismatch with azula's built-in VPSchedule.
    estimator_type = type(score_estimator).__name__
    if "VP" in estimator_type and "VE" not in estimator_type:
        azula_vp = VPSchedule()
        alpha_sbi_at_1, _ = sbi_schedule(torch.tensor(1.0))
        alpha_azula_at_1, _ = azula_vp(torch.tensor(1.0))
        mismatch = (alpha_sbi_at_1 - alpha_azula_at_1).abs().item()
        print(
            f"  SBISchedule vs azula VPSchedule at t=1: |alpha_diff| = {mismatch:.4f}"
        )
        print(f"    sbi alpha(1)  = {alpha_sbi_at_1.item():.6f}")
        print(f"    azula alpha(1) = {alpha_azula_at_1.item():.6f}")
        print("    -> These differ because sbi uses beta_min/beta_max integration")
        print("       while azula uses alpha_min quadratic schedule.")


# ---------------------------------------------------------------------------
# Benchmark orchestration
# ---------------------------------------------------------------------------


def run_benchmark(
    task_name: str = "two_moons",
    sde_types: list[str] | None = None,
    num_sims: int = 10_000,
    num_samples: int = 1_000,
    step_counts: list[int] | None = None,
    seeds: list[int] | None = None,
    observation_indices: list[int] | None = None,
    device: str = "cpu",
) -> list[dict]:
    """Run the full azula vs sbi sampler benchmark.

    Args:
        task_name: Mini-sbibm task name.
        sde_types: SDE types to benchmark (e.g., ["ve", "vp"]).
        num_sims: Number of training simulations.
        num_samples: Number of posterior samples per evaluation.
        step_counts: List of discretization step counts to test.
        seeds: Random seeds for statistical averaging.
        observation_indices: Mini-sbibm observation indices for evaluation.
        device: Torch device string.

    Returns:
        List of result dicts, one per (sde_type, sampler, steps, seed, obs) config.
    """
    sde_types = sde_types or ["ve", "vp"]
    step_counts = step_counts or [50, 100, 200, 500]
    seeds = seeds or [0, 1, 2]
    observation_indices = observation_indices or [1, 2, 3]

    task = get_task(task_name)

    # Azula sampler configurations: (display_name, class, kwargs, nfe_per_step)
    azula_samplers: list[tuple[str, type, dict, int]] = [
        ("DDIM (eta=0)", DDIMSampler, {"eta": 0.0}, 1),
        ("DDPM", DDPMSampler, {}, 1),
        ("Euler (azula)", EulerSampler, {}, 1),
        ("Heun", HeunSampler, {}, 2),  # 2 denoiser calls per step
    ]

    all_results: list[dict] = []

    for sde_type in sde_types:
        print(f"\n{'=' * 70}")
        print(f"SDE type: {sde_type.upper()}")
        print(f"{'=' * 70}")

        # --- Train NPSE (shared across all samplers for this SDE type) ---
        print(f"\nTraining NPSE ({sde_type}) on {task_name} with {num_sims} sims...")
        torch.manual_seed(42)
        prior = task.get_prior()
        theta, x = task.get_data(num_sims)

        inference = NPSE(prior, sde_type=sde_type, device=device)
        inference.append_simulations(theta.to(device), x.to(device))
        score_estimator = inference.train(show_train_summary=False)
        posterior = inference.build_posterior(score_estimator, sample_with="sde")

        # --- Validate noise schedule alignment ---
        print("\nSchedule validation:")
        validate_schedule_alignment(score_estimator)

        # --- Load reference posterior samples from mini-sbibm ---
        ref_samples = {
            idx: task.get_reference_posterior_samples(idx)
            for idx in observation_indices
        }
        observations = {idx: task.get_observation(idx) for idx in observation_indices}

        # --- Warmup (avoid JIT overhead in timing) ---
        warmup_x = observations[observation_indices[0]].to(device)
        _ = posterior.sample((10,), x=warmup_x, steps=10, show_progress_bars=False)

        # --- Benchmark sbi's default Euler-Maruyama ---
        print("\nBenchmarking sbi Euler-Maruyama...")
        for steps in step_counts:
            for seed in seeds:
                torch.manual_seed(seed)
                np.random.seed(seed)
                for obs_idx in observation_indices:
                    x_o = observations[obs_idx].to(device)
                    ref = ref_samples[obs_idx][:num_samples]

                    t0 = time.perf_counter()
                    samples = sample_with_sbi(posterior, x_o, num_samples, steps)
                    elapsed = time.perf_counter() - t0

                    samples_cpu = samples.cpu()
                    c2st_val = c2st(ref, samples_cpu).item()
                    mmd_val = unbiased_mmd_squared(ref, samples_cpu).item()

                    all_results.append({
                        "sde_type": sde_type,
                        "sampler": "Euler-Maruyama (sbi)",
                        "steps": steps,
                        "nfe": steps,
                        "seed": seed,
                        "obs_idx": obs_idx,
                        "c2st": c2st_val,
                        "mmd": mmd_val,
                        "time_s": elapsed,
                    })

        # --- Benchmark azula samplers ---
        for sampler_name, sampler_cls, sampler_kwargs, nfe_mult in azula_samplers:
            print(f"Benchmarking {sampler_name}...")
            for steps in step_counts:
                for seed in seeds:
                    torch.manual_seed(seed)
                    np.random.seed(seed)
                    for obs_idx in observation_indices:
                        x_o = observations[obs_idx].to(device)
                        ref = ref_samples[obs_idx][:num_samples]

                        denoiser = SBIDenoiser(score_estimator, x_o)

                        t0 = time.perf_counter()
                        samples = sample_with_azula(
                            denoiser, sampler_cls, num_samples, steps, sampler_kwargs
                        )
                        elapsed = time.perf_counter() - t0

                        samples_cpu = samples.cpu()
                        c2st_val = c2st(ref, samples_cpu).item()
                        mmd_val = unbiased_mmd_squared(ref, samples_cpu).item()

                        all_results.append({
                            "sde_type": sde_type,
                            "sampler": sampler_name,
                            "steps": steps,
                            "nfe": steps * nfe_mult,
                            "seed": seed,
                            "obs_idx": obs_idx,
                            "c2st": c2st_val,
                            "mmd": mmd_val,
                            "time_s": elapsed,
                        })

    return all_results


# ---------------------------------------------------------------------------
# Results formatting
# ---------------------------------------------------------------------------


def format_results(results: list[dict]) -> str:
    """Aggregate results across seeds and observations, format as a table.

    Reports mean +/- std for C2ST and MMD, and mean wall-clock time.

    Args:
        results: List of per-run result dicts from run_benchmark.

    Returns:
        Formatted results table as a string.
    """
    # Group by (sde_type, sampler, steps)
    groups: dict[tuple, list[dict]] = {}
    for r in results:
        key = (r["sde_type"], r["sampler"], r["steps"], r["nfe"])
        groups.setdefault(key, []).append(r)

    header = (
        f"{'SDE':<5} | {'Sampler':<22} | {'Steps':>5} | {'NFE':>5} | "
        f"{'C2ST (mean +/- std)':>22} | {'MMD (mean +/- std)':>22} | {'Time (s)':>8}"
    )
    sep = "-" * len(header)

    lines = [sep, header, sep]

    current_sde = None
    for key in sorted(groups.keys()):
        sde_type, sampler, steps, nfe = key
        runs = groups[key]

        if sde_type != current_sde:
            if current_sde is not None:
                lines.append(sep)
            current_sde = sde_type

        c2st_vals = [r["c2st"] for r in runs]
        mmd_vals = [r["mmd"] for r in runs]
        time_vals = [r["time_s"] for r in runs]

        c2st_mean, c2st_std = np.mean(c2st_vals), np.std(c2st_vals)
        mmd_mean, mmd_std = np.mean(mmd_vals), np.std(mmd_vals)
        time_mean = np.mean(time_vals)

        lines.append(
            f"{sde_type:<5} | {sampler:<22} | {steps:>5} | {nfe:>5} | "
            f"{c2st_mean:>8.4f} +/- {c2st_std:<8.4f} | "
            f"{mmd_mean:>8.6f} +/- {mmd_std:<8.6f} | "
            f"{time_mean:>8.2f}"
        )

    lines.append(sep)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark azula diffusion samplers vs sbi's Euler-Maruyama.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--task",
        default="two_moons",
        choices=["two_moons", "slcp"],
        help="Mini-sbibm task to benchmark on.",
    )
    parser.add_argument(
        "--sde-types",
        nargs="+",
        default=["ve", "vp"],
        help="SDE types to train and evaluate.",
    )
    parser.add_argument(
        "--num-sims",
        type=int,
        default=10_000,
        help="Number of training simulations.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=1_000,
        help="Number of posterior samples per evaluation.",
    )
    parser.add_argument(
        "--steps",
        nargs="+",
        type=int,
        default=[50, 100, 200, 500],
        help="Discretization step counts to test.",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[0, 1, 2],
        help="Random seeds for statistical averaging.",
    )
    parser.add_argument(
        "--obs-indices",
        nargs="+",
        type=int,
        default=[1, 2, 3],
        help="Mini-sbibm observation indices.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Torch device (cpu or cuda).",
    )
    args = parser.parse_args()

    print(f"Task: {args.task}")
    print(f"SDE types: {args.sde_types}")
    print(f"Training sims: {args.num_sims}")
    print(f"Eval samples: {args.num_samples}")
    print(f"Step counts: {args.steps}")
    print(f"Seeds: {args.seeds}")
    print(f"Observations: {args.obs_indices}")
    print(f"Device: {args.device}")

    results = run_benchmark(
        task_name=args.task,
        sde_types=args.sde_types,
        num_sims=args.num_sims,
        num_samples=args.num_samples,
        step_counts=args.steps,
        seeds=args.seeds,
        observation_indices=args.obs_indices,
        device=args.device,
    )

    print("\n\nResults:")
    print(format_results(results))

    # Note on NFE: Heun makes 2 denoiser (score) evaluations per step, so
    # NFE = 2 * steps. All other samplers use 1 evaluation per step (NFE = steps).
    # For fair efficiency comparisons, use NFE rather than step count.
    print(
        "\nNote: NFE = number of function (score) evaluations. "
        "Heun uses 2 per step; all others use 1."
    )


if __name__ == "__main__":
    main()
