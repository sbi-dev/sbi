#!/usr/bin/env python
"""
Smoke script for comprehensive FMPE z-scoring comparison.

This script compares all available z-scoring and Gaussian baseline configurations
for FMPE on shifted data. It is intended for manual validation and will be removed
in a follow-up commit along with the initial_pr_formula and position baseline options.

Usage:
    python tests/smoke_fmpe_zscore_comparison.py

"""

import torch
from torch import ones

from sbi.inference import FMPE
from sbi.utils import BoxUniform
from sbi.utils.metrics import c2st


def run_single_comparison(seed, verbose=True):
    """Run a single C2ST comparison with a given seed."""
    torch.manual_seed(seed)

    # Setup: shifted data where theta ~ U(95, 105), mean ~100
    num_dim = 2
    prior = BoxUniform(95.0 * ones(num_dim), 105.0 * ones(num_dim))

    num_sims = 1000
    theta_train = prior.sample((num_sims,))
    x_train = theta_train + 0.5 * torch.randn_like(theta_train)
    x_o = torch.tensor([[100.0, 100.0]])

    # Reference posterior samples
    torch.manual_seed(seed + 1000)
    reference_samples = x_o + 0.5 * torch.randn(1000, 2)

    # Configuration definitions
    configs = {
        "no_zscore": {
            "z_score_theta": None,
            "gaussian_baseline": False,
            "z_score_method": "true_marginal",
        },
        "zscore_true_marginal": {
            "z_score_theta": "independent",
            "gaussian_baseline": False,
            "z_score_method": "true_marginal",
        },
        "zscore_initial_pr": {
            "z_score_theta": "independent",
            "gaussian_baseline": False,
            "z_score_method": "initial_pr_formula",
        },
        "baseline_velocity": {
            "z_score_theta": "independent",
            "gaussian_baseline": "velocity",
            "z_score_method": "true_marginal",
        },
        "baseline_position": {
            "z_score_theta": "independent",
            "gaussian_baseline": "position",
            "z_score_method": "true_marginal",
        },
        "baseline_position_raw": {
            "z_score_theta": "independent",
            "gaussian_baseline": "position_raw",
            "z_score_method": "true_marginal",
        },
    }

    results = {}

    for name, config in configs.items():
        if verbose:
            print(f"  {name}...", end=" ", flush=True)
        torch.manual_seed(seed)

        inference = FMPE(prior, show_progress_bars=False, **config)
        inference.append_simulations(theta_train.clone(), x_train.clone())
        inference.train(max_num_epochs=300, show_train_summary=False)

        samples = inference.build_posterior().sample(
            (1000,), x=x_o, show_progress_bars=False, reject_outside_prior=False
        )
        c2st_value = float(c2st(reference_samples, samples))
        results[name] = c2st_value

        if verbose:
            status = "PASS" if c2st_value < 0.55 else "FAIL"
            print(f"{c2st_value:.3f} ({status})")

    return results


def run_comparison(num_seeds=1, seeds=None):
    """Run comprehensive C2ST comparison with multiple seeds for error bars."""
    import numpy as np

    if seeds is None:
        seeds = [42 + i * 100 for i in range(num_seeds)]

    all_results = {
        name: []
        for name in [
            "no_zscore",
            "zscore_true_marginal",
            "zscore_initial_pr",
            "baseline_velocity",
            "baseline_position",
            "baseline_position_raw",
        ]
    }

    for i, seed in enumerate(seeds):
        print(f"\n--- Run {i + 1}/{len(seeds)} (seed={seed}) ---")
        results = run_single_comparison(seed, verbose=True)
        for name, value in results.items():
            all_results[name].append(value)

    # Compute statistics
    print("\n" + "=" * 70)
    print("SUMMARY (sorted by mean C2ST, lower is better)")
    print("=" * 70)

    summary = []
    for name, values in all_results.items():
        mean = np.mean(values)
        std = np.std(values)
        summary.append((name, mean, std, values))

    for name, mean, std, values in sorted(summary, key=lambda x: x[1]):
        status = "PASS" if mean < 0.55 else "FAIL"
        if len(values) > 1:
            print(f"  {name:25s} C2ST = {mean:.3f} +/- {std:.3f}  [{status}]")
        else:
            print(f"  {name:25s} C2ST = {mean:.3f}  [{status}]")

    return all_results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="FMPE Z-Scoring Comparison")
    parser.add_argument(
        "--num-seeds", type=int, default=1, help="Number of seeds for error bars"
    )
    args = parser.parse_args()

    print("FMPE Z-Scoring Comprehensive Comparison")
    print("=" * 70)

    # Run empirical C2ST comparison
    print("\n" + "=" * 70)
    print(f"EMPIRICAL C2ST COMPARISON ({args.num_seeds} seed(s))")
    print("=" * 70)
    results = run_comparison(num_seeds=args.num_seeds)

    print("\nDone!")
