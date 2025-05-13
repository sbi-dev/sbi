# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

import warnings
from typing import Callable, Dict, List, Tuple, Union

import torch
from scipy.stats import kstest, uniform
from torch import Tensor, ones, zeros
from torch.distributions import Uniform
from tqdm.auto import tqdm

from sbi.inference import DirectPosterior, VectorFieldPosterior
from sbi.inference.posteriors.base_posterior import NeuralPosterior
from sbi.inference.posteriors.vi_posterior import VIPosterior
from sbi.utils.diagnostics_utils import (
    get_posterior_samples_on_batch,
    remove_nans_and_infs_in_x,
)
from sbi.utils.metrics import c2st


def run_sbc(
    thetas: Tensor,
    xs: Tensor,
    posterior: NeuralPosterior,
    num_posterior_samples: int = 1000,
    reduce_fns: Union[
        str,
        Callable[[Tensor, Tensor], Tensor],
        List[Callable[[Tensor, Tensor], Tensor]],
    ] = "marginals",
    num_workers: int = 1,
    show_progress_bar: bool = True,
    use_batched_sampling: bool = True,
    **kwargs,
) -> Tuple[Tensor, Tensor]:
    """Run simulation-based calibration (SBC) or expected coverage.

    Note: This function implements two versions of coverage diagnostics:

    - Setting ``reduce_fns = "marginals"`` performs SBC as proposed in Talts et al.
      (see https://arxiv.org/abs/1804.06788).
    - Setting ``reduce_fns = posterior.log_prob`` performs sample-based expected
      coverage as proposed in Deistler et al.
      (see https://arxiv.org/abs/2210.04815).

    Args:
        thetas: Ground-truth parameters for SBC, simulated from the prior.
        xs: Observed data for SBC, simulated from thetas.
        posterior: A posterior obtained from sbi.
        num_posterior_samples: Number of approximate posterior samples used for ranking.
        reduce_fns: Function used to reduce the parameter space into 1D.
            Simulation-based calibration can be recovered by setting this to the
            string `"marginals"`. Sample-based expected coverage can be recovered
            by setting it to `posterior.log_prob` (as a Callable).
        num_workers: Number of CPU cores to use in parallel.
        show_progress_bar: Whether to display a progress bar over SBC runs.
        use_batched_sampling: Whether to use batched sampling for posterior samples.

    Returns:
        ranks: Ranks of the ground truth parameters under the inferred posterior.
        dap_samples: Samples from the data-averaged posterior.
    """
    # Remove NaNs and infinities from the input data.
    thetas, xs = remove_nans_and_infs_in_x(thetas, xs)

    num_sbc_samples = thetas.shape[0]

    # Validate input parameters.
    _validate_sbc_inputs(thetas, xs, num_sbc_samples, num_posterior_samples)

    # Handle deprecated parameter.
    if "sbc_batch_size" in kwargs:
        warnings.warn(
            "`sbc_batch_size` is deprecated and will be removed in future versions."
            " Use `num_workers` instead.",
            DeprecationWarning,
            stacklevel=2,
        )

    # Get posterior samples, batched or parallelized.
    posterior_samples = get_posterior_samples_on_batch(
        xs,
        posterior,
        (num_posterior_samples,),
        num_workers,
        show_progress_bar,
        use_batched_sampling=use_batched_sampling,
    )

    # Take a random draw from each posterior to get data-averaged posterior samples.
    dap_samples = posterior_samples[0, :, :]
    assert dap_samples.shape == (num_sbc_samples, thetas.shape[1]), "Wrong DAP shape."

    # Create wrapper for reduce_fns if using a VIPosterior that ensures it is trained
    # before applying the reduce function.
    if isinstance(posterior, VIPosterior):

        def make_vipost_wrapper(original_reduce_fn: Callable) -> Callable:
            """Returns a wrapped reduce function for VIPosterior."""

            def wrapped_reduce_fn(theta: Tensor, x: Tensor) -> Tensor:
                """Wrap reduce function to ensure VIPosterior is trained."""
                # Train the posterior on the current x if needed
                posterior.set_default_x(x)
                posterior.train(show_progress_bar=False)
                return original_reduce_fn(theta, x)

            return wrapped_reduce_fn

        # Apply wrapper if reduce_fns is a single callable
        if callable(reduce_fns) and not isinstance(reduce_fns, str):
            reduce_fns = make_vipost_wrapper(reduce_fns)
        # Apply wrapper if reduce_fns is a list of callables
        elif isinstance(reduce_fns, list):
            reduce_fns = [make_vipost_wrapper(fn) for fn in reduce_fns]

    # Calculate ranks
    ranks = _run_sbc(thetas, xs, posterior_samples, reduce_fns, show_progress_bar)

    return ranks, dap_samples


def _validate_sbc_inputs(
    thetas: Tensor, xs: Tensor, num_sbc_samples: int, num_posterior_samples: int
) -> None:
    """Validate inputs for the SBC procedure."""
    if num_sbc_samples < 100:
        warnings.warn(
            "Number of SBC samples should be on the order of 100s to give reliable "
            "results.",
            stacklevel=2,
        )

    if num_posterior_samples < 100:
        warnings.warn(
            "Number of posterior samples for ranking should be on the order "
            "of 100s to give reliable SBC results.",
            stacklevel=2,
        )

    if thetas.shape[0] != xs.shape[0]:
        raise ValueError("Unequal number of parameters and observations.")


def _run_sbc(
    thetas: Tensor,
    xs: Tensor,
    posterior_samples: Tensor,
    reduce_fns: Union[
        str,
        Callable[[Tensor, Tensor], Tensor],
        List[Callable[[Tensor, Tensor], Tensor]],
    ] = "marginals",
    show_progress_bar: bool = True,
) -> Tensor:
    """Calculate ranks for SBC or expected coverage.

    Args:
        thetas: Ground-truth parameters.
        xs: Observed data corresponding to thetas.
        posterior_samples: Samples from posterior distribution.
        reduce_fns: Functions to reduce parameter space to 1D.
        show_progress_bar: Whether to show progress bar.

    Returns:
        Tensor of ranks for each parameter and reduction function.
    """
    num_sbc_samples = thetas.shape[0]

    # Construct reduce functions for SBC or expected coverage.
    reduce_fns = _prepare_reduce_functions(reduce_fns, thetas.shape[1])

    # Initialize ranks tensor.
    ranks = torch.zeros((num_sbc_samples, len(reduce_fns)))

    # Iterate over all SBC samples and calculate ranks.
    for sbc_idx, (true_theta, x_i) in tqdm(
        enumerate(zip(thetas, xs, strict=False)),
        total=num_sbc_samples,
        disable=not show_progress_bar,
        desc=f"Calculating ranks for {num_sbc_samples} SBC samples",
    ):
        # For each reduce_fn (e.g., per marginal for SBC)
        for dim_idx, reduce_fn in enumerate(reduce_fns):
            # Rank posterior samples against true parameter, reduced to 1D
            ranks[sbc_idx, dim_idx] = (
                (
                    reduce_fn(posterior_samples[:, sbc_idx, :], x_i)
                    < reduce_fn(true_theta.unsqueeze(0), x_i)
                )
                .sum()
                .item()
            )

    return ranks


def _prepare_reduce_functions(
    reduce_fns: Union[
        str,
        Callable[[Tensor, Tensor], Tensor],
        List[Callable[[Tensor, Tensor], Tensor]],
    ],
    param_dim: int,
) -> List[Callable[[Tensor, Tensor], Tensor]]:
    """Prepare reduction functions for SBC analysis.

    Args:
        reduce_fns: Function(s) to reduce parameters to 1D.
        param_dim: Dimensionality of parameter space.

    Returns:
        List of callable reduction functions.
    """
    # For SBC, we simply take the marginals for each parameter dimension.
    if isinstance(reduce_fns, str):
        if reduce_fns != "marginals":
            raise ValueError(
                "`reduce_fn` must either be the string `marginals` or a Callable or a "
                "List of Callables."
            )
        return [eval(f"lambda theta, x: theta[:, {i}]") for i in range(param_dim)]

    if isinstance(reduce_fns, Callable):
        return [reduce_fns]

    return reduce_fns


def get_nltp(thetas: Tensor, xs: Tensor, posterior: NeuralPosterior) -> Tensor:
    """Return negative log prob of true parameters under the posterior.

    NLTP: Negative log probabilities of true parameters under the approximate posterior.
    The expectation of NLTP over samples from the prior and the simulator defines
    an upper bound for accuracy of the ground-truth posterior (without having
    access to it, see Lueckmann et al. 2021, Appendix for details).

    If calculated for many thetas (>100), NLTP can be used as a comparable measure
    of posterior accuracy when comparing inference methods or settings.

    Note: This is interpretable only for normalized log probs, i.e., when using (S)NPE.

    Args:
        thetas: Parameters (sampled from the prior) for which to calculate NLTP values.
        xs: Simulated data corresponding to thetas.
        posterior: Inferred posterior for which to calculate NLTP.

    Returns:
        nltp: Negative log probs of true parameters under approximate posteriors.
    """
    nltp = torch.zeros(thetas.shape[0])
    unnormalized_log_prob = not isinstance(
        posterior, (DirectPosterior, VectorFieldPosterior)
    )

    for idx, (tho, xo) in enumerate(zip(thetas, xs, strict=False)):
        # Log prob of true params under posterior
        if unnormalized_log_prob:
            nltp[idx] = -posterior.potential(tho, x=xo)
        else:
            nltp[idx] = -posterior.log_prob(tho, x=xo)

    if unnormalized_log_prob:
        warnings.warn(
            "Note that log probs of the true parameters under the posteriors are not "
            "normalized because the posterior used is likelihood-based.",
            stacklevel=2,
        )

    return nltp


def check_sbc(
    ranks: Tensor,
    prior_samples: Tensor,
    dap_samples: Tensor,
    num_posterior_samples: int = 1000,
    num_c2st_repetitions: int = 1,
) -> Dict[str, Tensor]:
    """Return uniformity checks and data-averaged posterior checks for SBC.

    Args:
        ranks: Ranks for each SBC run and for each model parameter,
            shape (N, dim_parameters)
        prior_samples: N samples from the prior
        dap_samples: N samples from the data-averaged posterior
        num_posterior_samples: Number of posterior samples used for SBC ranking.
        num_c2st_repetitions: Number of times C2ST is repeated to estimate robustness.

    Returns:
        Dictionary containing:
        - ks_pvals: p-values of the Kolmogorov-Smirnov test of uniformity,
          one for each dim_parameters.
        - c2st_ranks: C2ST accuracy between ranks and uniform baseline,
          one for each dim_parameters.
        - c2st_dap: C2ST accuracy between prior and DAP samples, single value.
    """
    if ranks.shape[0] < 100:
        warnings.warn(
            "You are computing SBC checks with less than 100 samples. These checks "
            "should be based on a large number of test samples theta_o, x_o. We "
            "recommend using at least 100.",
            stacklevel=2,
        )

    # Run uniformity checks
    ks_pvals = check_uniformity_frequentist(ranks, num_posterior_samples)
    c2st_ranks = check_uniformity_c2st(
        ranks, num_posterior_samples, num_repetitions=num_c2st_repetitions
    )

    # Compare prior and data-averaged posterior
    c2st_scores_dap = check_prior_vs_dap(prior_samples, dap_samples)

    return {
        "ks_pvals": ks_pvals,
        "c2st_ranks": c2st_ranks,
        "c2st_dap": c2st_scores_dap,
    }


def check_prior_vs_dap(prior_samples: Tensor, dap_samples: Tensor) -> Tensor:
    """Returns the C2ST accuracy between prior and data-averaged posterior samples.

    C2ST is calculated for each dimension separately.

    According to simulation-based calibration, the inference method is well-calibrated
    if the data-averaged posterior samples follow the same distribution as the prior,
    i.e., if the C2ST score is close to 0.5. If it is not, then this suggests that the
    inference method is not well-calibrated (see Talts et al, "Simulation-based
    calibration" for details).

    Args:
        prior_samples: Samples from the prior distribution.
        dap_samples: Samples from the data-averaged posterior.

    Returns:
        Tensor of C2ST scores for each parameter dimension.
    """
    if prior_samples.shape != dap_samples.shape:
        raise ValueError("Prior and DAP samples must have the same shape")

    return torch.tensor([
        c2st(s1.unsqueeze(1), s2.unsqueeze(1))
        for s1, s2 in zip(prior_samples.T, dap_samples.T, strict=False)
    ])


def check_uniformity_frequentist(ranks: Tensor, num_posterior_samples: int) -> Tensor:
    """Return p-values for uniformity of the ranks using Kolmogorov-Smirnov test.

    Args:
        ranks: Ranks for each SBC run and for each model parameter,
            shape (N, dim_parameters)
        num_posterior_samples: Number of posterior samples used for SBC ranking.

    Returns:
        ks_pvals: p-values of the Kolmogorov-Smirnov test of uniformity,
            one for each dim_parameters.
    """
    kstest_pvals = torch.tensor(
        [
            kstest(rks, uniform(loc=0, scale=num_posterior_samples).cdf)[1]
            for rks in ranks.T
        ],
        dtype=torch.float32,
    )

    return kstest_pvals


def check_uniformity_c2st(
    ranks: Tensor, num_posterior_samples: int, num_repetitions: int = 1
) -> Tensor:
    """Return C2ST scores for uniformity of the ranks.

    Run a C2ST between ranks and uniform samples.

    Args:
        ranks: Ranks for each SBC run and for each model parameter,
            shape (N, dim_parameters)
        num_posterior_samples: Number of posterior samples used for SBC ranking.
        num_repetitions: Repetitions of C2ST tests to estimate classifier variance.

    Returns:
        c2st_ranks: C2ST accuracy between ranks and uniform baseline,
            one for each dim_parameters.
    """
    # Run C2ST multiple times to estimate stability
    c2st_scores = torch.tensor([
        [
            c2st(
                rks.unsqueeze(1),
                Uniform(zeros(1), num_posterior_samples * ones(1)).sample(
                    torch.Size((ranks.shape[0],))
                ),
            )
            for rks in ranks.T
        ]
        for _ in range(num_repetitions)
    ])

    # Use variance over repetitions to estimate robustness of C2ST
    c2st_std = c2st_scores.std(0, correction=0 if num_repetitions == 1 else 1)
    if (c2st_std > 0.05).any():
        warnings.warn(
            f"C2ST score variability is larger than 0.05: std={c2st_std}, "
            "result may be unreliable. Consider increasing the number of samples.",
            stacklevel=2,
        )

    # Return the mean over repetitions as C2ST score estimate
    return c2st_scores.mean(0)
