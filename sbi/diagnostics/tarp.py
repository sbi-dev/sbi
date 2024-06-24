"""
    Implementation taken from Lemos et al, 'Sampling-Based Accuracy Testing of
    Posterior Estimators for General Inference' https://arxiv.org/abs/2302.03026

    The TARP diagnostic is a global diagnostic which can be used to check a
    trained posterior against a set of true values of theta.
"""

import warnings
from typing import Callable, Optional, Tuple

# import numpy as np
import torch
from joblib import Parallel, delayed
from sbi.inference.posteriors.base_posterior import NeuralPosterior
from sbi.inference.posteriors.vi_posterior import VIPosterior
from sbi.simulators.simutils import tqdm_joblib
from sbi.utils.metrics import l1, l2
from scipy.stats import kstest
from torch import Tensor
from tqdm.auto import tqdm


def infer_posterior_on_batch(
    xs: Tensor,
    posterior: NeuralPosterior,
    num_posterior_samples: int = 1000,
) -> Tensor:
    """
    Infer samples of a posterior distribution on a batch of inputs.

    Parameters:
    ----------
    xs : Tensor
        The input data batch.
    posterior : NeuralPosterior
        The neural posterior to use for inference.
    num_posterior_samples : int, optional
        The number of posterior samples to draw for each input, by default 1000.

    Returns:
    -------
    Tensor
        A tensor of shape (num_posterior_samples, N, P) where N is the number of
        samples given by xs and P is the output dimension of the neural
        posterior estimator.
    """

    samples = []

    for idx in range(xs.shape[0]):
        # unsqueeze for potential higher-dimensional data.
        xo = xs[idx].unsqueeze(0)
        # VI posterior needs to be trained on the current xo.
        if isinstance(posterior, VIPosterior):
            posterior.set_default_x(xo)
            posterior.train()

        # Draw posterior samples and save one for the data average posterior.
        ths = posterior.sample((num_posterior_samples,), x=xo, show_progress_bars=False)
        # Note: one could calculate coverage values here

        samples.append(ths.unsqueeze(1))

    return torch.cat(samples, dim=1)


# this function currently does not perform any TARP related operation
# the purpose of the function is (a) to align with the sbc interface and
# (b) to provide the data which is required to run TARP
def prepare_estimates(
    xs: Tensor,
    posterior: NeuralPosterior,
    num_posterior_samples: int = 1000,
    num_workers: int = 1,
    infer_batch_size: int = 1,
    show_progress_bar: bool = True,
) -> Tensor:
    """
    Perform inference on batched x values using the provided posterior.
    the purpose of the function is (a) to align with the sbc interface and
    (b) to provide the data which is required to run TARP.

    Args:
        xs: observed data for tarp, simulated from thetas.
        posterior: a posterior obtained from sbi.
        num_posterior_samples: number of approximate posterior samples used
            for ranking.
        num_workers: number of CPU cores to use in parallel for running
            infer_batch_size inferences.
        infer_batch_size: batch size for workers.
        show_progress_bar: whether to display a progress bar

    Returns:
        samples: posterior samples obtained by performing inference on xs
            under the posterior

    """
    num_sim_samples = xs.shape[0]
    xs_batches = torch.split(xs, infer_batch_size, dim=0)

    if num_workers != 1:
        # Parallelize the sequence of batches across workers.
        # We use the solution proposed here: https://stackoverflow.com/a/61689175
        # to update the pbar only after the workers finished a task.
        with tqdm_joblib(
            tqdm(
                xs_batches,
                disable=not show_progress_bar,
                desc=f"Performing {num_sim_samples} posterior runs in"
                f"{len(xs_batches)} batches.",
                total=len(xs_batches),
            )
        ) as _:
            samples: Tensor
            samples = Parallel(
                n_jobs=num_workers
            )(  # pyright: ignore[reportAssignmentType]
                delayed(infer_posterior_on_batch)(
                    xs_batch, posterior, num_posterior_samples
                )
                for xs_batch in xs_batches
            )
    else:
        pbar = tqdm(
            total=num_sim_samples,
            disable=not show_progress_bar,
            desc=f"Running {num_sim_samples} samples for tarp analysis.",
        )

        with pbar:
            samples = []
            for xs_batch in xs_batches:
                samples.append(
                    infer_posterior_on_batch(xs_batch, posterior, num_posterior_samples)
                )
                pbar.update(infer_batch_size)
            samples = torch.cat(samples, dim=1)

    return samples


def _check_references(references: Tensor, theta: Tensor) -> Tensor:
    """construct references"""

    num_dims = theta.shape[-1]
    num_sims = theta.shape[0]

    if not isinstance(references, Tensor):
        # obtain min/max per dimension of theta
        lo = (
            torch.min(theta, dim=-2).values.min(axis=0).values
        )  # should be 0 if normalized
        hi = (
            torch.max(theta, dim=-2).values.max(axis=0).values
        )  # should be 1 if normalized

        refpdf = torch.distributions.Uniform(low=lo, high=hi)
        # sample one reference point for each entry in theta
        references = refpdf.sample((1, num_sims))
    else:
        if len(references.shape) == 2:
            # add singleton dimension in front
            references = references.unsqueeze(0)

        if len(references.shape) == 3 and references.shape[0] != 1:
            raise ValueError(
                f"""references must be a 2D array with a singular first
                    dimension, received {references.shape}"""
            )

        if references.shape[-2] != num_sims:
            raise ValueError(
                f"references must have the same number samples as samples,"
                f"received {references.shape[-2]} != {num_sims}"
            )

        if references.shape[-1] != num_dims:
            raise ValueError(
                "references must have the same number of dimensions as "
                f"samples or theta, received {references.shape[-1]}"
                f"!= {num_dims}"
            )

    return references


def run_tarp(
    samples: Tensor,
    theta: Tensor,
    references: Optional[Tensor] = None,
    distance: Callable = l2,
    num_bins: Optional[int] = None,
    do_norm: bool = False,
) -> Tuple[Tensor, Tensor]:
    """
    Estimates coverage of samples given true values theta with the TARP method.
    Reference: `Lemos, Coogan et al 2023 <https://arxiv.org/abs/2302.03026>`_

    The TARP diagnostic is a global diagnostic which can be used to check a
    trained posterior against a set of true values of theta.

    Args:
        samples: The predicted parameter samples to compute the coverage of,
                 these samples are expected to have shape
                 ``(num_samples, num_sims, num_dims)``. These are obtained by
                 sampling a trained posterior `num_samples` times. Multiple
                 (posterior) samples for one observation are encouraged.
        theta: The true parameter value theta. Theta is expected to
                 have shape ``(num_sims, num_dims)``.
        references: the reference points to use for the coverage regions, with
                shape ``(1, num_sims, num_dims)``, or ``None``.
                If ``None``, then reference points are chosen randomly from
                the unit hypercube over the parameter space given by theta.
                In other words, reference samples are drawn from the
                following ``Uniform(low=theta.min(dim=-1),high=theta.max(dim=-1))``.
        distance: the distance metric to use when computing the distance.
                Should be a callable function that accepts two tensors and
                computes the distance between them, e.g. given two tensors
                of shape ``(batch, 3)`` and ``(batch,3)``, this function should
                return ``(batch,1)`` distance values.
                Possible values: ``sbi.utils.metrics.l1`` or
                ``sbi.utils.metrics.l2``. ``l2`` is the default.
        num_bins: number of bins to use for the credibility values.
                If ``None``, then ``num_sims // 10`` bins are used.
        do_norm : whether to normalize parameters before coverage test
                (Default = True)

    Returns:
        ecp: Expected coverage probability (``ecp``)
        alpha: credibility values

    """
    # TARP assumes that the predicted thetas are sampled from the "true"
    # PDF num_samples times
    theta = theta.detach() if len(theta.shape) != 2 else theta.detach().unsqueeze(0)
    samples = samples.detach()

    assert (
        theta.shape[-2:] == samples.shape[-2:]
    ), f"shapes of theta {theta.shape[-2:]} and samples {samples.shape[-2:]} do not fit"

    num_samples = samples.shape[0]  # samples per simulation
    num_sims = samples.shape[-2]
    num_dims = samples.shape[-1]

    if num_bins is None:
        num_bins = num_sims // 10

    if theta.shape[-2] != num_sims:
        raise ValueError("theta must have the same number of rows as samples")
    if theta.shape[-1] != num_dims:
        raise ValueError("theta must have the same number of columns as samples")

    if do_norm:
        lo = torch.min(theta, dim=-2, keepdims=True).values  # min along num_sims
        hi = torch.max(theta, dim=-2, keepdims=True).values  # max along num_sims
        samples = (samples - lo) / (hi - lo + 1e-10)
        theta = (theta - lo) / (hi - lo + 1e-10)

    references = _check_references(references, theta)
    assert len(references.shape) == len(
        samples.shape
    ), f"references {references.shape} != samples {samples.shape}"

    # distances between references and samples
    sample_dists = distance(references.expand(num_samples, -1, -1), samples)

    # distances between references and true values
    theta_dists = distance(references, theta)

    # compute coverage, f in algorithm 2
    coverage_values = torch.sum(sample_dists < theta_dists, axis=0) / num_samples
    hist, bin_edges = torch.histogram(coverage_values, density=True, bins=num_bins)
    stepsize = bin_edges[1] - bin_edges[0]
    ecp = torch.cumsum(hist, dim=0) * stepsize

    return torch.cat([Tensor([0]), ecp]), bin_edges


def check_tarp(
    ecp: Tensor,
    alpha: Tensor,
) -> Tuple[float, float]:
    """check the obtained TARP credibitlity levels and
    expected coverage probabilities. This will help to uncover underdispersed,
    well covering or overdispersed posteriors.

    Args:
        samples: The predicted parameter samples to compute the coverage of,
                 these samples are expected to have shape
                 ``(num_samples, num_sims, num_dims)``. These are obtained by
                 sampling a trained posterior `num_samples` times. Multiple
                 (posterior) samples for one observation are encouraged.
        theta: The true parameter value theta. Theta is expected to
                 have shape ``(num_sims, num_dims)``.

    Returns:
        atc: area to curve, this number should be close to ``1``, values
             larger than ``1.`` indicated overdispersed distributions (i.e.
             the estimated posterior is too wide), values smaller than ``1``
             indicate underdispersed distributions (i.e. the estimated posterior
             is too narrow).
        ks prob: p-value for a two sample Kolmogorov-Smirnov test. The null
             hypothesis is that the two distributions (ecp and alpha) are
             identical. If they were, the p-value should be close 1. Commonly,
             people reject the null if p-value is below 0.05!
    """

    atc = (ecp - alpha).sum()
    kstest_pvals = torch.tensor(
        kstest(ecp.numpy(), alpha.numpy())[1],
        dtype=torch.float32,
    )

    return atc, kstest_pvals
