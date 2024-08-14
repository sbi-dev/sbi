"""
Implementation taken from Lemos et al, 'Sampling-Based Accuracy Testing of
Posterior Estimators for General Inference' https://arxiv.org/abs/2302.03026

The TARP diagnostic is a global diagnostic which can be used to check a
trained posterior against a set of true values of theta.
"""

from typing import Callable, Optional, Tuple

import torch
from scipy.stats import kstest
from torch import Tensor

from sbi.inference.posteriors.base_posterior import NeuralPosterior
from sbi.utils.diagnostics_utils import get_posterior_samples_on_batch
from sbi.utils.metrics import l2


def run_tarp(
    thetas: Tensor,
    xs: Tensor,
    posterior: NeuralPosterior,
    references: Optional[Tensor] = None,
    num_posterior_samples: int = 1000,
    num_workers: int = 1,
    show_progress_bar: bool = True,
    distance: Callable = l2,
    num_bins: Optional[int] = 30,
    z_score_theta: bool = True,
) -> Tuple[Tensor, Tensor]:
    """
    Estimates coverage of samples given true values thetas with the TARP method.
    Reference: `Lemos, Coogan et al 2023 <https://arxiv.org/abs/2302.03026>`_

    The TARP diagnostic is a global diagnostic which can be used to check a
    trained posterior against a set of true values of theta.

    Args:
        thetas: ground-truth parameters for tarp, simulated from the prior.
        xs: observed data for tarp, simulated from thetas.
        posterior: a posterior obtained from sbi.
        num_posterior_samples: number of approximate posterior samples used for ranking.
        num_workers: number of CPU cores to use in parallel for running num_sbc_samples
            inferences.
        show_progress_bar: whether to display a progress over sbc runs.
        distance: the distance metric to use when computing the distance.
            Should be a callable function that accepts two tensors and
            computes the distance between them, e.g. given two tensors
            of shape ``(batch, 3)`` and ``(batch,3)``, this function should
            return ``(batch,1)`` distance values.
            Possible values: ``sbi.utils.metrics.l1`` or
            ``sbi.utils.metrics.l2``. ``l2`` is the default.
        num_bins: number of bins to use for the credibility values.
            If ``None``, then ``num_sims // 10`` bins are used.
        z_score_theta : whether to normalize parameters before coverage test.

    Returns:
        ecp: Expected coverage probability (``ecp``), see equation 4 of the paper
        alpha: credibility values, see equation 2 of the paper
    """
    num_tarp_samples, dim_theta = thetas.shape

    posterior_samples = get_posterior_samples_on_batch(
        xs,
        posterior,
        (num_posterior_samples,),
        num_workers,
        show_progress_bar=show_progress_bar,
    )
    assert posterior_samples.shape == (
        num_posterior_samples,
        num_tarp_samples,
        dim_theta,
    ), f"Wrong posterior samples shape for TARP: {posterior_samples.shape}"

    # Sample reference points uniformly if not provided
    if references is None:
        references = get_tarp_references(thetas)

    return _run_tarp(
        posterior_samples, thetas, references, distance, num_bins, z_score_theta
    )


def _run_tarp(
    posterior_samples: Tensor,
    thetas: Tensor,
    references: Tensor,
    distance: Callable = l2,
    num_bins: Optional[int] = 30,
    z_score_theta: bool = False,
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
        z_score_theta : whether to normalize parameters before coverage test.

    Returns:
        ecp: Expected coverage probability (``ecp``), see equation 4 of the paper
        alpha: grid of credibility values, see equation 2 of the paper

    """
    num_posterior_samples, num_tarp_samples, _ = posterior_samples.shape

    assert (
        references.shape == thetas.shape
    ), "references must have the same shape as thetas"

    if num_bins is None:
        num_bins = num_tarp_samples // 10

    if z_score_theta:
        lo = thetas.min(dim=0, keepdim=True).values  # min over batch
        hi = thetas.max(dim=0, keepdim=True).values  # max over batch
        posterior_samples = (posterior_samples - lo) / (hi - lo + 1e-10)
        thetas = (thetas - lo) / (hi - lo + 1e-10)

    # distances between references and samples
    sample_dists = distance(references, posterior_samples)

    # distances between references and true values
    theta_dists = distance(references, thetas)

    # compute coverage, f in algorithm 2
    coverage_values = (
        torch.sum(sample_dists < theta_dists, dim=0) / num_posterior_samples
    )
    hist, alpha_grid = torch.histogram(coverage_values, density=True, bins=num_bins)
    # calculate empirical CDF via cumsum and normalize
    ecp = torch.cumsum(hist, dim=0) / hist.sum()
    # add 0 to the beginning of the ecp curve to match the alpha grid
    ecp = torch.cat([Tensor([0]), ecp])

    return ecp, alpha_grid


def get_tarp_references(thetas: Tensor) -> Tensor:
    """Returns reference points for the TARP diagnostic, sampled from a uniform."""

    # obtain min/max per dimension of theta
    lo = thetas.min(dim=0).values  # min for each theta dimension
    hi = thetas.max(dim=0).values  # max for each theta dimension

    refpdf = torch.distributions.Uniform(low=lo, high=hi)

    # sample one reference point for each entry in theta
    return refpdf.sample(torch.Size([thetas.shape[0]]))


def check_tarp(
    ecp: Tensor,
    alpha: Tensor,
) -> Tuple[float, float]:
    r"""check the obtained TARP credibitlity levels and
    expected coverage probabilities. This will help to uncover underdispersed,
    well covering or overdispersed posteriors.

    Args:
        ecp: expected coverage probabilities computed with the TARP method,
            i.e. first output of ``run_tarp``.
        alpha: credibility levels $\alpha$, i.e. second output of ``run_tarp``.

    Returns:
        atc: area to curve, the difference between the ecp and alpha curve for
            alpha values larger than 0.5. This number should be close to ``0``.
            Values larger than ``0`` indicated overdispersed distributions (i.e.
            the estimated posterior is too wide). Values smaller than ``0``
            indicate underdispersed distributions (i.e. the estimated posterior
            is too narrow). Note, this property of the ecp curve can also
            indicate if the posterior is biased, see figure 2 of the paper for
            details (https://arxiv.org/abs/2302.03026).
        ks prob: p-value for a two sample Kolmogorov-Smirnov test. The null
             hypothesis of this test is that the two distributions (ecp and
             alpha) are identical, i.e. are produced by one common CDF. If they
             were, the p-value should be close to ``1``. Commonly, people reject
             the null if p-value is below 0.05!
    """

    # get the index of the middle of the alpha grid
    midindex = alpha.shape[0] // 2
    # area to curve: difference between ecp and alpha above 0.5.
    atc = (ecp[midindex:] - alpha[midindex:]).sum().item()

    # Kolmogorov-Smirnov test between ecp and alpha
    kstest_pvals: float = kstest(ecp.numpy(), alpha.numpy())[1]  # type: ignore

    return atc, kstest_pvals
