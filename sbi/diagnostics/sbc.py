# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

import warnings
from typing import Callable, Dict, List, Union

import torch
from scipy.stats import kstest, uniform
from torch import Tensor, ones, zeros
from torch.distributions import Uniform
from tqdm.auto import tqdm

from sbi.inference import DirectPosterior
from sbi.inference.posteriors.base_posterior import NeuralPosterior
from sbi.inference.posteriors.vi_posterior import VIPosterior
from sbi.utils.diagnostics_utils import get_posterior_samples_on_batch
from sbi.utils.metrics import c2st


def run_sbc(
    thetas: Tensor,
    xs: Tensor,
    posterior: NeuralPosterior,
    num_posterior_samples: int = 1000,
    reduce_fns: Union[str, Callable, List[Callable]] = "marginals",
    num_workers: int = 1,
    show_progress_bar: bool = True,
    use_batched_sampling: bool = True,
    **kwargs,
):
    """Run simulation-based calibration (SBC) (parallelized across sbc runs).

    Note: This function implements two versions of coverage diagnostics:
     - setting reduce_fns = "marginals" performs SBC as proposed in Talts et
       al., https://arxiv.org/abs/1804.06788.
    - setting reduce_fns = posterior.log_prob performs sample-based expected
      coverage as proposed in Deistler et al., https://arxiv.org/abs/2210.04815.

    Args:
        thetas: ground-truth parameters for sbc, simulated from the prior.
        xs: observed data for sbc, simulated from thetas.
        posterior: a posterior obtained from sbi. num_posterior_samples: number
            of approximate posterior samples used for ranking.
        reduce_fns: Function used to reduce the parameter space into 1D.
            Simulation-based calibration can be recovered by setting this to the
            string `marginals`. Sample-based expected coverage can be recovered
            by setting it to `posterior.log_prob` (as a Callable).
        num_workers: number of CPU cores to use in parallel for running
            `num_sbc_samples` inferences.
        show_progress_bar: whether to display a progress over sbc runs.
        use_batched_sampling: whether to use batched sampling for posterior
            samples.

    Returns:
        ranks: ranks of the ground truth parameters under the inferred
        dap_samples: samples from the data averaged posterior.
    """
    num_sbc_samples = thetas.shape[0]

    if num_sbc_samples < 100:
        warnings.warn(
            "Number of SBC samples should be on the order of 100s to give realiable "
            "results.",
            stacklevel=2,
        )
    if num_posterior_samples < 100:
        warnings.warn(
            "Number of posterior samples for ranking should be on the order "
            "of 100s to give reliable SBC results.",
            stacklevel=2,
        )

    assert (
        thetas.shape[0] == xs.shape[0]
    ), "Unequal number of parameters and observations."

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

    # take a random draw from each posterior to get data averaged posterior samples.
    dap_samples = posterior_samples[0, :, :]
    assert dap_samples.shape == (num_sbc_samples, thetas.shape[1]), "Wrong dap shape."

    ranks = _run_sbc(
        thetas, xs, posterior_samples, posterior, reduce_fns, show_progress_bar
    )

    return ranks, dap_samples


def _run_sbc(
    thetas: Tensor,
    xs: Tensor,
    posterior_samples: Tensor,
    posterior: NeuralPosterior,
    reduce_fns: Union[str, Callable, List[Callable]] = "marginals",
    show_progress_bar: bool = True,
) -> Tensor:
    """Calculate ranks for SBC or expected coverage."""
    num_sbc_samples = thetas.shape[0]

    # construct reduce functions for SBC or expected coverage
    # For SBC, we simply take the marginals for each parameter dimension.
    if isinstance(reduce_fns, str):
        assert reduce_fns == "marginals", (
            "`reduce_fn` must either be the string `marginals` or a Callable or a List "
            "of Callables."
        )
        reduce_fns = [
            eval(f"lambda theta, x: theta[:, {i}]") for i in range(thetas.shape[1])
        ]

    # For a Callable (e.g., expected coverage) we put it into a list for unified
    # handling below.
    if isinstance(reduce_fns, Callable):
        reduce_fns = [reduce_fns]

    ranks = torch.zeros((num_sbc_samples, len(reduce_fns)))
    # Iterate over all sbc samples and calculate ranks.
    for sbc_idx, (true_theta, x_i) in tqdm(
        enumerate(zip(thetas, xs)),
        total=num_sbc_samples,
        disable=not show_progress_bar,
        desc=f"Calculating ranks for {num_sbc_samples} sbc samples.",
    ):
        # For VIPosteriors, we need to train on each x.
        if isinstance(posterior, VIPosterior):
            posterior.set_default_x(x_i)
            posterior.train(show_progress_bar=False)

        # For each reduce_fn (e.g., per marginal for SBC)
        for dim_idx, reduce_fn in enumerate(reduce_fns):
            # rank posterior samples against true parameter, reduced to 1D.
            ranks[sbc_idx, dim_idx] = (
                (
                    reduce_fn(posterior_samples[:, sbc_idx, :], x_i)
                    < reduce_fn(true_theta.unsqueeze(0), x_i)
                )
                .sum()
                .item()
            )

    return ranks


def get_nltp(thetas: Tensor, xs: Tensor, posterior: NeuralPosterior) -> Tensor:
    """Return negative log prob of true parameters under the posterior.

    NLTP: negative log probs of true parameters under the approximate posterior.
    The expectation of NLTP over samples from the prior and the simulator defines
    an upper bound for accuracy of the ground-truth posterior (without having
    access to it, see Lueckmann et al. 2021, Appendix for details).
    Thus, if the one calculates NLTP for many thetas (say >100), one can use it as a
    comparable measure of posterior accuracy when comparing inference methods, or
    settings (even without access to the ground-truth posterior)

    Note that this is interpretable only for normalized log probs, i.e., when
    using (S)NPE.

    Args:
        thetas: parameters (sampled from the prior) for which to calculate NLTP values.
        xs: simulated data corresponding to thetas.
        posterior: inferred posterior for which to calculate NLTP.

    Returns:
        nltp: negative log probs of true parameters under approximate posteriors.
    """
    nltp = torch.zeros(thetas.shape[0])
    unnormalized_log_prob = not isinstance(posterior, DirectPosterior)

    for idx, (tho, xo) in enumerate(zip(thetas, xs)):
        # Log prob of true params under posterior.
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
    """Return uniformity checks and data averaged posterior checks for SBC.

    Args:
        ranks: ranks for each sbc run and for each model parameter, i.e.,
            shape (N, dim_parameters)
        prior_samples: N samples from the prior
        dap_samples: N samples from the data averaged posterior
        num_posterior_samples: number of posterior samples used for sbc ranking.
        num_c2st_repetitions: number of times c2st is repeated to estimate robustness.

    Returns (all in a dictionary):
        ks_pvals: p-values of the Kolmogorov-Smirnov test of uniformity,
            one for each dim_parameters.
        c2st_ranks: C2ST accuracy of between ranks and uniform baseline,
            one for each dim_parameters.
        c2st_dap: C2ST accuracy between prior and dap samples, single value.
    """
    if ranks.shape[0] < 100:
        warnings.warn(
            "You are computing SBC checks with less than 100 samples. These checks"
            " should be based on a large number of test samples theta_o, x_o. We"
            " recommend using at least 100.",
            stacklevel=2,
        )

    ks_pvals = check_uniformity_frequentist(ranks, num_posterior_samples)
    c2st_ranks = check_uniformity_c2st(
        ranks, num_posterior_samples, num_repetitions=num_c2st_repetitions
    )
    c2st_scores_dap = check_prior_vs_dap(prior_samples, dap_samples)

    return dict(
        ks_pvals=ks_pvals,
        c2st_ranks=c2st_ranks,
        c2st_dap=c2st_scores_dap,
    )


def check_prior_vs_dap(prior_samples: Tensor, dap_samples: Tensor) -> Tensor:
    """Returns the c2st accuracy between prior and data avaraged posterior samples.

    c2st is calculated for each dimension separately.

    According to simulation-based calibration, the inference methods is well-calibrated
    if the data averaged posterior samples follow the same distribution as the prior,
    i.e., if the c2st score is close to 0.5. If it is not, then this suggests that the
    inference method is not well-calibrated (see Talts et al, "Simulation-based
    calibration" for details).
    """

    assert prior_samples.shape == dap_samples.shape

    return torch.tensor([
        c2st(s1.unsqueeze(1), s2.unsqueeze(1))
        for s1, s2 in zip(prior_samples.T, dap_samples.T)
    ])


def check_uniformity_frequentist(ranks, num_posterior_samples) -> Tensor:
    """Return p-values for uniformity of the ranks.

    Calculates Kolomogorov-Smirnov test using scipy.

    Args:
        ranks: ranks for each sbc run and for each model parameter, i.e.,
            shape (N, dim_parameters)
        num_posterior_samples: number of posterior samples used for sbc ranking.

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
    ranks, num_posterior_samples, num_repetitions: int = 1
) -> Tensor:
    """Return c2st scores for uniformity of the ranks.

    Run a c2st between ranks and uniform samples.

    Args:
        ranks: ranks for each sbc run and for each model parameter, i.e.,
            shape (N, dim_parameters)
        num_posterior_samples: number of posterior samples used for sbc ranking.
        num_repetitions: repetitions of C2ST tests estimate classifier variance.

    Returns:
        c2st_ranks: C2ST accuracy of between ranks and uniform baseline,
            one for each dim_parameters.
    """

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

    # Use variance over repetitions to estimate robustness of c2st.
    c2st_std = c2st_scores.std(0, correction=0 if num_repetitions == 1 else 1)
    if (c2st_std > 0.05).any():
        warnings.warn(
            f"C2ST score variability is larger than {0.05}: std={c2st_scores.std(0)}, "
            "result may be unreliable. Consider increasing the number of samples.",
            stacklevel=2,
        )

    # Return the mean over repetitions as c2st score estimate.
    return c2st_scores.mean(0)
