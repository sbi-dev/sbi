import warnings
from typing import Dict, Tuple

import torch
from joblib import Parallel, delayed
from scipy.stats import kstest, uniform
from torch import Tensor, ones, zeros
from torch.distributions import Uniform
from tqdm.auto import tqdm

from sbi.inference import DirectPosterior
from sbi.inference.posteriors.base_posterior import NeuralPosterior
from sbi.simulators.simutils import tqdm_joblib
from sbi.utils.metrics import c2st


def run_sbc(
    thetas: Tensor,
    xs: Tensor,
    posterior: NeuralPosterior,
    num_posterior_samples: int = 1000,
    num_workers: int = 1,
    sbc_batch_size: int = 1,
    show_progress_bar: bool = True,
) -> Tuple[Tensor, Tensor]:
    """Run simulation-based calibration (parallelized across sbc runs).

    Returns sbc ranks, log probs of the true parameters under the posterior and samples
    from the data averaged posterior, one for each sbc run, respectively.

    Args:
        thetas: ground-truth parameters for sbc, simulated from the prior.
        xs: observed data for sbc, simulated from thetas.
        posterior: a posterior obtained from sbi.
        num_posterior_samples: number of approximate posterior samples used for ranking.
        num_workers: number of CPU cores to use in parallel for running num_sbc_samples inferences.
        sbc_batch_size: batch size for workers.
        show_progress_bar: whether to display a progress over sbc runs.

    Returns:
        ranks: ranks of the ground truth parameters under the inferred posterior.
        dap_samples: samples from the data averaged posterior.
    """
    num_sbc_samples = thetas.shape[0]

    if num_sbc_samples < 1000:
        warnings.warn(
            """Number of SBC samples should be on the order of 100s to give realiable
            results. We recommend using 300."""
        )
    if num_posterior_samples < 100:
        warnings.warn(
            """Number of posterior samples for ranking should be on the order
            of 100s to give reliable SBC results. We recommend using at least 300."""
        )

    assert (
        thetas.shape[0] == xs.shape[0]
    ), "Unequal number of parameters and observations."

    thetas_batches = torch.split(thetas, sbc_batch_size, dim=0)
    xs_batches = torch.split(xs, sbc_batch_size, dim=0)

    if num_workers > 1:
        # Parallelize the sequence of batches across workers.
        # We use the solution proposed here: https://stackoverflow.com/a/61689175
        # to update the pbar only after the workers finished a task.
        with tqdm_joblib(
            tqdm(
                thetas_batches,
                disable=not show_progress_bar,
                desc=f"""Running {num_sbc_samples} sbc runs in {len(thetas_batches)}
                    batches.""",
                total=len(thetas_batches),
            )
        ) as progress_bar:
            sbc_outputs = Parallel(n_jobs=num_workers)(
                delayed(sbc_on_batch)(
                    thetas_batch, xs_batch, posterior, num_posterior_samples
                )
                for thetas_batch, xs_batch in zip(thetas_batches, xs_batches)
            )
    else:
        pbar = tqdm(
            total=num_sbc_samples,
            disable=not show_progress_bar,
            desc=f"Running {num_sbc_samples} sbc samples.",
        )

        with pbar:
            sbc_outputs = []
            for thetas_batch, xs_batch in zip(thetas_batches, xs_batches):
                sbc_outputs.append(
                    sbc_on_batch(
                        thetas_batch,
                        xs_batch,
                        posterior,
                        num_posterior_samples,
                    )
                )
                pbar.update(sbc_batch_size)

    # Aggregate results.
    ranks = []
    dap_samples = []
    for out in sbc_outputs:
        ranks.append(out[0])
        dap_samples.append(out[1])

    ranks = torch.cat(ranks)
    dap_samples = torch.cat(dap_samples)

    return ranks, dap_samples


def sbc_on_batch(
    thetas: Tensor, xs: Tensor, posterior: NeuralPosterior, num_posterior_samples: int
) -> Tuple[Tensor, Tensor]:
    """Return SBC results for a batch of SBC parameters and data from prior.

    Args:
        thetas: ground truth parameters.
        xs: corresponding observations.
        posterior: sbi posterior.
        num_posterior_samples: number of samples to draw from the posterior in each sbc run.

    Returns
        ranks: ranks of true parameters vs. posterior samples under the specified RV,
            for each posterior dimension.
        log_prob_thetas: log prob of true parameters under the approximate posterior.
            Note that this is interpretable only for normalized log probs, i.e., when
            using (S)NPE.
        dap_samples: samples from the data averaged posterior for the current batch,
            i.e., a single sample from each approximate posterior.
    """

    dap_samples = torch.zeros_like(thetas)
    ranks = torch.zeros_like(thetas)

    for idx, (tho, xo) in enumerate(zip(thetas, xs)):
        # Draw posterior samples and save one for the data average posterior.
        ths = posterior.sample((num_posterior_samples,), x=xo, show_progress_bars=False)

        # Save one random sample for data average posterior (dap).
        dap_samples[idx] = ths[0]

        # rank for each posterior dimension as in Talts et al. section 4.1.
        for dim in range(thetas.shape[1]):
            ranks[idx, dim] = (ths[:, dim] < tho[dim]).sum().item()

    return ranks, dap_samples


def get_nltp(thetas: Tensor, xs: Tensor, posterior: NeuralPosterior) -> Tensor:
    """Return negative log prob of true parameters under the posterior.

    NLTP: negative log probs of true parameters under the approximate posterior. Its
    mean over many N gives a lower bound on the posterior accuracy and can be used to
    compare different methods.

    Args:
        thetas: parameters for which to calculate NLTP, sampled from the prior.
        xs: simulated data corresponding to thetas.
        posterior: inferred posterior for which to calculate NLTP.

    Returns:
        nltp: negative log probs of true parameters under approximate posteriors.
            If N is large, e.g., N>100, then nltp can be used as a comparative measure
            of posterior accuracy, e.g., for comparing multiple inference methods, or
            hyperparameter settings (see num_posterior_samplesueckmann et al. 2021,
            appendix for details).
            Note that this is interpretable only for normalized log probs, i.e., when
            using (S)NPE.
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
            """Note that log probs of the true parameters under the posteriors
        are not normalized because the posterior used is likelihood-based."""
        )

    return nltp


def check_sbc(
    ranks: Tensor,
    prior_samples: Tensor,
    dap_samples: Tensor,
    num_posterior_samples: int,
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
            """You are computing SBC checks with less than 100 samples. These checks
            should be based on a large number of test samples theta_o, x_o. We
            recommend using at least 100."""
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
    """Return the c2st accuracy between prior and data avaraged posterior samples.

    c2st is calculated for each dimension separately.
    """

    assert prior_samples.shape == dap_samples.shape

    return torch.tensor(
        [
            c2st(s1.unsqueeze(1), s2.unsqueeze(1))
            for s1, s2 in zip(prior_samples.T, dap_samples.T)
        ]
    )


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

    c2st_scores = torch.tensor(
        [
            [
                c2st(
                    rks.unsqueeze(1),
                    Uniform(zeros(1), num_posterior_samples * ones(1)).sample(
                        (ranks.shape[0],)
                    ),
                )
                for rks in ranks.T
            ]
            for _ in range(num_repetitions)
        ]
    )

    # Use variance over repetitions to estimate robustness of c2st.
    if (c2st_scores.std(0) > 0.05).any():
        warnings.warn(
            "C2ST score variability is larger {0.05}, result may be unreliable."
        )

    # Return the mean over repetitions as c2st score estimate.
    return c2st_scores.mean(0)
