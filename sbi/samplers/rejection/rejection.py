import logging
import warnings
from typing import Any, Callable, Dict, Optional, Tuple, Union

import torch
import torch.distributions.transforms as torch_tf
from torch import Tensor, as_tensor, nn
from torch.distributions import Distribution
from tqdm.auto import tqdm

from sbi.utils.sbiutils import gradient_ascent


def rejection_sample(
    potential_fn: Callable,
    proposal: Any,
    theta_transform: Optional[torch_tf.Transform] = None,
    num_samples: int = 1,
    show_progress_bars: bool = False,
    warn_acceptance: float = 0.01,
    max_sampling_batch_size: int = 10_000,
    num_samples_to_find_max: int = 10_000,
    num_iter_to_find_max: int = 100,
    m: float = 1.2,
    device: str = "cpu",
) -> Tuple[Tensor, Tensor]:
    r"""Return samples from a `potential_fn` obtained via rejection sampling.

    This function uses rejection sampling with samples from posterior in order to
        1) obtain posterior samples within the prior support, and
        2) calculate the fraction of accepted samples as a proxy for correcting the
           density during evaluation of the posterior.

    Args:
        potential_fn: The potential to sample from. The potential should be passed as
            the logarithm of the desired distribution.
        proposal: The proposal from which to draw candidate samples. Must have a
            `sample()` and a `log_prob()` method.
        num_samples: Desired number of samples.
        show_progress_bars: Whether to show a progressbar during sampling.
        warn_acceptance: A minimum acceptance rate under which to warn about slowness.
        max_sampling_batch_size: Batch size for drawing samples from the posterior.
            Takes effect only in the second iteration of the loop below, i.e., in case
            of leakage or `num_samples>max_sampling_batch_size`. Larger batch size
            speeds up sampling.
        num_samples_to_find_max: Number of samples that are used to find the maximum
            of the `potential_fn / proposal` ratio.
        num_iter_to_find_max: Number of gradient ascent iterations to find the maximum
            of the `potential_fn / proposal` ratio.
        m: Multiplier to the maximum ratio between potential function and the
            proposal. This factor is applied after already having scaled the proposal
            with the maximum ratio of the `potential_fn / proposal` ratio. A higher
            value will ensure that the samples are indeed from the correct
            distribution, but will increase the fraction of rejected samples and thus
            computation time.
        device: Device on which to sample.

    Returns:
        Accepted samples and acceptance rate as scalar Tensor.
    """
    if theta_transform is None:
        theta_transform = torch_tf.IndependentTransform(
            torch_tf.identity_transform, reinterpreted_batch_ndims=1
        )

    samples_to_find_max = proposal.sample((num_samples_to_find_max,))

    # Define a potential as the ratio between target distribution and proposal.
    def potential_over_proposal(theta):
        return potential_fn(theta) - proposal.log_prob(theta)

    # Search for the maximum of the ratio.
    _, max_log_ratio = gradient_ascent(
        potential_fn=potential_over_proposal,
        inits=samples_to_find_max,
        theta_transform=theta_transform,
        num_iter=num_iter_to_find_max,
        learning_rate=0.01,
        num_to_optimize=max(1, int(num_samples_to_find_max / 10)),
        show_progress_bars=False,
    )

    if m < 1.0:
        warnings.warn("A value of m < 1.0 will lead to systematically wrong results.")

    class ScaledProposal:
        """
        Proposal for rejection sampling which is strictly larger than the potential_fn.
        """

        def __init__(self, proposal: Any, max_log_ratio: Tensor, log_m: Tensor):
            self.proposal = proposal
            self.max_log_ratio = max_log_ratio
            self.log_m = log_m

        def sample(self, sample_shape: torch.Size, **kwargs) -> Tensor:
            """
            Samples from the `ScaledProposal` are samples from the `proposal`.
            """
            return self.proposal.sample((sample_shape,), **kwargs)

        def log_prob(self, theta: Tensor, **kwargs) -> Tensor:
            """
            The log-prob is scaled such that the proposal is always above the potential.
            """
            return (
                self.proposal.log_prob(theta, **kwargs)
                + self.max_log_ratio
                + self.log_m
            )

    proposal = ScaledProposal(proposal, max_log_ratio, torch.log(torch.as_tensor(m)))

    with torch.no_grad():
        # Progress bar can be skipped, e.g. when sampling after each round just for
        # logging.
        pbar = tqdm(
            disable=not show_progress_bars,
            total=num_samples,
            desc=f"Drawing {num_samples} posterior samples",
        )

        num_sampled_total, num_remaining = 0, num_samples
        accepted, acceptance_rate = [], float("Nan")
        leakage_warning_raised = False

        # To cover cases with few samples without leakage:
        sampling_batch_size = min(num_samples, max_sampling_batch_size)
        while num_remaining > 0:

            # Sample and reject.
            candidates = proposal.sample(sampling_batch_size).reshape(
                sampling_batch_size, -1
            )

            target_proposal_ratio = torch.exp(
                potential_fn(candidates) - proposal.log_prob(candidates)
            )
            uniform_rand = torch.rand(target_proposal_ratio.shape).to(device)
            samples = candidates[target_proposal_ratio > uniform_rand]

            accepted.append(samples)

            # Update.
            num_sampled_total += sampling_batch_size
            num_remaining -= samples.shape[0]
            pbar.update(samples.shape[0])

            # To avoid endless sampling when leakage is high, we raise a warning if the
            # acceptance rate is too low after the first 1_000 samples.
            acceptance_rate = (num_samples - num_remaining) / num_sampled_total

            # For remaining iterations (leakage or many samples) continue
            # sampling with fixed batch size, reduced in cased the number
            # of remaining samples is low. The `max(..., 1e-12)` is to avoid division
            # by zero if acceptance rate is zero.
            sampling_batch_size = min(
                max_sampling_batch_size,
                max(int(1.5 * num_remaining / max(acceptance_rate, 1e-12)), 100),
            )
            if (
                num_sampled_total > 1000
                and acceptance_rate < warn_acceptance
                and not leakage_warning_raised
            ):
                logging.warning(
                    f"""Only {acceptance_rate:.3%} proposal samples were accepted. It
                        may take a long time to collect the remaining {num_remaining}
                        samples. Consider interrupting (Ctrl-C) and switching to a
                        different sampling method with
                        `build_posterior(..., sample_with='mcmc')`. or
                        `build_posterior(..., sample_with='vi')`."""
                )
                leakage_warning_raised = True  # Ensure warning is raised just once.

        pbar.close()

        # When in case of leakage a batch size was used there could be too many samples.
        samples = torch.cat(accepted)[:num_samples]
        assert (
            samples.shape[0] == num_samples
        ), "Number of accepted samples must match required samples."

    return samples, as_tensor(acceptance_rate)


@torch.no_grad()
def accept_reject_sample(
    proposal: Union[nn.Module, Distribution],
    accept_reject_fn: Callable,
    num_samples: int,
    show_progress_bars: bool = False,
    warn_acceptance: float = 0.01,
    sample_for_correction_factor: bool = False,
    max_sampling_batch_size: int = 10_000,
    proposal_sampling_kwargs: Dict = {},
    alternative_method: Optional[str] = None,
    **kwargs,
) -> Tuple[Tensor, Tensor]:
    r"""Returns samples from a proposal according to a acception criterion.

    This is relevant for snpe methods and flows for which the posterior tends to have
    mass outside the prior support.

    This function could in principle be integrated into `rejection_sample()`. However,
    to keep the warnings clean, to avoid additional code for integration, and confusing
    if-cases, we decided to keep two separate functions.

    This function uses rejection sampling with samples from posterior in order to
        1) obtain posterior samples within the prior support, and
        2) calculate the fraction of accepted samples as a proxy for correcting the
           density during evaluation of the posterior.

    Args:
        posterior_nn: Neural net representing the posterior.
        accept_reject_fn: Function that evaluatuates which samples are accepted or
            rejected. Must take a batch of parameters and return a boolean tensor which
            indicates which parameters get accepted.
        num_samples: Desired number of samples.
        show_progress_bars: Whether to show a progressbar during sampling.
        warn_acceptance: A minimum acceptance rate under which to warn about slowness.
        sample_for_correction_factor: True if this function was called by
            `leakage_correction()`. False otherwise. Will be used to adapt the leakage
             warning and to decide whether we have to search for the maximum.
        max_sampling_batch_size: Batch size for drawing samples from the posterior.
            Takes effect only in the second iteration of the loop below, i.e., in case
            of leakage or `num_samples>max_sampling_batch_size`. Larger batch size
            speeds up sampling.
        proposal_sampling_kwargs: Arguments that are passed to `proposal.sample()`.
        alternative_method: An alternative method for sampling from the restricted
            proposal. E.g., for SNPE, we suggest to sample with MCMC if the rejection
            rate is too high. Used only for printing during a potential warning.
        kwargs: Absorb additional unused arguments that can be passed to
            `rejection_sample()`. Warn if not empty.

    Returns:
        Accepted samples and acceptance rate as scalar Tensor.
    """

    if kwargs:
        logging.warn(
            f"You passed arguments to `rejection_sampling_parameters` that "
            f"are unused when you do not specify a `proposal` in the same "
            f"dictionary. The unused arguments are: {kwargs}"
        )

    # Progress bar can be skipped, e.g. when sampling after each round just for
    # logging.
    pbar = tqdm(
        disable=not show_progress_bars,
        total=num_samples,
        desc=f"Drawing {num_samples} posterior samples",
    )

    num_sampled_total, num_remaining = 0, num_samples
    accepted, acceptance_rate = [], float("Nan")
    leakage_warning_raised = False

    # To cover cases with few samples without leakage:
    sampling_batch_size = min(num_samples, max_sampling_batch_size)
    while num_remaining > 0:

        # Sample and reject.
        # This if-case is annoying, but it will be gone when we move away from
        # nflows and towards a flows-framework which takes a tuple as sample_size.
        if isinstance(proposal, nn.Module):
            candidates = proposal.sample(
                sampling_batch_size, **proposal_sampling_kwargs  # type: ignore
            ).reshape(sampling_batch_size, -1)
        else:
            candidates = proposal.sample(
                (sampling_batch_size,), **proposal_sampling_kwargs  # type: ignore
            )  # type: ignore

        # SNPE-style rejection-sampling when the proposal is the neural net.
        are_accepted = accept_reject_fn(candidates)

        samples = candidates[are_accepted]

        accepted.append(samples)

        # Update.
        num_sampled_total += sampling_batch_size
        num_remaining -= samples.shape[0]
        pbar.update(samples.shape[0])

        # To avoid endless sampling when leakage is high, we raise a warning if the
        # acceptance rate is too low after the first 1_000 samples.
        acceptance_rate = (num_samples - num_remaining) / num_sampled_total

        # For remaining iterations (leakage or many samples) continue
        # sampling with fixed batch size, reduced in cased the number
        # of remaining samples is low. The `max(..., 1e-12)` is to avoid division
        # by zero if acceptance rate is zero.
        sampling_batch_size = min(
            max_sampling_batch_size,
            max(int(1.5 * num_remaining / max(acceptance_rate, 1e-12)), 100),
        )
        if (
            num_sampled_total > 1000
            and acceptance_rate < warn_acceptance
            and not leakage_warning_raised
        ):
            if sample_for_correction_factor:
                logging.warning(
                    f"""Drawing samples from posterior to estimate the normalizing
                        constant for `log_prob()`. However, only
                        {acceptance_rate:.3%} posterior samples are within the
                        prior support. It may take a long time to collect the
                        remaining {num_remaining} samples.
                        Consider interrupting (Ctrl-C) and either basing the
                        estimate of the normalizing constant on fewer samples (by
                        calling `posterior.leakage_correction(x_o,
                        num_rejection_samples=N)`, where `N` is the number of
                        samples you want to base the
                        estimate on (default N=10000), or not estimating the
                        normalizing constant at all
                        (`log_prob(..., norm_posterior=False)`. The latter will
                        result in an unnormalized `log_prob()`."""
                )
            else:
                warn_msg = f"""Only {acceptance_rate:.3%} proposal samples are
                    accepted. It may take a long time to collect the remaining
                    {num_remaining} samples. """
                if alternative_method is not None:
                    warn_msg += f"""Consider interrupting (Ctrl-C) and switching to
                    `{alternative_method}`."""
                logging.warning(warn_msg)

            leakage_warning_raised = True  # Ensure warning is raised just once.

    pbar.close()

    # When in case of leakage a batch size was used there could be too many samples.
    samples = torch.cat(accepted)[:num_samples]
    assert (
        samples.shape[0] == num_samples
    ), "Number of accepted samples must match required samples."

    return samples, as_tensor(acceptance_rate)
