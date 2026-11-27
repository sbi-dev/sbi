# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

import logging
import math
import time
import warnings
from typing import Callable, Dict, Optional, Tuple

import torch
import torch.distributions.transforms as torch_tf
from torch import Tensor, as_tensor

from sbi.sbi_types import AcceptRejectFn, Proposal, SampleProposal
from sbi.utils.pbar import is_nested, nested_pbar_context, sampling_desc, tqdm
from sbi.utils.sbiutils import gradient_ascent

logger = logging.getLogger(__name__)


def rejection_sample(
    potential_fn: Callable,
    proposal: Proposal,
    theta_transform: Optional[torch_tf.Transform] = None,
    num_samples: int = 1,
    show_progress_bars: bool = False,
    warn_acceptance: float = 0.01,
    max_sampling_batch_size: int = 10_000,
    num_samples_to_find_max: int = 10_000,
    num_iter_to_find_max: int = 100,
    m: float = 1.2,
    max_sampling_time: Optional[float] = None,
    return_partial_on_timeout: bool = False,
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
        max_sampling_time: Optional maximum allowed sampling time (in seconds).
            If this time is exceeded, rejection sampling is aborted and a RuntimeError
            is raised (unless `return_partial_on_timeout=True`). This prevents jobs
            from stalling indefinitely when the acceptance rate is extremely low.
        return_partial_on_timeout: If True and `max_sampling_time` is exceeded, return
            the samples collected so far instead of raising a RuntimeError. A warning
            will be issued indicating the partial return. Default is False.
        device: Device on which to sample.

    Returns:
        Accepted samples and acceptance rate as scalar Tensor.
    """
    if theta_transform is None:
        theta_transform = torch_tf.IndependentTransform(
            torch_tf.identity_transform, reinterpreted_batch_ndims=1
        )

    with nested_pbar_context():
        samples_to_find_max = proposal.sample((num_samples_to_find_max,))

    # Define a potential as the ratio between target distribution and proposal.
    # One value per sample: a 1-D proposal without `Independent` returns (N, 1).
    def potential_over_proposal(theta):
        num_theta = theta.shape[0]
        return potential_fn(theta).reshape(num_theta) - proposal.log_prob(
            theta
        ).reshape(num_theta)

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
        warnings.warn(
            "A value of m < 1.0 will lead to systematically wrong results.",
            stacklevel=2,
        )

    class ScaledProposal:
        """
        Proposal for rejection sampling which is strictly larger than the potential_fn.
        """

        def __init__(self, proposal: Proposal, max_log_ratio: Tensor, log_m: Tensor):
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
            disable=not show_progress_bars or is_nested(),
            total=num_samples,
            desc=sampling_desc(num_samples, "rejection"),
        )

        num_sampled_total, num_remaining = 0, num_samples
        accepted, acceptance_rate = [], float("Nan")
        leakage_warning_raised = False

        # To cover cases with few samples without leakage:
        sampling_batch_size = min(num_samples, max_sampling_batch_size)
        start_time = time.time()
        while num_remaining > 0:
            if (
                max_sampling_time is not None
                and (time.time() - start_time) > max_sampling_time
            ):
                num_collected = sum(s.shape[0] for s in accepted)
                if return_partial_on_timeout and num_collected > 0:
                    pbar.close()
                    warnings.warn(
                        f"Timeout exceeded after collecting {num_collected}/"
                        f"{num_samples} samples. Returning partial results.",
                        stacklevel=2,
                    )
                    samples = torch.cat(accepted)
                    return samples, as_tensor(acceptance_rate)
                msg = (
                    "Sampling aborted early because rejection sampling exceeded "
                    "max_sampling_time."
                )
                if num_sampled_total > 0 and acceptance_rate < warn_acceptance:
                    msg += (
                        " This is likely due to extremely low acceptance. Consider "
                        "switching to MCMC or VI, or checking for model "
                        "misspecification."
                    )
                raise RuntimeError(msg)

            # Sample and reject.
            with nested_pbar_context():
                candidates = proposal.sample(sampling_batch_size).reshape(
                    sampling_batch_size, -1
                )

            target_proposal_ratio = torch.exp(
                potential_fn(candidates).reshape(sampling_batch_size)
                - proposal.log_prob(candidates).reshape(sampling_batch_size)
            )
            uniform_rand = torch.rand(target_proposal_ratio.shape).to(device)
            samples = candidates[target_proposal_ratio > uniform_rand]

            accepted.append(samples)

            # Update.
            num_sampled_total += sampling_batch_size
            pbar.update(min(samples.shape[0], num_remaining))
            num_remaining -= samples.shape[0]

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
                and num_remaining > 0
            ):
                logger.warning(
                    f"Only {acceptance_rate:.3%} proposal samples were accepted. It "
                    f"may take a long time to collect the remaining {num_remaining} "
                    "samples. You can prevent long runtimes by setting "
                    "`max_sampling_time` to limit runtime. Alternatively, consider "
                    "interrupting (Ctrl-C) and switching to a different sampling "
                    "method with `build_posterior(..., sample_with='mcmc')` or "
                    "`build_posterior(..., sample_with='vi')`."
                )
                leakage_warning_raised = True  # Ensure warning is raised just once.

        pbar.close()

        # When in case of leakage a batch size was used there could be too many samples.
        samples = torch.cat(accepted)[:num_samples]
        assert samples.shape[0] == num_samples, (
            "Number of accepted samples must match required samples."
        )

    return samples, as_tensor(acceptance_rate)


@torch.no_grad()
def accept_reject_sample(
    proposal: SampleProposal,
    accept_reject_fn: AcceptRejectFn,
    num_samples: int,
    num_xos: int = 1,
    show_progress_bars: bool = False,
    warn_acceptance: float = 0.01,
    max_sampling_batch_size: int = 10_000,
    proposal_sampling_kwargs: Optional[Dict] = None,
    alternative_method: Optional[str] = None,
    max_sampling_time: Optional[float] = None,
    return_partial_on_timeout: bool = False,
    low_acceptance_hint: Optional[str] = None,
    suggest_sampling_options: bool = True,
    **kwargs,
) -> Tuple[Tensor, Tensor]:
    r"""Returns samples from a proposal according to an acceptance criterion.

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
        proposal: A callable following the `SampleProposal` protocol, i.e., takes
            `sample_shape` as first argument (and kwargs as needed). Returns samples
            from the proposal distribution with shape (*sample_shape, event_dim).
        accept_reject_fn: A callable following the `AcceptRejectFn` protocol that
            evaluates which samples are accepted or rejected. Takes a batch of
            parameters and returns a boolean tensor indicating which are accepted.
        num_samples: Desired number of samples.
        num_xos: Number of conditions for batched_sampling (currently only accepting
            one batch dimension for the condition).
        show_progress_bars: Whether to show a progressbar during sampling.
        warn_acceptance: A minimum acceptance rate under which to warn about slowness.
        max_sampling_batch_size: Batch size for drawing samples from the posterior.
            Takes effect only in the second iteration of the loop below, i.e., in case
            of leakage or `num_samples>max_sampling_batch_size`. Larger batch size
            speeds up sampling.
        proposal_sampling_kwargs: Arguments that are passed to `proposal.sample()`.
        alternative_method: An alternative method for sampling from the restricted
            proposal. E.g., for SNPE, we suggest to sample with MCMC if the rejection
            rate is too high. Used only for printing during a potential warning.
        max_sampling_time: Optional maximum allowed sampling time (in seconds).
            If exceeded, the sampling loop is interrupted and a RuntimeError is raised
            unless `return_partial_on_timeout=True`. This prevents infinite or
            excessively slow rejection sampling runs, e.g. in cases of heavy leakage
            or extremely low acceptance rates.
        return_partial_on_timeout: If True and `max_sampling_time` is exceeded, return
            the samples collected so far instead of raising a RuntimeError. A warning
            will be issued indicating the partial return. Default is False.
        low_acceptance_hint: What a low acceptance rate means for the caller, e.g.
            leakage for NPE. Appended to the low-acceptance warnings, and to the
            timeout error when the acceptance rate is below `warn_acceptance`.
        suggest_sampling_options: Whether the low-acceptance warning and the timeout
            error suggest `max_sampling_time` and `reject_outside_prior=False`. Set it
            to False when the caller does not offer these options.
        kwargs: Absorb additional unused arguments that can be passed to
            `rejection_sample()`. Warn if not empty.

    Returns:
        Accepted samples of shape `(sample_dim, batch_dim, *event_shape)`, and
        acceptance rates for each observation.
    """
    if kwargs:
        logger.warning(
            f"You passed arguments to `rejection_sampling_parameters` that "
            f"are unused when you do not specify a `proposal` in the same "
            f"dictionary. The unused arguments are: {kwargs}"
        )

    # Progress bar can be skipped, e.g. when sampling after each round just for
    # logging.
    if proposal_sampling_kwargs is None:
        proposal_sampling_kwargs = {}

    num_remaining = num_samples

    # NOTE: We might want to change this to a more general approach in the future.
    # Currently limited to a single "batch_dim" for the condition.
    # But this would require giving the method the condition_shape explicitly...
    if "condition" in proposal_sampling_kwargs:
        num_xos = proposal_sampling_kwargs["condition"].shape[0]

    pbar = tqdm(
        disable=not show_progress_bars or is_nested(),
        total=num_samples,
        desc=sampling_desc(num_samples, "rejection", num_xos=num_xos),
    )

    accepted = [[] for _ in range(num_xos)]
    acceptance_rate = torch.full((num_xos,), float("Nan"))
    leakage_warning_raised = False

    # To cover cases with few samples without leakage:
    sampling_batch_size = min(num_samples, max_sampling_batch_size)
    num_sampled_total = torch.zeros(num_xos, dtype=torch.long)
    num_samples_possible = 0
    # Shape of one draw without the sample dim: `(*event_shape)` for flat proposals,
    # `(num_xos, *event_shape)` for batched ones. Set at each draw.
    draw_shape = torch.Size()

    start_time = time.time()
    while num_remaining > 0:
        if (
            max_sampling_time is not None
            and (time.time() - start_time) > max_sampling_time
        ):
            # Check if we have any samples collected
            num_collected = min(
                sum(s.shape[0] for s in accepted[i]) for i in range(num_xos)
            )
            if return_partial_on_timeout and num_collected > 0:
                pbar.close()
                warnings.warn(
                    f"Timeout exceeded after collecting {num_collected}/{num_samples}"
                    f" samples. Returning partial results.",
                    stacklevel=2,
                )
                # Return partial samples with proper shape
                samples = [
                    torch.cat(accepted[i], dim=0)[:num_collected]
                    for i in range(num_xos)
                ]
                samples = torch.stack(samples, dim=1)
                samples = samples.reshape(num_collected, *draw_shape)
                return samples, as_tensor(acceptance_rate, device=samples.device)
            msg = (
                "Sampling aborted early because rejection sampling exceeded "
                "max_sampling_time."
            )
            if (
                num_samples_possible > 0
                and acceptance_rate.min().item() < warn_acceptance
            ):
                msg += " This is likely due to extremely low acceptance."
                if suggest_sampling_options:
                    msg += (
                        " You can disable rejection sampling using "
                        "`reject_outside_prior=False` to draw samples directly from "
                        "the trained estimator."
                    )
                if alternative_method is not None:
                    msg += f" Consider switching to `{alternative_method}`."
                msg += " Low acceptance can also point to model misspecification."
                if low_acceptance_hint is not None:
                    msg += f" {low_acceptance_hint}"
            raise RuntimeError(msg)

        # Sample and reject.
        with nested_pbar_context():
            candidates = proposal(
                torch.Size((sampling_batch_size,)),
                **proposal_sampling_kwargs,
            )
        draw_shape = candidates.shape[1:]
        # SNPE-style rejection-sampling when the proposal is the neural net.
        are_accepted = accept_reject_fn(candidates)

        # Reshape necessary in certain cases which do not follow the shape conventions
        # of the "DensityEstimator" class.
        are_accepted = are_accepted.reshape(sampling_batch_size, num_xos)
        candidates_to_reject = candidates.reshape(
            sampling_batch_size, num_xos, *candidates.shape[candidates.ndim - 1 :]
        )

        for i in range(num_xos):
            num_missing = num_samples - int(num_sampled_total[i])
            if num_missing > 0:
                rows = are_accepted[:, i].nonzero()[:num_missing, 0]
                accepted[i].append(candidates_to_reject[rows, i])

        # Update.
        # Note: For any condition of shape (*batch_shape, *condition_shape), the
        # samples will be of shape(sampling_batch_size,*batch_shape, *event_shape)
        # and hence work in dim = 0.
        num_accepted = are_accepted.sum(dim=0)
        num_sampled_total += num_accepted.to(num_sampled_total.device)
        num_samples_possible += sampling_batch_size
        num_collected = min(int(num_sampled_total.min()), num_samples)
        pbar.update(num_collected - (num_samples - num_remaining))
        num_remaining = num_samples - num_collected

        # To avoid endless sampling when leakage is high, we raise a warning if the
        # acceptance rate is too low after the first 1_000 samples.
        acceptance_rate = num_sampled_total / num_samples_possible
        min_acceptance_rate = acceptance_rate.min().item()

        # For remaining iterations (leakage or many samples) continue
        # sampling with fixed batch size, reduced in cased the number
        # of remaining samples is low. The `max(..., 1e-12)` is to avoid division
        # by zero if acceptance rate is zero.
        sampling_batch_size = min(
            max_sampling_batch_size,
            max(int(1.5 * num_remaining / max(min_acceptance_rate, 1e-12)), 100),
        )
        if (
            num_samples_possible > (sampling_batch_size - 1)
            and min_acceptance_rate < warn_acceptance
            and not leakage_warning_raised
            and num_remaining > 0
        ):
            warn_msg = (
                f"Only {min_acceptance_rate:.3%} proposal samples are accepted. "
                "It may take a long time to collect the remaining "
                f"{num_remaining} samples."
            )
            if suggest_sampling_options:
                warn_msg += (
                    " You can prevent very long runtimes by setting "
                    "`max_sampling_time` to limit runtime, or disabling "
                    "rejection sampling (e.g. via `reject_outside_prior=False` "
                    "in `posterior.sample()` when available)."
                )
            if alternative_method is not None:
                warn_msg += (
                    " Alternatively, consider interrupting (Ctrl-C) and "
                    f"switching to `{alternative_method}`."
                )
            if low_acceptance_hint is not None:
                warn_msg += f" {low_acceptance_hint}"
            logger.warning(warn_msg)

            leakage_warning_raised = True  # Ensure warning is raised just once.

    pbar.close()

    # When in case of leakage a batch size was used there could be too many samples.
    samples = [torch.cat(accepted[i], dim=0)[:num_samples] for i in range(num_xos)]
    samples = torch.stack(samples, dim=1)
    samples = samples.reshape(num_samples, *draw_shape)
    assert samples.shape[0] == num_samples, (
        "Number of accepted samples must match required samples."
    )

    return samples, as_tensor(acceptance_rate, device=samples.device)


@torch.no_grad()
def count_accepted(
    proposal: Callable[[int, Tensor], Tensor],
    accept_reject_fn: AcceptRejectFn,
    num_xos: int,
    num_accepted: int,
    max_num_draws: int,
    max_sampling_batch_size: int = 10_000,
    show_progress_bars: bool = False,
) -> Tuple[Tensor, Tensor]:
    r"""Count accepted proposal draws per observation to estimate acceptance rates.

    For every observation, draws until `num_accepted` draws are accepted or
    `max_num_draws` draws are made. The draws are counted, not kept. Observations go
    in chunks that fit `num_accepted` draws each into one call of `proposal`, and one
    call gets at most `max_sampling_batch_size` draws, so memory does not grow with
    the number of draws or observations.

    Args:
        proposal: Called as `proposal(num_draws, indices)` with an index tensor of
            observations, returns `num_draws` draws for each of them, of shape
            `(num_draws, len(indices), *event_shape)`.
        accept_reject_fn: A callable following the `AcceptRejectFn` protocol that
            returns which draws are accepted.
        num_xos: Number of observations.
        num_accepted: Number of accepted draws after which an observation stops.
        max_num_draws: Number of draws after which an observation stops, whatever it
            accepted.
        max_sampling_batch_size: Maximum number of draws per call of `proposal`.
        show_progress_bars: Whether to show the number of finished observations.

    Returns:
        Number of accepted draws and number of draws, one per observation.
    """
    if not 1 <= num_accepted <= max_num_draws or max_sampling_batch_size < 1:
        raise ValueError(
            f"Cannot collect {num_accepted} accepted draws with at most "
            f"{max_num_draws} draws and {max_sampling_batch_size} draws at once: "
            "`num_rejection_samples` must be at least 1 and at most `max_num_draws`, "
            "and the batch size at least 1."
        )
    accepted = torch.zeros(num_xos, dtype=torch.long, device="cpu")
    drawn = torch.zeros(num_xos, dtype=torch.long, device="cpu")
    pbar = tqdm(
        disable=not show_progress_bars or is_nested(),
        total=num_xos,
        desc=f"Counting accepted draws for {num_xos} observation(s)",
    )
    chunk_size = max(1, max_sampling_batch_size // num_accepted)
    for pending in torch.arange(num_xos, device="cpu").split(chunk_size):
        # Pending observations get the same draws in every step, so one count holds.
        num_drawn = 0
        while pending.numel() > 0:
            # Aim at the slowest pending observation, within the batch and the cap.
            fewest = int(accepted[pending].min())
            num_draws = num_accepted - fewest
            if fewest > 0:
                num_draws = math.ceil(num_draws * num_drawn / fewest)
            elif num_drawn > 0:
                num_draws = 2 * num_drawn
            num_draws = min(
                num_draws,
                max_sampling_batch_size // pending.numel(),
                max_num_draws - num_drawn,
            )
            are_accepted = accept_reject_fn(proposal(num_draws, pending))
            accepted[pending] += are_accepted.reshape(num_draws, -1).sum(dim=0).cpu()
            num_drawn += num_draws

            done = (accepted[pending] >= num_accepted) | (num_drawn >= max_num_draws)
            drawn[pending[done]] = num_drawn
            pbar.update(int(done.sum()))
            pending = pending[~done]
    pbar.close()
    return accepted, drawn
