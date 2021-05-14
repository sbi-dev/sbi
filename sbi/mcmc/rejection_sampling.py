from typing import Any

import torch
from torch import Tensor, exp, log, rand

from sbi.utils import optimize_potential_fn


def rejection_sample(
        num_samples: torch.Size,
        potential_fn: Any,
        proposal: Any,
        num_samples_to_find_max: int = 10_000,
        m: float = 2.0,
        sampling_batch_size: int = 10_000
    ):
        r"""
        Return samples from a distribution via rejection sampling.

        This function is used in any case by SNLE and SNRE, but can also be used by SNPE
        in order to deal with strong leakage. Depending on the inference method, a
        different potential function for the rejection sampler is required.

        Args:
            num_samples: Desired number of samples.
            potential_fn: Potential function used for rejection sampling.
            proposal: Proposal distribution for rejection sampling.
            num_samples_to_find_max: Number of samples that are used to find the maximum 
                of the `potential_fn / proposal` ratio. 
            m: Multiplier to the maximum ratio between potential function and the 
                proposal. A higher value will ensure that the samples are indeed from 
                the posterior, but will increase the rejection ratio and thus 
                computation time.
            sampling_batch_size: Batchsize of samples being drawn from 
                the proposal at every iteration.

        Returns:
            Tensor of shape (num_samples, shape_of_single_theta).
        """

        find_max = proposal.sample((num_samples_to_find_max,))

        def potential_over_proposal(theta):
            return torch.squeeze(potential_fn(find_max)) - proposal.log_prob(find_max)

        _, max_ratio = optimize_potential_fn(
            potential_fn=potential_over_proposal, 
            inits=find_max, 
            dist_specifying_bounds=proposal,
            num_iter=100,
            learning_rate=0.01,
            num_to_optimize=min(1, int(num_samples_to_find_max / 10)), 
            show_progress_bars=False,
        )

        num_accepted = 0
        all_ = []
        while num_accepted < num_samples:
            candidates = proposal.sample((sampling_batch_size,))
            target_log_probs = torch.squeeze(potential_fn(candidates))
            proposal_log_probs = proposal.log_prob(candidates) + max_ratio
            target_proposal_ratio = exp(target_log_probs - proposal_log_probs)
            acceptance = rand(target_proposal_ratio.shape)
            accepted = candidates[torch.squeeze(target_proposal_ratio) > torch.squeeze(acceptance)]
            num_accepted += accepted.shape[0]
            all_.append(accepted)
        samples = torch.cat(all_)[:num_samples]
        return samples
