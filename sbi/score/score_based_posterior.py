# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Affero General Public License v3, see <https://www.gnu.org/licenses/>.

from functools import partial
from typing import Any, Callable, Dict, List, Optional, Union
from warnings import warn

import numpy as np
import torch
from torch import Tensor, nn


from sbi.inference.posteriors.base_posterior import NeuralPosterior
from sbi.types import Shape
from sbi.utils import del_entries, optimize_potential_fn, rejection_sample
from sbi.utils.torchutils import ScalarFloat, atleast_2d, ensure_theta_batched

from score_sampling import get_sampling_fn 
from score_log_prob import get_likelihood_fn


class ScoreBasedPosterior(NeuralPosterior):
    r"""TODO
    """

    def __init__(
        self,
        method_family: str,
        neural_net: nn.Module,
        prior,
        sde,
        x_shape: torch.Size,
        sample_with: str = "mcmc",
        mcmc_method: str = "slice_np",
        mcmc_parameters: Optional[Dict[str, Any]] = None,
        rejection_sampling_parameters: Optional[Dict[str, Any]] = None,
        vi_parameters: Optional[Dict[str, Any]] = None,
        device: str = "cpu",
    ):
        """
        Args:
            method_family: One of snpe, snl, snre_a, snre_b or sspe.
            neural_net: A classifier for SNRE, a density estimator for SNPE and SNL, a
            score net for SSPE.
            prior: Prior distribution with `.log_prob()` and `.sample()`.
            x_shape: Shape of the simulated data. It can differ from the
                observed data the posterior is conditioned on later in the batch
                dimension. If it differs, the additional entries are interpreted as
                independent and identically distributed data / trials. I.e., the data is
                assumed to be generated based on the same (unknown) model parameters or
                experimental condations.
            sample_with: OTHER SAMPLING FUNCTIONS
            device: Training device, e.g., cpu or cuda:0.
        """
        self.sde = sde
        kwargs = del_entries(locals(), entries=("self", "__class__", "sde"))
        super().__init__(**kwargs)

    def log_prob(
        self, theta: Tensor, x: Optional[Tensor] = None, log_prob_fn_type="ode_logprob",track_gradients: bool = False
    ) -> Tensor:
        r"""Returns the log-probability of $p(x|\theta) \cdot p(\theta).$

        This is a unbiased estimate of the log probability when using the probability
        flow formulation.

        Args:
            theta: Parameters $\theta$.
            x: Conditioning context for posterior $p(\theta|x)$. If not provided,
                fall back onto `x` passed to `set_default_x()`.
            track_gradients: Whether the returned tensor supports tracking gradients.
                This can be helpful for e.g. sensitivity analysis, but increases memory
                consumption.

        Returns:
            `(len(θ),)`-shaped log-probability $\log(p(x|\theta) \cdot p(\theta))$.

        """

        # TODO Train exited here, entered after sampling?
        self.net.eval()

        theta, x = self._prepare_theta_and_x_for_log_prob_(theta, x)

        if log_prob_fn_type == "ode_logprob":
            log_prob_fn = get_likelihood_fn(self.sde)
            return log_prob_fn(self.net, theta, context=x)
        else:
            raise NotImplementedError("To implement")


    def sample(
        self,
        sample_shape: Shape = torch.Size(),
        x: Optional[Tensor] = None,
        show_progress_bars: bool = True,
        sample_with: Optional[str] = None,
        mcmc_method: Optional[str] = None,
        sde_parameters: Optional[Dict[str,Any]] = dict(),
        mcmc_parameters: Optional[Dict[str, Any]] = None,
        rejection_sampling_parameters: Optional[Dict[str, Any]] = None,
        vi_parameters: Optional[Dict[str, Any]] = None,

    ) -> Tensor:
        r"""
        Return samples from posterior distribution $p(\theta|x)$ with MCMC.

        Args:
            sample_shape: Desired shape of samples that are drawn from posterior. If
                sample_shape is multidimensional we simply draw `sample_shape.numel()`
                samples and then reshape into the desired shape.
            x: Conditioning context for posterior $p(\theta|x)$. If not provided,
                fall back onto `x` passed to `set_default_x()`.
            show_progress_bars: Whether to show sampling progress monitor.
            sample_with: Method to use for sampling from the posterior. Must be one of
                [`mcmc` | `rejection`].
            mcmc_method: Optional parameter to override `self.mcmc_method`.
            mcmc_parameters: Dictionary overriding the default parameters for MCMC.
                The following parameters are supported: `thin` to set the thinning
                factor for the chain, `warmup_steps` to set the initial number of
                samples to discard, `num_chains` for the number of chains,
                `init_strategy` for the initialisation strategy for chains; `prior`
                will draw init locations from prior, whereas `sir` will use Sequential-
                Importance-Resampling using `init_strategy_num_candidates` to find init
                locations.
            rejection_sampling_parameters: Dictionary overriding the default parameters
                for rejection sampling. The following parameters are supported:
                `proposal` as the proposal distribtution (default is the prior).
                `max_sampling_batch_size` as the batchsize of samples being drawn from
                the proposal at every iteration. `num_samples_to_find_max` as the
                number of samples that are used to find the maximum of the
                `potential_fn / proposal` ratio. `num_iter_to_find_max` as the number
                of gradient ascent iterations to find the maximum of that ratio. `m` as
                multiplier to that ratio.
            vi_parameters: TODO WRITE

        Returns:
            Samples from posterior.
        """

        self.net.eval()

        sample_with = sample_with if sample_with is not None else self._sample_with
        x, num_samples = self._prepare_for_sample(x, sample_shape)

        if sample_with == "sde":
            if "sampler_name" in sde_parameters:
                sampler_name = sde_parameters.pop("sampler_name")
            else:
                sampler_name="ode_sampler"
            sampling_fn = get_sampling_fn(sampler_name,self.sde)
            return sampling_fn(self.net, sample_shape, x, **sde_parameters)
        else:
            raise NotImplementedError("To implement")

    @property
    def _num_trained_rounds(self) -> int:
        return self._trained_rounds

    @_num_trained_rounds.setter
    def _num_trained_rounds(self, trained_rounds: int) -> None:
        """
        Sets the number of trained rounds and updates the purpose.

        When the number of trained rounds is 1 and the algorithm is SNRE_A, then the
        log_prob will be normalized, as specified in the purpose.

        The reason we made this a property is that the purpose gets updated
        automatically whenever the number of rounds is updated.
        """
        self._trained_rounds = trained_rounds

        normalized_or_not = (
            ""
            if (self._method_family == "snre_a" and self._trained_rounds == 1)
            else "_unnormalized_ "
        )
        self._purpose = (
            f"It provides MCMC to .sample() from the posterior and "
            f"can evaluate the {normalized_or_not}posterior density with .log_prob()."
        )

    @staticmethod
    def _score_over_trials(
        x: Tensor,
        theta: Tensor,
        net: nn.Module,
        track_gradients: bool = False,
    ) -> Tensor:
        r"""Return log ratios summed over iid trials of `x`.

        Note: `x` can be a batch with batch size larger 1. Batches in x are assumed to
        be iid trials, i.e., data generated based on the same paramters / experimental
        conditions.

        Repeats `x` and $\theta$ to cover all their combinations of batch entries.

        Args:
            x: batch of iid data.
            theta: batch of parameters
            net: neural net representing the classifier to approximate the ratio.
            track_gradients: Whether to track gradients.

        Returns:
            log_ratio_trial_sum: log ratio for each parameter, summed over all
                batch entries (iid trials) in `x`.
        """

        theta_repeated, x_repeated = NeuralPosterior._match_theta_and_x_batch_shapes(
            theta=theta, x=atleast_2d(x)
        )
        assert (
            x_repeated.shape[0] == theta_repeated.shape[0]
        ), "x and theta must match in batch shape."
        assert (
            next(net.parameters()).device == x.device and x.device == theta.device
        ), f"device mismatch: net, x, theta: {next(net.parameters()).device}, {x.device}, {theta.device}."

        # Calculate ratios in one batch.
        with torch.set_grad_enabled(track_gradients):
            score_trial_batch = net([theta_repeated, x_repeated])
            # Reshape to (x-trials x parameters), sum over trial-log likelihoods.
            score_trial_batch_sum = score_trial_batch.reshape(x.shape[0], -1).sum(0)

        return score_trial_batch

