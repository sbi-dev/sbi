from typing import Optional, Union

import torch
from torch import Tensor
from torch.distributions import Distribution

from sbi.inference.posteriors.direct_posterior import DirectPosterior
from sbi.neural_nets.estimators.tabpfn_flow import TabPFNFlow
from sbi.sbi_types import Shape


class FilteredDirectPosterior(DirectPosterior):
    r"""Direct posterior with context kNN selection for TabPFN estimators.

    For every queried condition `x`, this posterior selects nearest-neighbor context
    simulations in embedded condition space and updates the underlying `TabPFNFlow`
    context before delegating to `DirectPosterior` sampling / log-probability logic.
    """

    def __init__(
        self,
        posterior_estimator: TabPFNFlow,
        prior: Distribution,
        full_context_input: Tensor,
        full_context_condition: Tensor,
        max_sampling_batch_size: int = 10_000,
        device: Optional[Union[str, torch.device]] = None,
        x_shape: Optional[torch.Size] = None,
        enable_transform: bool = True,
        context_nn_k: int = 2048,
        context_nn_enabled: bool = True,
    ):
        if context_nn_k <= 0:
            raise ValueError(
                f"context_nn_k must be greater than 0, got {context_nn_k}."
            )

        super().__init__(
            posterior_estimator=posterior_estimator,
            prior=prior,
            max_sampling_batch_size=max_sampling_batch_size,
            device=device,
            x_shape=x_shape,
            enable_transform=enable_transform,
        )

        self.context_nn_k = int(context_nn_k)
        self.context_nn_enabled = bool(context_nn_enabled)
        self._full_context_input, self._full_context_condition = (
            full_context_input,
            full_context_condition,
        )

    def _set_context_for_x_o(self, x_o: Tensor) -> None:

        # TODO potentially add checking

        condition_embedded = self.posterior_estimator.embed_x_o(x_o)
        num_context = self._full_context_condition.shape[0]
        k = min(self.context_nn_k, num_context)

        if k >= num_context:
            self.posterior_estimator.set_context(
                self._full_context_input, self._full_context_condition
            )
            return

        # TODO double check, does this really work??
        distances = torch.cdist(condition_embedded, self._full_context_condition, p=2)
        nn_indices = torch.topk(distances, k=k, largest=False, dim=1).indices
        unique_indices = torch.unique(nn_indices.reshape(-1), sorted=False)

        self.posterior_estimator.set_context(
            self._full_context_input[unique_indices],
            self._full_context_condition[unique_indices],
        )

    def sample(
        self,
        sample_shape: Shape = torch.Size(),
        x: Optional[Tensor] = None,
        max_sampling_batch_size: int = 10_000,
        show_progress_bars: bool = True,
        reject_outside_prior: bool = True,
        max_sampling_time: Optional[float] = None,
        return_partial_on_timeout: bool = False,
    ) -> Tensor:
        x_for_context = self._x_else_default_x(x)
        self._set_context_for_x_o(x_for_context)
        return super().sample(
            sample_shape=sample_shape,
            x=x,
            max_sampling_batch_size=max_sampling_batch_size,
            show_progress_bars=show_progress_bars,
            reject_outside_prior=reject_outside_prior,
            max_sampling_time=max_sampling_time,
            return_partial_on_timeout=return_partial_on_timeout,
        )

    def sample_batched(
        self,
        sample_shape: Shape,
        x: Tensor,
        max_sampling_batch_size: int = 10_000,
        show_progress_bars: bool = True,
        reject_outside_prior: bool = True,
        max_sampling_time: Optional[float] = None,
        return_partial_on_timeout: bool = False,
    ) -> Tensor:
        raise NotImplementedError

    def log_prob(
        self,
        theta: Tensor,
        x: Optional[Tensor] = None,
        norm_posterior: bool = True,
        track_gradients: bool = False,
        leakage_correction_params: Optional[dict] = None,
    ) -> Tensor:
        x_for_context = self._x_else_default_x(x)
        self._set_context_for_x_o(x_for_context)
        return super().log_prob(
            theta=theta,
            x=x,
            norm_posterior=norm_posterior,
            track_gradients=track_gradients,
            leakage_correction_params=leakage_correction_params,
        )

    def log_prob_batched(
        self,
        theta: Tensor,
        x: Tensor,
        norm_posterior: bool = True,
        track_gradients: bool = False,
        leakage_correction_params: Optional[dict] = None,
    ) -> Tensor:
        raise NotImplementedError

    # TODO can we support map?
    def map(
        self,
        x=None,
        num_iter=1000,
        num_to_optimize=100,
        learning_rate=0.01,
        init_method="posterior",
        num_init_samples=1000,
        save_best_every=10,
        show_progress_bars=False,
        force_update=False,
    ):
        raise NotImplementedError
