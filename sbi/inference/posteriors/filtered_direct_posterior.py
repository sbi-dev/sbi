# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

import warnings
from typing import Callable, Literal, Optional, Union

import torch
from torch import Tensor
from torch.distributions import Distribution

from sbi.inference.posteriors.direct_posterior import DirectPosterior
from sbi.neural_nets.estimators.tabpfn_flow import TabPFNFlow
from sbi.sbi_types import Shape

FilterMode = Literal["knn", "first"]
FilterFn = Callable[[Tensor, Tensor, Tensor, int], Tensor]
FilterType = Union[FilterMode, FilterFn]


class FilteredDirectPosterior(DirectPosterior):
    r"""Direct posterior with context filtering for TabPFN estimators.

    For every queried condition `x`, this posterior selects a subset of context
    simulations and updates the underlying `TabPFNFlow` context before delegating to
    `DirectPosterior` sampling / log-probability logic.
    """

    def __init__(
        self,
        estimator: TabPFNFlow,
        prior: Distribution,
        full_context_input: Tensor,
        full_context_condition: Tensor,
        max_sampling_batch_size: int = 10_000,
        device: Optional[Union[str, torch.device]] = None,
        x_shape: Optional[torch.Size] = None,
        enable_transform: bool = True,
        filter_type: FilterType = "knn",
        filter_size: int = 2048,
    ):
        r"""Initialize a direct posterior with observation-dependent context filtering.

        Args:
            estimator: TabPFN-based posterior estimator used for evaluation.
            prior: Prior distribution over parameters.
            full_context_input: Full set of context inputs (typically `theta`).
            full_context_condition: Full set of context conditions (typically `x`).
            max_sampling_batch_size: Maximum number of samples drawn per internal batch.
            device: Device on which posterior computations are performed.
            x_shape: Optional event shape for observations.
            enable_transform: Whether to use unconstrained-space transforms for MAP.
            filter_type: Context filtering strategy. Either `"knn"`, `"first"`,
                or a callable returning selected indices.
            filter_size: Maximum number of context points retained per observation.
        """
        if filter_size <= 1:
            raise ValueError(f"filter_size must be greater than 1, got {filter_size}.")

        super().__init__(
            posterior_estimator=estimator,
            prior=prior,
            max_sampling_batch_size=max_sampling_batch_size,
            device=device,
            x_shape=x_shape,
            enable_transform=enable_transform,
        )

        self.filter_size = int(filter_size)
        self.filtering = filter_type
        self._full_context_input = full_context_input
        self._full_context_condition = full_context_condition
        self._full_context_condition_embedded = estimator.embed(full_context_condition)

    def _validate_filter_indices(self, indices: Tensor, num_context: int) -> Tensor:
        """Validate and normalize context indices returned by a filter."""

        if indices.numel() < 2:
            raise ValueError("Filtering function must return at least two indices.")

        indices = indices.to(device=self._full_context_input.device, dtype=torch.long)
        unique_indices = torch.unique(indices, sorted=False)
        if unique_indices.numel() < indices.numel():
            warnings.warn(
                "Filtering function returned duplicate indices. Duplicates were "
                "removed before setting context.",
                stacklevel=2,
            )

        return unique_indices

    def _select_context_indices(self, condition_embedded: Tensor) -> Tensor:
        """Select context indices according to the configured filtering strategy."""
        num_context = self._full_context_condition_embedded.shape[0]
        k = min(self.filter_size, num_context)

        if k >= num_context:
            return torch.arange(num_context, device=self._full_context_input.device)

        if isinstance(self.filtering, str):
            if self.filtering == "knn":
                indices = _knn_filter_indices(
                    condition_embedded, self._full_context_condition_embedded, k
                )
            elif self.filtering == "first":
                indices = _first_filter_indices(k, self._full_context_input.device)
            else:
                raise RuntimeError(f"Unsupported filtering mode: {self.filtering}")

            return self._validate_filter_indices(indices, num_context)

        indices = self.filtering(
            condition_embedded,
            self._full_context_input,
            self._full_context_condition_embedded,
            k,
        )
        return self._validate_filter_indices(indices, num_context)

    def _set_context_for_x_o(self, x_o: Tensor) -> None:
        """Filter and set estimator context for a single queried observation."""
        condition_embedded = self.posterior_estimator.embed(x_o)
        unique_indices = self._select_context_indices(condition_embedded)

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
        r"""Sample from the posterior after setting context for the queried `x`.

        Args:
            sample_shape: Shape of the returned sample batch.
            x: Observation to condition on. Uses the default observation if `None`.
            max_sampling_batch_size: Maximum internal sampling batch size.
            show_progress_bars: Whether to display progress bars.
            reject_outside_prior: Whether to reject samples outside prior support.
            max_sampling_time: Optional timeout in seconds.
            return_partial_on_timeout: Whether to return collected samples on timeout.

        Returns:
            Samples from the filtered direct posterior.
        """
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
        """Batched sampling is not supported for observation-dependent filtering."""
        raise NotImplementedError(
            "Filtering makes the context observation dependent. "
            "Batched inference requires sharing context, "
            "which is currently not supported."
        )

    def log_prob(
        self,
        theta: Tensor,
        x: Optional[Tensor] = None,
        norm_posterior: bool = True,
        track_gradients: bool = False,
        leakage_correction_params: Optional[dict] = None,
    ) -> Tensor:
        r"""Evaluate posterior log-probability after setting context for `x`.

        Args:
            theta: Parameters at which to evaluate log-probability.
            x: Observation to condition on. Uses the default observation if `None`.
            norm_posterior: Whether to include leakage correction normalization.
            track_gradients: Whether to evaluate with gradient tracking.
            leakage_correction_params: Optional parameters for leakage correction.

        Returns:
            Posterior log-probabilities for ``theta`` conditioned on ``x``.
        """
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
        """Batched log-probability is unsupported with per-observation filtering."""
        raise NotImplementedError(
            "Filtering makes the context observation dependent. "
            "Batched inference requires sharing context, "
            "which is currently not supported."
        )

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
        """MAP is not supported because gradient-based optimization is unavailable."""
        raise NotImplementedError(
            "Computing the MAP requires gradients, which are currently not supported "
            "for NPE-PFN."
        )


def _knn_filter_indices(
    condition_embedded: Tensor,
    full_context_condition: Tensor,
    filter_size: int,
) -> Tensor:
    """Return flattened k-nearest-neighbor context indices."""
    distances = torch.cdist(condition_embedded, full_context_condition, p=2)
    nn_indices = torch.topk(distances, k=filter_size, largest=False, dim=1).indices
    return nn_indices.reshape(-1)


def _first_filter_indices(filter_size: int, device: torch.device) -> Tensor:
    """Return indices of the first `filter_size` context entries."""
    return torch.arange(filter_size, device=device)
