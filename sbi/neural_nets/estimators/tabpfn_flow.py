# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

import logging
import warnings
from pathlib import Path
from typing import Mapping, Optional, Tuple, Union

import torch
from torch import Tensor, nn

try:
    from tabpfn import TabPFNRegressor
    from tabpfn.model_loading import prepend_cache_path

    _TABPFN_AVAILABLE = True
except ImportError:
    _TABPFN_AVAILABLE = False

from sbi.neural_nets.estimators.base import ConditionalDensityEstimator
from sbi.neural_nets.estimators.shape_handling import reshape_to_batch_event
from sbi.sbi_types import Shape

_HAS_LOGGED_TABPFN_LICENSE: bool = False
logger = logging.getLogger(__name__)


class TabPFNFlow(ConditionalDensityEstimator):
    r"""TabPFN-based conditional density estimator via autoregressive evaluation.

    This estimator keeps a context dataset and evaluates
    :math:`p(\theta \mid x)` by autoregressive factorization over dimensions of
    :math:`\theta`.
    """

    def __init__(
        self,
        input_shape: torch.Size,
        condition_shape: torch.Size,
        embedding_net: Optional[nn.Module] = None,
        regressor_init_kwargs: Optional[Mapping] = None,
        max_context_size: int = 10_000,
    ) -> None:
        r"""Initialize a TabPFN-based conditional density estimator.

        Args:
            input_shape: Event shape of the modeled input variable.
            condition_shape: Event shape of the conditioning variable.
            embedding_net: Optional embedding module applied to conditions before
                TabPFN regression. Defaults to identity.
            regressor_init_kwargs: Optional keyword arguments forwarded to
                `TabPFNRegressor`. Defaults to TabPFN v2. The `model_path` key
                accepts version shorthands ``"v2"`` / ``"2"`` / ``"v2.0"``
                (default), ``"v2.5"`` / ``"2.5"``, ``"v2.6"`` / ``"2.6"``, full
                checkpoint filenames, or ``"auto"`` for the latest version.
            max_context_size: Maximum number of context pairs that can be stored.
        """
        if not _TABPFN_AVAILABLE:
            raise ImportError(
                "TabPFN is required for TabPFNFlow but is not installed. "
                "Install it with: pip install 'sbi[tabpfn]'"
            )
        super().__init__(
            net=nn.Identity(),
            input_shape=input_shape,
            condition_shape=condition_shape,
        )
        # Anchor buffer: gives infer_module_device a device to find
        self.register_buffer('_device_anchor', torch.zeros(1))

        self._embedding_net = (
            embedding_net if embedding_net is not None else nn.Identity()
        )
        self._regressor_init_kwargs = dict(regressor_init_kwargs or {})
        self._regressor_init_kwargs["model_path"] = self._resolve_model_path(
            self._regressor_init_kwargs.get("model_path", "v2")
        )
        self._model = TabPFNRegressor(**self._regressor_init_kwargs)
        # Log license information once per process when a TabPFN model is initialized.
        self._log_license(self._model.model_path)

        self.max_context_size = int(max_context_size)
        self._warn_if_context_exceeds_recommended(
            self._model.model_path, self.max_context_size
        )
        self._input_numel = int(torch.Size(input_shape).numel())
        # Plain CPU tensors — not buffers. TabPFN's numpy API requires CPU, so these
        # must never be moved with the module to another device.
        self._context_input: Optional[Tensor] = None
        self._context_condition: Optional[Tensor] = None

    @property
    def embedding_net(self) -> nn.Module:
        r"""Return the embedding network."""
        return self._embedding_net

    def to(self, device: Union[str, torch.device]) -> "TabPFNFlow":
        """Move the module to `device` in place.

        Moves the embedding net and device anchor to `device` and updates
        TabPFNRegressor's target device so subsequent fit() calls run on the
        same device.

        Note: Context tensors (_context_input, _context_condition) remain on
        CPU as required by TabPFN's numpy API.
        """
        super().to(device)
        # Sync TabPFNRegressor so its next fit() call uses the correct device.
        self._model.device = device
        return self

    def set_context(
        self, input_context: Tensor, condition_context: Tensor
    ) -> "TabPFNFlow":
        r"""Set the context dataset used by the TabPFN autoregressive model.

        Args:
            input_context: Context inputs of shape `(context_batch, *input_shape)`.
            condition_context: Context conditions of shape
                `(context_batch, *condition_shape)`.
        """
        input_context = reshape_to_batch_event(
            input_context, event_shape=self.input_shape
        )
        condition_context = reshape_to_batch_event(
            condition_context, event_shape=self.condition_shape
        )

        embedded_condition = self._embed_condition(condition_context)

        self.set_context_flat(
            input_context_flat=input_context.reshape(input_context.shape[0], -1).cpu(),
            condition_context_flat=embedded_condition.reshape(
                embedded_condition.shape[0], -1
            ).cpu(),
        )
        return self

    def _require_context(self) -> Tuple[Tensor, Tensor]:
        """Return stored context tensors or raise if no context is available."""
        if self._context_input is None or self._context_condition is None:
            raise RuntimeError(
                "No context is set. "
                "Call `set_context(input_context, condition_context)` "
                "before calling `sample` or `log_prob`."
            )

        return self._context_input, self._context_condition

    def _embed_condition(self, condition: Tensor) -> Tensor:
        """Validate, embed, flatten, and move conditions to CPU."""
        self._check_condition_shape(condition)
        with torch.no_grad():
            embedded = self._embedding_net(condition)
        return embedded.reshape(embedded.shape[0], -1).cpu()

    def embed(self, condition: Tensor) -> Tensor:
        r"""Public wrapper for preparing embedded, flattened conditions."""
        return self._embed_condition(condition)

    def set_context_flat(
        self,
        input_context_flat: Tensor,
        condition_context_flat: Tensor,
    ) -> "TabPFNFlow":
        r"""Set flattened context directly.

        Both tensors must be 2D and are always stored on CPU regardless of input
        device, because TabPFN's numpy API requires CPU tensors.

        Args:
            input_context_flat: 2D tensor of shape `(context_batch, input_numel)`.
            condition_context_flat: 2D tensor of shape
                `(context_batch, condition_embed_numel)`.
        """

        if input_context_flat.shape[0] > self.max_context_size:
            raise ValueError(
                "Context batch size exceeds the configured maximum in `set_context`: "
                f"got {input_context_flat.shape[0]}, maximum is "
                f"{self.max_context_size}."
            )

        if input_context_flat.ndim != 2 or condition_context_flat.ndim != 2:
            raise ValueError(
                "Expected 2D flattened context tensors for input and condition, "
                "but got "
                f"shapes {tuple(input_context_flat.shape)} and "
                f"{tuple(condition_context_flat.shape)}."
            )

        if input_context_flat.shape[0] != condition_context_flat.shape[0]:
            raise ValueError(
                "Context input and condition must have the same batch dimension, but "
                f"got {input_context_flat.shape[0]} and "
                f"{condition_context_flat.shape[0]}."
            )

        if input_context_flat.shape[1] != self._input_numel:
            raise ValueError(
                "Expected flattened input context with second dimension "
                f"{self._input_numel}, "
                f"but got {input_context_flat.shape[1]}."
            )

        self._context_input = input_context_flat.to("cpu")
        self._context_condition = condition_context_flat.to("cpu")
        return self

    def _autoregressive_log_prob(
        self, input_flat: Tensor, condition_flat: Tensor, eps: float = 1e-15
    ) -> Tensor:
        r"""Evaluate autoregressive log probability for flattened inputs.

        Args:
            input_flat: Flattened input tensor of shape `(batch, input_numel)`.
            condition_flat: Flattened embedded conditions of shape
                `(batch, condition_embed_numel)`.
            eps: Small positive value used to replace `-inf` log-probabilities.

        Returns:
            Log probabilities of shape `(batch,)`.
        """
        context_input, context_condition = self._require_context()
        joint_data = torch.cat([context_condition, context_input], dim=1)

        dim_condition = context_condition.shape[1]
        log_prob = torch.zeros(input_flat.shape[0])
        test_joint = torch.cat([condition_flat, input_flat], dim=1)
        log_eps = torch.log(torch.tensor(eps))

        for dim_idx in range(self._input_numel):
            features_end = dim_condition + dim_idx
            target_idx = dim_condition + dim_idx

            self._model.fit(joint_data[:, :features_end], joint_data[:, target_idx])
            pred_dist = self._model.predict(
                test_joint[:, :features_end], output_type="full", quantiles=[]
            )

            bar_dist = pred_dist["criterion"]
            dim_log_prob = -bar_dist(
                pred_dist["logits"].to(bar_dist.borders.device),
                test_joint[:, target_idx].to(bar_dist.borders.device),
            )
            dim_log_prob = dim_log_prob.to("cpu")

            dim_log_prob = torch.where(
                dim_log_prob == float("-inf"),
                log_eps,
                dim_log_prob,
            )
            log_prob += dim_log_prob

        return log_prob

    def _autoregressive_sample(
        self, condition_flat: Tensor, with_log_prob: bool = False, eps: float = 1e-15
    ) -> tuple[Tensor, Tensor]:
        r"""Draw autoregressive samples for flattened embedded conditions.

        Args:
            condition_flat: Flattened embedded conditions of shape
                `(batch, condition_embed_numel)`.
            with_log_prob: Whether to accumulate and return sample log-probabilities.
            eps: Small positive value used to replace `-inf` log-probabilities.

        Returns:
            A tuple `(samples_flat, log_probs)` where `samples_flat` has shape
            `(batch, input_numel)` and `log_probs` is `None` unless
            `with_log_prob=True`.
        """
        context_input, context_condition = self._require_context()
        joint_data = torch.cat([context_condition, context_input], dim=1)

        dim_condition = context_condition.shape[1]
        autoregressive_inputs = condition_flat
        log_prob = torch.zeros(condition_flat.shape[0]) if with_log_prob else None
        log_eps = torch.log(torch.tensor(eps))

        for dim_idx in range(self._input_numel):
            features_end = dim_condition + dim_idx
            target_idx = dim_condition + dim_idx

            self._model.fit(joint_data[:, :features_end], joint_data[:, target_idx])
            pred_dist = self._model.predict(
                autoregressive_inputs[:, :features_end],
                output_type="full",
                quantiles=[],
            )

            sampled_dim = pred_dist["criterion"].sample(pred_dist["logits"])

            if with_log_prob and log_prob is not None:
                bar_dist = pred_dist["criterion"]
                dim_log_prob = -bar_dist(
                    pred_dist["logits"].to(bar_dist.borders.device),
                    sampled_dim.to(bar_dist.borders.device),
                )
                dim_log_prob = dim_log_prob.to("cpu")

                dim_log_prob = torch.where(
                    dim_log_prob == float("-inf"),
                    log_eps,
                    dim_log_prob,
                )
                log_prob += dim_log_prob

            autoregressive_inputs = torch.cat(
                [autoregressive_inputs, sampled_dim[:, None]], dim=1
            )

        return autoregressive_inputs[:, dim_condition:], log_prob

    def log_prob(self, input: Tensor, condition: Tensor, eps: float = 1e-15) -> Tensor:
        r"""Return log probabilities of input given condition.

        Args:
            input: Tensor of shape `(sample_dim, batch_dim, *input_shape)`.
            condition: Tensor of shape `(batch_dim, *condition_shape)`.

        Returns:
            Tensor of shape `(sample_dim, batch_dim)`.
        """

        self._check_input_shape(input)
        condition_flat = self._embed_condition(condition)

        input_sample_dim = input.shape[0]
        input_batch_dim = input.shape[1]
        condition_batch_dim = condition.shape[0]

        if input_batch_dim != condition_batch_dim:
            raise ValueError(
                f"Batch shape of condition ({condition_batch_dim}) and input "
                f"({input_batch_dim}) do not match."
            )

        input_flat = input.reshape(input_sample_dim * input_batch_dim, -1)
        repeated_condition = condition_flat.repeat(input_sample_dim, 1)

        log_probs_flat = self._autoregressive_log_prob(
            input_flat=input_flat, condition_flat=repeated_condition, eps=eps
        )
        log_probs = log_probs_flat.reshape(input_sample_dim, input_batch_dim)

        return log_probs.to(device=input.device)

    def sample(
        self, sample_shape: Shape, condition: Tensor, eps: float = 1e-15
    ) -> Tensor:
        r"""Return samples from the conditional density estimator.

        Args:
            sample_shape: Shape of the samples to return.
            condition: Conditions of shape `(batch_dim, *condition_shape)`.

        Returns:
            Samples of shape `(*sample_shape, batch_dim, *input_shape)`.
        """
        sample_shape = torch.Size(sample_shape)
        num_samples = sample_shape.numel()
        condition_flat = self._embed_condition(condition)
        batch_dim = condition.shape[0]

        repeated_condition = condition_flat.repeat(num_samples, 1)

        samples_flat, _ = self._autoregressive_sample(
            repeated_condition, with_log_prob=False, eps=eps
        )

        samples = samples_flat.reshape((*sample_shape, batch_dim, *self.input_shape))
        return samples.to(device=condition.device)

    def sample_and_log_prob(
        self, sample_shape: torch.Size, condition: Tensor, eps: float = 1e-15, **kwargs
    ) -> Tuple[Tensor, Tensor]:
        r"""Return samples and corresponding log probabilities.

        Args:
            sample_shape: Shape of the samples to return.
            condition: Conditions of shape `(batch_dim, *condition_shape)`.
            eps: Small positive value used to replace `-inf` log-probabilities.
            **kwargs: Unused; accepted for API compatibility.

        Returns:
            A tuple `(samples, log_probs)` with shapes
            `(*sample_shape, batch_dim, *input_shape)` and
            `(*sample_shape, batch_dim)`.
        """
        sample_shape = torch.Size(sample_shape)
        num_samples = sample_shape.numel()
        condition_flat = self._embed_condition(condition)
        batch_dim = condition.shape[0]

        repeated_condition = condition_flat.repeat(num_samples, 1)

        samples_flat, log_probs_flat = self._autoregressive_sample(
            repeated_condition, with_log_prob=True, eps=eps
        )
        if log_probs_flat is None:
            raise RuntimeError("Expected log probabilities when with_log_prob=True.")

        samples = samples_flat.reshape((*sample_shape, batch_dim, *self.input_shape))
        log_probs = log_probs_flat.reshape((*sample_shape, batch_dim))

        return samples.to(device=condition.device), log_probs.to(
            device=condition.device
        )

    def loss(self, input: Tensor, condition: Tensor, **kwargs) -> Tensor:
        r"""Return loss for training."""

        raise NotImplementedError(
            "Loss for potential fine-tuning is not implemented yet."
        )

    @staticmethod
    def _warn_if_context_exceeds_recommended(model_path, max_context_size: int) -> None:
        """Warn if max_context_size exceeds the recommended limit."""
        name = Path(model_path).name.lower()
        if "v2.5" in name or "v2.6" in name:
            recommended = 50_000
        elif "v2" in name:
            recommended = 10_000
        else:
            return
        if max_context_size > recommended:
            warnings.warn(
                f"max_context_size={max_context_size} exceeds the recommended "
                f"maximum of {recommended} for TabPFN version {name}.",
                UserWarning,
                stacklevel=3,
            )

    @staticmethod
    def _resolve_model_path(model_path: str) -> str:
        """Normalize version shorthands and resolve to a cache-prefixed absolute path.

        Accepted shorthands (case-insensitive): "v2"/"2"/"v2.0"/"2.0" (default),
        "v2.5"/"2.5", "v2.6"/"2.6". Bare checkpoint filenames are resolved into
        the platform cache directory. Absolute paths and "auto" pass through unchanged.
        """
        _SHORTHANDS: dict[str, str] = {
            "v2": "tabpfn-v2-regressor.ckpt",
            "2": "tabpfn-v2-regressor.ckpt",
            "v2.0": "tabpfn-v2-regressor.ckpt",
            "2.0": "tabpfn-v2-regressor.ckpt",
            "v2.5": "tabpfn-v2.5-regressor-v2.5_default.ckpt",
            "2.5": "tabpfn-v2.5-regressor-v2.5_default.ckpt",
            "v2.6": "tabpfn-v2.6-regressor-v2.6_default.ckpt",
            "2.6": "tabpfn-v2.6-regressor-v2.6_default.ckpt",
        }
        resolved = _SHORTHANDS.get(str(model_path).strip().lower(), model_path)
        # Prepend cache dir for bare filenames so the model is stored system-wide,
        # not in the current working directory.
        if resolved != "auto" and not Path(resolved).is_absolute():
            return prepend_cache_path(resolved)
        return resolved

    def _log_license(self, model_path: str | Path) -> None:
        # remove ckpt suffix if present for cleaner loggin
        model_path = Path(model_path).with_suffix("").with_suffix("")
        global _HAS_LOGGED_TABPFN_LICENSE
        if not _HAS_LOGGED_TABPFN_LICENSE:
            logger.warning(
                f"TabPFN {Path(model_path).name} is a NONCOMMERCIAL model. "
                "Usage of this artifact (including through the sbi package) "
                "is not permitted for commercial tasks unless granted "
                "explicit permission by the model authors (PriorLabs)."
            )  # Aligning with TabPFNv25 license
            _HAS_LOGGED_TABPFN_LICENSE = True
