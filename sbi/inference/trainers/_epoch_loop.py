"""
Internal epoch-loop helper.

Small and import-light utility that base trainers can delegate to. This module
is intentionally framework-agnostic and avoids runtime torch imports; callers
pass in the model/optimizer/scheduler objects and the loss function.

This is a draft skeleton meant to show the surface area and responsibilities.
Implementations can replace the NotImplementedError with the concrete loop.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Tuple

from ._contracts import LossArgs, TrainConfig

# Aliases (kept loose here to avoid imports)
Batch = Tuple[Any, Any]  # e.g., (theta, x) or any batch structure your loss_fn expects
Metrics = Dict[str, float]
MetricsLogger = Callable[[str, Metrics, int], None]


def run_training_loop(
    model: Any,
    optimizer: Any,
    dataloaders: Any,
    *,
    compute_loss: Callable[[Batch, Optional[LossArgs]], Any],
    loss_args: Optional[LossArgs],
    config: TrainConfig,
    metrics_logger: Optional[MetricsLogger] = None,
    scheduler: Any = None,
    grad_clip_max_norm: Optional[float] = None,
) -> Metrics:
    """Run the train/val epoch loop.

    Parameters
    - model: torch.nn.Module-like object (no hard import here).
    - optimizer: torch.optim.Optimizer-like.
        - dataloaders: object with attributes `train` and optional `val`,
            each iterable over batches.
    - compute_loss: callable taking (batch, loss_args) -> scalar-like (Tensor or float).
    - loss_args: typed loss arguments (NPE/NRE/VF) or None.
    - config: TrainConfig hyperparameters.
    - metrics_logger: optional callable (scope, metrics, step) -> None.
    - scheduler: optional LR scheduler with step() method.
    - grad_clip_max_norm: optional gradient clipping threshold.

    Returns
    - A metrics dict for the final (or best) epoch.

    Notes
    - This draft defines the contract and event points. Replace the body with a concrete
      implementation that iterates epochs and batches, computes losses, logs metrics,
      steps optimizer/scheduler, and handles early stopping.
    """
    del (
        model,
        optimizer,
        dataloaders,
        compute_loss,
        loss_args,
        config,
        metrics_logger,
        scheduler,
        grad_clip_max_norm,
    )

    raise NotImplementedError(
        "run_training_loop is a draft surface; implement the concrete epoch loop "
        "in a follow-up PR."
    )


__all__ = ["run_training_loop", "Batch", "Metrics", "MetricsLogger"]
