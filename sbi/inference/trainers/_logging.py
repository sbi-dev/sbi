"""
Internal logging adapters for the training loop.

We expose a tiny callable interface:

    metrics_logger(scope, metrics, step) -> None

- scope: usually "train", "val", or "epoch" (free-form string).
- metrics: dict of scalar floats, e.g., {"loss": 0.123}.
- step: integer step. For batch logs, this is the global batch step; for
  epoch logs, this is the epoch index.

Design goals
- Keep it dependency-free: no direct imports of TensorBoard, WandB, MLflow.
- Provide tiny adapters that accept an already-initialized writer/module and
  return a metrics_logger callable.
- Default to a no-op when the caller doesn’t care about logging.

Examples
--------
Stdout
    logger = to_stdout(prefix="run42")
    logger("train", {"loss": 0.12}, step=100)

TensorBoard
    writer = SummaryWriter(log_dir)
    logger = to_tensorboard(writer, scope_prefix="npe")
    logger("train", {"loss": 0.12}, step=100)  # adds scalar "npe/train/loss"

WandB
    import wandb
    wandb.init(...)
    logger = to_wandb(wandb, scope_prefix="nre")
    logger("val", {"loss": 0.09}, step=5)  # wandb.log with step=5

MLflow
    import mlflow
    mlflow.start_run()
    logger = to_mlflow(mlflow, scope_prefix="vf")
    logger("epoch", {"loss": 0.08}, step=5)  # mlflow.log_metric with step
"""

from __future__ import annotations

from typing import Callable, Dict

Metrics = Dict[str, float]
MetricsLogger = Callable[[str, Metrics, int], None]


def no_op() -> MetricsLogger:
    """Return a logger that ignores all inputs.

    Useful as a default to avoid conditionals in calling code.
    """

    def _log(scope: str, metrics: Metrics, step: int) -> None:  # noqa: ARG001
        return None

    return _log


def to_stdout(prefix: str = "") -> MetricsLogger:
    """Create a logger that prints metrics to stdout.

    Parameters
    - prefix: optional string prepended to each line for quick filtering.
    """

    def _log(scope: str, metrics: Metrics, step: int) -> None:
        joined = ", ".join(f"{k}={v:.6g}" for k, v in metrics.items())
        pfx = f"{prefix} " if prefix else ""
        print(f"{pfx}{scope} step={step}: {joined}")

    return _log


def to_tensorboard(writer, *, scope_prefix: str | None = None) -> MetricsLogger:
    """Create a logger backed by a TensorBoard writer.

    Parameters
    - writer: object exposing `add_scalar(tag, scalar_value, global_step)`.
    - scope_prefix: optional prefix added before the scope, e.g., "npe".
        Resulting tag looks like "{scope_prefix}/{scope}/{metric}".
    """

    def _log(scope: str, metrics: Metrics, step: int) -> None:
        base = f"{scope_prefix}/" if scope_prefix else ""
        for k, v in metrics.items():
            writer.add_scalar(f"{base}{scope}/{k}", float(v), step)

    return _log


def to_wandb(wandb_module, *, scope_prefix: str | None = None) -> MetricsLogger:
    """Create a logger that forwards metrics to Weights & Biases.

    Parameters
    - wandb_module: the imported `wandb` module (or a Run with `.log`).
        We avoid importing it here; pass it in from the caller.
    - scope_prefix: optional prefix for names, e.g., "nre".

    Behavior
    - Calls `wandb.log({f"{prefix}{scope}/{k}": v, ...}, step=step)`.
    - The `step` argument ensures correct x-axis alignment in WandB.
    """

    def _log(scope: str, metrics: Metrics, step: int) -> None:
        base = f"{scope_prefix}/" if scope_prefix else ""
        payload = {f"{base}{scope}/{k}": float(v) for k, v in metrics.items()}
        # Support both wandb module and run: both expose `log`.
        wandb_module.log(payload, step=step)

    return _log


def to_mlflow(mlflow_module, *, scope_prefix: str | None = None) -> MetricsLogger:
    """Create a logger that forwards metrics to MLflow.

    Parameters
    - mlflow_module: the imported `mlflow` module (or client) that exposes
        `log_metric(key, value, step=step)`.
    - scope_prefix: optional prefix for names, e.g., "vf".

    Behavior
    - Calls `mlflow.log_metric(f"{prefix}{scope}/{key}", value, step=step)` for
        every metric.
    """

    def _log(scope: str, metrics: Metrics, step: int) -> None:
        base = f"{scope_prefix}/" if scope_prefix else ""
        for k, v in metrics.items():
            mlflow_module.log_metric(f"{base}{scope}/{k}", float(v), step=step)

    return _log


__all__ = [
    "Metrics",
    "MetricsLogger",
    "no_op",
    "to_stdout",
    "to_tensorboard",
    "to_wandb",
    "to_mlflow",
]
