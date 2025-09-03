# Epoch loop extraction — lightweight design

This document specifies a small, dependency-free epoch loop helper that trims `base.py` without changing any user APIs.

Goals
- Keep public `train(...kwargs)` unchanged in all trainers.
- Shrink `base.py` by extracting only the hot loop.
- Use typed contracts (`TrainConfig`, `StartIndexContext`, `LossArgsX`) to fix current brittleness.
- No new frameworks, factories, or callback systems.

Non-goals
- Do not introduce a generic trainer class or a callback protocol today.
- Do not add external dependencies (Lightning, Ignite, etc.).

## Single function contract

```python
from sbi.inference.trainers._contracts import TrainConfig, LossArgs
from sbi.inference.trainers._logging import MetricsLogger

# dataloaders: object with attributes `.train` and optional `.val` (iterables of batches)
# compute_loss: (batch, loss_args) -> scalar loss (Tensor-like)

def run_training_loop(
    model,
    optimizer,
    dataloaders,
    *,
    compute_loss,
    loss_args: LossArgs | None,
    config: TrainConfig,
    metrics_logger: MetricsLogger | None = None,
    scheduler=None,
    grad_clip_max_norm: float | None = None,
) -> dict[str, float]:
    """Run training and optional validation epochs, return final/best metrics."""
    ...
```

Responsibilities
- Iterate epochs and batches for train (and val if present).
- Compute loss via `compute_loss(batch, loss_args)` and step optimizer.
- Apply gradient clipping if `grad_clip_max_norm` is set.
- Step scheduler if provided.
- Log metrics via `metrics_logger(scope, metrics, step)` if provided.
- Handle minimal early-stop/checkpoint inline (see below).
- Return metrics for the final or best epoch (implementation choice, documented).

Inputs
- model: a torch.nn.Module-like object (no hard import here).
- optimizer: a torch.optim.Optimizer-like object.
- dataloaders: has `.train` and optional `.val`, each iterable over batches.
- compute_loss: callable supplied by the family (NPE/NRE/VF) that knows how to compute its loss.
- loss_args: typed per-family args (e.g., `LossArgsNPE`), passed through to `compute_loss`.
- config: `TrainConfig` with batch size, lr, validation_fraction, early-stop window, max epochs, etc.
- metrics_logger: optional; see `_logging` adapters; defaults to a no-op.
- scheduler: optional; if supplied, stepped each epoch or batch (documented in impl).
- grad_clip_max_norm: optional scalar for `clip_grad_norm_`.

Minimal early-stop & checkpoint (inline)
- Track best validation metric (or training if no validation) and an integer patience counter from `config.stop_after_epochs`.
- If no improvement for `patience` epochs, break the loop.
- On improvement, snapshot `best_state = model.state_dict()` (and optionally optimizer/scheduler state if needed).
- If `config.resume_training` and a previous snapshot exists, load it before the loop; if file paths are provided, persist to disk; otherwise keep in-memory.

How trainers use it (no public API changes)
1) Build `TrainConfig` and `StartIndexContext` inside the subclass’s existing `train(...kwargs)`.
2) Compute the start index (as today), create/refresh the model and optimizer.
3) Build dataloaders from the replay buffer and `config.validation_fraction`.
4) Provide a small `compute_loss(batch, loss_args)` that wraps the family-specific loss.
5) Choose a logger (stdout, TensorBoard, WandB, MLflow) or no-op.
6) Call `run_training_loop(...)`; then wrap the trained model as the estimator (e.g., `NeuralPosterior`).

Example (sketch)
```python
final_metrics = run_training_loop(
    model=net,
    optimizer=opt,
    dataloaders=make_dls(cfg.training_batch_size, cfg.validation_fraction),
    compute_loss=lambda b, la: npe_loss(b, la),
    loss_args=LossArgsNPE(proposal=proposal, calibration_kernel=kernel),
    config=cfg,
    metrics_logger=to_tensorboard(writer, scope_prefix="npe"),
    scheduler=sched,
    grad_clip_max_norm=cfg.clip_max_norm,
)
posterior = build_estimator(net)
```

Edge cases
- validation_fraction == 0: skip validation; early-stop disabled or based on train loss only (document decision in impl).
- clip_max_norm is None or 0: skip clipping.
- resume_training & retrain_from_scratch both True: retrain wins; log a warning.

Future evolution (only if needed)
- If loop grows, extract early-stopping/checkpoint helpers into tiny utilities.
- If feature needs outgrow this, consider a thin adapter to Ignite/Lightning instead of building our own engine.
