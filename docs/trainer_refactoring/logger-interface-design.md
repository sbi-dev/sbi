# Logger interface — lightweight design

A tiny, dependency-free logger callable used by the epoch loop.

Contract
- Type: `MetricsLogger = Callable[[str, dict[str, float], int], None]`
- Call shape: `logger(scope, metrics, step)`
  - scope: "train", "val", or "epoch" (free-form string allowed)
  - metrics: mapping of scalar floats, e.g., {"loss": 0.123}
  - step: global batch step for batch logs, or epoch index for epoch logs

Guiding principles
- No hard dependencies; adapters take initialized backends.
- Defaults to a no-op to keep call sites branch-free.
- Naming convention: `{optional_prefix}/{scope}/{metric}` so logs are consistent across backends.

Built-in adapters (in `sbi/inference/trainers/_logging.py`)
- `no_op()` → ignores logs.
- `to_stdout(prefix="")` → prints: `"{prefix} {scope} step={step}: k=v, ..."`.
- `to_tensorboard(writer, scope_prefix=None)` → `writer.add_scalar("{prefix}{scope}/{k}", v, step)`.
- `to_wandb(wandb_module, scope_prefix=None)` → `wandb.log({"{prefix}{scope}/{k}": v, ...}, step=step)`.
- `to_mlflow(mlflow_module, scope_prefix=None)` → `mlflow.log_metric("{prefix}{scope}/{k}", v, step=step)`.

Quick examples
```python
from sbi.inference.trainers._logging import (
    no_op, to_stdout, to_tensorboard, to_wandb, to_mlflow
)

# Stdout
logger = to_stdout(prefix="run42")
logger("train", {"loss": 0.12}, step=100)

# TensorBoard
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter("./tb")
tb_logger = to_tensorboard(writer, scope_prefix="npe")

# WandB (dependency passed in)
import wandb
wandb.init(project="sbi")
wb_logger = to_wandb(wandb, scope_prefix="nre")

# MLflow (dependency passed in)
import mlflow
mlflow.start_run()
mf_logger = to_mlflow(mlflow, scope_prefix="vf")
```

How the loop uses it
- At train batch end: `logger("train", {"loss": loss}, step=global_step)`
- At val batch end: `logger("val", {"loss": val_loss}, step=global_step)`
- At epoch end: `logger("epoch", {"loss": epoch_loss, "val_loss": best}, step=epoch_idx)`

Future evolution (only if needed)
- Add a thin Protocol if richer lifecycle events are required (train start/end, best model).
- Provide a CSV adapter if offline logging is requested.
