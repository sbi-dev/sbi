# EP-01: Pluggable Training Infrastructure for sbi

Status: Discussion
Feedback: See GitHub Discussion → [EP-01 Discussion](https://github.com/sbi-dev/sbi/discussions/new?category=ideas)

## Summary

This enhancement proposal describes the refactoring of `sbi` neural network training
infrastructure to address technical debt, improve maintainability, and provide users
with better experiment tracking capabilities. The refactoring will introduce typed
configuration objects, extract a unified training loop, and implement pluggable
interfaces for logging and early stopping while maintaining full backward compatibility.

## Motivation

The `sbi` package has evolved organically over five years to serve researchers in
simulation-based inference. This growth has resulted in several architectural issues in
the training infrastructure:

1. **Code duplication**: Training logic is duplicated across NPE, NLE, NRE, and other
   inference methods, spanning over 3,000 lines of code
2. **Type safety issues**: Unstructured dictionary parameters lead to runtime errors and
   poor IDE support
3. **Limited logging options**: Users are restricted to TensorBoard with no easy way to
   integrate their preferred experiment tracking tools
4. **Missing features**: No built-in early stopping, leading to wasted compute and
   potential overfitting
5. **Integration barriers**: Community contributions like PR #1629 cannot be easily
   integrated due to architectural constraints

These issues affect both users and maintainers, slowing down development and making the
codebase harder to work with.

## Goals

### For Users

- **Seamless experiment tracking**: Support for TensorBoard, WandB, MLflow, and stdout
  without changing existing code
- **Clear API for early stopping strategies**: Better docs for patience-based
  (implemented internally), plus possibly a lightweight interface to external early
  stopping strategies.
- **Better debugging**: Typed configurations with clear error messages
- **Zero migration effort**: Full backward compatibility with existing code

### For Maintainers

- **Reduced code duplication**: Extract shared training logic into reusable components
- **Type safety**: Prevent entire classes of bugs through static typing
- **Easier feature integration**: Clean interfaces for community contributions
- **Future extensibility**: Lightweight interfaces for external tools, e.g., logging and
  early stoppping.

## Non-Goals

We want to avoid removing code duplication just for the sake of reduced LOC. Sometimes,
code duplication is required and preferred in favor of overcomplicated large class
structures with unclear separation of concerns. In other words, having clear API interfaces is
more important than reducing code duplication.

We also want to avoid adding complexity by aiming implementing all possible features.
E.g., we probably should not implement all kinds of early stopping tools internally
because this will add maintainance and documentation burden for us, and overhead for the
user to understand the API and the docs. Instead, we should implement either a
lightweight interface that allows to plug external early stopping tool, or implement
just a basic version in our internal training (like we do now), and refer to the
flexible training interface when a user wants to use other approaches.

## Design

These are very rough sketches of how this could look like. They should be open for
discussion and can be changed substantially when we implement this (s.t. to the
non-goals defined above).

### Current API

```python
# Current: Each method has its own training implementation, mixing general training options with method-specific loss options.
inference = NPE(prior=prior)
inference.train(
    training_batch_size=50,
    learning_rate=5e-4,
    validation_fraction=0.1,
    stop_after_epochs=20,
    max_num_epochs=2**31-1,
    clip_max_norm=5.0,
    exclude_invalid_x=True,
    resume_training=False,
    show_train_summary=False,
)
```

### Proposed API

```python
# Proposed: Cleaner API with typed configurations
from sbi.training import TrainConfig, LossArgs

# Configure training (with IDE autocomplete and validation)
train_config = TrainConfig(
    batch_size=50,
    learning_rate=5e-4,
    max_epochs=1000,
    device="cuda"
)

# Method-specific loss configuration
loss_args = LossArgsNPE(exclude_invalid_x=True)

# Train with clean API
inference = NPE(prior=prior)
inference.train(train_config, loss_args)
```

### Logging Interface

Users can seamlessly switch between logging backends:

```python
from sbi.training import LoggingConfig

# Choose your backend - no other code changes needed
logging = LoggingConfig(backend="wandb", project="my-experiment")
# or: LoggingConfig(backend="tensorboard", log_dir="./runs")
# or: LoggingConfig(backend="mlflow", experiment_name="sbi-run")
# or: LoggingConfig(backend="stdout")  # default

inference.train(train_config, loss_args, logging=logging)
```

### Early Stopping

Multiple strategies available out of the box:

```python
from sbi.training import EarlyStopping

# Stop when validation loss plateaus
early_stop = EarlyStopping.validation_loss(patience=20, min_delta=1e-4)

# Stop when learning rate drops too low
early_stop = EarlyStopping.lr_threshold(min_lr=1e-6)

inference.train(train_config, loss_args, early_stopping=early_stop)
```

### Backward Compatibility

All existing code continues to work:

```python
# Old API still supported - no breaking changes
inference.train(training_batch_size=100, learning_rate=1e-3)

# Mix old and new as needed during migration
inference.train(
    training_batch_size=100,  # old style
    logging=LoggingConfig(backend="wandb")  # new feature
)
```

### Unified Backend

All inference methods share the same training infrastructure:

```python
# NPE, NLE, NRE all use the same configuration
npe = NPE(prior=prior)
npe.train(train_config, loss_args)

nle = NLE(prior=prior)
nle.train(train_config, loss_args)
```

## Example: Complete Training Pipeline

```python
from sbi import utils
from sbi.inference import NPE
from sbi.training import TrainConfig, LossArgsNPE, LoggingConfig, EarlyStopping

# Setup simulation
prior = utils.BoxUniform(low=-2*torch.ones(2), high=2*torch.ones(2))
simulator = lambda theta: theta + 0.1 * torch.randn_like(theta)

# Configure training with type safety and autocomplete
config = TrainConfig(
    batch_size=100,
    learning_rate=1e-3,
    max_epochs=1000
)

# Setup logging and early stopping
logging = LoggingConfig(backend="wandb", project="sbi-experiment")
early_stop = EarlyStopping.validation_loss(patience=20)

# Train with new features
inference = NPE(prior=prior)
theta, x = utils.simulate_for_sbi(simulator, prior, num_simulations=10000)
inference.append_simulations(theta, x)

neural_net = inference.train(
    config,
    LossArgsNPE(exclude_invalid_x=True),
    logging=logging,
    early_stopping=early_stop
)
```

## Next steps

Centralizing training logic in `base.py` has historically increased the size and
responsibilities of the `NeuralInference` “god class”. As a natural next step, we
propose extracting the entire training loop into a standalone function that takes the
configured options and training components, and returns the trained network (plus
optional artifacts), e.g., something like:

```python
def run_training(
        config: TrainConfig,
        model: torch.nn.Module,
        loss_fn: Callable[..., torch.Tensor],
        train_loader: DataLoader,
        val_loader: DataLoader | None = None,
        optimizer: torch.optim.Optimizer | None = None,
        scheduler: torch.optim.lr_scheduler._LRScheduler | None = None,
        callbacks: Sequence[Callback] | None = None,  # logging, early stopping, etc.
        device: str | torch.device | None = None,
) -> tuple[torch.nn.Module, TrainingSummary]:
        """Runs the unified training loop and returns the trained model and summary."""
```

Benefits:

- Shrinks `NeuralInference` and makes responsibilities explicit.
- Improves testability (train loop covered independently; inference classes can be
    tested with lightweight mocks).
- Enables pluggable logging/early-stopping via callbacks without entangling method-
    specific logic.
- Keeps backward compatibility: inference classes compose `run_training()` internally
    while still exposing the existing `.train(...)` entry point.

This should be tackled in a follow-up EP or PR that would introduce `run_training()`
(and a minimal `Callback` protocol), migrate NPE/NLE/NRE to call it, and add focused
unit tests for the training runner.

## Feedback Wanted

We welcome feedback and implementation interest in GitHub Discussions:

1. Which logging backends are most important?
2. What early stopping strategies would be useful?
3. Any concerns about the proposed API?
4. What do you think about the external training function?

- Discussion thread: [EP-01 Discussion](https://github.com/sbi-dev/sbi/discussions/new?category=ideas)

## References

- [PR #1629](https://github.com/sbi-dev/sbi/pull/1629): Community early stopping implementation
- [NUMFOCUS SDG Proposal](https://github.com/numfocus/small-development-grant-proposals/issues/60): Related funding proposal
