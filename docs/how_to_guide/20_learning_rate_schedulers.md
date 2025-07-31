# How to use learning rate schedulers

Learning rate schedulers can significantly improve training convergence and final model performance by automatically adjusting the learning rate during training. As of `sbi` v0.25.0, all inference methods (NPE, NLE, and NRE) support PyTorch's built-in learning rate schedulers.

## Quick start

The simplest way to use a learning rate scheduler is to specify it by name when calling `.train()`:

```python
from sbi.inference import NPE
import torch
from torch.distributions import MultivariateNormal

# Set up basic inference
prior = MultivariateNormal(torch.zeros(2), torch.eye(2))
inference = NPE(prior=prior)

# Train with exponential learning rate decay
posterior = inference.train(
    lr_scheduler="exponential",
    lr_scheduler_kwargs={"gamma": 0.95},  # Decay rate
    learning_rate=1e-3,
    max_num_epochs=100
)
```

## Supported schedulers

`sbi` supports the following PyTorch learning rate schedulers:

- **`"plateau"`**: `ReduceLROnPlateau` - Reduces LR when validation loss plateaus
- **`"exponential"`**: `ExponentialLR` - Exponential decay: `lr = lr * gamma^epoch`
- **`"step"`**: `StepLR` - Step-wise decay at fixed intervals
- **`"multistep"`**: `MultiStepLR` - Step-wise decay at multiple milestones
- **`"cosine"`**: `CosineAnnealingLR` - Cosine annealing schedule
- **`"cyclic"`**: `CyclicLR` - Cyclical learning rates

## Configuration methods

### Method 1: String shortcuts

Use predefined scheduler types with custom parameters:

```python
# Plateau scheduler (recommended for most cases)
posterior = inference.train(
    lr_scheduler="plateau",
    lr_scheduler_kwargs={
        "factor": 0.5,      # Reduce LR by half
        "patience": 10,     # Wait 10 epochs before reducing
        "min_lr": 1e-6,     # Don't go below this LR
    }
)

# Exponential decay
posterior = inference.train(
    lr_scheduler="exponential",
    lr_scheduler_kwargs={"gamma": 0.9}  # 10% decay per epoch
)
```

### Method 2: Dictionary configuration

Use a dictionary to specify both the scheduler type and its parameters:

```python
scheduler_config = {
    "type": "cosine",
    "T_max": 50,        # Half-period of cosine
    "eta_min": 1e-5,    # Minimum learning rate
}

posterior = inference.train(
    lr_scheduler=scheduler_config,
    learning_rate=1e-3,
    max_num_epochs=100
)
```

## Early stopping with minimum learning rate

You can automatically stop training when the learning rate becomes too small:

```python
posterior = inference.train(
    lr_scheduler="exponential",
    lr_scheduler_kwargs={"gamma": 0.85},
    learning_rate=1e-3,
    min_lr_threshold=1e-5,    # Stop when LR < 1e-5
    max_num_epochs=200,       # Maximum epochs
    stop_after_epochs=50      # Also stop if no validation improvement
)
```

This prevents over-training with very small learning rates and can speed up your workflow.

## Scheduler-specific examples

### ReduceLROnPlateau (Recommended)

Best for most use cases - adapts to training progress:

```python
posterior = inference.train(
    lr_scheduler="plateau",
    lr_scheduler_kwargs={
        "factor": 0.3,          # Aggressive reduction
        "patience": 5,          # Reduce after 5 epochs without improvement
        "threshold": 1e-4,      # Consider improvement only if > threshold
        "cooldown": 3,          # Wait 3 epochs after reduction before next reduction
    },
    learning_rate=5e-3,         # Start with higher LR
    max_num_epochs=150
)
```

### CyclicLR for exploration

Cyclical learning rates can help escape local minima:

```python
posterior = inference.train(
    lr_scheduler="cyclic",
    lr_scheduler_kwargs={
        "base_lr": 1e-5,        # Minimum in cycle
        "max_lr": 1e-3,         # Maximum in cycle
        "step_size_up": 10,     # Epochs to go from base to max
        "mode": "triangular",   # Triangular cycle
    },
    learning_rate=1e-4,         # Starting LR (becomes base_lr)
    max_num_epochs=100
)
```

### MultiStepLR for staged training

Reduce learning rate at specific milestones:

```python
posterior = inference.train(
    lr_scheduler="multistep",
    lr_scheduler_kwargs={
        "milestones": [30, 60, 90],  # Reduce at these epochs
        "gamma": 0.1,                # Reduce by 10x each time
    },
    learning_rate=1e-2,              # Start with high LR
    max_num_epochs=120
)
```

## Monitoring learning rate changes

The training summary automatically tracks learning rate changes:

```python
posterior = inference.train(
    lr_scheduler="exponential",
    lr_scheduler_kwargs={"gamma": 0.95},
    show_train_summary=True
)

# Access learning rate history
learning_rates = inference._summary["learning_rates"]
epochs = inference._summary["epochs_trained"]

# Plot learning rate schedule
import matplotlib.pyplot as plt
plt.plot(epochs, learning_rates)
plt.xlabel("Epoch")
plt.ylabel("Learning Rate")
plt.yscale("log")
plt.title("Learning Rate Schedule")
plt.show()
```

## Compatibility with all inference methods

Learning rate schedulers work identically across all `sbi` inference methods:

```python
from sbi.inference import NPE, NLE_A, NRE_A, MNLE, NRE_B, NRE_C, BNRE

# All methods support the same scheduler interface
methods = [NPE, NLE_A, NRE_A, MNLE, NRE_B, NRE_C, BNRE]

for Method in methods:
    inference = Method(prior=prior)
    inference.append_simulations(theta, x)

    # Same scheduler syntax for all methods
    estimator = inference.train(
        lr_scheduler="plateau",
        lr_scheduler_kwargs={"factor": 0.5, "patience": 8},
        max_num_epochs=50
    )
```

## Resuming training with schedules

When resuming training, the scheduler state is preserved:

```python
# Initial training
inference.train(
    lr_scheduler="exponential",
    lr_scheduler_kwargs={"gamma": 0.9},
    max_num_epochs=50
)

current_lr = inference._summary["learning_rates"][-1]
print(f"Learning rate after 50 epochs: {current_lr}")

# Resume training - scheduler continues from previous state
inference.train(
    resume_training=True,
    max_num_epochs=100  # Will run epochs 51-100
)

final_lr = inference._summary["learning_rates"][-1]
print(f"Final learning rate: {final_lr}")
```

## Best practices

1. **Start with ReduceLROnPlateau**: It's adaptive and works well for most problems
2. **Use higher initial learning rates**: Schedulers allow you to start aggressive and decay
3. **Monitor the learning rate**: Check `inference._summary["learning_rates"]` to verify behavior
4. **Set min_lr_threshold**: Prevent training with ineffectively small learning rates
5. **Combine with early stopping**: Use both `stop_after_epochs` and `min_lr_threshold`

## Troubleshooting

**Learning rate not changing**: Ensure you're using the correct parameter names for your chosen scheduler. Check the PyTorch documentation for the specific scheduler.

**Training stops too early**: The `min_lr_threshold` might be too high, or ReduceLROnPlateau patience might be too low.

**No improvement with scheduler**: Try different scheduler types or adjust hyperparameters. Some problems benefit more from cyclical schedules, others from monotonic decay.

**Backward compatibility**: Existing code without schedulers continues to work unchanged - schedulers are completely optional.
