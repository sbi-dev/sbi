# Training Refactor PR: Architectural Assessment and Plan

This document reviews the "training-refactor" changes that move the training loop into the `NeuralInference` base class and introduce new template methods across trainer classes. It covers the current architecture, a critique against SOLID design principles, and a concrete plan to harden the contracts with typed contexts and unified signatures.

## Summary

- Benefit: removes duplicated epoch loops across NPE/NLE/NRE/VF trainers; centralizes optimizer/loop/early-stopping/logging.
- Risk: increases the responsibility of the base class ("god class"), and introduces type-contract issues and brittle kwargs patterns.
- Proposal: keep the DRY benefits, but formalize the contracts:
  - Unify abstract method signatures with a StartIndexContext dataclass.
  - Replace magic `loss_kwargs` dicts with typed LossArgs objects per trainer family.
  - Standardize `train` signature via a `TrainConfig` and generic `LossArgsT` to restore LSP and type safety.

---

## Repository & module overview (abridged)

- `sbi/inference`: Orchestrates SBI pipelines.
  - `trainers/`: Implements training orchestration for method families (NPE/NLE/NRE/VF). New refactor centralizes the loop in `base.py`.
  - `posteriors/`: Sampling/evaluation backends (MCMC, VI, importance, rejection, direct, vector-field).
  - `potentials/`: Connect estimators to posterior samplers.
- `sbi/neural_nets`: Estimators and builders used by trainers.
- `sbi/utils`: Device, simulation, data handling, z-scoring, logging, etc.

This separation is sensible: estimators are model-level; trainers orchestrate data + optimization; posteriors and potentials compose the inference runtime.

---

## Trainers submodule: state after refactor

- `NeuralInference` gains:
  - Generic over estimator type.
  - Central `_run_training_loop` encapsulating optimizer setup (Adam), train/val epochs, clipping, early stopping, summary/TensorBoard logging.
  - Hooks: `_get_losses`, `_get_start_index`, `_initialize_neural_network`.
  - Default `_train_epoch`, `_validate_epoch`, `_summarize_epoch` (VF overrides to implement EMA + validation time logic).
- Subclasses (NPE/NLE/NRE/VF):
  - Implement the hooks.
  - Prepare `loss_kwargs` bags (e.g., `num_atoms`, `proposal`, `calibration_kernel`, `times`) and call `_run_training_loop`.

---

## Key findings

### 1 Conflated responsibilities

`NeuralInference` already handled: data/masks/round storage, dataloaders, posterior building for multiple samplers, progress, TB writer. The refactor adds optimizer policy and the full training loop. This centralization reduces duplication, but increases coupling and change surface.

Impact: changes in the base loop affect all trainers; testing and evolution get riskier.

### 2 LSP/signature issues

- Abstract `_get_start_index(self, discard_prior_samples: bool)` in base vs subclass overrides requiring additional params (`force_first_round_loss`, `resume_training`). This violates substitution and would fail static type checks.
- Subclass `train(...)` add required params (e.g., `num_atoms`, `validation_times`), which breaks a uniform base contract and hinders type tooling.

### 3 Weak abstraction via ad-hoc dicts

- `loss_kwargs` is a magic bag: keys like `num_atoms`, `proposal`, `calibration_kernel`, `times` are passed implicitly. This is error-prone and undiscoverable.
- VF special-cases validation times by filtering the dict during training—control flow via untyped keys.

### 4 Error-prone contracts

- Optimizer is hardcoded (Adam) in base; no hook for alternative policies.
- Mutating trainer state (`self.optimizer`, `self.epoch`, `self._val_loss`) is tightly coupled to resume logic; expectations aren’t explicit.
- The loop returns `deepcopy(self._neural_net)`—subclasses must remember to follow the same pattern.

---

## Recommendations (to include in this PR)

These changes retain the refactor’s DRY benefits while hardening contracts and type-safety.

### A) Unify abstract method signatures via a context object

Introduce a typed context for start-index policies that accommodates all families without changing signatures over time.

```python
from dataclasses import dataclass
from typing import Optional

@dataclass(frozen=True)
class StartIndexContext:
    discard_prior_samples: bool
    # SNPE/VF specifics
    force_first_round_loss: Optional[bool] = None
    resume_training: Optional[bool] = None
    # SNPE specifics
    use_non_atomic_loss: Optional[bool] = None
    ran_final_round: Optional[bool] = None
```

Base and subclasses implement one signature.

Benefits: no more signature divergence; evolution is done by extending the dataclass, not method signatures.

### B) Replace `loss_kwargs` dict with typed loss args

Define per-family LossArgs dataclasses (or TypedDicts) and make the trainer generic over its loss-args type.

```python
from dataclasses import dataclass
from typing import Optional, Callable, Union, Generic, TypeVar
from torch import Tensor
from torch.distributions import Distribution

# Common unions for proposal
ProposalLike = Optional[Union[Distribution, "NeuralPosterior"]]

@dataclass(frozen=True)
class LossArgsNRE:
    num_atoms: int

@dataclass(frozen=True)
class LossArgsNPE:
    proposal: ProposalLike
    calibration_kernel: Callable[[Tensor], Tensor]
    force_first_round_loss: bool

@dataclass(frozen=True)
class LossArgsVF:
    proposal: ProposalLike
    calibration_kernel: Callable[[Tensor], Tensor]
    force_first_round_loss: bool
    times: Optional[Tensor]  # Fixed times for validation/EMA logic

LossArgsT = TypeVar("LossArgsT")
```

Update base hook and loop signatures:

```python
from typing import Sequence, Optional
from torch import Tensor

class NeuralInference(Generic[EstimatorT, LossArgsT]):
    # ...existing code...

    @abstractmethod
    def _get_losses(self, batch: Sequence[Tensor], loss_args: Optional[LossArgsT]) -> Tensor:
        """Compute per-sample losses for a batch; must not mutate batch."""

    def _run_training_loop(
        self,
        train_loader,
        val_loader,
        max_num_epochs: int,
        stop_after_epochs: int,
        learning_rate: float,
        resume_training: bool,
        clip_max_norm: Optional[float],
        show_train_summary: bool,
        loss_args: Optional[LossArgsT] = None,
        summarization_kwargs: Optional[dict] = None,
    ) -> EstimatorT:
        # unchanged logic; pass loss_args through to _train_epoch/_validate_epoch
        ...
```

Subclasses then implement `_get_losses` with their typed args, avoiding magic keys.

### C) Standardize the core training path with `TrainConfig` + typed loss args

Keep public subclass `train(...kwargs)` unchanged while introducing a single, typed core in the base via a protected method. Soften the abstract `train` contract in base to `*args, **kwargs` to avoid LSP friction during the transition.

```python
@dataclass(frozen=True)
class TrainConfig:
    training_batch_size: int = 200
    learning_rate: float = 5e-4
    validation_fraction: float = 0.1
    stop_after_epochs: int = 20
    max_num_epochs: int = 1000
    clip_max_norm: Optional[float] = 5.0
    resume_training: bool = False
    retrain_from_scratch: bool = False
    show_train_summary: bool = False

class NeuralInference(Generic[EstimatorT, LossArgsT]):
    @abstractmethod
    def train(self, *args, **kwargs) -> EstimatorT:
        """Public entry point remains abstract and user-facing in subclasses.
        We soften the signature here to avoid LSP issues while we migrate to a typed core.
        """

    def _train_with_config(
        self,
        *,
        config: TrainConfig,
        loss_args: Optional[LossArgsT] = None,
        dataloader_kwargs: Optional[dict] = None,
        start_index_ctx: Optional[StartIndexContext] = None,
    ) -> EstimatorT:
        # 1) compute start idx via _get_start_index(start_index_ctx or default)
        # 2) dataloaders
        # 3) _initialize_neural_network
        # 4) _run_training_loop(..., loss_args=loss_args)
        ...
```

Subclass example (NRE):

```python
class RatioEstimatorTrainer(NeuralInference[RatioEstimator, LossArgsNRE]):
    def train(
        self,
        num_atoms: int,
        *,
        training_batch_size: int = 200,
        learning_rate: float = 5e-4,
        validation_fraction: float = 0.1,
        stop_after_epochs: int = 20,
        max_num_epochs: Optional[int] = None,
        clip_max_norm: Optional[float] = 5.0,
        discard_prior_samples: bool = False,
        retrain_from_scratch: bool = False,
        resume_training: bool = False,
        show_train_summary: bool = False,
        dataloader_kwargs: Optional[dict] = None,
    ) -> RatioEstimator:
        cfg = TrainConfig(
            training_batch_size=training_batch_size,
            learning_rate=learning_rate,
            validation_fraction=validation_fraction,
            stop_after_epochs=stop_after_epochs,
            max_num_epochs=max_num_epochs or 10_000,
            clip_max_norm=clip_max_norm,
            resume_training=resume_training,
            retrain_from_scratch=retrain_from_scratch,
            show_train_summary=show_train_summary,
        )
        idx_ctx = StartIndexContext(discard_prior_samples=discard_prior_samples)
        loss_args = LossArgsNRE(num_atoms=num_atoms)
        return self._train_with_config(
            config=cfg,
            loss_args=loss_args,
            dataloader_kwargs=dataloader_kwargs,
            start_index_ctx=idx_ctx,
        )
```

NPE/NLE/VF similarly provide their `LossArgs` and `StartIndexContext` fields (e.g., `force_first_round_loss`, `times`).

---

## Migration & back-compat

- Keep current user-facing parameters in subclass `train(...)` methods; internally construct `TrainConfig`, `StartIndexContext`, and `LossArgsX` and call the base `train`.
- No external API break for library users; changes are internal to the trainer contracts and typing.
- Tests to add/update:
  - Start-index behavior across families (SNPE atomic vs non-atomic; SNPE-A last-round; NLE/NRE discard behavior).
  - VF validation-times expansion (shapes and sample counts).
  - Resume vs retrain_from_scratch flows.
  - Type checks (pyright/mypy if used) for the new dataclasses and generics.

---

## Optional follow-ups (post-merge)

1) Extract an `EpochTrainer` component (composition over inheritance): encapsulate optimizer, epoch loop, clipping, early stopping, and logging behind a small contract.
2) Introduce a `Logger` interface to decouple TB writer from the loop.
3) Encapsulate start-index logic as a `DataSelectionPolicy` for explicit, testable rules per method family.
4) Add `_make_optimizer` and optional scheduler hook to base for flexibility.

---

## Risks & mitigations

- Risk: Introducing generics and dataclasses adds boilerplate.
  - Mitigation: Subclass convenience overloads keep user ergonomics. Stronger typing reduces future regressions.
- Risk: VF validation-times control leaks into general loop.
  - Mitigation: Keep VF overrides for `_train_epoch`/`_summarize_epoch` or add a dedicated validation context; avoid mixing control flags into generic loss args.
- Risk: Base remains large.
  - Mitigation: Plan follow-up extraction of `EpochTrainer` and `Logger` to shrink base responsibilities.

---

## Appendix: Before/after snippets

### Unifying `_get_start_index`

Before (divergent signatures):

```python
# base
@abstractmethod
def _get_start_index(self, discard_prior_samples: bool) -> int: ...

# npe/vf subclasses
def _get_start_index(self, discard_prior_samples: bool, force_first_round_loss: bool, resume_training: bool) -> int:
    ...
```

After (single signature via context):

```python
@abstractmethod
def _get_start_index(self, ctx: StartIndexContext) -> int:
    # Example in SNPE:
    start_idx = int(ctx.discard_prior_samples and self._round > 0)
    if (ctx.use_non_atomic_loss or ctx.ran_final_round):
        start_idx = self._round
    return start_idx
```

### Replacing `loss_kwargs` with typed args

Before:

```python
loss_kwargs = {"num_atoms": num_atoms}
train_net = self._run_training_loop(..., loss_kwargs=loss_kwargs)
```

After:

```python
loss_args = LossArgsNRE(num_atoms=num_atoms)
train_net = self._run_training_loop(..., loss_args=loss_args)
```

Subclass `_get_losses`:

```python
def _get_losses(self, batch: Sequence[Tensor], loss_args: Optional[LossArgsNRE]) -> Tensor:
    theta, x = batch[0].to(self._device), batch[1].to(self._device)
    assert loss_args is not None
    return self._loss(theta, x, num_atoms=loss_args.num_atoms)
```

### Standardized `train` with config

Before:

```python
# per-subclass differing signatures and argument sets
trainer.train(num_atoms=10, learning_rate=5e-4, validation_fraction=0.1, ...)
```

After (subclass wrapper builds config and args):

```python
cfg = TrainConfig(training_batch_size=200, learning_rate=5e-4, ...)
idx_ctx = StartIndexContext(discard_prior_samples=True)
loss_args = LossArgsNRE(num_atoms=10)
trainer.train(cfg, loss_args=loss_args, start_index_ctx=idx_ctx)
```

---

## Conclusion

The refactor direction is good and clearly reduces duplication. The changes proposed here preserve those gains while fixing LSP violations and replacing brittle kwargs with typed, discoverable contracts. This makes the training infrastructure safer to extend and easier to maintain.
