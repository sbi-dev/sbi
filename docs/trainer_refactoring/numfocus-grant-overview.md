# sbi: Training stack modernization — NumFOCUS small dev grant overview

## What is sbi?

`sbi` is a Python/PyTorch toolkit for simulation-based inference (SBI): learning
posteriors or likelihood(-ratios) from pairs of parameters and simulator
observations. It is widely used in the sciences to calibrate mechanistic models
when closed-form likelihoods are unavailable.

## Package organization (high level)

- `sbi/inference/trainers/` — Training orchestration per estimator family:
  - NPE (Neural Posterior Estimation), NLE (Neural Likelihood Estimation),
    NRE (Neural Ratio Estimation), VF (Vector-Field variants).
  - A base trainer provides shared utilities; family-specific trainers implement
    losses and particularities (e.g., SNPE calibration kernels, VF time
    handling).
- `sbi/inference/posteriors/` — Posterior wrappers around trained neural nets
  (e.g., `NeuralPosterior`) providing `.sample`, `.log_prob`, and convenience
  features (`set_default_x`, ensembles, MCMC posterior).
- `sbi/neural_nets/` — Network architectures used by the trainers.
- `sbi/samplers/`, `sbi/utils/`, `sbi/diagnostics/` — Sampling utilities,
  shared helpers, and diagnostics/evaluation.

Typical training flow today
1) A trainer (e.g., NPE) receives replayed (theta, x) pairs.
2) It builds a neural estimator, runs an inlined epoch loop, and computes a
   family-specific loss.
3) The trained network is wrapped as a `NeuralPosterior` (or corresponding
   estimator) returned to the user.

While this design works well, the current base trainer class mixes many concerns
("god class") and relies on untyped kwargs across families, which makes
extension and maintenance harder.

---

## Project summary

We will modernize and simplify sbi’s training stack by:
- Introducing typed trainer contracts (StartIndexContext, TrainConfig, LossArgsX)
  to replace brittle kwargs and ad-hoc signatures.
- Extracting a small, dependency-free epoch loop helper (run_training_loop) to
  reduce duplication and shrink base.py.
- Providing a lightweight logger callable with adapters for stdout, TensorBoard,
  WandB, and MLflow—no new hard deps.
- Adding a minimal early-stopper/checkpoint pattern inline (optional adapters later).
- Preserving all user-facing `train(...)` APIs to ensure zero disruption.

This work decouples orchestration from method-specific logic (NPE/NLE/NRE/VF),
improves type safety and maintainability, and makes instrumentation and
reproducibility first-class.

## Motivation and impact

- Reduce “god class” and technical debt: Central training code in base.py is hard
  to extend and review. Extracting the epoch loop and formalizing contracts
  reduces complexity and prevents LSP violations.
- Eliminate silent bugs: Typed dataclasses remove untyped “kwargs soup,” catching
  misconfigurations early.
- Make contributions easier: A smaller, focused loop with clear contracts lowers
  the barrier for new trainers or features (e.g., validation hooks).
- Improve reproducibility: A unified loop and standard logging interface make
  results traceable across backends and CI.
- Keep users stable: Public `train(...)` signatures remain unchanged; all
  improvements are internal and opt-in (e.g., logging adapters).

## Objectives and deliverables

- Contracts (internal publicness, re-export optional)
  - StartIndexContext, TrainConfig, LossArgsNRE/NPE/VF, LossArgs union.
  - Update hooks to use typed inputs; remove ad-hoc dicts.
- Epoch loop extraction
  - `sbi/inference/trainers/_epoch_loop.py` with `run_training_loop(...)` used by trainers.
  - Base delegates loop; retains start-index logic, dataloaders, model/optim init,
    estimator wrapping.
- Logging interface
  - `sbi/inference/trainers/_logging.py` with `metrics_logger` callable and adapters:
    - no_op, to_stdout, to_tensorboard, to_wandb, to_mlflow.
- Minimal early stop/checkpoint (inline)
  - Best-metric tracking, patience counter, best state snapshot.
  - Optional file-based persistence behind a tiny helper (later if needed).
- Tests and docs
  - Unit tests for start-index behavior, VF validation-times, logging adapters.
  - Developer docs for contracts, loop, and logging; migration notes (no external API change).
  - Example snippets for TB/WandB/MLflow wiring.

## Scope and timeline (10–12 weeks)

- Week 1–2: Finalize and land typed contracts; align subclass hooks; small PRs to de-risk.
- Week 3–5: Extract epoch loop; update trainers to delegate; ensure parity with tests.
- Week 6–7: Add logging adapters; wire optional logging in base; document usage.
- Week 8: Minimal early-stop/checkpoint inline; tests for best-state/patience.
- Week 9–10: Documentation, examples, and polishing; ensure back-compat; CI hardening.
- Week 11–12: Stabilization, profiling, and community feedback; small follow-ups.

## Success criteria

- Maintain public API: existing user code continues to work unchanged.
- Reduction in base.py complexity: centralized loop logic moved to `_epoch_loop`;
  measurable LoC reduction and clearer responsibilities.
- Type safety: contracts replace untyped kwargs across trainers; mypy/pyright
  passes on new surfaces.
- Parity: test suite passes; specific tests cover start-index rules, VF times
  expansion, logging outputs.
- Instrumentation: users can log to stdout, TensorBoard, WandB, MLflow by passing
  a single callable.

## Risks and mitigation

- Hidden coupling in current loop: Mitigate via incremental PRs, full test runs,
  and targeted parity tests.
- Import cycles: Contracts live in `trainers/_contracts.py` with TYPE_CHECKING
  imports only.
- Performance regressions: Maintain inner-loop structure; benchmark on
  representative tasks; no new dependencies.
- Scope creep: Keep early-stopper/checkpoint minimal and inline; extract helpers
  only if complexity grows.

## Sustainability

- Back-compat preserved; internal contracts documented and covered by tests.
- Minimal surface area and no hard deps keep maintenance low.
- Design leaves a path to integrate a mature engine (Ignite/Lightning) later via
  a thin adapter if community demand grows.
