# SBI Architecture

## Trainer Hierarchy

```
NeuralInference (base)
├── PosteriorEstimatorTrainer → NPE_A, NPE_B, NPE_C, MNPE
├── RatioEstimatorTrainer → NRE_A, NRE_B, NRE_C, BNRE
├── LikelihoodEstimatorTrainer → NLE_A, MNLE
├── VectorFieldTrainer → FMPE, NPSE
├── NPE_PFN (training-free, TabPFN-based; subclasses NeuralInference directly)
└── MarginalTrainer (marginal / unconditional density)
```

## Estimator API

Trainers produce a trained `Estimator` — an `nn.Module` subclass with a uniform `forward` / `loss` / `log_prob` / `sample` interface that downstream posteriors and potentials consume. Concrete implementations live under `sbi/neural_nets/estimators/` (plus `sbi/neural_nets/ratio_estimators.py` for the ratio family).

```
ConditionalEstimator (base, ABC)
├── ConditionalDensityEstimator    # NPE / NLE; backs DirectPosterior and likelihood potentials
│   └── MixedDensityEstimator      # MNPE / MNLE (mixed continuous + discrete)
├── ConditionalVectorFieldEstimator # FMPE / NPSE; backs VectorFieldPosterior
│   └── ConditionalScoreEstimator  # score-based (NPSE)
└── RatioEstimator                  # NRE; sibling file sbi/neural_nets/ratio_estimators.py

UnconditionalEstimator (base, ABC)
└── UnconditionalDensityEstimator   # MarginalTrainer output
```

## Posterior Types

```
NeuralPosterior (base)
├── DirectPosterior           # Direct sampling from density estimator
├── MCMCPosterior             # MCMC via Pyro/PyMC (HMC, NUTS)
├── RejectionPosterior        # Rejection sampling
├── ImportanceSamplingPosterior
├── VIPosterior               # Variational inference
├── VectorFieldPosterior      # ODE / SDE solver-based sampling
└── EnsemblePosterior         # Ensemble of posteriors
```

## Sampler Backends

Each posterior class delegates to a sampling backend under `sbi/samplers/`:

```
sbi/samplers/
├── mcmc/          # used by MCMCPosterior (slice, pymc, init strategies)
├── vi/            # used by VIPosterior (rKL / fKL divergences, quality control)
├── rejection/     # used by RejectionPosterior + direct-posterior rejection paths
├── importance/    # used by ImportanceSamplingPosterior (incl. SIR)
├── ode_solvers/   # used by VectorFieldPosterior — deterministic (ODE)
└── score/         # used by VectorFieldPosterior — stochastic (SDE)
```

## Training Pipeline

Shown for NPE; the same `append_simulations → train → build_posterior` flow applies to
NLE / NRE / FMPE / NPSE. See the [Implemented Methods tutorial](https://sbi.readthedocs.io/en/latest/tutorials/16_implemented_methods.html)
for the other inference types.

```python
trainer = NPE(prior=prior, density_estimator="maf")
trainer.append_simulations(theta, x)
trainer.train()
posterior = trainer.build_posterior()
samples = posterior.sample((num_samples,), x=x_observed)
```

## Key Design Patterns

1. **Factory Pattern:** `neural_nets/factory.py` creates estimators from string specs
2. **Protocol-Based Polymorphism:** `ConditionalEstimatorBuilder[EstimatorType]` — a builder protocol generic over the estimator it returns (e.g. `ConditionalEstimatorBuilder[RatioEstimator]`)
3. **Potential Function Abstraction:** Decouples inference from sampling
4. **Device Management:** Automatic device detection and consistency

## Dependencies

Core: **PyTorch** (neural networks), **Zuko** (modern normalizing flows),
**nflows** (classical flow architectures). Optional extras: **Pyro-ppl** / **PyMC**
(MCMC backends). See `pyproject.toml` for the authoritative version constraints.
