# `sbi`: simulation-based inference toolkit

`sbi` provides access to simulation-based inference methods via a user-friendly
interface:

```python
# simulation
theta = prior.sample((1000,))
x = simulator(theta)

# training
inference = SNPE(prior).append_simulations(theta, x)
inference.train()

# inference
posterior = inference.build_posterior()
posterior_samples = posterior.sample((1000,), x=x_o)
```

## Overview

**To get started, install the `sbi` package with:**

```commandline
pip install sbi
```

for more advanced install options, see our [Install Guide](install.md).

Then, check out our material:

<div class="grid cards" markdown>

-  :dart: [__Motivation and approach__](#motivation-and-approach)
   <br/><br/>
   *General motivation for the SBI framework and methods included in `sbi`.*

-  :rocket: [__Tutorials__](tutorials/)
   <br/><br/>
   *Various examples illustrating how to use the `sbi` package.*

-  :building_construction: [__Reference API__](reference/)
   <br/><br/>
   *The detailed description of the package classes and functions.*

-  :book: [__Citation__](citation.md)
   <br/><br/>
   *How to cite the `sbi` package.*

</div>

## Motivation and approach

Many areas of science and engineering make extensive use of complex, stochastic,
numerical simulations to describe the structure and dynamics of the processes being
investigated.

A key challenge in simulation-based science is constraining these simulation models'
parameters, which are interpretable quantities, with observational data. Bayesian
inference provides a general and powerful framework to invert the simulators, i.e.
describe the parameters that are consistent both with empirical data and prior
knowledge.

In the case of simulators, a key quantity required for statistical inference, the
likelihood of observed data given parameters, $\mathcal{L}(\theta) = p(x_o|\theta)$, is
typically intractable, rendering conventional statistical approaches inapplicable.

`sbi` implements powerful machine-learning methods that address this problem. Roughly,
these algorithms can be categorized as:

- Neural Posterior Estimation (amortized `NPE` and sequential `SNPE`),
- Neural Likelihood Estimation (`(S)NLE`), and
- Neural Ratio Estimation (`(S)NRE`).

Depending on the characteristics of the problem, e.g. the dimensionalities of the
parameter space and the observation space, one of the methods will be more suitable.

![](./static/goal.png)

**Goal: Algorithmically identify mechanistic models that are consistent with data.**

Each of the methods above needs three inputs: A candidate mechanistic model,
prior knowledge or constraints on model parameters, and observational data (or
summary statistics thereof).

The methods then proceed by

1. sampling parameters from the prior followed by simulating synthetic data from
   these parameters,
2. learning the (probabilistic) association between data (or data features) and
   underlying parameters, i.e. to learn statistical inference from simulated
   data. How this association is learned differs between the above methods, but
   all use deep neural networks.
3. This learned neural network is then applied to empirical data to derive the
   full space of parameters consistent with the data and the prior, i.e. the
   posterior distribution. The posterior assigns high probability to parameters
   that are consistent with both the data and the prior, and low probability to
   inconsistent parameters. While SNPE directly learns the posterior
   distribution, SNLE and SNRE need an extra MCMC sampling step to construct a
   posterior.
4. If needed, an initial estimate of the posterior can be used to adaptively
   generate additional informative simulations.

See [Cranmer, Brehmer, Louppe (2020)](https://doi.org/10.1073/pnas.1912789117) for a recent
review on simulation-based inference.

## Getting started with the `sbi` package

Once `sbi` is installed, inference can be run in a single line of code

```python
posterior = infer(simulator, prior, method='SNPE', num_simulations=1000)
```

or in a few lines for more flexibility:

```python
inference = SNPE(prior=prior)
_ = inference.append_simulations(theta, x).train()
posterior = inference.build_posterior()
```

`sbi` lets you choose from a variety of _amortized_ and _sequential_ SBI methods:

Amortized methods return a posterior that can be applied to many different
observations without retraining, whereas sequential methods focus the inference
on one particular observation to be more simulation-efficient.


For an overview of implemented methods see [the Inference API's reference](
reference/inference/), or checkout or [GitHub page](https://github.com/mackelab/sbi).
