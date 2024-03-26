# Tutorials for using the `sbi` toolbox

These `sbi` tutorials are aimed at two sepatate groups
1.  _users_, e.g., domain scientists that aim to get an introduction to the method to then apply it to their (mechanistic) models
2. _contributers_ who develop methods and/or plan to contribute to the `sbi` toolbox

Before running the notebooks, follow our instructions to [install sbi](../README.md).
The numbers of the notebooks are not informative of the order, please follow this structure depending on which group you identify with.

## I want to start applying `sbi` (_user_)
Before going through the tutorial notebooks, make sure to read through the **Overview, Motivation and Approach** below.

- [Getting started](00_getting_started_flexible.ipynb) introduces the `sbi` package and its core functionality.
- [Inferring parameters for multiple observations](01_gaussian_amortized.ipynb) introduces the concept of amortization, i.e., that we do not need to retrain our inference procedure for different observations.
- [The example for a scientific simulator from neuroscience (Hodgkin-Huxley)](../examples/00_HH_simulator.ipynb), shows how `sbi` can be applied to scientific use cases building on the previous two examples.
- [Inferring parameters for a single observation ](03_multiround_inference.ipynb) introduces the concept of multi round inference for a single observation to be more sampling efficient.

[All implemented methods](16_implemented_methods.ipynb) provides an overview of the implemented inference methods and how to call them.

Once you have familiarised yourself with the methods and identified how to apply SBI to your use case, ensure you work through the **Diagnostics** tutorials linked below, to identify failure cases and assess the quality of your inference.


## I develop methods for `sbi` (_contributer_)

### Introduction
- [Getting started](00_getting_started_flexible.ipynb) introduces the `sbi` package and its core functionality.
- [Inferring parameters for multiple observations ](01_gaussian_amortized.ipynb)introduces the concept of amortization.
- [All implemented methods](16_implemented_methods.ipynb) provides an overview of the implemented inference methods and how to call them.

### Advanced:
- [Multi-round inference](03_multiround_inference.ipynb)
- [Sampling algorithms in sbi](11_sampler_interface.ipynb)
- [Custom density estimators](04_density_estimators.ipynb)
- [Learning summary statistics](05_embedding_net.ipynb)
- [SBI with trial-based data](14_iid_data_and_permutation_invariant_embeddings.ipynb)
- [Handling invalid simulations](08_restriction_estimator.ipynb)
- [Crafting summary statistics](10_crafting_summary_statistics.ipynb)

### Diagnostics:
- [Posterior predictive checks](12_diagnostics_posterior_predictive_check.ipynb)
- [Simulation-based calibration](13_diagnostics_simulation_based_calibration.ipynb)
- [Density plots and MCMC diagnostics with ArviZ](15_mcmc_diagnostics_with_arviz.ipynb)

### Analysis:
- [Conditional distributions](07_conditional_distributions.ipynb)
- [Posterior sensitivity analysis](09_sensitivity_analysis.ipynb) shows how to perform a sensitivity analysis of a model.

### Examples:
- [Hodgkin-Huxley example](../examples/00_HH_simulator.ipynb)
- [Decision making model](../examples/01_decision_making_model.ipynb)

Please first read our [contributer guide](../CONTRIBUTING.md) and our [code of conduct](../CODE_OF_CONDUCT.md).




## Overview


`sbi` lets you choose from a variety of _amortized_ and _sequential_ SBI methods:

Amortized methods return a posterior that can be applied to many different observations without retraining,
whereas sequential methods focus the inference on one particular observation to be more simulation-efficient.
For an overview of implemented methods see below, or checkout our [GitHub page](https://github.com/mackelab/sbi).

- To learn about the general motivation behind simulation-based inference, and the
  inference methods included in `sbi`, read on below.

- For example applications to canonical problems in neuroscience, browse the recent
  research article [Training deep neural density estimators to identify mechanistic models of neural dynamics](https://doi.org/10.7554/eLife.56261).



## Motivation and approach

Many areas of science and engineering make extensive use of complex, stochastic,
numerical simulations to describe the structure and dynamics of the processes being
investigated.

A key challenge in simulation models for science, is constraining the parameters of these models, which are intepretable quantities, with observational data. Bayesian
inference provides a general and powerful framework to invert the simulators, i.e.
describe the parameters which are consistent both with empirical data and prior
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

**Goal: Algorithmically identify mechanistic models which are consistent with data.**

Each of the methods above needs three inputs: A candidate mechanistic model, prior
knowledge or constraints on model parameters, and observational data (or summary statistics
thereof).

The methods then proceed by

1. sampling parameters from the prior followed by simulating synthetic data from
   these parameters,
2. learning the (probabilistic) association between data (or
   data features) and underlying parameters, i.e. to learn statistical inference from
   simulated data. The way in which this association is learned differs between the
   above methods, but all use deep neural networks.
3. This learned neural network is then applied to empirical data to derive the full
   space of parameters consistent with the data and the prior, i.e. the posterior
   distribution. High posterior probability is assigned to parameters which are
   consistent with both the data and the prior, low probability to inconsistent
   parameters. While SNPE directly learns the posterior distribution, SNLE and SNRE need
   an extra MCMC sampling step to construct a posterior.
4. If needed, an initial estimate of the posterior can be used to adaptively generate
   additional informative simulations.

## Publications

See [Cranmer, Brehmer, Louppe (2020)](https://doi.org/10.1073/pnas.1912789117) for a recent
review on simulation-based inference.

The following papers offer additional details on the inference methods implemented in `sbi`.
You can find a tutorial on how to run each of these methods [here](https://sbi-dev.github.io/sbi/tutorial/16_implemented_methods/).

### Posterior estimation (`(S)NPE`)

- **Fast ε-free Inference of Simulation Models with Bayesian Conditional Density Estimation**<br> by Papamakarios & Murray (NeurIPS 2016) <br>[[Paper]](https://papers.nips.cc/paper/6084-fast-free-inference-of-simulation-models-with-bayesian-conditional-density-estimation.pdf) [[BibTeX]](https://papers.nips.cc/paper/6084-fast-free-inference-of-simulation-models-with-bayesian-conditional-density-estimation/bibtex)

- **Flexible statistical inference for mechanistic models of neural dynamics** <br> by Lueckmann, Goncalves, Bassetto, Öcal, Nonnenmacher & Macke (NeurIPS 2017) <br>[[PDF]](https://papers.nips.cc/paper/6728-flexible-statistical-inference-for-mechanistic-models-of-neural-dynamics.pdf) [[BibTeX]](https://papers.nips.cc/paper/6728-flexible-statistical-inference-for-mechanistic-models-of-neural-dynamics/bibtex)

- **Automatic posterior transformation for likelihood-free inference**<br>by Greenberg, Nonnenmacher & Macke (ICML 2019) <br>[[Paper]](http://proceedings.mlr.press/v97/greenberg19a/greenberg19a.pdf)

- **Truncated proposals for scalable and hassle-free simulation-based inference** <br> by Deistler, Goncalves & Macke (NeurIPS 2022) <br>[[Paper]](https://arxiv.org/abs/2210.04815)


### Likelihood-estimation (`(S)NLE`)

- **Sequential neural likelihood: Fast likelihood-free inference with autoregressive flows**<br>by Papamakarios, Sterratt & Murray (AISTATS 2019) <br>[[PDF]](http://proceedings.mlr.press/v89/papamakarios19a/papamakarios19a.pdf) [[BibTeX]](https://gpapamak.github.io/bibtex/snl.bib)

- **Variational methods for simulation-based inference** <br> by Glöckler, Deistler, Macke (ICLR 2022) <br>[[Paper]](https://arxiv.org/abs/2203.04176)

- **Flexible and efficient simulation-based inference for models of decision-making** <br> by Boelts, Lueckmann, Gao, Macke (Elife 2022) <br>[[Paper]](https://elifesciences.org/articles/77220)


### Likelihood-ratio-estimation (`(S)NRE`)

- **Likelihood-free MCMC with Amortized Approximate Likelihood Ratios**<br>by Hermans, Begy & Louppe (ICML 2020) <br>[[PDF]](http://proceedings.mlr.press/v119/hermans20a/hermans20a.pdf)

- **On Contrastive Learning for Likelihood-free Inference**<br>Durkan, Murray & Papamakarios (ICML 2020) <br>[[PDF]](http://proceedings.mlr.press/v119/durkan20a/durkan20a.pdf)

- **Towards Reliable Simulation-Based Inference with Balanced Neural Ratio Estimation**<br>by Delaunoy, Hermans, Rozet, Wehenkel & Louppe (NeurIPS 2022) <br>[[PDF]](https://arxiv.org/pdf/2208.13624.pdf)

- **Contrastive Neural Ratio Estimation**<br>Benjamin Kurt Miller, Christoph Weniger, Patrick Forré (NeurIPS 2022) <br>[[PDF]](https://arxiv.org/pdf/2210.06170.pdf)

### Utilities

- **Restriction estimator**<br>by Deistler, Macke & Goncalves (PNAS 2022) <br>[[Paper]](https://www.pnas.org/doi/10.1073/pnas.2207632119)

- **Simulation-based calibration**<br>by Talts, Betancourt, Simpson, Vehtari, Gelman (arxiv 2018) <br>[[Paper]](https://arxiv.org/abs/1804.06788))

- **Expected coverage (sample-based)**<br>as computed in Deistler, Goncalves, Macke [[Paper]](https://arxiv.org/abs/2210.04815) and in Rozet, Louppe [[Paper]](https://matheo.uliege.be/handle/2268.2/12993)
