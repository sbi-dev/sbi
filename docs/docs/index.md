# `sbi`: simulation-based inference toolkit

`sbi` is a Python package for simulation-based inference, designed to meet the needs of
both researchers and practitioners. Whether you need fine-grained control or an
easy-to-use interface, `sbi` has you covered.

With `sbi`, you can perform parameter inference using Bayesian inference: Given a
simulator that models a real-world process, SBI estimates the full posterior
distribution over the simulator’s parameters based on observed data. This distribution
indicates the most likely parameter values while additionally quantifying uncertainty
and revealing potential interactions between parameters.

`sbi` provides access to simulation-based inference methods via a user-friendly
interface:

```python
import torch
from sbi.inference import NPE

# define shifted Gaussian simulator.
def simulator(θ): return θ + torch.randn_like(θ)
# draw parameters from Gaussian prior.
θ = torch.randn(1000, 2)
# simulate data
x = simulator(θ)

# choose sbi method and train
inference = NPE()
inference.append_simulations(θ, x).train()

# do inference given observed data
x_o = torch.ones(2)
posterior = inference.build_posterior()
samples = posterior.sample((1000,), x=x_o)
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

-  :rocket: [__Tutorials and Examples__](tutorials/index.md)
   <br/><br/>
   *Various examples illustrating how to<br/> [get
   started](tutorials/00_getting_started.md) or use the `sbi` package.*

-  :building_construction: [__Reference API__](reference/index.md)
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
   inconsistent parameters. While NPE directly learns the posterior
   distribution, NLE and NRE need an extra MCMC sampling step to construct a
   posterior.
4. If needed, an initial estimate of the posterior can be used to adaptively
   generate additional informative simulations.

See [Cranmer, Brehmer, Louppe (2020)](https://doi.org/10.1073/pnas.1912789117)
for a recent review on simulation-based inference.

## Implemented algorithms

`sbi` implements a variety of _amortized_ and _sequential_ SBI methods.

Amortized methods return a posterior that can be applied to many different
observations without retraining (e.g., NPE), whereas sequential methods focus
the inference on one particular observation to be more simulation-efficient
(e.g., SNPE).

Below, we list all implemented methods and the corresponding publications. To see
how to access these methods in `sbi`, check out our [Inference API's reference](
reference/inference.md) and the [tutorial on implemented
methods](tutorials/16_implemented_methods.md).

### Posterior estimation (`(S)NPE`)

- **Fast ε-free Inference of Simulation Models with Bayesian Conditional Density
  Estimation**<br> by Papamakarios & Murray (NeurIPS 2016)
  <br>[[PDF]](https://papers.nips.cc/paper/6084-fast-free-inference-of-simulation-models-with-bayesian-conditional-density-estimation.pdf)
  [[BibTeX]](https://papers.nips.cc/paper/6084-fast-free-inference-of-simulation-models-with-bayesian-conditional-density-estimation/bibtex)

- **Flexible statistical inference for mechanistic models of neural dynamics**
  <br> by Lueckmann, Goncalves, Bassetto, Öcal, Nonnenmacher & Macke (NeurIPS
  2017)
  <br>[[PDF]](https://papers.nips.cc/paper/6728-flexible-statistical-inference-for-mechanistic-models-of-neural-dynamics.pdf)
  [[BibTeX]](https://papers.nips.cc/paper/6728-flexible-statistical-inference-for-mechanistic-models-of-neural-dynamics/bibtex)

- **Automatic posterior transformation for likelihood-free inference**<br> by Greenberg, Nonnenmacher & Macke (ICML 2019) <br>[[PDF]](http://proceedings.mlr.press/v97/greenberg19a/greenberg19a.pdf) [[BibTeX]](data:text/plain;charset=utf-8,%0A%0A%0A%0A%0A%0A%40InProceedings%7Bpmlr-v97-greenberg19a%2C%0A%20%20title%20%3D%20%09%20%7BAutomatic%20Posterior%20Transformation%20for%20Likelihood-Free%20Inference%7D%2C%0A%20%20author%20%3D%20%09%20%7BGreenberg%2C%20David%20and%20Nonnenmacher%2C%20Marcel%20and%20Macke%2C%20Jakob%7D%2C%0A%20%20booktitle%20%3D%20%09%20%7BProceedings%20of%20the%2036th%20International%20Conference%20on%20Machine%20Learning%7D%2C%0A%20%20pages%20%3D%20%09%20%7B2404--2414%7D%2C%0A%20%20year%20%3D%20%09%20%7B2019%7D%2C%0A%20%20editor%20%3D%20%09%20%7BChaudhuri%2C%20Kamalika%20and%20Salakhutdinov%2C%20Ruslan%7D%2C%0A%20%20volume%20%3D%20%09%20%7B97%7D%2C%0A%20%20series%20%3D%20%09%20%7BProceedings%20of%20Machine%20Learning%20Research%7D%2C%0A%20%20address%20%3D%20%09%20%7BLong%20Beach%2C%20California%2C%20USA%7D%2C%0A%20%20month%20%3D%20%09%20%7B09--15%20Jun%7D%2C%0A%20%20publisher%20%3D%20%09%20%7BPMLR%7D%2C%0A%20%20pdf%20%3D%20%09%20%7Bhttp%3A%2F%2Fproceedings.mlr.press%2Fv97%2Fgreenberg19a%2Fgreenberg19a.pdf%7D%2C%0A%20%20url%20%3D%20%09%20%7Bhttp%3A%2F%2Fproceedings.mlr.press%2Fv97%2Fgreenberg19a.html%7D%2C%0A%20%20abstract%20%3D%20%09%20%7BHow%20can%20one%20perform%20Bayesian%20inference%20on%20stochastic%20simulators%20with%20intractable%20likelihoods%3F%20A%20recent%20approach%20is%20to%20learn%20the%20posterior%20from%20adaptively%20proposed%20simulations%20using%20neural%20network-based%20conditional%20density%20estimators.%20However%2C%20existing%20methods%20are%20limited%20to%20a%20narrow%20range%20of%20proposal%20distributions%20or%20require%20importance%20weighting%20that%20can%20limit%20performance%20in%20practice.%20Here%20we%20present%20automatic%20posterior%20transformation%20(APT)%2C%20a%20new%20sequential%20neural%20posterior%20estimation%20method%20for%20simulation-based%20inference.%20APT%20can%20modify%20the%20posterior%20estimate%20using%20arbitrary%2C%20dynamically%20updated%20proposals%2C%20and%20is%20compatible%20with%20powerful%20flow-based%20density%20estimators.%20It%20is%20more%20flexible%2C%20scalable%20and%20efficient%20than%20previous%20simulation-based%20inference%20techniques.%20APT%20can%20operate%20directly%20on%20high-dimensional%20time%20series%20and%20image%20data%2C%20opening%20up%20new%20applications%20for%20likelihood-free%20inference.%7D%0A%7D%0A)

- **BayesFlow: Learning complex stochastic models with invertible neural
  networks**<br> by Radev, S. T., Mertens, U. K., Voss, A., Ardizzone, L., & Köthe, U. (IEEE transactions on neural networks and learning systems 2020)<br>
  [[Paper]](https://ieeexplore.ieee.org/abstract/document/9298920)

- **Truncated proposals for scalable and hassle-free simulation-based
  inference** <br> by Deistler, Goncalves & Macke (NeurIPS 2022)
  <br>[[Paper]](https://arxiv.org/abs/2210.04815)

- **Flow matching for scalable simulation-based inference**<br> by Dax, M., Wildberger,
  J., Buchholz, S., Green, S. R., Macke, J. H., & Schölkopf, B. (NeurIPS, 2023)<br>
  [[Paper]](https://arxiv.org/abs/2305.17161)

- **Compositional Score Modeling for Simulation-Based Inference**<br> by Geffner, T.,
  Papamakarios, G., & Mnih, A. (ICML 2023)<br>
  [[Paper]](https://proceedings.mlr.press/v202/geffner23a.html)

### Likelihood-estimation (`(S)NLE`)

- **Sequential neural likelihood: Fast likelihood-free inference with
  autoregressive flows**<br> by Papamakarios, Sterratt & Murray (AISTATS 2019)
  <br>[[PDF]](http://proceedings.mlr.press/v89/papamakarios19a/papamakarios19a.pdf)
  [[BibTeX]](https://gpapamak.github.io/bibtex/snl.bib)

- **Variational methods for simulation-based inference** <br> by Glöckler,
  Deistler, Macke (ICLR 2022) <br>[[Paper]](https://arxiv.org/abs/2203.04176)

- **Flexible and efficient simulation-based inference for models of
  decision-making** <br> by Boelts, Lueckmann, Gao, Macke (Elife 2022)
  <br>[[Paper]](https://elifesciences.org/articles/77220)

### Likelihood-ratio-estimation (`(S)NRE`)

- **Likelihood-free MCMC with Amortized Approximate Likelihood Ratios**<br> by
  Hermans, Begy & Louppe (ICML 2020)
  <br>[[PDF]](http://proceedings.mlr.press/v119/hermans20a/hermans20a.pdf)

- **On Contrastive Learning for Likelihood-free Inference**<br> by Durkan,
  Murray & Papamakarios (ICML 2020)
  <br>[[PDF]](http://proceedings.mlr.press/v119/durkan20a/durkan20a.pdf)

- **Towards Reliable Simulation-Based Inference with Balanced Neural Ratio
  Estimation**<br> by Delaunoy, Hermans, Rozet, Wehenkel & Louppe (NeurIPS 2022)
  <br>[[PDF]](https://arxiv.org/pdf/2208.13624.pdf)

- **Contrastive Neural Ratio Estimation**<br> by Benjamin Kurt Miller, Christoph
  Weniger & Patrick Forré (NeurIPS 2022)
  <br>[[PDF]](https://arxiv.org/pdf/2210.06170.pdf)

### Diagnostics

- **Simulation-based calibration**<br> by Talts, Betancourt, Simpson, Vehtari,
  Gelman (arxiv 2018)<br>[[Paper]](https://arxiv.org/abs/1804.06788)

- **Expected coverage (sample-based)**<br> as computed in Deistler, Goncalves, &
  Macke (NeurIPS 2022)<br>[[Paper]](https://arxiv.org/abs/2210.04815) and in
  Rozet & Louppe [[Paper]](https://matheo.uliege.be/handle/2268.2/12993)

- **Local C2ST**<br> by Linhart, Gramfort & Rodrigues (NeurIPS
  2023)<br>[[Paper](https://arxiv.org/abs/2306.03580)]

- **TARP**<br> by Lemos, Coogan, Hezaveh & Perreault-Levasseur (ICML
  2023)<br>[[Paper]](https://arxiv.org/abs/2302.03026)
