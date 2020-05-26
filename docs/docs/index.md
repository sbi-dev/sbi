# `sbi`

`sbi`: A Python toolbox to perform simulation-based inference using density-estimation approaches.

The focus of `sbi` is _Sequential Neural Posterior Estimation_ (SNPE). In SNPE, a neural network is trained to perform Bayesian inference on simulated data.

- To see illustrations of SNPE on canonical problems in neuroscience, read our preprint:
[Training deep neural density estimators to identify mechanistic models of neural dynamics](https://www.biorxiv.org/content/10.1101/838383v3).
- To learn more about the general motivation behind simulation-based inference, and algorithms included in `sbi`, keep on reading.


## Motivation and approach

Many areas of science and engineering make extensive use of complex, stochastic, numerical simulations to describe the structure and dynamics of the processes being investigated. A key challenge in simulation-based science is linking simulation models to empirical data: Bayesian inference provides a general and powerful framework for identifying the set of parameters which are consistent both with empirical data and prior knowledge.

One of the key quantities required for statistical inference, the likelihood of observed data given parameters, $\mathcal{L}(\theta) = p(x_o|\theta)$, is typically intractable for
simulation-based models, rendering conventional statistical approaches inapplicable.

Sequential Neural Posterior Estimation (SNPE) is a powerful machine-learning technique to address this problem.


![](./static/goal.png)

**Goal: Algorithmically identify mechanistic models which are consistent with data.**

SNPE takes three inputs: A candidate mechanistic model, prior knowledge or constraints on model parameters, and data (or summary statistics). SNPE proceeds by:

1. sampling parameters from the prior and simulating synthetic datasets from these parameters, and
2. using a deep density estimation neural network to learn the (probabilistic) association between data (or data features) and underlying parameters, i.e. to learn statistical inference from simulated data.
3. This density estimation network is then applied to empirical data to derive the full space of parameters consistent with the data and the prior, i.e. the posterior distribution. High posterior probability is assigned to parameters which are consistent with both the data and the prior, low probability to inconsistent parameters.
4. If needed, an initial estimate of the posterior can be used to adaptively generate additional informative simulations.


## Publications

Algorithms included in `sbi` were published in the following papers, which provide additional information:


- **Fast ε-free Inference of Simulation Models with Bayesian Conditional Density Estimation**<br> by Papamakarios & Murray (NeurIPS 2016) <br>[[PDF]](https://papers.nips.cc/paper/6084-fast-free-inference-of-simulation-models-with-bayesian-conditional-density-estimation.pdf) [[BibTeX]](https://papers.nips.cc/paper/6084-fast-free-inference-of-simulation-models-with-bayesian-conditional-density-estimation/bibtex)

- **Flexible statistical inference for mechanistic models of neural dynamics** <br> by Lueckmann, Goncalves, Bassetto, Öcal, Nonnenmacher & Macke (NeurIPS 2017) <br>[[PDF]](https://papers.nips.cc/paper/6728-flexible-statistical-inference-for-mechanistic-models-of-neural-dynamics.pdf) [[BibTeX]](https://papers.nips.cc/paper/6728-flexible-statistical-inference-for-mechanistic-models-of-neural-dynamics/bibtex)

- **Automatic posterior transformation for likelihood-free inference**<br>by Greenberg, Nonnenmacher & Macke (ICML 2019) <br>[[PDF]](http://proceedings.mlr.press/v97/greenberg19a/greenberg19a.pdf) [[BibTeX]](data:text/plain;charset=utf-8,%0A%0A%0A%0A%0A%0A%40InProceedings%7Bpmlr-v97-greenberg19a%2C%0A%20%20title%20%3D%20%09%20%7BAutomatic%20Posterior%20Transformation%20for%20Likelihood-Free%20Inference%7D%2C%0A%20%20author%20%3D%20%09%20%7BGreenberg%2C%20David%20and%20Nonnenmacher%2C%20Marcel%20and%20Macke%2C%20Jakob%7D%2C%0A%20%20booktitle%20%3D%20%09%20%7BProceedings%20of%20the%2036th%20International%20Conference%20on%20Machine%20Learning%7D%2C%0A%20%20pages%20%3D%20%09%20%7B2404--2414%7D%2C%0A%20%20year%20%3D%20%09%20%7B2019%7D%2C%0A%20%20editor%20%3D%20%09%20%7BChaudhuri%2C%20Kamalika%20and%20Salakhutdinov%2C%20Ruslan%7D%2C%0A%20%20volume%20%3D%20%09%20%7B97%7D%2C%0A%20%20series%20%3D%20%09%20%7BProceedings%20of%20Machine%20Learning%20Research%7D%2C%0A%20%20address%20%3D%20%09%20%7BLong%20Beach%2C%20California%2C%20USA%7D%2C%0A%20%20month%20%3D%20%09%20%7B09--15%20Jun%7D%2C%0A%20%20publisher%20%3D%20%09%20%7BPMLR%7D%2C%0A%20%20pdf%20%3D%20%09%20%7Bhttp%3A%2F%2Fproceedings.mlr.press%2Fv97%2Fgreenberg19a%2Fgreenberg19a.pdf%7D%2C%0A%20%20url%20%3D%20%09%20%7Bhttp%3A%2F%2Fproceedings.mlr.press%2Fv97%2Fgreenberg19a.html%7D%2C%0A%20%20abstract%20%3D%20%09%20%7BHow%20can%20one%20perform%20Bayesian%20inference%20on%20stochastic%20simulators%20with%20intractable%20likelihoods%3F%20A%20recent%20approach%20is%20to%20learn%20the%20posterior%20from%20adaptively%20proposed%20simulations%20using%20neural%20network-based%20conditional%20density%20estimators.%20However%2C%20existing%20methods%20are%20limited%20to%20a%20narrow%20range%20of%20proposal%20distributions%20or%20require%20importance%20weighting%20that%20can%20limit%20performance%20in%20practice.%20Here%20we%20present%20automatic%20posterior%20transformation%20(APT)%2C%20a%20new%20sequential%20neural%20posterior%20estimation%20method%20for%20simulation-based%20inference.%20APT%20can%20modify%20the%20posterior%20estimate%20using%20arbitrary%2C%20dynamically%20updated%20proposals%2C%20and%20is%20compatible%20with%20powerful%20flow-based%20density%20estimators.%20It%20is%20more%20flexible%2C%20scalable%20and%20efficient%20than%20previous%20simulation-based%20inference%20techniques.%20APT%20can%20operate%20directly%20on%20high-dimensional%20time%20series%20and%20image%20data%2C%20opening%20up%20new%20applications%20for%20likelihood-free%20inference.%7D%0A%7D%0A)

- **On Contrastive Learning for Likelihood-free Inference**<br>by Durkan,
  Murray, Papamakarios (arXiv 2020) <br>[[PDF]](https://arxiv.org/abs/2002.03712)

We refer to these algorithms as SNPE-A, SNPE-B, and SNPE-C/APT, respectively.


As an alternative to directly estimating the posterior on parameters given data, it is also possible to estimate the likelihood of data given parameters, and then subsequently draw posterior samples using MCMC ([Papamakarios, Sterratt & Murray, 2019](http://proceedings.mlr.press/v89/papamakarios19a/papamakarios19a.pdf)[^1], [Lueckmann, Karaletsos, Bassetto, Macke, 2019](http://proceedings.mlr.press/v96/lueckmann19a/lueckmann19a.pdf)). Depending on the problem, approximating the likelihood can be more or less effective than SNPE techniques.

See [Cranmer, Brehmer, Louppe (2019)](https://arxiv.org/abs/1911.01429) for a recent review on simulation-based inference and our recent preprint [Training deep neural density estimators to identify mechanistic models of neural dynamics (Goncalves et al., 2019)](https://www.biorxiv.org/content/10.1101/838383v1) for applications to canonical problems in neuroscience.

[^1]: Code for SNL is available from the [original repository](https://github.com/gpapamak/snl) or as a [python 3 package](https://github.com/mnonnenm/SNL_py3port/tree/master).
