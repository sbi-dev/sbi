---
title: 'sbi - a toolkit for simulation-based inference'
tags:
  - simulation science
  - likelihood-free inference
  - bayesian inference
  - system identification
  - parameter identification
authors: 
 - name: Alvaro Tejero-Cantero
   orcid: 0000-0002-8768-4227
   affiliation: "e, 1"
 - name: Jan Boelts
   orcid: 0000-0003-4979-7092
   affiliation: "e, 1"
 - name: Michael Deistler
   orcid: 0000-0002-3573-0404
   affiliation: "e, 1"
 - name: Jan-Matthis Lueckmann
   orcid: 0000-0003-4320-4663
   affiliation: "e, 1"
 - name: Conor Durkan
   orcid: 0000-0001-9333-7777
   affiliation: "e, 2"
 - name: Pedro J. Gonçalves
   orcid: 0000-0002-6987-4836
   affiliation: "1, 3"
 - name: David S. Greenberg
   orcid: 0000-0002-8515-0459
   affiliation: "1, 4"
 - name: Jakob H. Macke
   orcid: 0000-0001-5154-8912
   affiliation: "1, 5, 6"
affiliations:
 - name: Equally contributing authors
   index: e
 - name: Computational Neuroengineering, Department of Electrical and Computer Engineering, Technical University of Munich
   index: 1
 - name: School of Informatics, University of Edinburgh
   index: 2
 - name: Neural Systems Analysis, Center of Advanced European Studies and Research (caesar), Bonn 
   index: 3
 - name:  Model-Driven Machine Learning, Centre for Materials and Coastal Research, Helmholtz-Zentrum Geesthacht
   index: 4
 - name: Machine Learning in Science, University of Tübingen
   index: 5
 - name: Empirical Inference, Max Planck Institute for Intelligent Systems, Tübingen
   index: 6
date: 23 June 2020.
bibliography: paper.bib
---

# Summary
Scientists and engineers employ stochastic numerical simulators to model empirically observed phenomena. In contrast to purely statistical models, simulators express scientific principles that provide powerful inductive biases, improve generalization to new data or scenarios and allow for fewer, more interpretable and domain-relevant parameters. 
Despite these advantages, tuning a simulator’s parameters so that its outputs match data is challenging. Simulation-based inference (SBI) seeks to identify parameter sets that a) are compatible with prior knowledge and b) match empirical observations. Importantly, SBI does not seek to recover a single ‘best’ data-compatible parameter set, but rather to identify all high probability regions of parameter space that explain observed data, and thereby to quantify parameter uncertainty. In Bayesian terminology, SBI aims to retrieve the posterior distribution over the parameters of interest. In contrast to conventional Bayesian inference, SBI is also applicable when one can run model simulations, but no formula or algorithm exists for evaluating the probability of data given parameters, i.e. the likelihood.

We present `sbi`, a PyTorch-based package that implements SBI algorithms based on neural networks. `sbi` facilitates inference on black-box simulators for practising scientists and engineers by providing a unified interface to state-of-the-art algorithms together with documentation and tutorials.

# Motivation
Bayesian inference is a principled approach for determining parameters consistent with empirical observations: Given a prior over parameters, a stochastic simulator, and observations, it returns a posterior distribution. In cases where the simulator likelihood _can_ be evaluated, many methods for approximate Bayesian inference exist [e.g., @metropolis1953; @neal2003; @graham2017; @le2016; @baydin2020]. For more general simulators, however, evaluating the likelihood of data given parameters might be computationally intractable. Traditional algorithms for this 'likelihood-free' setting [@cranmer2019] are based on Monte-Carlo rejection [@pritchard1999; @sisson2007], an approach known as  _Approximate Bayesian Computation_ (ABC). More recently, algorithms based on neural networks have been developed [@papamakarios2016; @lueckmann2017; @papamakarios2019a; @greenberg2019; @hermans2019]. These algorithms are not based on rejecting simulations, but rather train deep neural conditional density estimators or classifiers on simulated data. 
To aid in effective application of these algorithms to a wide range of problems, `sbi` closely integrates with PyTorch and offers state-of-the-art neural network-based SBI algorithms [@papamakarios2019a; @hermans2019; @greenberg2019] with flexible choice of network architectures and flow-based density estimators. With `sbi`, researchers can easily implement new neural inference algorithms, benefiting from the infrastructure to manage simulators and a unified posterior representation. Users, in turn, can profit from a single inference interface that allows them to either use their own custom neural network, or choose from a growing library of preconfigured options provided with the package.

## Related software and use in research

We are aware of several mature packages that implement SBI algorithms. `elfi` [@elfi2018] is a package offering BOLFI, a Gaussian process-based algorithm [@gutmann2015], and some classical ABC algorithms. The package `carl` [@louppe2016] implements the algorithm described in @cranmer2015carl. Two other SBI packages, currently under development, are `hypothesis` [@hypothesis-repo] and `pydelfi` [@pydelfi-repo]. `pyabc` [@Klinger2018] and `ABCpy` [@abcpy-repo] are two packages offering a diversity of ABC algorithms.

`sbi` is closely integrated with PyTorch [@paszke2019] and uses `nflows` [@nflows-repo] for flow-based density estimators. `sbi` builds on experience accumulated developing `delfi` [@delfi-repo], which it succeeds. `delfi` was based on `theano` [@theano] (development discontinued) and developed both for SBI research [@greenberg2019; @lueckmann2017] and for scientific applications [@goncalves2019]. The `sbi` codebase started as a fork of `lfi` [@lfi-repo], developed for @durkan2020.

# Description
`sbi` currently implements three families of neural inference algorithms:

* Sequential Neural _Posterior_ Estimation (SNPE) trains a deep neural density estimator that directly estimates the posterior distribution of parameters given data. Afterwards, it can sample parameter sets from the posterior, or evaluate the posterior density on any parameter set. Currently, SNPE-C [@greenberg2019] is implemented in `sbi`. 

* Sequential Neural _Likelihood_ Estimation (SNLE) [@papamakarios2019a] trains a deep neural density estimator of the likelihood, which then allows to sample from the posterior using e.g. MCMC.

* Sequential Neural _Ratio_ Estimation (SNRE) [@hermans2019; @durkan2020] trains a classifier to estimate density ratios, which in turn can be used to sample from the posterior e.g. with MCMC. 

The inference step returns a `NeuralPosterior` object that represents the uncertainty about the parameters conditional on an observation, i.e. the posterior distribution. This object can be sampled from —and if the chosen algorithm allows, evaluated— with the same API as a standard PyTorch probability distribution.

An important challenge in making SBI algorithms usable by a broader community is to deal with diverse, often pre-existing, complex simulators. `sbi` works with any simulator as long as it can be wrapped in a Python callable. Furthermore, `sbi` ensures that custom simulators work well with neural networks, e.g. by performing automatic shape inference, standardizing inputs or handling failed simulations. To maximize simulator performance, `sbi` leverages vectorization where available and optionally parallelizes simulations using `joblib` [@joblib]. Moreover, if dimensionality reduction of the simulator output is desired, `sbi` can use a trainable summarizing network to extract relevant features from raw simulator output and spare the user manual feature engineering.

In addition to the full-featured interface, `sbi` provides also a _simple_ interface which consists of a single function call with reasonable defaults. This allows new users to get familiarized with simulation-based inference and quickly obtain results without having to define custom networks or tune hyperparameters.

With `sbi`, we aim to support scientific discovery and computational engineering by making Bayesian inference applicable to the widest class of models (simulators with no likelihood available), and practical for complex problems. We have designed an open architecture and adopted community-oriented development practices in order to invite other machine-learning researchers to join us in this long-term vision.

# Acknowledgements

This work has been supported by the German Federal Ministry of Education and Research (BMBF, project \`ADIMEM', FKZ 01IS18052 A-D), the German Research Foundation (DFG) through  SFB 1089 \`Synaptic Microcircuits', SPP 2041 `Computational Connectomics' and Germany’s Excellence Strategy – EXC-Number 2064/1 – Project number 390727645.

Conor Durkan was supported by the EPSRC Centre for Doctoral Training in Data Science, funded by the UK Engineering and Physical Sciences Research Council (grant EP/L016427/1) and the University of Edinburgh.

We are grateful to Artur Bekasov, George Papamakarios and Iain Murray for making `nflows` [@nflows-repo] available, a package for normalizing flow-based density estimation which `sbi` leverages extensively.

# References
