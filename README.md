![Tests](https://github.com/mackelab/sbi/workflows/Tests/badge.svg?branch=master)
[![PyPI version](https://badge.fury.io/py/sbi.svg)](https://badge.fury.io/py/sbi)
[![Contributions welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg?style=flat)](https://github.com/mackelab/sbi/blob/master/docs/docs/contribute.md)
[![GitHub license](https://img.shields.io/github/license/mackelab/sbi)](https://github.com/mackelab/sbi/blob/master/LICENSE.txt)
## sbi: simulation-based inference

`sbi` is a PyTorch package for simulation-based inference. Simulation-based inference is
the process of finding the parameters of a simulator from observations.

`sbi` takes a Bayesian approach and returns a full posterior distribution
over the parameters, conditional on the observations. This posterior can be amortized (i.e.
useful for any observation) or focused (i.e. tailored to a particular observation), with different
computational trade-offs.

`sbi` offers a simple interface for one-line posterior inference

```python
from sbi inference import infer
# import your simulator, define your prior over the parameters
parameter_posterior = infer(simulator, prior, method='SNPE')
```
See below for the available methods of inference, `SNPE`, `SNRE` and `SNLE`.

## Installation
We recommend to use a [`conda`](https://docs.conda.io/en/latest/miniconda.html) virtual
environment. The Python version must be 3.7 or higher.
If you don't have one, it should work with the following steps
```shell
# 1. install miniconda from https://docs.conda.io/en/latest/miniconda.html
$ (dependent on your system, follow instructions on website)
# 2. create an environment for sbi (indicate Python 3.7 or higher); activate it
$ conda create -n sbi_env python=3.7 && conda activate sbi_env
```
Then install `sbi`:
```shell
$ pip install sbi
```
To test the installation, drop into a python prompt and run 
```python
from sbi.example.minimal import simple
posterior = simple()
print(posterior)
```

## Inference methods
The following methods are currently available

#### Sequential Neural Posterior Estimation (SNPE)

* [`SNPE_C`](https://www.mackelab.org/sbi/reference/#sbi.inference.snpe.snpe_c.SNPE_C) or `APT` from Greenberg D, Nonnenmacher M, and Macke J [_Automatic
  PosteriorTransformation for likelihood-free
  inference_](https://arxiv.org/abs/1905.07488) (ICML 2020).

<!-- 
- **Fast ε-free Inference of Simulation Models with Bayesian Conditional Density
  Estimation**<br> by Papamakarios G. and Murray I. (NeurIPS 2016)
  <br>[[PDF]](https://papers.nips.cc/paper/6084-fast-free-inference-of-simulation-models-with-bayesian-conditional-density-estimation.pdf)
  [[BibTeX]](https://papers.nips.cc/paper/6084-fast-free-inference-of-simulation-models-with-bayesian-conditional-density-estimation/bibtex).
- Papamakarios, George, and Iain Murray. 2016. “Fast ε-Free Inference of Simulation
  Models with Bayesian Conditional Density Estimation.” In ArXiv:1605.06376 [Cs, Stat]. http://arxiv.org/abs/1605.06376.

  -->

#### Sequential Neural Ratio Estimation (SNRE)

* [`SNRE_A`](https://www.mackelab.org/sbi/reference/#sbi.inference.snre.snre_a.SNRE_A) or `AALR` from Hermans J, Begy V, and Louppe G. [_Likelihood-free Inference with Amortized Approximate Likelihood Ratios_](https://arxiv.org/abs/1903.04057) (ICML 2020).

* [`SNRE_B`](https://www.mackelab.org/sbi/reference/#sbi.inference.snre.snre_b.SNRE_B) or `SRE` from Durkan C, Murray I, and Papamakarios G. [_On Contrastive Learning for Likelihood-free Inference_](https://arxiv.org/abs/2002.03712) (ICML 2020).

#### Sequential Neural Likelihood Estimation (SNRE)
* [`SNLE_A`](https://www.mackelab.org/sbi/reference/#sbi.inference.snle.snle_a.SNLE_A) or just `SNL` from Papamakarios G, Sterrat DC and Murray I [_Sequential
  Neural Likelihood_](https://arxiv.org/abs/1805.07226) (AISTATS 2019).
>

## Acknowledgements

`sbi` is the successor (using PyTorch) of the 
[`delfi`](https://github.com/mackelab/delfi) package. It was started as a fork of Conor
M. Durkan's `lfi`. `sbi` runs as a community project; development is coordinated at the
[mackelab](https://uni-tuebingen.de/en/research/core-research/cluster-of-excellence-machine-learning/research/research/cluster-research-groups/professorships/machine-learning-in-science/).

We would like to hear how it is working for your simulation as well as receive bug
reports, pull requests and other feedback (see
[contribute](http://www.mackelab.org/sbi/contribute/)).

See also [credits](https://github.com/mackelab/sbi/blob/master/docs/docs/credits.md).

## Support

`sbi` has been developed in the context of the [ADIMEM
grant](https://fit.uni-tuebingen.de/Activity/Details?id=6097), project A. ADIMEM is a
BMBF grant awarded to groups at the Technical University of Munich, University of
Tübingen and Research Center caesar of the Max Planck Gesellschaft.

