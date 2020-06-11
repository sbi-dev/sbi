![Tests](https://github.com/mackelab/sbi/workflows/Tests/badge.svg?branch=master)
[![PyPI version](https://badge.fury.io/py/sbi.svg)](https://badge.fury.io/py/sbi)
[![Contributions welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg?style=flat)](https://github.com/mackelab/sbi/blob/master/docs/docs/contribute.md)
[![GitHub license](https://img.shields.io/github/license/mackelab/sbi)](https://github.com/mackelab/sbi/blob/master/LICENSE.txt)
## sbi: simulation-based inference

`sbi` is a PyTorch package for simulation-based inference. Simulation-based inference is
the process of finding the parameters of a simulator given also a prior and
observations. `sbi` takes a Bayesian approach and returns a full posterior distribution
over the parameters, conditional on the observations. This posterior can be focused on a
specific observation of interest, or amortized and ready to be evaluated and sampled for
any value of the observation.

`sbi` offers a simple interface for one-line posterior inference

```python
from sbi inference import infer
# import your simulator, define your prior
parameter_posterior = infer(simulator, prior, method='SNPE')
```
See below for the available methods of inference, `SNPE`, `SNRE` and `SNLE`.

## Installation

```
pip install sbi
```

## Inference methods
The following methods are currently available

### Sequential Neural Posterior Estimation (SNPE)

* `SNPE_C` or [`APT`](https://github.com/mackelab/delfi) from Greenberg D, Nonnenmacher M, and Macke J [_Automatic
  PosteriorTransformation for likelihood-free
  inference_](https://arxiv.org/abs/1905.07488)(ICML 2020).

<!-- 
- **Fast ε-free Inference of Simulation Models with Bayesian Conditional Density
  Estimation**<br> by Papamakarios G. and Murray I. (NeurIPS 2016)
  <br>[[PDF]](https://papers.nips.cc/paper/6084-fast-free-inference-of-simulation-models-with-bayesian-conditional-density-estimation.pdf)
  [[BibTeX]](https://papers.nips.cc/paper/6084-fast-free-inference-of-simulation-models-with-bayesian-conditional-density-estimation/bibtex).
- Papamakarios, George, and Iain Murray. 2016. “Fast ε-Free Inference of Simulation
  Models with Bayesian Conditional Density Estimation.” In ArXiv:1605.06376 [Cs, Stat]. http://arxiv.org/abs/1605.06376.

  -->

### Sequential Neural Ratio Estimation (SNRE)

* `SNRE_A` or `AALR` from Hermans J, Begy V, and Louppe G. [_Likelihood-free Inference with Amortized Approximate Likelihood Ratios_](https://arxiv.org/abs/1903.04057)(ICML 2020).

* `SNRE_B` or [`SRE`](https://github.com/bayesiains/lfi) from Durkan C, Murray I, and Papamakarios G. [_On Contrastive Learning for Likelihood-free Inference_](https://arxiv.org/abs/2002.03712)(ICML 2020).

### Sequential Neural Likelihood Estimation (SNRE)
* `SNLE_A` or just [`SNL`](https://github.com/gpapamak/snl) from Papamakarios G, Sterrat DC and Murray I [_Sequential
  Neural Likelihood_](https://arxiv.org/abs/1805.07226)(AISTATS 2019).
>
## Developing

Clone the repo and install all the dependencies using the `environment.yml` file to
create a conda environment: `conda env create -f environment.yml`. If you already have
an `sbi` environment and want to refresh dependencies, just run `conda env update -f
environment.yml --prune`.

Alternatively, you can install via `setup.py` using `pip install -e ".[dev]"` (the dev
flag installs development and testing dependencies).

Issues and pull requests here are welcome! See [contribution
guidelines](https://github.com/mackelab/sbi/blob/master/docs/docs/contribute.md) for more.

## Examples

Examples are collected in notebooks in `examples/`.

## Binary files and Jupyter notebooks

### Using `sbi`

We use git lfs to store large binary files. Those files are not downloaded by cloning the repository, but you have to pull them separately. To do so follow installation instructions here [https://git-lfs.github.com/](https://git-lfs.github.com/). In particular, in a freshly cloned repository on a new machine, you will need to run both `git-lfs install` and `git-lfs pull`.

### Contributing to `sbi`

We use a filename filter to track lfs files. Once you installed and pulled git lfs you can add a file to git lfs by appending `_gitlfs` to the basename, e.g., `oldbase_gitlfs.npy`. Then add the file to the index, commit, and it will be tracked by git lfs.

Additionally, to avoid large diffs due to Jupyter notebook outputs we are using `nbstripout` to remove output from notebooks before every commit. The `nbstripout` package is downloaded automatically during installation of `sbi`. However, **please make sure to set up the filter yourself**, e.g., through `nbstriout --install` or with different options as described [here](https://github.com/kynan/nbstripout).

## Acknowledgements

`sbi` was started from [`lfi`](https://github.com/conormdurkan/lfi) by Conor M Durkan.
It is currently developed at the [mackelab](https://uni-tuebingen.de/en/research/core-research/cluster-of-excellence-machine-learning/research/research/cluster-research-groups/professorships/machine-learning-in-science/).

See [credits](https://github.com/mackelab/sbi/blob/master/docs/docs/credits.md).
