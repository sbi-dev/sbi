[![PyPI version](https://badge.fury.io/py/sbi.svg)](https://badge.fury.io/py/sbi)
[![Contributions welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg?style=flat)](https://github.com/mackelab/sbi/blob/master/CONTRIBUTING.md)
[![Tests](https://github.com/mackelab/sbi/workflows/Tests/badge.svg?branch=main)](https://github.com/mackelab/sbi/actions)
[![codecov](https://codecov.io/gh/mackelab/sbi/branch/main/graph/badge.svg)](https://codecov.io/gh/mackelab/sbi)
[![GitHub license](https://img.shields.io/github/license/mackelab/sbi)](https://github.com/mackelab/sbi/blob/master/LICENSE.txt)
[![DOI](https://joss.theoj.org/papers/10.21105/joss.02505/status.svg)](https://doi.org/10.21105/joss.02505)

## sbi: simulation-based inference
[Getting Started](https://www.mackelab.org/sbi/tutorial/00_getting_started/) | [Documentation](https://www.mackelab.org/sbi/)

`sbi` is a PyTorch package for simulation-based inference. Simulation-based inference is  
the process of finding parameters of a simulator from observations.

`sbi` takes a Bayesian approach and returns a full posterior distribution
over the parameters, conditional on the observations. This posterior can be amortized (i.e.
useful for any observation) or focused (i.e. tailored to a particular observation), with different
computational trade-offs.

`sbi` offers a simple interface for one-line posterior inference.

```python
from sbi.inference import infer
# import your simulator, define your prior over the parameters
parameter_posterior = infer(simulator, prior, method='SNPE', num_simulations=100)
```
See below for the available methods of inference, `SNPE`, `SNRE` and `SNLE`.


## Installation

`sbi` requires Python 3.6 or higher. We recommend to use a [`conda`](https://docs.conda.io/en/latest/miniconda.html) virtual
environment ([Miniconda installation instructions](https://docs.conda.io/en/latest/miniconda.html])). If `conda` is installed on the system, an environment for
installing `sbi` can be created as follows:
```commandline
# Create an environment for sbi (indicate Python 3.6 or higher); activate it
$ conda create -n sbi_env python=3.7 && conda activate sbi_env
```

Independent of whether you are using `conda` or not, `sbi` can be installed using `pip`:
```commandline
$ pip install sbi
```

To test the installation, drop into a python prompt and run
```python
from sbi.examples.minimal import simple
posterior = simple()
print(posterior)
```

## Inference Algorithms

The following algorithms are currently available:

#### Sequential Neural Posterior Estimation (SNPE)

* [`SNPE_A`](https://www.mackelab.org/sbi/reference/#sbi.inference.snpe.snpe_a.SNPE_A) from Papamakarios G and Murray I [_Fast ε-free Inference of Simulation Models with Bayesian Conditional Density Estimation_](https://proceedings.neurips.cc/paper/2016/hash/6aca97005c68f1206823815f66102863-Abstract.html) (NeurIPS 2016).
  
* [`SNPE_C`](https://www.mackelab.org/sbi/reference/#sbi.inference.snpe.snpe_c.SNPE_C) or `APT` from Greenberg D, Nonnenmacher M, and Macke J [_Automatic
  Posterior Transformation for likelihood-free
  inference_](https://arxiv.org/abs/1905.07488) (ICML 2019).


#### Sequential Neural Likelihood Estimation (SNLE)
* [`SNLE_A`](https://www.mackelab.org/sbi/reference/#sbi.inference.snle.snle_a.SNLE_A) or just `SNL` from Papamakarios G, Sterrat DC and Murray I [_Sequential
  Neural Likelihood_](https://arxiv.org/abs/1805.07226) (AISTATS 2019).


#### Sequential Neural Ratio Estimation (SNRE)

* [`SNRE_A`](https://www.mackelab.org/sbi/reference/#sbi.inference.snre.snre_a.SNRE_A) or `AALR` from Hermans J, Begy V, and Louppe G. [_Likelihood-free Inference with Amortized Approximate Likelihood Ratios_](https://arxiv.org/abs/1903.04057) (ICML 2020).

* [`SNRE_B`](https://www.mackelab.org/sbi/reference/#sbi.inference.snre.snre_b.SNRE_B) or `SRE` from Durkan C, Murray I, and Papamakarios G. [_On Contrastive Learning for Likelihood-free Inference_](https://arxiv.org/abs/2002.03712) (ICML 2020).

#### Sequential Neural Variational Inference (SNVI)

* [`SNVI`](https://www.mackelab.org/sbi/reference/#sbi.inference.posteriors.vi_posterior) from Glöckler M, Deistler M, Macke J, [_Variational methods for simulation-based inference_](https://openreview.net/forum?id=kZ0UYdhqkNY) (ICLR 2022). 

## Feedback and Contributions

We would like to hear how `sbi` is working for your inference problems as well as receive bug reports, pull requests and other feedback (see
[contribute](http://www.mackelab.org/sbi/contribute/)).


## Acknowledgements

`sbi` is the successor (using PyTorch) of the
[`delfi`](https://github.com/mackelab/delfi) package. It was started as a fork of Conor
M. Durkan's `lfi`. `sbi` runs as a community project; development is coordinated at the
[mackelab](https://uni-tuebingen.de/en/research/core-research/cluster-of-excellence-machine-learning/research/research/cluster-research-groups/professorships/machine-learning-in-science/). See also [credits](https://github.com/mackelab/sbi/blob/master/docs/docs/credits.md).


## Support

`sbi` has been supported by the German Federal Ministry of Education and Research (BMBF) through the project ADIMEM, FKZ 01IS18052 A-D). [ADIMEM](https://fit.uni-tuebingen.de/Project/Details?id=9199) is a collaborative project between the groups of Jakob Macke (Uni Tübingen), Philipp Berens (Uni Tübingen), Philipp Hennig (Uni Tübingen) and Marcel Oberlaender (caesar Bonn) which aims to develop inference methods for mechanistic models.


## License

[Affero General Public License v3 (AGPLv3)](https://www.gnu.org/licenses/)


## Citation
If you use `sbi` consider citing the [sbi software paper](https://doi.org/10.21105/joss.02505), in addition to the original research articles describing the specifc sbi-algorithm(s) you are using: 

```
@article{tejero-cantero2020sbi,
  doi = {10.21105/joss.02505},
  url = {https://doi.org/10.21105/joss.02505},
  year = {2020},
  publisher = {The Open Journal},
  volume = {5},
  number = {52},
  pages = {2505},
  author = {Alvaro Tejero-Cantero and Jan Boelts and Michael Deistler and Jan-Matthis Lueckmann and Conor Durkan and Pedro J. Gonçalves and David S. Greenberg and Jakob H. Macke},
  title = {sbi: A toolkit for simulation-based inference},
  journal = {Journal of Open Source Software}
}
```
