[![PyPI version](https://badge.fury.io/py/sbi.svg)](https://badge.fury.io/py/sbi)
[![Contributions welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg?style=flat)](https://github.com/sbi-dev/sbi/blob/master/CONTRIBUTING.md)
[![Tests](https://github.com/sbi-dev/sbi/workflows/Tests/badge.svg?branch=main)](https://github.com/sbi-dev/sbi/actions)
[![codecov](https://codecov.io/gh/sbi-dev/sbi/branch/main/graph/badge.svg)](https://codecov.io/gh/sbi-dev/sbi)
[![GitHub license](https://img.shields.io/github/license/sbi-dev/sbi)](https://github.com/sbi-dev/sbi/blob/master/LICENSE.txt)
[![DOI](https://joss.theoj.org/papers/10.21105/joss.02505/status.svg)](https://doi.org/10.21105/joss.02505)

## sbi: simulation-based inference

[Getting Started](https://sbi-dev.github.io/sbi/tutorial/00_getting_started/) | [Documentation](https://sbi-dev.github.io/sbi/)

`sbi` is a PyTorch package for simulation-based inference. Simulation-based inference is the process of finding parameters of a simulator from observations.

`sbi` takes a Bayesian approach and returns a full posterior distribution over the parameters of the simulator, conditional on the observations.
The package implements a variety of inference algorithms, including _amortized_ and _sequential_ methods.
Amortized methods return a posterior that can be applied to many different observations without retraining; sequential methods focus the inference on one particular observation to be more simulation-efficient.
See below for an overview of implemented methods.

`sbi` offers a simple interface for posterior inference in a few lines of code

```python
from sbi.inference import SNPE
# import your simulator, define your prior over the parameters
# sample parameters theta and observations x
inference = SNPE(prior=prior)
_ = inference.append_simulations(theta, x).train()
posterior = inference.build_posterior()
```

## Installation

`sbi` requires Python 3.8 or higher. A GPU is not required, but can lead to speed-up in some cases. We recommend to use a [`conda`](https://docs.conda.io/en/latest/miniconda.html) virtual
environment ([Miniconda installation instructions](https://docs.conda.io/en/latest/miniconda.html)). If `conda` is installed on the system, an environment for installing `sbi` can be created as follows:

```commandline
# Create an environment for sbi (indicate Python 3.8 or higher); activate it
$ conda create -n sbi_env python=3.10 && conda activate sbi_env
```

Independent of whether you are using `conda` or not, `sbi` can be installed using `pip`:

```commandline
pip install sbi
```

To test the installation, drop into a python prompt and run

```python
from sbi.examples.minimal import simple
posterior = simple()
print(posterior)
```

## Tutorials

For first time users, you can now head over to the turorials and get going with [Getting Started](https://sbi-dev.github.io/sbi/tutorial/00_getting_started/).

## Inference Algorithms

The following inference algorithms are currently available. You can find instructions on how to run each of these methods [here](https://sbi-dev.github.io/sbi/tutorial/16_implemented_methods/).

### Neural Posterior Estimation: amortized (NPE) and sequential (SNPE)

* [`SNPE_A`](https://sbi-dev.github.io/sbi/reference/#sbi.inference.snpe.snpe_a.SNPE_A) (including amortized single-round `NPE`) from Papamakarios G and Murray I [_Fast ε-free Inference of Simulation Models with Bayesian Conditional Density Estimation_](https://proceedings.neurips.cc/paper/2016/hash/6aca97005c68f1206823815f66102863-Abstract.html) (NeurIPS 2016).

* [`SNPE_C`](https://sbi-dev.github.io/sbi/reference/#sbi.inference.snpe.snpe_c.SNPE_C) or `APT` from Greenberg D, Nonnenmacher M, and Macke J [_Automatic
  Posterior Transformation for likelihood-free
  inference_](https://arxiv.org/abs/1905.07488) (ICML 2019).

* `TSNPE` from Deistler M, Goncalves P, and Macke J [_Truncated proposals for scalable and hassle-free simulation-based inference_](https://arxiv.org/abs/2210.04815) (NeurIPS 2022).

### Neural Likelihood Estimation: amortized (NLE) and sequential (SNLE)

* [`SNLE_A`](https://sbi-dev.github.io/sbi/reference/#sbi.inference.snle.snle_a.SNLE_A) or just `SNL` from Papamakarios G, Sterrat DC and Murray I [_Sequential
  Neural Likelihood_](https://arxiv.org/abs/1805.07226) (AISTATS 2019).

### Neural Ratio Estimation: amortized (NRE) and sequential (SNRE)

* [`(S)NRE_A`](https://sbi-dev.github.io/sbi/reference/#sbi.inference.snre.snre_a.SNRE_A) or `AALR` from Hermans J, Begy V, and Louppe G. [_Likelihood-free Inference with Amortized Approximate Likelihood Ratios_](https://arxiv.org/abs/1903.04057) (ICML 2020).

* [`(S)NRE_B`](https://sbi-dev.github.io/sbi/reference/#sbi.inference.snre.snre_b.SNRE_B) or `SRE` from Durkan C, Murray I, and Papamakarios G. [_On Contrastive Learning for Likelihood-free Inference_](https://arxiv.org/abs/2002.03712) (ICML 2020).

* [`BNRE`](https://sbi-dev.github.io/sbi/reference/#sbi.inference.snre.bnre.BNRE) from Delaunoy A, Hermans J, Rozet F, Wehenkel A, and Louppe G. [_Towards Reliable Simulation-Based Inference with Balanced Neural Ratio Estimation_](https://arxiv.org/abs/2208.13624) (NeurIPS 2022).

* [`(S)NRE_C`](https://sbi-dev.github.io/sbi/reference/#sbi.inference.snre.snre_c.SNRE_C) or `NRE-C` from Miller BK, Weniger C, Forré P. [_Contrastive Neural Ratio Estimation_](https://arxiv.org/abs/2210.06170) (NeurIPS 2022).

### Neural Variational Inference, amortized (NVI) and sequential (SNVI)

* [`SNVI`](https://sbi-dev.github.io/sbi/reference/#sbi.inference.posteriors.vi_posterior) from Glöckler M, Deistler M, Macke J, [_Variational methods for simulation-based inference_](https://openreview.net/forum?id=kZ0UYdhqkNY) (ICLR 2022).

### Mixed Neural Likelihood Estimation (MNLE)

* [`MNLE`](https://sbi-dev.github.io/sbi/reference/#sbi.inference.snle.mnle.MNLE) from Boelts J, Lueckmann JM, Gao R, Macke J, [_Flexible and efficient simulation-based inference for models of decision-making_](https://elifesciences.org/articles/77220) (eLife 2022).

## Feedback and Contributions

We welcome any feedback on how `sbi` is working for your inference problems (see [Discussions](https://github.com/sbi-dev/sbi/discussions)) and are happy to receive bug reports, pull requests and other feedback (see
[contribute](http://sbi-dev.github.io/sbi/contribute/)).
We wish to maintain a positive community, please read our [Code of Conduct](CODE_OF_CONDUCT.md).

## Acknowledgements

`sbi` is the successor (using PyTorch) of the
[`delfi`](https://github.com/mackelab/delfi) package. It was started as a fork of Conor
M. Durkan's `lfi`. `sbi` runs as a community project. See also [credits](https://github.com/sbi-dev/sbi/blob/master/docs/docs/credits.md).

## Support

`sbi` has been supported by the German Federal Ministry of Education and Research (BMBF) through project ADIMEM (FKZ 01IS18052 A-D), project SiMaLeSAM (FKZ 01IS21055A) and the Tübingen AI Center (FKZ 01IS18039A).

## License

[Affero General Public License v3 (AGPLv3)](https://www.gnu.org/licenses/)

## Citation

If you use `sbi` consider citing the [sbi software paper](https://doi.org/10.21105/joss.02505), in addition to the original research articles describing the specific sbi-algorithm(s) you are using.

```latex
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

The above citation refers to the original version of the `sbi` project and has a persistent DOI.
Additionally, new releases of `sbi` are citable via [Zenodo](https://zenodo.org/record/3993098), where we create a new DOI for every release.
