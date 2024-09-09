[![PyPI version](https://badge.fury.io/py/sbi.svg)](https://badge.fury.io/py/sbi)
[![Contributions welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg?style=flat)](https://github.com/sbi-dev/sbi/blob/master/CONTRIBUTING.md)
[![Tests](https://github.com/sbi-dev/sbi/workflows/Tests/badge.svg?branch=main)](https://github.com/sbi-dev/sbi/actions)
[![codecov](https://codecov.io/gh/sbi-dev/sbi/branch/main/graph/badge.svg)](https://codecov.io/gh/sbi-dev/sbi)
[![GitHub license](https://img.shields.io/github/license/sbi-dev/sbi)](https://github.com/sbi-dev/sbi/blob/master/LICENSE.txt)
[![DOI](https://joss.theoj.org/papers/10.21105/joss.02505/status.svg)](https://doi.org/10.21105/joss.02505)

## `sbi`: Simulation-Based Inference

[Getting Started](https://sbi-dev.github.io/sbi/latest/tutorials/00_getting_started/) |
[Documentation](https://sbi-dev.github.io/sbi/)

`sbi` is a Python package for simulation-based inference, designed to meet the needs of
both researchers and practitioners. Whether you need fine-grained control or an
easy-to-use interface, `sbi` has you covered.

With `sbi`, you can perform parameter inference using Bayesian inference: Given a
simulator that models a real-world process, SBI estimates the full posterior
distribution over the simulator’s parameters based on observed data. This distribution
indicates the most likely parameter values while additionally quantifying uncertainty
and revealing potential interactions between parameters.

### Key Features of `sbi`

`sbi` offers a blend of flexibility and ease of use:

- **Low-Level Interfaces**: For those who require maximum control over the inference
  process, `sbi` provides low-level interfaces that allow you to fine-tune many aspects
  of your workflow.
- **High-Level Interfaces**: If you prefer simplicity and efficiency, `sbi` also offers
  high-level interfaces that enable quick and easy implementation of complex inference
  tasks.

In addition, `sbi` supports a wide range of state-of-the-art inference algorithms (see
below for a list of implemented methods):

- **Amortized Methods**: These methods enable the reuse of posterior estimators across
  multiple observations without the need to retrain.
- **Sequential Methods**: These methods focus on individual observations, optimizing the
  number of simulations required.

Beyond inference, `sbi` also provides:

- **Validation Tools**: Built-in methods to validate and verify the accuracy of your
  inferred posteriors.
- **Plotting and Analysis Tools**: Comprehensive functions for visualizing and analyzing
  results, helping you interpret the posterior distributions with ease.

Getting started with `sbi` is straightforward, requiring only a few lines of code:

```python
from sbi.inference import NPE
# Given: parameters theta and corresponding simulations x
inference = NPE(prior=prior)
inference.append_simulations(theta, x).train()
posterior = inference.build_posterior()
```

### Installation

`sbi` requires Python 3.9 or higher. While a GPU isn't necessary, it can improve
performance in some cases. We recommend using a virtual environment with
[`conda`](https://docs.conda.io/en/latest/miniconda.html) for an easy setup.

To install `sbi`, follow these steps:

1. **Create a Conda Environment** (if using Conda):

   ```bash
   conda create -n sbi_env python=3.9 && conda activate sbi_env
   ```

2. **Install `sbi`**: Independent of whether you are using `conda` or not, `sbi` can be
   installed using `pip`:

  ```commandline
  pip install sbi
  ```

3. **Test the installation**:
Open a Python prompt and run

```python
from sbi.examples.minimal import simple
posterior = simple()
print(posterior)
```

## Tutorials

If you're new to `sbi`, we recommend starting with our [Getting
Started](https://sbi-dev.github.io/sbi/latest/tutorials/00_getting_started/) tutorial.

You can also access and run these tutorials directly in your browser by opening
[Codespace](https://docs.github.com/en/codespaces/overview). To do so, click the green
“Code” button on the GitHub repository and select “Open with Codespaces.” This provides
a fully functional environment where you can explore `sbi` through Jupyter notebooks.

## Inference Algorithms

The following inference algorithms are currently available. You can find instructions on
how to run each of these methods
[here](https://sbi-dev.github.io/sbi/latest/tutorials/16_implemented_methods/).

### Neural Posterior Estimation: amortized (NPE) and sequential (SNPE)

* [`(S)NPE_A`](https://sbi-dev.github.io/sbi/latest/reference/#sbi.inference.trainers.npe.npe_a.NPE_A)
  (including amortized single-round `NPE`) from Papamakarios G and Murray I [_Fast
  ε-free Inference of Simulation Models with Bayesian Conditional Density
  Estimation_](https://proceedings.neurips.cc/paper/2016/hash/6aca97005c68f1206823815f66102863-Abstract.html)
  (NeurIPS 2016).

* [`(S)NPE_C`](https://sbi-dev.github.io/sbi/latest/reference/#sbi.inference.trainers.npe.npe_c.NPE_C)
  or `APT` from Greenberg D, Nonnenmacher M, and Macke J [_Automatic Posterior
  Transformation for likelihood-free inference_](https://arxiv.org/abs/1905.07488) (ICML
  2019).

* `TSNPE` from Deistler M, Goncalves P, and Macke J [_Truncated proposals for scalable
  and hassle-free simulation-based inference_](https://arxiv.org/abs/2210.04815)
  (NeurIPS 2022).

* [`FMPE`](https://sbi-dev.github.io/sbi/latest/reference/#sbi.inference.trainers.fmpe.fmpe.FMPE)
  from Wildberger, J., Dax, M., Buchholz, S., Green, S., Macke, J. H., & Schölkopf, B.
  [_Flow matching for scalable simulation-based
  inference_](https://proceedings.neurips.cc/paper_files/paper/2023/hash/3663ae53ec078860bb0b9c6606e092a0-Abstract-Conference.html).
  (NeurIPS 2023).

* [`NPSE`](https://sbi-dev.github.io/sbi/latest/reference/#sbi.inference.trainers.npse.npse.NPSE) from
  Geffner, T., Papamakarios, G., & Mnih, A. [_Compositional score modeling for
  simulation-based inference_](https://proceedings.mlr.press/v202/geffner23a.html).
  (ICML 2023)

### Neural Likelihood Estimation: amortized (NLE) and sequential (SNLE)

* [`(S)NLE`](https://sbi-dev.github.io/sbi/latest/reference/#sbi.inference.trainers.nle.nle_a.NLE_A)
  or just `SNL` from Papamakarios G, Sterrat DC and Murray I [_Sequential Neural
  Likelihood_](https://arxiv.org/abs/1805.07226) (AISTATS 2019).

### Neural Ratio Estimation: amortized (NRE) and sequential (SNRE)

* [`(S)NRE_A`](https://sbi-dev.github.io/sbi/latest/reference/#sbi.inference.trainers.nre.nre_a.NRE_A)
  or `AALR` from Hermans J, Begy V, and Louppe G. [_Likelihood-free Inference with
  Amortized Approximate Likelihood Ratios_](https://arxiv.org/abs/1903.04057) (ICML
  2020).

* [`(S)NRE_B`](https://sbi-dev.github.io/sbi/latest/reference/#sbi.inference.trainers.nre.nre_b.NRE_B)
  or `SRE` from Durkan C, Murray I, and Papamakarios G. [_On Contrastive Learning for
  Likelihood-free Inference_](https://arxiv.org/abs/2002.03712) (ICML 2020).

* [`(S)NRE_C`](https://sbi-dev.github.io/sbi/latest/reference/#sbi.inference.trainers.nre.nre_c.NRE_C)
  or `NRE-C` from Miller BK, Weniger C, Forré P. [_Contrastive Neural Ratio
  Estimation_](https://arxiv.org/abs/2210.06170) (NeurIPS 2022).

* [`BNRE`](https://sbi-dev.github.io/sbi/latest/reference/#sbi.inference.trainers.nre.bnre.BNRE) from
  Delaunoy A, Hermans J, Rozet F, Wehenkel A, and Louppe G. [_Towards Reliable
  Simulation-Based Inference with Balanced Neural Ratio
  Estimation_](https://arxiv.org/abs/2208.13624) (NeurIPS 2022).


### Neural Variational Inference, amortized (NVI) and sequential (SNVI)

* [`SNVI`](https://sbi-dev.github.io/sbi/latest/reference/#sbi.inference.posteriors.vi_posterior)
  from Glöckler M, Deistler M, Macke J, [_Variational methods for simulation-based
  inference_](https://openreview.net/forum?id=kZ0UYdhqkNY) (ICLR 2022).

### Mixed Neural Likelihood Estimation (MNLE)

* [`MNLE`](https://sbi-dev.github.io/sbi/latest/reference/#sbi.inference.trainers.nle.mnle.MNLE) from
  Boelts J, Lueckmann JM, Gao R, Macke J, [_Flexible and efficient simulation-based
  inference for models of decision-making_](https://elifesciences.org/articles/77220)
  (eLife 2022).

## Feedback and Contributions

We welcome any feedback on how `sbi` is working for your inference problems (see
[Discussions](https://github.com/sbi-dev/sbi/discussions)) and are happy to receive bug
reports, pull requests, and other feedback (see
[contribute](https://sbi-dev.github.io/sbi/latest/contribute/)). We wish to maintain a positive
community; please read our [Code of Conduct](CODE_OF_CONDUCT.md).

## Acknowledgments

`sbi` is the successor (using PyTorch) of the
[`delfi`](https://github.com/mackelab/delfi) package. It started as a fork of Conor M.
Durkan's `lfi`. `sbi` runs as a community project. See also
[credits](https://github.com/sbi-dev/sbi/blob/master/docs/docs/credits.md).

## Support

`sbi` has been supported by the German Federal Ministry of Education and Research (BMBF)
through project ADIMEM (FKZ 01IS18052 A-D), project SiMaLeSAM (FKZ 01IS21055A) and the
Tübingen AI Center (FKZ 01IS18039A). Since 2024, `sbi` is supported by the appliedAI
Institute for Europe.

## License

[Apache License Version 2.0 (Apache-2.0)](https://www.apache.org/licenses/LICENSE-2.0)

## Citation

If you use `sbi` consider citing the [sbi software
paper](https://doi.org/10.21105/joss.02505), in addition to the original research
articles describing the specific sbi-algorithm(s) you are using.

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

The above citation refers to the original version of the `sbi` project and has a
persistent DOI. Additionally, new releases of `sbi` are citable via
[Zenodo](https://zenodo.org/record/3993098), where we create a new DOI for every
release.
