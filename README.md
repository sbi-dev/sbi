![Tests](https://github.com/mackelab/sbi/workflows/Tests/badge.svg?branch=master)

## Warning: pre-release stage

SBI is currently under very active development leading up to a first stable release on 12th June.

Some aspects of the interface will change, and the documentation for running the inference methods (`SnpeB, SnpeC (APT), SRE, SNL`) is not accessible  at the moment through regular Python introspection - you'll have to look at the superclasses ([`SnpeBase`](https://github.com/mackelab/sbi/blob/master/sbi/inference/snpe/snpe_base.py), [`NeuralInference`](https://github.com/mackelab/sbi/blob/master/sbi/inference/base.py)). Authorship information is also out of date and licensing still pending (it will be free software).

If you'd still like to give it a spin before release, please, by all means! We're glad to engage in conversation about it, please file an issue if you encounter unexpected behaviour or wonder about specific functionality.

## Description

Building on code for "On Contrastive Learning for Likelihood-free Inference" in <https://github.com/conormdurkan/lfi>.

Features neural likelihood-free methods from

> Papamakarios et al., _Sequential Neural Likelihood_ (SNL), 2019. [[arXiv]](https://arxiv.org/abs/1805.07226)
>
>Greenberg et al., _Automatic Posterior Transformation_ (SNPE-C), 2019. [[arXiv]](https://arxiv.org/abs/1905.07488)
>
>Hermans et al., _Likelihood-free Inference with Amortized Approximate Likelihood Ratios_ (SRE), 2019.  [[arXiv]](https://arxiv.org/abs/1903.04057)
>
>Durkan et al., _On Contrastive Learning for Likelihood-free Inference_, 2020 [[arXiv]](https://arxiv.org/abs/2002.03712) 

## Setup

Clone the repo and install all the dependencies using the `environment.yml` file to create a conda environment: `conda env create -f environment.yml`. If you already have an `sbi` environment and want to refresh dependencies, just run `conda env update -f environment.yml --prune`.

Alternatively, you can install via `setup.py` using `pip install -e ".[dev]"` (the dev flag installs development and testing dependencies).

## Examples

Examples are collected in notebooks in `examples/`.

## Binary files and Jupyter notebooks

### Using `sbi`

We use git lfs to store large binary files. Those files are not downloaded by cloning the repository, but you have to pull them separately. To do so follow installation instructions here [https://git-lfs.github.com/](https://git-lfs.github.com/). In particular, in a freshly cloned repository on a new machine, you will need to run both `git-lfs install` and `git-lfs pull`.

### Contributing to `sbi`

We use a filename filter to track lfs files. Once you installed and pulled git lfs you can add a file to git lfs by appending `_gitlfs` to the basename, e.g., `oldbase_gitlfs.npy`. Then add the file to the index, commit, and it will be tracked by git lfs.

Additionally, to avoid large diffs due to Jupyter notebook outputs we are using `nbstripout` to remove output from notebooks before every commit. The `nbstripout` package is downloaded automatically during installation of `sbi`. However, **please make sure to set up the filter yourself**, e.g., through `nbstriout --install` or with different options as described [here](https://github.com/kynan/nbstripout).

## Acknowledgements

This code builds heavily on previous work by [Conor Durkan](https://conormdurkan.github.io/), [George Papamakarios](https://gpapamak.github.io/) and [Artur Bekasov](https://arturbekasov.github.io/).
Relevant repositories include [bayesiains/nsf](https://github.com/bayesiains/nsf) and [conormdurkan/lfi](https://github.com/conormdurkan/lfi). 
