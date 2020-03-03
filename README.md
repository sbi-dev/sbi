![Tests](https://github.com/mackelab/sbi/workflows/Tests/badge.svg?branch=master)

## Description

Building on code for "On Contrastive Learning for Likelihood-free Inference" in <https://github.com/conormdurkan/lfi.>

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

## Binary files and jupyter notebooks

We use git lfs to store large binary files. To use git lfs follow installation instructions here [https://git-lfs.github.com/](https://git-lfs.github.com/). In particular, in a freshly cloned repository on a new machine, you will need both `git-lfs install` and `git-lfs pull`. We use a filename filter to track lfs files. Therefore, to add a file to git lfs let the filename contain `_gitlfs_`.

Small binary files and jupyter notebook will not be tracked by git lfs for now. Instead we are using `nbstripout` to remove output from jupyter notebooks before pushing to remote. The `nbstripout` package is downloaded automatically during installation of `sbi`, but **please make sure to install it locally** as described [here](https://github.com/kynan/nbstripout).

## Acknowledgements

This code builds heavily on previous work by [Conor Durkan](https://conormdurkan.github.io/), [George Papamakarios](https://gpapamak.github.io/) and [Artur Bekasov](https://arturbekasov.github.io/).
Relevant repositories include [bayesiains/nsf](https://github.com/bayesiains/nsf) and [conormdurkan/lfi](https://github.com/conormdurkan/lfi). 
