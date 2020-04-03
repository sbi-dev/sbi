# Installation

## Quick start

Clone the repo and install all the dependencies using the `environment.yml` file to create a conda environment: `conda env create -f environment.yml`. If you already have an `sbi` environment and want to refresh dependencies, just run `conda env update -f environment.yml --prune`.

Alternatively, you can install via `setup.py` using `pip install -e ".[dev]"` (the dev flag installs development and testing dependencies).

```
git clone https://github.com/mackelab/sbi.git
cd sbi
pip install -e ".dev"
```
