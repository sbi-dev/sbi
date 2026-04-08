(installation)=
# Installation

`sbi` requires Python 3.10 or higher. We recommend using Python 3.12 for the best
experience. A GPU is not required, but can lead to speed-up in some cases.

## Recommended: Installation with uv

We recommend using [`uv`](https://docs.astral.sh/uv)
for package and environment management. If you haven't installed `uv` yet, follow
the [installation instructions](https://docs.astral.sh/uv/getting-started/installation/).

Create a virtual environment and install `sbi`:

```bash
# Create a virtual environment with Python 3.12
uv venv --python 3.12

# Activate the environment (on macOS/Linux)
source .venv/bin/activate

# On Windows
# .venv\Scripts\activate

# Install sbi
uv pip install sbi
```

### Optional dependencies

Pyro and PyMC MCMC samplers are optional. Install them as needed:

```bash
uv pip install "sbi[pyro]"   # for Pyro samplers (HMC, NUTS)
uv pip install "sbi[pymc]"   # for PyMC samplers (HMC, NUTS, Slice)
uv pip install "sbi[all]"    # for all optional dependencies
```

## Alternative installation methods

### Using pip

If you prefer using `pip` directly:

```bash
python -m pip install sbi
```

For Pyro or PyMC MCMC samplers, install the corresponding extras:

```bash
python -m pip install "sbi[pyro]"   # or "sbi[pymc]", or "sbi[all]"
```

### Using conda

To install into a conda environment:

```bash
# Create an environment for sbi (Python 3.10 or higher)
conda create -n sbi_env python=3.12 && conda activate sbi_env

# Install sbi from conda-forge
conda install --channel conda-forge sbi
```

For Pyro or PyMC MCMC samplers, install them as additional conda packages:

```bash
conda install --channel conda-forge pyro-ppl  # for Pyro samplers
conda install --channel conda-forge pymc       # for PyMC samplers
```

### Using pixi

To install and add `sbi` to a project with [`pixi`](https://pixi.sh/):

```bash
pixi add sbi
```

For Pyro or PyMC samplers, add them separately: `pixi add pyro-ppl` or `pixi add pymc`.

## Testing the installation

To test the installation, drop into a Python prompt and run:

```python
from sbi.examples.minimal import simple
posterior = simple()
print(posterior)
```
