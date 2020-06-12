import os

from setuptools import find_packages, setup

desc = """`sbi` is a PyTorch package for simulation-based inference. Simulation-based
inference is the process of finding the parameters of a simulator from observations.
`sbi` takes a Bayesian approach and returns a full posterior distribution over the
parameters, conditional on the observations.

`sbi` offers a simple interface for one-line posterior inference

```python
from sbi inference import infer
# import your simulator, define your prior on the parameters
parameter_posterior = infer(simulator, prior, method='SNPE')
```

`sbi` is a community project that started from Conor M. Durkan's `lfi` and continued
development at the [MackeLab](https://mackelab.org). We would like to hear how it is
working for your simulation as well as receive bug reports, pull requests and other
feedback at [github.com/mackelab/sbi](https://github.com/mackelab/sbi).
"""


PROJECT_PATH = os.path.dirname(os.path.abspath(__file__))

version_namespace = {}
for line in open(os.path.join(PROJECT_PATH, "sbi", "__version__.py")):
    if line.startswith("__version__ = "):
        exec(line, version_namespace)

setup(
    name="sbi",
    version="0.8rc2",  # version_namespace["__version__"],
    description="Simulation-based inference.",
    long_description=desc,
    long_description_content_type="text/markdown",
    keywords="bayesian parameter inference system_identification simulator PyTorch",
    url="http://mackelab.org/sbi",
    author=(
        "Álvaro Tejero-Cantero, Jakob H. Macke, Jan-Matthis Lückmann,"
        " Conor M. Durkan, Michael Deistler, Jan Bölts."
    ),
    author_email="sbi@mackelab.org",
    packages=find_packages(exclude=["tests"]),
    license="AGPLv3",
    install_requires=[
        "joblib",
        "matplotlib",
        "nbstripout",
        "numpy",
        "pillow",
        "pyro-ppl",
        "pyknos==0.11",
        "scipy",
        "tensorboard",
        "torch>=1.4.0, !=1.5.0",  # See issue #37703 in PyTorch 1.5.0.
        "tqdm",
    ],
    extras_require={
        "dev": [
            "autoflake",
            "black",
            "deepdiff",
            "flake8",
            "isort",
            "mkdocs",
            "mkdocs-material",
            "markdown-include",
            "mkdocs-redirects",
            "mkdocstrings",
            "nbconvert",
            "pep517", 
            "pytest",
            "pyyaml",
            "scikit-learn",
            "torchtestcase",
            "twine",
        ]
    },
    dependency_links=[],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Information Technology",
        "Intended Audience :: Science/Research",
        "Topic :: Adaptive Technologies",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Games/Entertainment :: Simulation",
        (
            "License :: OSI Approved :: GNU Affero General Public License v3 or later"
            "(AGPLv3+)"
        ),
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
    ],
)
