import os

from setuptools import find_packages, setup


PROJECT_PATH = os.path.dirname(os.path.abspath(__file__))

version_namespace = {}
for line in open(os.path.join(PROJECT_PATH, "sbi", "__version__.py")):
    if line.startswith("__version__ = "):
        exec(line, version_namespace)

setup(
    name="sbi",
    version="0.8rc2",  # version_namespace["__version__"],
    description="Simulation-based inference.",
    download_url="https://github.com/mackelab/sbi/archive/v0.8rc2.tar.gz",
    keywords="bayesian parameter inference system_identification simulator PyTorch",
    url="https://github.com/mackelab/sbi",
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
            "pytest",
            "pyyaml",
            "torchtestcase",
            "scikit-learn",
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
        "License :: OSI Approved :: GNU Affero General Public License v3 or later (AGPLv3+)",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
    ],
)
