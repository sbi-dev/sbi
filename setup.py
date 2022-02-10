#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Affero General Public License v3, see <https://www.gnu.org/licenses/>.
#
# Note: To use the 'upload' functionality of this file, you must:
#   $ pipenv install twine --dev

import io
import os
import sys
from shutil import rmtree

from setuptools import Command, find_packages, setup

# Package meta-data.
NAME = "sbi"
DESCRIPTION = "Simulation-based inference."
KEYWORDS = "bayesian parameter inference system_identification simulator PyTorch"
URL = "https://github.com/mackelab/sbi"
EMAIL = "sbi@mackelab.org"
AUTHOR = "Álvaro Tejero-Cantero, Jakob H. Macke, Jan-Matthis Lückmann, Conor M. Durkan, Michael Deistler, Jan Bölts"
REQUIRES_PYTHON = ">=3.6.0"

REQUIRED = [
    "joblib>=1.0.0",
    "matplotlib",
    "numpy",
    "pillow",
    "pyknos>=0.14.2",
    "pyro-ppl>=1.3.1",
    "scikit-learn",
    "scipy",
    "tensorboard",
    "torch>=1.8.0",
    "tqdm",
]

EXTRAS = {
    "dev": [
        "autoflake",
        "black",
        "deepdiff",
        "flake8",
        "isort",
        "jupyter",
        "mkdocs",
        "mkdocs-material",
        "markdown-include",
        "mkdocs-redirects",
        "mkdocstrings",
        "nbconvert",
        "pep517",
        "pytest",
        "pyyaml",
        "pyright",
        "torchtestcase",
        "twine",
    ],
}

here = os.path.abspath(os.path.dirname(__file__))

# Import the README and use it as the long-description.
try:
    with io.open(os.path.join(here, "README.md"), encoding="utf-8") as f:
        long_description = "\n" + f.read()
except FileNotFoundError:
    long_description = DESCRIPTION

# Load the package's __version__.py module as a dictionary.
about = {}
project_slug = NAME.lower().replace("-", "_").replace(" ", "_")
with open(os.path.join(here, project_slug, "__version__.py")) as f:
    exec(f.read(), about)


class UploadCommand(Command):
    """Support setup.py upload."""

    description = "Build and publish the package."
    user_options = []

    @staticmethod
    def status(s):
        """Prints things in bold."""
        print("\033[1m{0}\033[0m".format(s))

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        try:
            self.status("Removing previous builds…")
            rmtree(os.path.join(here, "dist"))
        except OSError:
            pass

        self.status("Building Source and Wheel (universal) distribution…")
        os.system("{0} setup.py sdist bdist_wheel --universal".format(sys.executable))

        self.status("Uploading the package to PyPI via Twine…")
        os.system("twine upload dist/*")

        self.status("Pushing git tags…")
        os.system("git tag v{0}".format(about["__version__"]))
        os.system("git push --tags")

        sys.exit()


setup(
    name=NAME,
    version=about["__version__"],
    description=DESCRIPTION,
    keywords=KEYWORDS,
    long_description=long_description,
    long_description_content_type="text/markdown",
    author=AUTHOR,
    author_email=EMAIL,
    python_requires=REQUIRES_PYTHON,
    url=URL,
    packages=find_packages(exclude=["tests", "*.tests", "*.tests.*", "tests.*"]),
    install_requires=REQUIRED,
    extras_require=EXTRAS,
    include_package_data=True,
    license="AGPLv3",
    classifiers=[
        # Full list: https://pypi.python.org/pypi?%3Aaction=list_classifiers
        "Development Status :: 3 - Alpha",
        "License :: OSI Approved :: GNU Affero General Public License v3 or later (AGPLv3+)",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "Topic :: Adaptive Technologies",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
    ],
    # $ setup.py publish support.
    cmdclass=dict(upload=UploadCommand),
)
