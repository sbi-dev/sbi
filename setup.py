import os
import subprocess

from setuptools import find_packages, setup

PROJECT_PATH = os.path.dirname(os.path.abspath(__file__))
VERSION = """
# This file is auto-generated with the version information during setup.py installation.
__version__ = '{}'
"""

# Add commit_sha
for line in open(os.path.join(PROJECT_PATH, "sbi", "__init__.py")):
    if line.startswith("version_prefix = "):
        version = line.strip().split()[2][1:-1]
commit_sha = ""
try:
    commit_sha = (
        subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=PROJECT_PATH
        )
        .decode("ascii")
        .strip()
    )
except Exception:
    pass

if commit_sha:
    version += "+{}".format(commit_sha)
with open(os.path.join(PROJECT_PATH, "sbi", "_version.py"), "w") as f:
    f.write(VERSION.format(version))

setup(
    name="sbi",
    version=version,
    description="Simulation-based inference",
    url="https://github.com/mackelab/sbi",
    author="Conor Durkan, George Papamakarios, Artur Bekasov",
    packages=find_packages(exclude=["tests"]),
    license="GPLv3",
    install_requires=[
        "matplotlib",
        "nbstripout",
        "numpy",
        "pyro-ppl",
        "pyknos@git+https://github.com/mackelab/pyknos.git",
        "scipy",
        "tensorboard",
        "torch",
        "tqdm",
    ],
    extras_require={
        "dev": [
            "autoflake",
            "black",
            "deepdiff",
            "flake8",
            "isort",
            "pytest",
            "pyyaml",
            "torchtestcase",
        ]
    },
    dependency_links=[],
)
