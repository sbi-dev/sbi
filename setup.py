import os

from setuptools import find_packages, setup


PROJECT_PATH = os.path.dirname(os.path.abspath(__file__))

version_namespace = {}
for line in open(os.path.join(PROJECT_PATH, "sbi", "__version__.py")):
    if line.startswith("__version__ = "):
        exec(line, version_namespace)

setup(
    name="sbi",
    version=version_namespace["__version__"],
    description="Simulation-based inference",
    url="https://github.com/mackelab/sbi",
    author="Conor Durkan, George Papamakarios, Artur Bekasov",
    packages=find_packages(exclude=["tests"]),
    license="GPLv3",
    install_requires=[
        "matplotlib",
        "nbstripout",
        "numpy",
        "pillow",
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
            "scikit-learn",
        ]
    },
    dependency_links=[],
)
