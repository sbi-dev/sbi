from setuptools import find_packages, setup

exec(open("sbi/version.py").read())

setup(
    name="sbi",
    version=__version__,
    description="Simulation-based inference",
    url="https://github.com/mackelab/sbi",
    author="Conor Durkan, George Papamakarios, Artur Bekasov",
    packages=find_packages(exclude=["tests"]),
    license="GPLv3",
    install_requires=[
        "matplotlib",
        "numpy",
        "pyro-ppl",
        "pyknos@git+https://github.com/mackelab/nflows-pr",
        "scipy",
        "tensorboard",
        "torch",
        "tqdm",
    ],
    extras_requires={
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
