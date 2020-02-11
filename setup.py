from setuptools import find_packages, setup

exec(open("lfi/version.py").read())

setup(
    name="lfi",
    version=__version__,
    description="LFI + CDE.",
    url="https://github.com/mackelab/lfi",
    author="Conor Durkan",
    packages=find_packages(exclude=["tests"]),
    license="GPLv3",
    install_requires=[
        "matplotlib",
        "numpy",
        "pillow",
        "pyro-ppl",
        "scipy",
        "tensorboard",
        "torch",
        "tqdm",
    ],
    extras_requires={
        "dev": ["autoflake", "black", "flake8", "isort", "pyyaml"],
        "testing": ["pytest", "deepdiff", "torchtestcase"],
    },
    dependency_links=[],
)
