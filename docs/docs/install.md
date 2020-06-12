# Installation

We recommend to use a [`conda`](https://docs.conda.io/en/latest/miniconda.html) virtual
environment with Python >= 3.7. If you don't have one, it should work with the following
steps
```shell
# 1. install miniconda from https://docs.conda.io/en/latest/miniconda.html
$ (dependent on your system, follow instructions on website)
# 2. create an environment for sbi (indicate Python 3.7 or higher); activate it
$ conda create -n sbi_env python=3.7 && conda activate sbi_env
```
Then install `sbi`:
```shell
$ pip install sbi
```
To test the installation, drop into a python prompt and run 
```python
from sbi.example.minimal import simple
posterior = simple()
print(posterior)
``` 