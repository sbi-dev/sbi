## User experiences, bugs, and feature requests

If you are using `sbi` to infer the parameters of a simulators, we would be delighted to
know how it worked for you. If it didn't work according to plan, please open up an issue
and tell us more about your use case: the dimensionality of the input parameters, the
type of simulator and the dimensionality of the output.

To report bugs and suggest features (including better documentation), please equally
head over to [issues on GitHub](https://github.com/mackelab/sbi/issues). 


## Code contributions: pull requests

In general, we use pull requests to make changes to `sbi`.

### Setting up a development environment

Clone [the repo](https://github.com/mackelab/sbi) and install all the dependencies using
the `environment.yml` file to create a conda environment: `conda env create -f
environment.yml`. If you already have an `sbi` environment and want to refresh
dependencies, just run `conda env update -f environment.yml --prune`.

Alternatively, you can install via `setup.py` using `pip install -e ".[dev]"` (the dev
flag installs development and testing dependencies).


### Contributing to code: style conventions

For docstrings and comments, we use [Google
Style](http://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings).

Code needs to pass through the following tools, which are installed alongside `sbi`:

**[black](https://github.com/psf/black)**: Automatic code formatting for Python. You can
run black manually from the console using `black .` in the top directory of the
repository, which will format all files. 

**[isort](https://github.com/timothycrosley/isort)**: Used to consistently order
imports. You can run isort manually from the console using `isort -y` in the top
directory.


### Online documentation

Part of the documentation is written as Jupyter notebooks. If you want to contribute to
these files, please be sure to strip output before committing, either by hand upon
saving or using an automated tool such as `nbstripout`.

The rest of [the documentation](http://mackelab.org/sbi) is written in markdown ([basic
markdown guide](https://guides.github.com/features/mastering-markdown/)).

You can directly fix mistakes and suggest clearer formulations in markdown files simply
by initiating a PR on the web. Click on [documentation
file](https://github.com/mackelab/sbi/tree/master/docs/docs) and look for the little
pencil at top left. 

The online documentation will be updated automatically a couple of minutes after we
merge your changes. If you want to test a local build of the documentation, take a look
at `docs/README.md`.