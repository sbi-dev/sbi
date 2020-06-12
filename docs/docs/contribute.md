## User experiences, bugs, and feature requests

If you are using `sbi` to infer the parameters of a simulators, we would be delighted to
know how it worked for you. If it didn't work according to plan, please open up an issue
and tell us more about your use case: the dimensionality of the input parameters, the
type of simulator and the dimensionality of the output.

To report bugs and suggest features (including better documentation), please equally
head over to [issues on GitHub](https://github.com/mackelab/sbi/issues). 


## Code contributions: pull requests.

In general, we use pull requests to make changes to `sbi`.

### Setting up a development environment

Clone [the repo](https://github.com/mackelab/sbi) and install all the dependencies using
the `environment.yml` file to create a conda environment: `conda env create -f
environment.yml`. If you already have an `sbi` environment and want to refresh
dependencies, just run `conda env update -f environment.yml --prune`.

Alternatively, you can install via `setup.py` using `pip install -e ".[dev]"` (the dev
flag installs development and testing dependencies).


### Code style

For docstrings and comments, we use [Google Style](http://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings).

Code needs to pass through the following tools, which are installed along with `sbi`:

**[black](https://github.com/psf/black)**: Automatic code formatting for Python. You can run black manually from the console using `black .` in the top directory of the repository, which will format all files. 

**[isort](https://github.com/timothycrosley/isort)**: Used to consistently order imports. You can run isort manually from the console using `isort -y` in the top directory.


### Documentation

The documentation is entirely written in markdown ([basic markdown guide](https://guides.github.com/features/mastering-markdown/)). It would be of great help, if you fixed mistakes, and edited where things are unclear. After a PR with documentation changes has been merged, the online documentation will be updated automatically in a couple of minutes. If you want to test a local build of the documentation, take a look at `docs/README.md`.

Examples are collected in notebooks in `examples/`.

### Binary files and Jupyter notebooks

#### Using `sbi` 

We use git lfs to store large binary files. Those files are not downloaded by cloning the repository, but you have to pull them separately. To do so follow installation instructions here [https://git-lfs.github.com/](https://git-lfs.github.com/). In particular, in a freshly cloned repository on a new machine, you will need to run both `git-lfs install` and `git-lfs pull`.

#### Contributing to `sbi`

We use a filename filter to track lfs files. Once you installed and pulled git lfs you can add a file to git lfs by appending `_gitlfs` to the basename, e.g., `oldbase_gitlfs.npy`. Then add the file to the index, commit, and it will be tracked by git lfs.

Additionally, to avoid large diffs due to Jupyter notebook outputs we are using `nbstripout` to remove output from notebooks before every commit. The `nbstripout` package is downloaded automatically during installation of `sbi`. However, **please make sure to set up the filter yourself**, e.g., through `nbstriout --install` or with different options as described [here](https://github.com/kynan/nbstripout).