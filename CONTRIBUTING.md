# Contribution guidelines

By participating in the `sbi` community, all members are expected to comply with our [Code
of Conduct](CODE_OF_CONDUCT.md). This ensures a positive and inclusive environment for
everyone involved.

## User experiences, bugs, and feature requests

If you are using `sbi` to infer the parameters of a simulator, we would be delighted to
know how it worked for you. If it didn't work according to plan, please open up an
[issue](https://github.com/sbi-dev/sbi/issues) or
[discussion](https://github.com/sbi-dev/sbi/discussions) and tell us more about your use
case: the dimensionality of the input parameters and of the output, as well as the setup
you used to run inference (i.e., number of simulations, number of rounds, etc.).

To report bugs and suggest features (including better documentation), please equally
head over to [issues on GitHub](https://github.com/sbi-dev/sbi/issues).

## Code contributions

Contributions to the `sbi` package are welcome! In general, we use pull requests to make
changes to `sbi`. So, if you are planning to make a contribution, please fork, create a
feature branch, and then make a Pull Request (PR) from your feature branch to the upstream `sbi`
([details](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/creating-a-pull-request-from-a-fork)).
To give credit to contributors, we consider adding contributors who repeatedly and
substantially contributed to `sbi` to the list of authors of the package at the end of
every year. Additionally, we mention all contributors in the releases.

### Development environment

Clone [the repo](https://github.com/sbi-dev/sbi) and install all the dependencies via
`pyproject.toml` using `pip install -e ".[dev]"` (the `-e` flag installs the package
editable mode, and the `dev` flag installs development and testing dependencies).
This requires at least Python 3.8.

We use [`pre-commit`](https://pre-commit.com) to ensure proper formatting and perform
linting (see below). Please install `pre-commit` locally using `pre-commit install`.

### Style conventions and testing

For docstrings and comments, we use [Google
Style](http://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings).

For code linting and formating, we use [`ruff`](https://docs.astral.sh/ruff/), which is
installed alongside `sbi`.

You can exclude slow tests and those which require a GPU with `pytest -m "not slow and not gpu"`.
Additionally, we recommend to run tests with `pytest -n auto -m "not slow and not gpu"` in parallel.
GPU tests should probably not be run this way.
If you see unexpected behavior (tests fail if they shouldn't), try to run them without `-n auto` and see if it persists.
When writing new tests and debugging things, it may make sense to also run them without `-n auto`.

When you create a PR onto `main`, our Continuous Integration (CI) actions on GitHub will perform the following
checks:

- **`ruff`** for linting and formatting (including `black`, `isort`, and `flake8`)
- **[`pyright`](https://github.com/Microsoft/pyright)** for static type checking.
- **`pytest`** for running a subset of fast tests from our test suite.

If any of these fail, try reproducing and solving the error locally:

- **`ruff`**: Make sure you have `pre-commit` installed locally with the same version as specified in the [requirements](pyproject.toml). Execute it
 using `pre-commit run --all-files`. `ruff` tends to give informative error
  messages that help you fix the problem.
- **`pyright`**: Run it locally using `pyright sbi/` and ensure you are using the same
  `pyright` version as used in the CI (which is the case if you have installed it with `pip install -e ".[dev]"`
  but note that you have to rerun it once someone updates the version in the `pyproject.toml`).
  - Known issues and fixes:
    - If using `**kwargs`, you either have to specify all possible types of `kwargs`, e.g. `**kwargs: Union[int, boolean]` or use `**kwargs: Any`
- **`pytest`**: On GitHub Actions you can see which test failed. Reproduce it locally, e.g., using `pytest -n auto tests/linearGaussian_snpe_test.py`. Note that this will run for a few minutes and should result in passes and expected fails (xfailed).
- commit and push again until CI tests pass. Don't hesitate to ask for help by
  commenting on the PR.

## Online documentation

Most of [the documentation](http://sbi-dev.github.io/sbi) is written in markdown ([basic markdown guide](https://guides.github.com/features/mastering-markdown/)).

You can directly fix mistakes and suggest clearer formulations in markdown files simply by initiating a PR on GitHub. Click on [documentation
file](https://github.com/sbi-dev/sbi/tree/master/docs/docs) and look for the little pencil at top right.
