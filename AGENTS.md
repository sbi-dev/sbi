# AGENTS.md — sbi

Shared guidance for coding agents working in this repository.

## Setup

```bash
uv venv --python 3.12
uv pip install -e ".[dev]"
```

Run validation commands through `uv run`; activating the virtual environment is
optional. See `docs/contributing.md` for the full development workflow and
`pyproject.toml` for the authoritative tool configuration.

## Validation

Start with checks targeted to the files and behavior you changed, then broaden as
appropriate:

```bash
uv run pytest tests/path_to_relevant_test.py
uv run pytest -n auto -m "not slow and not gpu"
uv run ruff check <changed Python paths>
uv run ruff format --check <changed Python paths>
uv run pyright sbi
uv run pre-commit run --files <changed files>
```

Tests live in `*_test.py` files. Available markers include `slow`, `gpu`, `mcmc`, and
`benchmark`; their definitions are in `pyproject.toml`.

## Code conventions

- Use Google-style docstrings.
- Put shared public type aliases in `sbi/sbi_types.py`.
- Keep string identifiers for estimators lowercase (for example, `"maf"`, `"nsf"`,
  and `"mdn"`).
- Add or update tests when behavior changes.
- Preserve unrelated local changes. If a checkout is already in use, work in a
  separate Git worktree.

## Code map

Inspect the current code rather than relying on a duplicated class inventory:

- `sbi/inference/__init__.py` — public inference methods
- `sbi/inference/trainers/` — training workflows
- `sbi/inference/posteriors/` — posterior implementations
- `sbi/inference/potentials/` — estimator-to-sampler bridges
- `sbi/neural_nets/estimators/` — estimator contracts and implementations
- `sbi/samplers/` — sampling backends

The documentation is built from `docs/`. Its curated `docs/llms.txt` index is served
at `https://sbi.readthedocs.io/en/latest/llms.txt` for tools that support it.
