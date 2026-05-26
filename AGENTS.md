# AGENTS.md — sbi

Instructions for AI coding agents (Claude Code, Codex, Copilot, Aider, etc.) working in this repository.

**Repository:** https://github.com/sbi-dev/sbi
**Documentation:** https://sbi.readthedocs.io/en/latest/
**Python:** >=3.10 | **License:** Apache 2.0

## Environment

```bash
source .venv/bin/activate
pip install -e ".[dev]"
```

## Testing

```bash
pytest tests/                              # Full suite
pytest -m "not slow and not gpu"           # Fast tests
pytest -n auto -m "not slow and not gpu"   # Fast + parallel
pytest --bm                                # mini-sbibm benchmarks
pytest --bm -n auto --bm-mode npe         # Single benchmark mode
```

**Markers:** `@pytest.mark.slow`, `.gpu`, `.mcmc`, `.benchmark`
**Test naming:** `*_test.py` files, `test_<description>` functions
**Fixtures (from `tests/conftest.py`):** `set_seed` (auto, seed=1), `mcmc_params_accurate` (20 chains), `mcmc_params_fast` (1 chain)

## Linting & Formatting

```bash
ruff check sbi/ --fix     # Lint (auto-fix)
ruff format sbi/          # Format
pyright sbi/              # Type check (basic mode)
pre-commit run --all-files
```

**Config:** Ruff line length 88, Pyright basic mode.

## Code Style

- Classes: `PascalCase` (e.g., `NPE_A`, `SNPE_C`)
- Functions/methods: `snake_case`, private: `_leading_underscore`
- Estimator strings: lowercase (`"maf"`, `"nsf"`, `"mdn"`)
- Type aliases: `sbi/sbi_types.py`
- Docstrings: Google style
- Import sorting: `ruff` / isort rules
- Per-file ignores: `__init__.py` allows unused imports; `test_*.py` allows star imports

## Important Files

- `ARCHITECTURE.md` — Domain glossary (trainer hierarchy, posterior types, design patterns)
- `sbi/inference/__init__.py` — Main inference exports
- `sbi/neural_nets/factory.py` — Network builder factory
- `sbi/utils/user_input_checks.py` — Input validation
- `sbi/__version__.py` — Version string
- `tests/conftest.py` — Test fixtures and configuration

## Repository Structure

- `sbi/` — Main package (inference, neural_nets, samplers, utils, analysis, diagnostics)
- `tests/` — Test suite (pytest)
- `docs/` — Sphinx documentation
- `mkdocs/` — MkDocs configuration

## Working with AI Agents

This repository is set up to be productive with any AI coding agent (Claude Code, Codex, Copilot, Aider, etc.). The instructions above are intentionally tool-agnostic — use whatever fits your workflow.

### Domain reference

Read `ARCHITECTURE.md` before naming domain concepts (in issue titles, refactor proposals, hypotheses, test names) so your output uses sbi's vocabulary rather than inventing synonyms.

### Documentation

The built docs site exposes an `llms.txt` index at `sbi.readthedocs.io/en/latest/llms.txt` — useful for agents that need to navigate the documentation rather than the codebase. The source file lives at `docs/llms.txt`.

### Issues and PRs

Issues and PRs are tracked on GitHub at `sbi-dev/sbi`. There is no prescribed agent-triage workflow — sbi uses standard maintainer-driven triage with the existing label set (`bug`, `enhancement`, `wontfix`, `help wanted`, etc.). When filing issues from an agent's output, use the `gh` CLI and write the issue in clear, human-reviewable form.

### AI assistance best practices

- Review AI-generated code carefully; do not accept blindly
- Ensure generated code is correct, efficient, and secure
- Add or update tests to verify behavior
- Mention AI assistance in PR/commit summaries when substantial

### Personal agent conventions

Individual maintainers may layer on their own AI-tooling conventions (custom skills, triage workflows, path-scoped rules) in their personal `.claude/`, `.codex/`, or equivalent local configs. These are not enforced repo-wide.
