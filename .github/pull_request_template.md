## What does this PR do?

Describe your changes in a few sentences. What problem does this solve, or what
feature does it add?

## Does this close any issues?

Link the issue this fixes, for example `Fixes #123`.

## Anything else we should know?

Open questions, things you would like feedback on, or relevant logs and
screenshots.

## Checklist

Put an `x` in the boxes that apply. If you are unsure about any of them, just
ask - we are happy to help.

- [ ] I have read the [contributing guide](https://sbi.readthedocs.io/en/latest/contributing.html).
- [ ] `uv run pytest -n auto -m "not slow and not gpu"` passes.
- [ ] `uv run pre-commit run --all-files` passes (ruff and formatting).
- [ ] `uv run pyright sbi` passes.
- [ ] I added or updated tests for the changed behavior.
- [ ] I used Google-style docstrings for new or changed public functions.
- [ ] (If applicable) I reported how long new tests run and marked slow ones
      with `pytest.mark.slow`.

<!-- If you used an AI assistant for a substantial part of this pull request,
please tell us briefly how. -->
