import os

import nbformat
import pytest
from nbclient.exceptions import CellExecutionError
from nbconvert.preprocessors import ExecutePreprocessor


def list_notebooks(directory: str) -> list:
    """Return sorted list of all notebooks in a directory."""
    notebooks = [
        os.path.join(directory, f)
        for f in os.listdir(directory)
        if f.endswith(".ipynb")
    ]
    return sorted(notebooks)


@pytest.mark.slow
@pytest.mark.parametrize("notebook_path", list_notebooks("tutorials/"))
def test_tutorials(notebook_path):
    """Test that all notebooks in the tutorials directory can be executed."""
    with open(notebook_path) as f:
        nb = nbformat.read(f, as_version=4)
        ep = ExecutePreprocessor(timeout=600, kernel_name='python3')
        print(f"Executing notebook {notebook_path}")
        try:
            ep.preprocess(nb, {'metadata': {'path': os.path.dirname(notebook_path)}})
        except CellExecutionError as e:
            # Conditional_distributions tutorial requires movie writer that is failing
            # in GitHub CI.
            if "Requested MovieWriter" in str(e):
                print("Skipping error in movie writer.")
            else:
                raise RuntimeError(
                    f"Error executing the notebook {notebook_path}: {e}"
                ) from e
        except Exception as e:
            raise RuntimeError(
                f"Error executing the notebook {notebook_path}: {e}"
            ) from e
