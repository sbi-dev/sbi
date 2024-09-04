# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

# Script to strip outputs of Jupyter notebooks except for plots and prints so that there
# are displayed nicely on the documentation website.

import os

import nbformat
from nbconvert.preprocessors import Preprocessor


class StripOutputExceptPlotsAndPrintsPreprocessor(Preprocessor):
    """Subclass of nbconvert.preprocessors.Preprocessor to strip notebook outputs"""

    def preprocess_cell(
        self, cell: nbformat.NotebookNode, resources: dict, cell_index: int
    ) -> tuple:
        """Preprocess a cell

        Note, this method will be called internall by the Preprocessor.preprocess
        method and needs to have this specific signature.

        Args:
            cell: The cell to preprocess
            resources: Additional resources used in the conversion process
            cell_index: The index of the cell
        """
        if cell.cell_type == 'code':
            # Retain outputs that contain either 'image/png' or 'stdout' (prints)
            cell.outputs = [
                output
                for output in cell.outputs
                if ('data' in output and 'image/png' in output['data'])
                or (
                    output.get('output_type') == 'stream'
                    and output.get('name') == 'stdout'
                )
            ]
        return cell, resources


def strip_output_except_plots_and_prints(notebook_path):
    """Strip the output from a Jupyter notebook, except for plots and prints"""

    # Read the notebook
    with open(notebook_path, 'r') as f:
        nb = nbformat.read(f, as_version=4)

    # Strip the output
    preprocessor = StripOutputExceptPlotsAndPrintsPreprocessor()
    nb, _ = preprocessor.preprocess(nb, {})

    # Write the notebook back to disk
    with open(notebook_path, 'w') as f:
        nbformat.write(nb, f)


def strip_notebooks_in_directory(directory):
    """Strip output from all notebooks in a directory, except for plots and prints"""

    # Walk through the directory
    for root, _, files in os.walk(directory):
        for file in files:
            # Filter for Jupyter notebooks
            if file.endswith(".ipynb"):
                notebook_path = os.path.join(root, file)
                strip_output_except_plots_and_prints(notebook_path)


# Add main to run the script from the command line for GitHub Actions
if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage: python strip_notebook_outputs.py <notebooks_directory>")
        sys.exit(1)

    notebooks_directory = sys.argv[1]
    strip_notebooks_in_directory(notebooks_directory)
