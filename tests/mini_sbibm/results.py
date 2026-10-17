# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

"""Run labels, results files and the results table of mini SBIBM."""

import re
import subprocess
from pathlib import Path
from typing import Callable
from urllib.parse import quote

import pandas as pd

# One results file per run label. A new run replaces the stored rows with the same
# key. A new run replaces all seeds of a case, so the key has no seed.
RESULT_KEY = ["label", "method", "task_name"]
RESULT_COLUMNS = [*RESULT_KEY, "seed", "num_simulations", "c2st", "mean_err", "std_err"]
METRIC_TITLES = {
    "c2st": "C2ST (best: 0.5)",
    "mean_err": "Posterior mean error, in reference std (best: 0)",
    "std_err": "Posterior std error, in reference std (best: 0)",
}


def _git(*args: str) -> str:
    """Return the output of a git command, or an empty string if it fails."""
    try:
        return subprocess.run(
            ["git", *args], capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return ""


def benchmark_label(label: str | None) -> str:
    """Return the run label: the given label, else the git branch or commit."""
    if label is not None:
        return label
    branch = _git("rev-parse", "--abbrev-ref", "HEAD")
    if branch and branch != "HEAD":
        return branch
    # A detached HEAD has no branch; the short commit hash keeps checkouts apart.
    return _git("rev-parse", "--short", "HEAD") or "default"


def results_file(results_dir: Path, label: str) -> Path:
    """Return the results file of one run label.

    Each label has its own file, so runs with different labels never overwrite each
    other, also when they finish at the same time.
    """
    # Percent-encoding keeps different labels apart, e.g. "a/b" and "a_b".
    return results_dir / f"results-{quote(label, safe='')}.csv"


def read_results(path: Path) -> pd.DataFrame | None:
    """Return the stored results, or None if there are none in the current format."""
    if not path.exists():
        return None
    try:
        # Only empty cells are missing, so labels like "NA" stay strings.
        results = pd.read_csv(
            path, dtype={"label": str}, keep_default_na=False, na_values=[""]
        )
    except (pd.errors.ParserError, pd.errors.EmptyDataError):
        return None
    if not set(RESULT_COLUMNS).issubset(results.columns):
        return None
    return results


def merge_results(results_dir: Path, results: pd.DataFrame) -> None:
    """Merge new results of one label into the results file of that label."""
    path = results_file(results_dir, results["label"].iloc[0])
    stored = read_results(path)
    if stored is not None:
        replaced = stored.set_index(RESULT_KEY).index.isin(
            results.set_index(RESULT_KEY).index
        )
        results = pd.concat([stored[~replaced], results])
    path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(path, index=False)


def strip_ansi_escape_codes(text):
    ansi_escape = re.compile(r'\x1b\[.*?m')
    return ansi_escape.sub('', text)


# Function to center text with ANSI colors, adjusting for escape codes
def center_colored_text(text, width):
    visible_length = len(strip_ansi_escape_codes(text))
    padding = max(0, (width - visible_length) // 2)
    return " " * padding + text + " " * (width - visible_length - padding)


def write_results_tables(write: Callable[[str], None], results_dir: Path) -> None:
    """Write the stored results of all labels, one table per metric.

    Rows are methods with their run label, columns are tasks. Several seeds of a
    case are shown as mean and standard deviation.
    """
    stored = [read_results(path) for path in sorted(results_dir.glob("results-*.csv"))]
    stored = [results for results in stored if results is not None]
    if not stored:
        write("No results found.")
        return
    results = pd.concat(stored)

    rows = results["method"] + " [" + results["label"]
    if results["num_simulations"].nunique() > 1:
        rows += ", " + results["num_simulations"].astype(str) + " sims"
    results = results.assign(row=rows + "]")

    for metric, title in METRIC_TITLES.items():
        stats = results.groupby(["row", "task_name"])[metric].agg(["mean", "std"])
        write("")
        write(f"== {title} ==")
        _write_metric_table(write, stats["mean"].unstack(), stats["std"].unstack())


def _write_metric_table(write, means: pd.DataFrame, spreads: pd.DataFrame) -> None:
    """Write one table with a row per method and label and a column per task.

    Within each task, the best (lowest) value is green and the worst red.
    """
    tasks = list(means.columns)
    texts = {}
    for row in means.index:
        for task in tasks:
            mean, spread = means.at[row, task], spreads.at[row, task]
            if pd.isna(mean):
                texts[row, task] = "N/A"
            elif pd.isna(spread):
                texts[row, task] = f"{mean:.3f}"
            else:
                texts[row, task] = f"{mean:.3f} ±{spread:.3f}"

    row_width = max(len(row) for row in means.index) + 2
    widths = {
        task: max(10, len(task), *(len(texts[row, task]) for row in means.index)) + 2
        for task in tasks
    }
    header = " " * row_width + "".join(task.center(widths[task]) for task in tasks)
    write(header)
    write("-" * len(header))

    for row in means.index:
        line = row.ljust(row_width)
        for task in tasks:
            mean = means.at[row, task]
            if pd.isna(mean):
                line += texts[row, task].center(widths[task])
                continue
            low, high = means[task].min(), means[task].max()
            normalized = (mean - low) / (high - low) if high > low else 0.5
            if normalized == 0.0:
                color = "\033[92m"  # Green for best
            elif normalized == 1.0:
                color = "\033[91m"  # Red for worst
            else:
                color = f"\033[9{int(2 + normalized * 3)}m"
            line += center_colored_text(
                f"{color}{texts[row, task]}\033[0m", widths[task]
            )
        write(line)
