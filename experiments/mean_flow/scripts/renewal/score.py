"""Scores every arm of the paper's experiment 3 with the paper's own metrics.

Run with the paper's interpreter. The paper's arms are read from the posterior draws
their dataset stages keep (`<results>/exp3_renewal/shards/progress/dataset_<k>/`),
sbi NPE from `sbi_npe.py`'s outputs. Per dataset: the RMSE of the posterior median of
log R_t against the true curve and the 90% coverage of log R_t (`rt_scores` in
`exp3_renewal.py`), and the draw-level NRMSE of the coefficients
(`helpers.metrics.nrmse` with the prior range, as `common.py` scores it). Then the
mean over datasets and paired differences against each NPE run.

    cd ~/meanflow-paper && .venv-pypi/bin/python <sbi>/.../renewal/score.py \\
        --data ~/meanflow-runs/renewal/data.npz \\
        --paper ~/meanflow-runs/renewal/results \\
        --npe ~/meanflow-runs/renewal/npe_s0.npz --out scores.json
"""

import argparse
import json
import statistics
import sys
from pathlib import Path

sys.path[:0] = [str(Path.cwd()), str(Path.cwd() / "experiments")]

import numpy as np  # noqa: E402
from exp3_renewal import rt_scores  # noqa: E402
from helpers.metrics import nrmse  # noqa: E402

from experiments.tasks.config_renewal import Renewal  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--data", type=Path, required=True)
parser.add_argument("--paper", type=Path, nargs="*", default=[], help="results dirs")
parser.add_argument("--npe", type=Path, nargs="*", default=[])
parser.add_argument("--out", type=Path, required=True)
args = parser.parse_args()

task = Renewal()
data = np.load(args.data)
truths = data["theta"]
arms: dict[str, list[np.ndarray | None]] = {}
for results in args.paper:
    progress = results / "exp3_renewal" / "shards" / "progress"
    for arm in ("mf_dit", "nf"):
        found = [
            progress / f"dataset_{k:03d}" / f"{arm}.npz" for k in range(len(truths))
        ]
        if any(p.exists() for p in found):
            arms[f"{arm} {results.name}"] = [
                np.load(p)["samples"] if p.exists() else None for p in found
            ]
for path in args.npe:
    arms[f"npe {path.stem}"] = list(np.load(path)["samples"])

scores = {}
for name, per_set in arms.items():
    rows = []
    for samples, truth in zip(per_set, truths, strict=True):
        if samples is None:
            rows.append(None)
            continue
        row = rt_scores(task, samples, truth)
        row["nrmse"] = nrmse(samples, truth, scale=task.upper - task.lower)["nrmse"]
        rows.append(row)
    scores[name] = rows
    done = [r for r in rows if r is not None]
    summary = ", ".join(
        f"{key} {statistics.mean(r[key] for r in done):.3f}"
        for key in ("rt_rmse", "rt_cov90", "nrmse")
    )
    print(f"{name:24s} {summary} over {len(done)} datasets")

for name, rows in scores.items():
    for npe in [k for k in scores if k.startswith("npe") and k != name]:
        diffs = [
            a["rt_rmse"] - b["rt_rmse"]
            for a, b in zip(rows, scores[npe], strict=True)
            if a is not None and b is not None
        ]
        if len(diffs) > 1:
            se = statistics.stdev(diffs) / len(diffs) ** 0.5
            print(
                f"rt_rmse {name} - {npe}: {statistics.mean(diffs):+.3f} ({se:.3f}), "
                f"n={len(diffs)}"
            )
args.out.write_text(json.dumps(scores, indent=1) + "\n")
