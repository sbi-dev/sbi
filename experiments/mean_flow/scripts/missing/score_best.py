"""C2ST of the mean flow at its best epoch (`paper_arms.py --mf-best-epoch`) on the
missing-data check, against the last-epoch mean flow and sbi NPE scored by `score.py`.

    python experiments/mean_flow/scripts/missing/score_best.py \\
        --runs ~/meanflow-runs/missing --best ~/meanflow-runs/missing-best
"""

import argparse
import json
import re
import statistics
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path.cwd()))

from sbi.utils.metrics import c2st  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--runs", type=Path, required=True, help="the earlier check")
parser.add_argument("--best", type=Path, required=True, help="the best-epoch run")
parser.add_argument("--num-samples", type=int, default=4000)
args = parser.parse_args()

reference = np.load(args.runs / "reference.npz")
scores = json.loads((args.runs / "scores.json").read_text())
NUM_SEEDS, NUM_SETS = 3, len(reference["x"])

rng = np.random.default_rng(0)
best = []
for seed in range(NUM_SEEDS):
    row = []
    for index in range(NUM_SETS):
        samples = np.load(args.best / f"paper_s{seed}" / f"paper_mf_dit_d{index}.npz")
        samples = samples["samples"]
        ref = reference["reference"][index]
        m = min(args.num_samples, len(samples))
        a = ref[rng.choice(len(ref), m, replace=False)]
        b = samples[rng.choice(len(samples), m, replace=False)]
        a, b = (torch.as_tensor(v, dtype=torch.float32) for v in (a, b))
        row.append(round(float(c2st(a, b)), 3))
    best.append(row)
    print(f"mf_best s{seed} mean {statistics.mean(row):.3f}  {row}", flush=True)
scores.update({f"mf_best s{s}": row for s, row in enumerate(best)})
(args.best / "scores.json").write_text(json.dumps(scores, indent=1))


def c2st_of(arm: str) -> np.ndarray:
    """C2ST per seed and dataset."""
    return np.array([scores[f"{arm} s{s}"] for s in range(NUM_SEEDS)], dtype=float)


npe, last, mf_best = c2st_of("sbi_npe"), c2st_of("mf_dit"), c2st_of("mf_best")
for name, values in (("sbi_npe", npe), ("mf last", last), ("mf best", mf_best)):
    print(f"{name}: C2ST {values.mean():.3f}, per seed {np.round(values.mean(1), 3)}")


def paired(label: str, a: np.ndarray, b: np.ndarray) -> None:
    """Mean difference over seeds per dataset, standard error over datasets."""
    per_set = (a - b).mean(axis=0)
    se = per_set.std(ddof=1) / np.sqrt(len(per_set))
    print(f"{label}: {per_set.mean():+.3f} ({se:.3f}), n={len(per_set)}")


paired("mf best - mf last", mf_best, last)
paired("mf best - sbi_npe", mf_best, npe)
paired("mf last - sbi_npe", last, npe)

NUTS_LINE = re.compile(
    r"nuts (converged|failed the convergence gate): .*?min bulk ESS ([\d,]+), "
    r"max Rhat ([\d.]+), (\d+) divergent, ([\d.e+]+) NFE, (\d+)s"
)
runs = []
for log in sorted((args.best / "logs").glob("s*_d*.txt")):
    match = NUTS_LINE.search(log.read_text())
    runs.append(match.groups() if match else ("missing",) + ("",) * 5)
converged = sum(r[0] == "converged" for r in runs)
walls = [int(r[5]) for r in runs if r[5]]
print(
    f"NUTS: {converged} of {len(runs)} converged, "
    f"max divergent {max(int(r[3]) for r in runs if r[3])}, "
    f"median {statistics.median(walls):.0f} s"
)
