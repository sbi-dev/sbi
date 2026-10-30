"""Paired C2ST differences, NUTS diagnostics and parameter RMSE for the missing-data
check, from the outputs of `score.py` and the run logs.

    python experiments/mean_flow/scripts/missing/stats.py ~/meanflow-runs/missing
"""

import json
import re
import statistics
import sys
from pathlib import Path

import numpy as np

runs = Path(sys.argv[1])
scores = json.loads((runs / "scores.json").read_text())
reference = np.load(runs / "reference.npz")
NUM_SEEDS, NUM_SETS = 3, len(reference["theta"])

NUTS_LINE = re.compile(
    r"setting (\w+): nuts (converged|failed the convergence gate): .*?"
    r"min bulk ESS ([\d,]+), max Rhat ([\d.]+), (\d+) divergent, "
    r"([\d.e+]+) NFE, (\d+)s"
)
nuts = {}
for log in sorted((runs / "logs").glob("paper_s*_d*.txt")):
    seed, index = map(int, re.match(r"paper_s(\d+)_d(\d+)", log.stem).groups())
    for match in NUTS_LINE.finditer(log.read_text()):
        arm, status, _, _, _, nfe, wall = match.groups()
        nuts[(arm, seed, index)] = (status == "converged", float(nfe), int(wall))

print("not converged:", [key for key, (ok, _, _) in nuts.items() if not ok])
for arm in ("mf_dit", "nf"):
    runs_of_arm = [value for key, value in nuts.items() if key[0] == arm]
    walls = [wall for _, _, wall in runs_of_arm]
    per_1k = [1000 * wall / nfe for _, nfe, wall in runs_of_arm]
    print(
        f"{arm}: {sum(ok for ok, _, _ in runs_of_arm)} of {len(runs_of_arm)} "
        f"converged, median {statistics.median(walls):.0f} s, "
        f"{statistics.median(per_1k):.1f} s per 1,000 NFE"
    )


def c2st_of(arm: str) -> np.ndarray:
    """C2ST per seed and dataset."""
    return np.array([scores[f"{arm} s{s}"] for s in range(NUM_SEEDS)], dtype=float)


npe, mf, nf = c2st_of("sbi_npe"), c2st_of("mf_dit"), c2st_of("nf")
for name, values in (("sbi_npe", npe), ("mf_dit", mf), ("nf", nf)):
    print(f"{name}: C2ST {values.mean():.3f}, per seed {np.round(values.mean(1), 3)}")


def paired(label: str, a: np.ndarray, b: np.ndarray, keep=None) -> None:
    """Mean difference over seeds per dataset, standard error over datasets."""
    diff = a - b if keep is None else np.where(keep, a - b, np.nan)
    per_set = np.nanmean(diff, axis=0)
    se = per_set.std(ddof=1) / np.sqrt(len(per_set))
    print(f"{label}: {per_set.mean():+.3f} ({se:.3f}), n={len(per_set)}")


converged = np.array([
    [nuts[("mf_dit", s, i)][0] for i in range(NUM_SETS)] for s in range(NUM_SEEDS)
])
paired("mf_dit - sbi_npe", mf, npe)
paired("mf_dit - sbi_npe, converged runs only", mf, npe, converged)
paired("nf - sbi_npe", nf, npe)
paired("mf_dit - nf", mf, nf)


def rmse(samples_per_set) -> float:
    """Posterior-mean RMSE against the true theta, averaged over datasets."""
    errors = [
        np.sqrt(((samples.mean(0) - theta) ** 2).mean())
        for samples, theta in zip(samples_per_set, reference["theta"], strict=True)
    ]
    return float(np.mean(errors))


print(f"RMSE reference {rmse(reference['reference']):.3f}")
for s in range(NUM_SEEDS):
    paper = runs / f"paper_s{s}"
    arms = {
        "npe": np.load(runs / f"sbi_npe_s{s}.npz")["samples"],
        "mf_dit": [
            np.load(paper / f"paper_mf_dit_d{i}.npz")["samples"]
            for i in range(NUM_SETS)
        ],
        "nf": [
            np.load(paper / f"paper_nf_d{i}.npz")["samples"] for i in range(NUM_SETS)
        ],
    }
    print(f"RMSE seed {s}: " + ", ".join(f"{k} {rmse(v):.3f}" for k, v in arms.items()))
