"""C2ST against the exact reference posteriors for every arm of the missing-data check,
per dataset, plus paired differences against sbi NPE.

    python experiments/mean_flow/scripts/missing/score.py --runs ~/meanflow-runs/missing
"""

import argparse
import json
import statistics
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path.cwd()))

from sbi.utils.metrics import c2st  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--runs", type=Path, required=True)
parser.add_argument("--num-samples", type=int, default=4000)
args = parser.parse_args()

reference = np.load(args.runs / "reference.npz")
num_sets = len(reference["x"])
arms: dict[str, list[np.ndarray | None]] = {}
for path in sorted(args.runs.glob("sbi_npe_s*.npz")):
    arms[f"sbi_npe {path.stem.split('_')[-1]}"] = list(np.load(path)["samples"])
for run in sorted(args.runs.glob("paper_s*")):
    for arm in ("mf_dit", "nf"):
        found = [run / f"paper_{arm}_d{i}.npz" for i in range(num_sets)]
        if any(p.exists() for p in found):
            arms[f"{arm} {run.name.split('_')[-1]}"] = [
                np.load(p)["samples"] if p.exists() else None for p in found
            ]

rng = np.random.default_rng(0)
n = args.num_samples
scores = {}
for name, per_set in arms.items():
    row = []
    for index, samples in enumerate(per_set):
        if samples is None:
            row.append(None)
            continue
        ref = reference["reference"][index]
        m = min(n, len(samples))
        a = ref[rng.choice(len(ref), m, replace=False)]
        b = samples[rng.choice(len(samples), m, replace=False)]
        a, b = (torch.as_tensor(v, dtype=torch.float32) for v in (a, b))
        row.append(round(float(c2st(a, b)), 3))
    scores[name] = row
    done = [v for v in row if v is not None]
    print(f"{name:16s} mean {statistics.mean(done):.3f} over {len(done)}  {row}")

for name, row in scores.items():
    if name.startswith("sbi_npe"):
        continue
    for npe in [k for k in scores if k.startswith("sbi_npe")]:
        pairs = zip(row, scores[npe], strict=True)
        diffs = [a - b for a, b in pairs if a is not None and b is not None]
        if len(diffs) > 1:
            se = statistics.stdev(diffs) / len(diffs) ** 0.5
            mean = statistics.mean(diffs)
            print(f"{name} - {npe}: {mean:+.3f} ({se:.3f}), n={len(diffs)}")
(args.runs / "scores.json").write_text(json.dumps(scores, indent=1))
