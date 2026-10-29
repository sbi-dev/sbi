"""Evaluation datasets and exact reference posteriors for the missing-data task.

    python experiments/mean_flow/scripts/missing/make_reference.py \
        --out ~/meanflow-runs/missing/reference.npz
"""

import argparse
from pathlib import Path

import numpy as np
from wave_task import datasets, reference_posterior

parser = argparse.ArgumentParser()
parser.add_argument("--num-datasets", type=int, default=10)
parser.add_argument("--draws", type=int, default=20_000)
parser.add_argument("--out", type=Path, required=True)
args = parser.parse_args()

sets = datasets(args.num_datasets)
references, ess = [], []
for index, d in enumerate(sets):
    draws, e = reference_posterior(
        d["x"], d["present"], args.draws, np.random.default_rng(50_000 + index)
    )
    references.append(draws)
    ess.append(e)
    print(
        f"dataset {index}: {int(d['present'].sum())} of 32 present, IS ESS {e:,.0f}, "
        f"posterior sd {np.round(draws.std(0), 3)}",
        flush=True,
    )
args.out.parent.mkdir(parents=True, exist_ok=True)
np.savez(
    args.out,
    theta=np.stack([d["theta"] for d in sets]),
    x=np.stack([d["x"] for d in sets]),
    present=np.stack([d["present"] for d in sets]),
    reference=np.stack(references),
    ess=np.array(ess),
)
