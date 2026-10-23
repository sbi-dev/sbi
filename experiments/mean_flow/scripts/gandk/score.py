"""Score the paper's arms and the sbi arms on g-and-k with the paper's own metric code.

Runs in the paper's environment and imports their code without changing it:

    cd ~/meanflow-paper && JAX_PLATFORMS=cpu .venv-pypi/bin/python -I \
        <sbi>/experiments/mean_flow/scripts/gandk/score.py \
        --paper-dir ~/meanflow-paper --results ~/meanflow-runs/phase1/results \
        --data ~/meanflow-runs/gandk-data --sbi-mcmc ~/meanflow-runs/sbi-mcmc \
        --out <sbi>/experiments/mean_flow/results/gandk

Reads every dataset shard of their experiment 2 (rows and draws of their arms) and
every `<sbi-mcmc>/<label>/d<k>.npz` of `sample_sbi.py`. An sbi cell gets the row
their `run_arm` builds (`mcmc_summary`, NFE accounting by `nfe_per_evaluation`),
then all cells are scored together by their `score_sample_metrics` against their
cached reference posteriors (C2ST, W1, MMD at one common draw count, cells below 500
draws left unscored) and aggregated over datasets by `aggregate_over_datasets`.

Run their `--aggregate` first, on the same device (CPU): this checks that pooling
leaves the paper arms' C2ST draw count and values as their aggregate has them.

Writes `cells.jsonl` (one row per arm and dataset) and `arms.json` (mean, sd and
values over datasets per arm) to `--out`, and prints the table.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--paper-dir", type=Path, required=True)
parser.add_argument("--results", type=Path, required=True)
parser.add_argument("--data", type=Path, required=True)
parser.add_argument("--sbi-mcmc", type=Path, required=True)
parser.add_argument("--out", type=Path, required=True)
args = parser.parse_args()

paper = args.paper_dir.resolve()
sys.path[:0] = [str(paper), str(paper / "experiments")]

import arviz as az  # noqa: E402
from common import (  # noqa: E402  (sets KERAS_BACKEND)
    ADAPTATION,
    aggregate_over_datasets,
    converged,
    mcmc_summary,
    prior_truth_provenance,
    provenance_matches,
    score_sample_metrics,
)
from sampling.nfe import nfe_per_evaluation  # noqa: E402
from sampling.reference import reference_posterior  # noqa: E402

from experiments.config_run import MIN_KEPT_DRAWS, N_DATASETS, SEED  # noqa: E402
from experiments.tasks.config_gnk import N_OBS, REFERENCE, GnK  # noqa: E402

task = GnK()
names = list(task.param_names)
results = args.results / "exp2_gnk"
datasets = np.load(args.data / "datasets.npz")


class Log:
    """The two members of their `RunContext` that `score_sample_metrics` uses."""

    metrics: dict = {}

    @staticmethod
    def log(message: str) -> None:
        print(message, flush=True)


rows, draws = {}, {}
for index in range(N_DATASETS):
    shard = results / "shards" / f"dataset_{index:03d}.json"
    if not shard.exists():
        print(f"no shard for dataset {index}")
        continue
    body = json.loads(shard.read_text())
    # their _matches_run, as their --aggregate applies it
    truth = prior_truth_provenance(datasets["truths"][index])
    if not (
        int(body.get("seed", -1)) == SEED
        and int(body.get("n_datasets", -1)) == N_DATASETS
        and not body.get("smoke")
        and provenance_matches(body.get("provenance"), truth)
    ):
        raise SystemExit(f"{shard} is not a shard of this run")
    arrays = np.load(shard.with_suffix(".npz"))
    for label, row in body["payload"]["arms"].items():
        rows[(label, index)] = row | {"source": "paper"}
        if label in arrays:
            draws[(label, index)] = arrays[label]


def training_info(model: str) -> dict:
    run = Path(model).parent
    history = json.loads((run / "history.json").read_text())
    return {
        "weights": Path(model).stem,
        "best_epoch": history.get("best_epoch"),
        "epochs_run": history.get("epochs_run"),
        "train_hours": round(history["train_seconds"] / 3600, 2),
        "parameters": history.get("parameters"),
    }


for run in sorted(p for p in args.sbi_mcmc.iterdir() if p.is_dir()):
    for cell in sorted(run.glob("d*.npz")):
        index = int(cell.stem[1:])
        r = np.load(cell)
        steps = int(r["nfe_per_density"])
        kind = str(r["kind"])
        if (run.name, index) in rows:
            raise SystemExit(f"label {run.name} collides with a paper arm")
        theta = r["theta"]
        posterior = {name: theta[..., j] for j, name in enumerate(names)}
        idata = az.from_dict(
            posterior=posterior,
            sample_stats={"n_steps": r["n_steps"], "diverging": r["diverging"]},
            warmup_sample_stats={
                "n_steps": r["warmup_n_steps"],
                "diverging": r["warmup_diverging"],
            },
            save_warmup=True,
        )
        nfe = sum(
            int(idata[g]["n_steps"].sum())
            for g in ("sample_stats", "warmup_sample_stats")
        )
        nfe *= nfe_per_evaluation(kind, steps if kind == "mf" else None, n_obs=N_OBS)
        row = {
            "arm": run.name,
            "source": str(r["source"]),
            "kind": kind,
            "sampler": "nuts",
            "steps": steps,
            "nfe_per_density": steps,
            "requested_draws": int(r["draws"]),
            "kept_draws_per_chain": int(theta.shape[1]),
            "budget_matched": True,
            "avg_leapfrogs": round(float(np.mean(r["n_steps"])), 2),
            "nuts_adaptation": ADAPTATION,
            "tune_per_chain": int(r["warmup_n_steps"].shape[1]),
            "logp_calls": int(r["logp_calls"]),
        } | mcmc_summary(
            idata,
            nfe=nfe,
            n_obs=N_OBS,
            param_names=names,
            wall=float(r["wall_seconds"]),
        )
        row |= training_info(str(r["model"]))
        rows[(run.name, index)] = row
        draws[(run.name, index)] = theta.reshape(-1, len(names))


def reference_for(key) -> np.ndarray:
    index = key[1]
    return reference_posterior(
        task,
        datasets["observations"][index],
        seed=SEED + index,
        cache_path=results / "cache" / f"reference_d{index}.npz",
        config=REFERENCE,
        verbose=False,
        initial_theta=datasets["truths"][index],
    )


n_kernel = score_sample_metrics(
    Log,
    rows,
    draws,
    reference_for,
    label_of=lambda key: f"{key[0]} on dataset {key[1]}",
    param_names=names,
    divergences=("c2st", "w1", "mmd"),
    min_draws=MIN_KEPT_DRAWS,
)
for row in rows.values():
    row["converged"] = converged(row)

# Pooling must not change the draw count the paper's own arms are scored at, or their
# C2ST would no longer be the one their --aggregate (and Table S6) reports.
theirs = json.loads((results / "metrics.json").read_text())
if n_kernel != theirs["scoring"]["n_kernel"]:
    raise SystemExit(
        f"pooled C2ST size {n_kernel} differs from their aggregate's "
        f"{theirs['scoring']['n_kernel']}"
    )
# their c2st_values list the pooled datasets in order; all 10 are pooled
for label, record in theirs["arms"].items():
    for index, value in enumerate(record.get("c2st_values", [])):
        ours = rows.get((label, index), {}).get("c2st")
        if ours is not None and abs(ours - value) > 1e-6:
            print(
                f"warning: {label} d{index} C2ST {ours:.4f}, "
                f"their aggregate {value:.4f}"
            )

args.out.mkdir(parents=True, exist_ok=True)
with open(args.out / "cells.jsonl", "w") as f:
    for (_, index), row in sorted(rows.items()):
        f.write(json.dumps({"dataset": index} | row, default=float) + "\n")

per_arm: dict[str, list] = {}
for (label, _), row in sorted(rows.items(), key=lambda kv: kv[0][1]):
    per_arm.setdefault(label, []).append(row)
arms = {label: aggregate_over_datasets(r) for label, r in per_arm.items()}
(args.out / "arms.json").write_text(
    json.dumps({"n_kernel": n_kernel, "arms": arms}, default=float, indent=1)
)

columns = [
    "n_datasets",
    "kept_draws",
    "min_bulk_ess",
    "ess_per_1k_nfe",
    "max_rhat",
    "c2st",
    "c2st_std",
    "wall_seconds",
]
print(f"\nC2ST etc. at {n_kernel} draws per side; mean over datasets")
print(f"{'arm':28s}" + "".join(f"{c:>15s}" for c in columns))
for label, agg in arms.items():
    cells = "".join(
        f"{agg[c]:15.4g}" if isinstance(agg.get(c), (int, float)) else f"{'-':>15s}"
        for c in columns
    )
    print(f"{label:28s}{cells}")
