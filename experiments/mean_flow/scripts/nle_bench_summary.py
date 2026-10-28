"""Tables from `nle_bench.py` rows: C2ST per arm, paired differences, cost.

python experiments/mean_flow/scripts/nle_bench_summary.py gpu.jsonl cpu.jsonl
"""

import json
import statistics
import sys
from collections import defaultdict

TASKS = ("two_moons", "linear_mvg_2d", "gaussian_linear", "slcp")


def label(row: dict) -> str:
    steps = f" S={row['num_steps']}" if row["num_steps"] is not None else ""
    return f"{row['arm']} {row['weights']}{steps}"


def mean_sd(values: list[float]) -> str:
    if len(values) == 1:
        return f"{values[0]:.3f}"
    return f"{statistics.mean(values):.3f} ± {statistics.stdev(values):.3f}"


def paired(rows: list[dict], a: str, b: str, n: int, task: str) -> str:
    """Mean of C2ST(a) - C2ST(b) over matching seeds and observations, with its
    standard error; negative means `a` is better."""
    by_key = defaultdict(dict)
    for r in rows:
        if r["num_simulations"] == n and r["task"] == task:
            by_key[r["seed"]][label(r)] = r["c2st_per_obs"]
    diffs = [
        x - y
        for arms in by_key.values()
        if a in arms and b in arms
        for x, y in zip(arms[a], arms[b], strict=True)
    ]
    if len(diffs) < 2:
        return "n/a"
    se = statistics.stdev(diffs) / len(diffs) ** 0.5
    return f"{statistics.mean(diffs):+.3f} ({se:.3f})"


def main() -> None:
    rows = []
    for path in sys.argv[1:]:
        with open(path) as f:
            rows += [json.loads(line) for line in f]
    for n in sorted({r["num_simulations"] for r in rows}):
        sub = [r for r in rows if r["num_simulations"] == n]
        print(f"## C2ST at {n:,} simulations (mean ± sd over seeds, [seeds])\n")
        print("| arm | " + " | ".join(TASKS) + " | parameters over tasks |")
        print("|---|" + "---|" * (len(TASKS) + 1))
        table = defaultdict(lambda: defaultdict(list))
        parameters = defaultdict(set)
        for r in sub:
            table[label(r)][r["task"]].append(r["c2st"])
            parameters[label(r)].add(r["parameters"])
        for name in sorted(table):
            cells = [
                f"{mean_sd(table[name][t])} [{len(table[name][t])}]"
                if table[name][t]
                else ""
                for t in TASKS
            ]
            low, high = min(parameters[name]), max(parameters[name])
            print(f"| {name} | " + " | ".join(cells) + f" | {low:,} to {high:,} |")

        print(f"\n## Paired C2ST differences at {n:,} (mean (se) over seeds x obs)\n")
        pairs = [
            ("mf_sbi best S=2", "mf_paper best S=2"),
            ("mf_sbi last S=2", "mf_paper last S=2"),
            ("abl_time best S=2", "mf_paper best S=2"),
            ("abl_actinit best S=2", "mf_paper best S=2"),
            ("abl_zeroout best S=2", "mf_paper best S=2"),
            ("abl_time last S=2", "mf_paper last S=2"),
            ("abl_actinit last S=2", "mf_paper last S=2"),
            ("abl_zeroout last S=2", "mf_paper last S=2"),
            ("mf_paper last S=2", "mf_paper best S=2"),
            ("mf_sbi last S=2", "mf_sbi best S=2"),
            ("mf_paper best S=2", "nsf_default sbi"),
            ("mf_paper last S=2", "nsf_default sbi"),
            ("mf_sbi best S=2", "nsf_default sbi"),
            ("mf_sbi best S=1", "mf_sbi best S=2"),
            ("nsf_paper best", "nsf_default sbi"),
            ("nsf_paper last", "nsf_paper best"),
        ]
        print("| a - b | " + " | ".join(TASKS) + " |")
        print("|---|" + "---|" * len(TASKS))
        for a, b in pairs:
            cells = [paired(sub, a, b, n, t) for t in TASKS]
            print(f"| {a} - {b} | " + " | ".join(cells) + " |")

        print(f"\n## Cost at {n:,} (median over tasks and seeds, seconds)\n")
        print("| arm | device | training | MCMC, 10 observations, S=1 / S=2 |")
        print("|---|---|---|---|")
        cost = defaultdict(lambda: defaultdict(list))
        for r in sub:
            cost[r["arm"]]["train"].append(r["train_time"])
            cost[r["arm"]][f"sample{r['num_steps']}"].append(r["sample_time"])
            cost[r["arm"]]["device"] = r["device"]
        for arm, c in sorted(cost.items()):
            sample = " / ".join(
                f"{statistics.median(c[k]):.0f}"
                for k in ("sample1", "sample2", "sampleNone")
                if c[k]
            )
            train = statistics.median(c["train"])
            print(f"| {arm} | {c['device']} | {train:.0f} | {sample} |")
        print()


if __name__ == "__main__":
    main()
