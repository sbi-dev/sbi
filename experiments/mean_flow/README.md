# Mean flow experiments

Evaluation of conditional mean flows (Arruda, Hasenauer, Radev 2026, arXiv:2610.21657) as a density estimator in sbi. Experimental branch; not meant for `main`.

| File | Contents |
|---|---|
| `paper_summary.md` | The method, the paper's four experiments and their results, how it relates to sbi |
| `results.md` | Our results: training regime, mini-sbibm at 2,000 simulations, consistency diagnostics, two moons at the paper's scale, bugs found, verdict |
| `gandk_reproduction_plan.md` | Plan for reproducing the paper's g-and-k comparison on a GPU machine, with a fair baseline |
| `scripts/bm_timing.py` | mini-sbibm runs with separate training and MCMC timing, NSF or mean flow, sbi trainer or paper recipe |
| `scripts/nle_bench.py` | NLE on mini-sbibm: paper and sbi mean-flow networks, single-difference variants between them, NSF; last and best epoch |
| `scripts/diagnostics.py` | Round trip, normalization across conditions, direct samples against MCMC on `log_prob` (two moons) |
| `scripts/mean_flow_paper_trainer.py` | The paper's training recipe with best-epoch selection on a held-out split |
| `scripts/training_variants.py` | Two moons NPE under different training regimes |
| `scripts/run_queue.py` | Runs a file of shell commands with N in parallel |
| `scripts/fetch_paper_code.py` | Downloads the paper's code from anonymous.4open.science |
| `results/` | Raw rows (`*.jsonl`, `*.json`) and the aggregated tables (`*summary.txt`) |

Run the scripts from the repository root, for example:

```bash
python experiments/mean_flow/scripts/bm_timing.py --mode nle --task two_moons \
    --estimator mean_flow --trainer paper --seed 0 --out results.jsonl
```

The estimator is `sbi.neural_nets.MeanFlowConfig` (`sbi/neural_nets/estimators/mean_flow_estimator.py`, `sbi/neural_nets/net_builders/mean_flow.py`, tests in `tests/mean_flow_estimator_test.py`).
