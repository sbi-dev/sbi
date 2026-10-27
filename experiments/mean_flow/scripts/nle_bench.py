"""NLE on mini-sbibm: the paper's mean-flow network, sbi's, single-difference
variants between them, and NSF, each scored at the last and the best epoch.

Same tasks, observations, metrics and default posterior as `bm_timing.py` (and
`tests/bm_test.py`). Mean flows and `nsf_paper` train with the paper's recipe
(`mean_flow_paper_trainer.py`) and are evaluated with both the best held-out and the
last-epoch weights; `nsf_default` trains with `NLE.train()` defaults.

The `abl_*` arms are the paper network (`MeanFlowConfig(variant="paper")`) with one
difference to sbi's `MeanFlowNet` switched to sbi's choice:

- `abl_time`: times $(s, t - s)$ through fixed Fourier features only (paper:
  $(s, s - t)$, trainable features plus the raw times);
- `abl_actinit`: GELU and PyTorch's default initialization, layer-norm epsilon 1e-5
  (paper: mish, Keras' He-normal, epsilon 1e-3);
- `abl_zeroout`: zero-initialized output layers (paper: Glorot uniform).

FiLM against adaptive layer norm is what remains between the paper network with all
three switched and `mf_sbi`.

    python experiments/mean_flow/scripts/nle_bench.py --task slcp --seed 0 \
        --num-simulations 2000 --arm mf_paper --out bench.jsonl
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch
from torch import Tensor, nn

sys.path.insert(0, str(Path.cwd()))
sys.path.insert(0, str(Path(__file__).parent))

from mean_flow_paper_trainer import train_paper_recipe  # noqa: E402

from sbi.inference import NLE  # noqa: E402
from sbi.neural_nets import MeanFlowConfig, NSFConfig  # noqa: E402
from sbi.neural_nets.net_builders.mean_flow import (  # noqa: E402
    PaperMeanFlowNet,
    _he_normal_,
    _mish,
)
from sbi.utils.metrics import c2st  # noqa: E402
from tests.mini_sbibm import get_task  # noqa: E402

NUM_EVALUATION_OBS = 10  # as tests/bm_test.py
NUM_SAMPLES = 1000
ARMS = (
    "mf_paper",
    "mf_sbi",
    "abl_time",
    "abl_actinit",
    "abl_zeroout",
    "nsf_paper",
    "nsf_default",
)


class _Block(nn.Module):
    """`_FiLMBlock` with a choice of activation and initialization."""

    def __init__(self, width: int, context_dim: int, paper_act_init: bool):
        super().__init__()
        init = _he_normal_ if paper_act_init else (lambda layer: layer)
        self.dense = init(nn.Linear(width, width))
        self.to_gamma = init(nn.Linear(context_dim, width))
        self.to_beta = init(nn.Linear(context_dim, width))
        self.norm = nn.LayerNorm(width, eps=1e-3 if paper_act_init else 1e-5)
        self.act = _mish if paper_act_init else nn.functional.gelu

    def forward(self, h: Tensor, context: Tensor) -> Tensor:
        update = self.act(self.dense(h))
        update = (1 + self.to_gamma(context)) * update + self.to_beta(context)
        return self.norm(h + update)


class AblationNet(nn.Module):
    """`PaperMeanFlowNet` with switches to sbi's time input, activation and
    initialization, or output initialization. With every switch on the paper's
    side it is `PaperMeanFlowNet` (same parameters, same function)."""

    def __init__(
        self,
        input_dim: int,
        condition_dim: int,
        paper_time: bool = True,
        paper_act_init: bool = True,
        zero_output: bool = False,
        hidden_features: int = 256,
        num_blocks: int = 5,
        num_head_blocks: int = 2,
        time_emb_dim: int = 32,
        fourier_scale: float = 0.3,
    ):
        super().__init__()
        self.paper_time, self.fourier_scale = paper_time, fourier_scale
        if paper_time:
            self.frequencies = nn.Parameter(0.05 * torch.randn(2, time_emb_dim // 2))
            time_dim = 2 + 2 * time_emb_dim
        else:
            self.register_buffer(
                "_frequencies", torch.randn(time_emb_dim // 2) * fourier_scale
            )
            time_dim = 2 * time_emb_dim
        init = _he_normal_ if paper_act_init else (lambda layer: layer)
        self.act = _mish if paper_act_init else nn.functional.gelu
        self.x_proj = init(nn.Linear(input_dim, hidden_features))
        self.c_proj = init(nn.Linear(condition_dim, hidden_features))
        self.merge_proj = init(nn.Linear(2 * hidden_features, hidden_features))
        context_dim = time_dim + hidden_features

        def blocks(n: int) -> nn.ModuleList:
            return nn.ModuleList(
                _Block(hidden_features, context_dim, paper_act_init) for _ in range(n)
            )

        self.trunk = blocks(num_blocks - num_head_blocks)
        self.heads = nn.ModuleDict({
            "mean": blocks(num_head_blocks),
            "velocity": blocks(num_head_blocks),
        })
        self.outputs = nn.ModuleDict({
            "mean": nn.Linear(hidden_features, input_dim),
            "velocity": nn.Linear(hidden_features, input_dim),
        })
        for output in self.outputs.values():
            if zero_output:
                nn.init.zeros_(output.weight)
            else:
                nn.init.xavier_uniform_(output.weight)
            nn.init.zeros_(output.bias)

    def _time_embedding(self, s: Tensor, t: Tensor) -> Tensor:
        if self.paper_time:
            times = torch.stack([s, s - t], dim=-1)
            projection = (
                2 * math.pi * self.fourier_scale * times[..., None] * self.frequencies
            )
            fourier = torch.cat([torch.sin(projection), torch.cos(projection)], -1)
            return torch.cat([times, fourier.flatten(-2)], dim=-1)
        embedded = []
        for times in (s, t - s):
            projection = 2 * math.pi * times[:, None] * self._frequencies
            embedded += [torch.sin(projection), torch.cos(projection)]
        return torch.cat(embedded, dim=-1)

    def forward(
        self, z: Tensor, condition_emb: Tensor, s: Tensor, t: Tensor, head="mean"
    ) -> Tensor:
        if head == "velocity":
            t = s
        condition_proj = self.c_proj(condition_emb)
        merged = torch.cat([self.x_proj(z), condition_proj], dim=-1)
        h = self.act(self.merge_proj(self.act(merged)))
        context = torch.cat([self._time_embedding(s, t), condition_proj], dim=-1)
        for block in [*self.trunk, *self.heads[head]]:
            h = block(h, context)
        return self.outputs[head](h)


def build(arm: str, input: Tensor, condition: Tensor):
    if arm == "mf_sbi":
        return MeanFlowConfig().build(input, condition)
    if arm.startswith("nsf"):
        return NSFConfig().build(input, condition)
    estimator = MeanFlowConfig(variant="paper").build(input, condition)
    if arm != "mf_paper":
        estimator.net = AblationNet(
            input.shape[1],
            condition.shape[1],
            paper_time=arm != "abl_time",
            paper_act_init=arm != "abl_actinit",
            zero_output=arm == "abl_zeroout",
        )
    return estimator


def check_ablation_net() -> None:
    """With every switch on the paper's side, `AblationNet` is `PaperMeanFlowNet`."""
    paper = PaperMeanFlowNet(3, 4)
    ablation = AblationNet(3, 4)
    ablation.load_state_dict(paper.state_dict())
    z, c, s, t = torch.randn(7, 3), torch.randn(7, 4), torch.rand(7), torch.rand(7)
    for head in ("mean", "velocity"):
        assert torch.allclose(paper(z, c, s, t, head), ablation(z, c, s, t, head))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True)
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-simulations", type=int, default=2000)
    parser.add_argument("--epochs", type=int, help="default 400 at 2,000, else 1000")
    parser.add_argument("--steps", default="1,2", help="mean-flow step counts")
    parser.add_argument("--device", default="cpu", help="training device")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--models", type=Path, help="directory for state dicts")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    check_ablation_net()
    epochs = args.epochs or (400 if args.num_simulations <= 2000 else 1000)

    task = get_task(args.task)
    torch.manual_seed(args.seed)
    theta, x = task.get_data(args.num_simulations)
    prior = task.get_prior()
    inference = NLE(prior, show_progress_bars=False)
    inference.append_simulations(theta, x)

    torch.manual_seed(args.seed + 1)
    start = time.perf_counter()
    if args.arm == "nsf_default":
        inference = NLE(prior, density_estimator=NSFConfig(), show_progress_bars=False)
        estimator = inference.append_simulations(theta, x).train()
        states = {"sbi": estimator.state_dict()}
        best_epoch = inference.summary["epochs_trained"][-1]
    else:
        estimator = build(args.arm, x, theta).to(args.device)
        states = {}
        _, best_epoch = train_paper_recipe(
            estimator, x.to(args.device), theta.to(args.device), epochs, states=states
        )
        estimator = estimator.cpu()
    train_time = time.perf_counter() - start
    if args.models is not None:
        args.models.mkdir(parents=True, exist_ok=True)
        name = f"{args.task}_{args.arm}_n{args.num_simulations}_s{args.seed}.pt"
        torch.save(states, args.models / name)

    steps = (
        [None]
        if args.arm.startswith("nsf")
        else [int(s) for s in args.steps.split(",")]
    )
    for weights, state in states.items():
        estimator.load_state_dict(state)
        estimator.eval()
        for num_steps in steps:
            if num_steps is not None:
                estimator.num_steps = num_steps
            # build_posterior works on a deep copy, so set num_steps before it.
            posterior = inference.build_posterior(estimator)
            sample_time, metrics = 0.0, []
            for idx in range(1, NUM_EVALUATION_OBS + 1):
                x_o = task.get_observation(idx)
                torch.manual_seed(args.seed * 1000 + idx)
                tic = time.perf_counter()
                samples = posterior.sample(
                    (NUM_SAMPLES,), x=x_o, show_progress_bars=False
                )
                sample_time += time.perf_counter() - tic
                reference = task.get_reference_posterior_samples(idx)
                reference_std = reference.std(0)
                metrics.append({
                    "c2st": float(c2st(reference[:NUM_SAMPLES], samples)),
                    "mean_err": float(
                        (
                            (samples.mean(0) - reference.mean(0)).abs() / reference_std
                        ).mean()
                    ),
                    "std_err": float(
                        ((samples.std(0) - reference_std).abs() / reference_std).mean()
                    ),
                })
            row = {
                "task": args.task,
                "arm": args.arm,
                "weights": weights,
                "num_steps": num_steps,
                "seed": args.seed,
                "num_simulations": args.num_simulations,
                "epochs": epochs if args.arm != "nsf_default" else None,
                "best_epoch": best_epoch,
                "device": args.device,
                "parameters": sum(p.numel() for p in estimator.parameters()),
                "train_time": round(train_time, 1),
                "sample_time": round(sample_time, 1),
                **{
                    key: round(sum(m[key] for m in metrics) / len(metrics), 4)
                    for key in metrics[0]
                },
                "c2st_per_obs": [round(m["c2st"], 3) for m in metrics],
            }
            with open(args.out, "a") as f:
                f.write(json.dumps(row) + "\n")
            print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
