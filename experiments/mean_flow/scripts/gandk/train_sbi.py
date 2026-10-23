"""Train one sbi likelihood estimator on the exported g-and-k data of Arruda et al.

    python experiments/mean_flow/scripts/gandk/train_sbi.py --arm mf_sbi --seed 0 \
        --data ~/meanflow-runs/gandk-data --out ~/meanflow-runs/sbi-train --device cuda

Arms (see `ARMS`): the sbi mean flow, its port of the paper's network, a coupling
flow sized like the paper's under the paper's coupling-flow protocol and under the
mean-flow protocol, and sbi's NSF and MAF defaults under `NLE.train()`.

The paper recipe is BayesFlow 2.0.14's offline default as their code uses it: AdamW
(weight decay 5e-3, epsilon 1e-7), learning rate warmed up linearly from 5e-5 to
5e-4 over 5% of the steps and then cosine-decayed to zero, every gradient clipped to
norm 1.5 on its own (Keras `clipnorm`), batch 256 reshuffled every epoch, the last
partial batch kept. The held-out loss is taken on their 1,024 validation
simulations; for the mean flow with the same noise every epoch.

Writes `<out>/<arm>_s<seed>/`: `last.pt` and `best.pt` (whole estimators, the
coupling-flow protocol's `best.pt` being what Keras restores), `history.json`.
"""

import argparse
import copy
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from sbi.inference import NLE
from sbi.neural_nets import MAFConfig, MeanFlowConfig, NSFConfig
from sbi.utils import BoxUniform

# BayesFlow's coupling flow on g-and-k: depth 5 of dual couplings (10 single
# couplings), 16-bin splines on [-3, 3], subnets of two 256-wide layers, dropout 0.05.
NSF_PAPER = NSFConfig(
    hidden_features=256,
    num_transforms=10,
    num_bins=16,
    tail_bound=3.0,
    num_blocks=1,
    dropout_probability=0.05,
)

#: arm -> (config, epochs, early-stopping patience, trainer)
ARMS = {
    "mf_sbi": (MeanFlowConfig(), 1000, None, "paper"),
    "mf_paper": (MeanFlowConfig(variant="paper"), 1000, None, "paper"),
    "nsf_nfprotocol": (NSF_PAPER, 200, 10, "paper"),
    "nsf_long": (NSF_PAPER, 1000, None, "paper"),
    "nsf_default": (NSFConfig(), None, None, "sbi"),
    "maf_default": (MAFConfig(), None, None, "sbi"),
}


def clip_each_(parameters, max_norm: float) -> None:
    """Keras `clipnorm`: scale every gradient to norm at most `max_norm` on its own."""
    grads = [p.grad for p in parameters if p.grad is not None]
    norms = torch._foreach_norm(grads)
    scales = [max_norm / torch.clamp(n, min=max_norm) for n in norms]
    torch._foreach_mul_(grads, scales)


def train_paper_recipe(estimator, x, theta, x_val, theta_val, epochs, patience, log):
    """Returns the history; leaves the last weights in `estimator`, the best in
    `history["best_state"]`."""
    num_batches = math.ceil(len(x) / 256)
    total = epochs * num_batches
    warmup = int(0.05 * total)
    optimizer = torch.optim.AdamW(
        estimator.parameters(), lr=5e-4, weight_decay=5e-3, eps=1e-7
    )

    def learning_rate_factor(step: int) -> float:
        if step < warmup:
            return 0.1 + 0.9 * step / warmup
        return 0.5 * (1 + math.cos(math.pi * (step - warmup) / (total - warmup)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, learning_rate_factor)
    parameters = list(estimator.parameters())

    def validation_loss() -> float:
        estimator.eval()
        with (
            torch.no_grad(),
            torch.random.fork_rng(devices=[x.device] if x.is_cuda else []),
        ):
            torch.manual_seed(12345)
            value = estimator.loss(x_val, theta_val).mean()
        estimator.train()
        return float(value)

    history = {"train_loss": [], "val_loss": [], "epoch_seconds": []}
    best, best_state, best_epoch, wait = math.inf, None, 0, 0
    estimator.train()
    for epoch in range(1, epochs + 1):
        started, running = time.perf_counter(), 0.0
        for idx in torch.randperm(len(x), device=x.device).split(256):
            loss = estimator.loss(x[idx], theta[idx]).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            clip_each_(parameters, 1.5)
            optimizer.step()
            scheduler.step()
            running += float(loss.detach()) * len(idx)
        current = validation_loss()
        history["train_loss"].append(running / len(x))
        history["val_loss"].append(current)
        history["epoch_seconds"].append(time.perf_counter() - started)
        wait += 1
        if current < best:
            best, best_epoch, wait = current, epoch, 0
            best_state = copy.deepcopy(estimator.state_dict())
        log(
            f"epoch {epoch}/{epochs} train {history['train_loss'][-1]:.4f} "
            f"val {current:.4f} best {best:.4f}@{best_epoch} "
            f"{history['epoch_seconds'][-1]:.1f}s"
        )
        if patience is not None and wait >= patience:
            log(f"early stop at epoch {epoch}, best epoch {best_epoch}")
            break
    history |= {
        "best_epoch": best_epoch,
        "best_val_loss": best,
        "epochs_run": epoch,
        "best_state": best_state,
    }
    return history


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=sorted(ARMS), required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--epochs", type=int, help="override, for smoke tests")
    parser.add_argument("--limit", type=int, help="first rows only, for smoke tests")
    args = parser.parse_args()

    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    out = args.out / f"{args.arm}_s{args.seed}"
    out.mkdir(parents=True, exist_ok=True)

    def log(message: str) -> None:
        with open(out / "train.txt", "a") as log_file:
            print(message, file=log_file)

    data = np.load(args.data / "train.npz")

    def tensor(name: str) -> torch.Tensor:
        return torch.as_tensor(data[name], dtype=torch.float32, device=args.device)

    x, theta = tensor("x_train")[: args.limit], tensor("theta_train")[: args.limit]
    x_val, theta_val = tensor("x_val"), tensor("theta_val")
    config, epochs, patience, trainer = ARMS[args.arm]
    epochs = args.epochs or epochs
    log(f"arm {args.arm} seed {args.seed} device {args.device} config {config!r}")

    started = time.perf_counter()
    if trainer == "paper":
        estimator = config.build(x, theta).to(args.device)
        log(f"parameters {sum(p.numel() for p in estimator.parameters()):,}")
        history = train_paper_recipe(
            estimator, x, theta, x_val, theta_val, epochs, patience, log
        )
        best_state = history.pop("best_state")
        torch.save(estimator.cpu(), out / "last.pt")
        estimator.load_state_dict(best_state)
        torch.save(estimator.cpu(), out / "best.pt")
    else:
        lower = torch.as_tensor(data["lower"], dtype=torch.float32)
        upper = torch.as_tensor(data["upper"], dtype=torch.float32)
        prior = BoxUniform(lower, upper, device=args.device)
        inference = NLE(
            prior,
            density_estimator=config,
            device=args.device,
            show_progress_bars=False,
        )
        estimator = inference.append_simulations(theta, x).train()
        summary = inference.summary
        history = {
            "train_loss": summary["training_loss"],
            "val_loss": summary["validation_loss"],
            "best_epoch": int(np.argmin(summary["validation_loss"]) + 1),
            "epochs_run": len(summary["validation_loss"]),
        }
        log(f"sbi train(): {history['epochs_run']} epochs")
        # sbi's train() keeps the best epoch; there is no separate last epoch.
        torch.save(estimator.cpu(), out / "best.pt")
    history["train_seconds"] = time.perf_counter() - started
    history["parameters"] = sum(p.numel() for p in estimator.parameters())
    (out / "history.json").write_text(json.dumps(history))
    log(f"done in {history['train_seconds'] / 3600:.2f} h")


if __name__ == "__main__":
    main()
