"""sbi NPE with the missingness mask appended, on the paper's experiment 3.

As `scripts/missing/sbi_npe.py`: the observation standardized per coordinate, missing
entries set to 0 and the 0/1 mask appended, a new pattern per training simulation
every epoch, 10% of the simulations held out with 10 fixed patterns each, Adam with lr
5e-4, batch 200, clipping 5.0, early stopping after 50 epochs, best epoch kept. The
data come from `export.py`; patterns follow the paper's rule (`Renewal.present_mask`:
missing at random, at least one coordinate kept).

    python experiments/mean_flow/scripts/renewal/sbi_npe.py --seed 0 \\
        --data ~/meanflow-runs/renewal/data.npz --out ~/meanflow-runs/renewal/npe_s0.npz
"""

import argparse
import copy
import time
from pathlib import Path

import numpy as np
import torch

from sbi.inference.posteriors import DirectPosterior
from sbi.neural_nets import NSFConfig


def present_mask(n: int, dim: int, rate: float, rng) -> np.ndarray:
    drawn = rng.random((n, dim))
    return (drawn >= rate) | (drawn >= drawn.max(axis=-1, keepdims=True))


def encode(x, present, mean, std, device) -> torch.Tensor:
    z = np.where(present, (x - mean) / std, 0.0)
    joint = np.concatenate([z, present], -1)
    return torch.as_tensor(joint, dtype=torch.float32, device=device)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--hidden-features", type=int, default=50)
    parser.add_argument("--draws", type=int, default=4000)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--max-epochs", type=int, default=2000)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    data = np.load(args.data)
    rate = float(data["missing_rate"])
    theta, x = data["theta_train"], data["x_train"]
    num, dim = x.shape

    rng = np.random.default_rng(args.seed)
    order = rng.permutation(num)
    theta, x = theta[order], x[order]
    mean, std = x.mean(0), x.std(0)
    num_val = num // 10

    def to_tensor(a):
        return torch.as_tensor(a, dtype=torch.float32, device=args.device)

    theta_val = np.repeat(theta[:num_val], 10, axis=0)
    x_val = np.repeat(x[:num_val], 10, axis=0)
    val_condition = encode(
        x_val, present_mask(len(x_val), dim, rate, rng), mean, std, args.device
    )
    val_input = to_tensor(theta_val)
    theta_train, x_train = theta[num_val:], x[num_val:]
    train_input = to_tensor(theta_train)

    def train_condition():
        present = present_mask(len(x_train), dim, rate, rng)
        return encode(x_train, present, mean, std, args.device)

    torch.manual_seed(args.seed + 1)
    config = NSFConfig(z_score_condition="none", hidden_features=args.hidden_features)
    estimator = config.build(train_input, train_condition()).to(args.device)
    optimizer = torch.optim.Adam(estimator.parameters(), lr=5e-4)
    best, best_state, since, epoch = float("inf"), None, 0, 0
    start = time.perf_counter()
    while epoch < args.max_epochs and since < args.patience:
        estimator.train()
        condition = train_condition()
        for batch in torch.randperm(len(train_input), device=args.device).split(200):
            optimizer.zero_grad()
            loss = estimator.loss(train_input[batch], condition[batch]).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(estimator.parameters(), 5.0)
            optimizer.step()
        estimator.eval()
        with torch.no_grad():
            val = float(estimator.loss(val_input, val_condition).mean())
        if val < best:
            best, best_state, since = val, copy.deepcopy(estimator.state_dict()), 0
        else:
            since += 1
        epoch += 1
    train_time = time.perf_counter() - start
    estimator.load_state_dict(best_state)
    estimator.eval()
    print(f"seed {args.seed}: {epoch} epochs, best val {best:.3f}, {train_time:.0f}s")

    prior = torch.distributions.Independent(
        torch.distributions.Normal(
            torch.zeros(theta.shape[1], device=args.device),
            torch.ones(theta.shape[1], device=args.device),
        ),
        1,
    )
    posterior = DirectPosterior(estimator, prior, device=args.device)
    samples, sample_time = [], 0.0
    for x_o, present in zip(data["x"], data["present"], strict=True):
        tic = time.perf_counter()
        draws = posterior.sample(
            (args.draws,),
            x=encode(x_o[None], present[None], mean, std, args.device),
            show_progress_bars=False,
        )
        sample_time += time.perf_counter() - tic
        samples.append(draws.cpu().numpy())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.out,
        samples=np.stack(samples),
        epochs=epoch,
        best_val=best,
        train_time=train_time,
        sample_time=sample_time,
    )


if __name__ == "__main__":
    main()
