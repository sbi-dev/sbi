"""sbi NPE with the missingness mask appended to the observation.

The observation is standardized per coordinate (statistics of the complete training
simulations, which under missingness at random equal those of the present entries in
expectation), missing entries are set to 0 and the 0/1 mask is appended, so the NSF is
conditioned on 64 numbers. Each epoch draws a new mask for every training simulation,
as the mean flow does per batch. The held-out 10% of simulations get 10 fixed masks
each. Adam with lr 5e-4, batch 200, gradient clipping 5.0 and early stopping after 50
epochs without improvement (capped at 2,000 epochs), best epoch kept: sbi's defaults
except for a longer patience.

    python experiments/mean_flow/scripts/missing/sbi_npe.py --seed 0 \
        --reference ~/meanflow-runs/missing/reference.npz \
        --out ~/meanflow-runs/missing/sbi_npe_s0.npz
"""

import argparse
import copy
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path.cwd()))
sys.path.insert(0, str(Path(__file__).parent))

from wave_task import NUM_PARAMS, T, present_mask, simulate  # noqa: E402

from sbi.inference.posteriors import DirectPosterior  # noqa: E402
from sbi.neural_nets import NSFConfig  # noqa: E402


def encode(x: np.ndarray, present: np.ndarray, mean, std) -> torch.Tensor:
    z = np.where(present, (x - mean) / std, 0.0)
    return torch.as_tensor(np.concatenate([z, present], -1), dtype=torch.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-simulations", type=int, default=8192)
    parser.add_argument("--draws", type=int, default=20_000)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--max-epochs", type=int, default=2000)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)

    rng = np.random.default_rng(args.seed)
    theta = rng.standard_normal((args.num_simulations, NUM_PARAMS))
    x = simulate(theta, rng)
    mean, std = x.mean(0), x.std(0)
    num_val = args.num_simulations // 10
    theta_val = np.repeat(theta[:num_val], 10, axis=0)
    x_val = np.repeat(x[:num_val], 10, axis=0)
    val_condition = encode(x_val, present_mask(len(x_val), rng), mean, std)
    val_input = torch.as_tensor(theta_val, dtype=torch.float32)
    theta_train, x_train = theta[num_val:], x[num_val:]
    train_input = torch.as_tensor(theta_train, dtype=torch.float32)

    torch.manual_seed(args.seed + 1)
    estimator = NSFConfig(z_score_condition="none").build(
        train_input, encode(x_train, present_mask(len(x_train), rng), mean, std)
    )
    optimizer = torch.optim.Adam(estimator.parameters(), lr=5e-4)
    best, best_state, since, epoch = float("inf"), None, 0, 0
    start = time.perf_counter()
    while epoch < args.max_epochs and since < args.patience:
        estimator.train()
        condition = encode(x_train, present_mask(len(x_train), rng), mean, std)
        for batch in torch.randperm(len(train_input)).split(200):
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

    reference = np.load(args.reference)
    prior = torch.distributions.Independent(
        torch.distributions.Normal(torch.zeros(NUM_PARAMS), torch.ones(NUM_PARAMS)), 1
    )
    posterior = DirectPosterior(estimator, prior)
    samples, sample_time = [], 0.0
    for x_o, present in zip(reference["x"], reference["present"], strict=True):
        assert x_o.shape == (T,)
        tic = time.perf_counter()
        draws = posterior.sample(
            (args.draws,),
            x=encode(x_o[None], present[None], mean, std),
            show_progress_bars=False,
        )
        sample_time += time.perf_counter() - tic
        samples.append(draws.numpy())
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
