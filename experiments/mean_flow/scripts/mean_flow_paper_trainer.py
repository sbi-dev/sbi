"""Training recipe of Arruda et al. (2026), App. C, shared by the mean-flow scripts."""

import copy
import math

import torch


def train_paper_recipe(
    estimator,
    input,
    condition,
    epochs,
    batch_size=256,
    validation_fraction=0.1,
    states=None,
):
    """AdamW (weight decay 5e-3), learning rate warmed up linearly over the first 5%
    of steps to 5e-4, then cosine decay to zero, gradient clipping at 1.5.

    Unlike the paper, which trains on 50k to 100k simulations, the best epoch on a
    held-out split is kept: at 2,000 simulations the full schedule overfits (on
    gaussian_linear the held-out loss rises from 7.4 to 36.6 by epoch 400).
    The held-out loss uses the same noise every epoch, so epochs are comparable.

    If `states` is a dict, it receives the `"best"` and the `"last"` state dicts.

    Returns:
        The estimator with the best weights, in eval mode, and the best epoch.
    """
    permutation = torch.randperm(len(input))
    num_validation = int(validation_fraction * len(input))
    validation, train = permutation[:num_validation], permutation[num_validation:]

    num_batches = math.ceil(len(train) / batch_size)
    total = epochs * num_batches
    warmup = max(1, int(0.05 * total))
    optimizer = torch.optim.AdamW(estimator.parameters(), lr=5e-4, weight_decay=5e-3)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda k: (0.1 + 0.9 * k / warmup)
        if k < warmup
        else 0.5 * (1 + math.cos(math.pi * (k - warmup) / (total - warmup))),
    )

    def validation_loss():
        estimator.eval()
        with torch.no_grad(), torch.random.fork_rng():
            torch.manual_seed(12345)
            value = estimator.loss(input[validation], condition[validation]).mean()
        estimator.train()
        return float(value)

    best_loss, best_state, best_epoch = float("inf"), None, 0
    estimator.train()
    for epoch in range(1, epochs + 1):
        for idx in train[torch.randperm(len(train))].split(batch_size):
            loss = estimator.loss(input[idx], condition[idx]).mean()
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(estimator.parameters(), 1.5)
            optimizer.step()
            scheduler.step()
        current = validation_loss()
        if current < best_loss:
            best_loss, best_epoch = current, epoch
            best_state = copy.deepcopy(estimator.state_dict())
    if states is not None:
        states["best"] = best_state
        states["last"] = copy.deepcopy(estimator.state_dict())
    estimator.load_state_dict(best_state)
    estimator.eval()
    return estimator, best_epoch
