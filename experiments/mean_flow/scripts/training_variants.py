"""Two moons NPE under different training regimes (results.md, "Training regime").

A: FMPE defaults. B: mean flow, sbi trainer, lr 1e-4, 600 epochs.
C, D: mean flow, paper recipe, 500 epochs (D with 10,000 simulations).

    python experiments/mean_flow/scripts/training_variants.py C 2000
"""

import math
import sys
import time

import torch

sys.path.insert(0, ".")
torch.set_num_threads(3)
from sbi.inference import FMPE, NPE  # noqa: E402
from sbi.neural_nets import MeanFlowConfig  # noqa: E402
from sbi.utils.metrics import c2st  # noqa: E402
from tests.mini_sbibm import get_task  # noqa: E402

variant, num_sims = sys.argv[1], int(sys.argv[2])
task = get_task("two_moons")
torch.manual_seed(0)
theta, x = task.get_data(num_sims)
prior = task.get_prior()
t0 = time.time()


def report(sample_fn, label):
    cs = []
    for i in range(1, 6):
        ref = task.get_reference_posterior_samples(i)[:1000]
        cs.append(float(c2st(ref, sample_fn(task.get_observation(i)))))
    per_observation = [round(c, 3) for c in cs]
    print(
        f"{variant} n={num_sims} {label}: c2st mean {sum(cs) / 5:.3f} "
        f"per-obs {per_observation}",
        flush=True,
    )


if variant == "A":
    inf = FMPE(prior, show_progress_bars=False)
    inf.append_simulations(theta, x).train()
    post = inf.build_posterior()
    print(
        f"A train {time.time() - t0:.0f}s epochs {inf.summary['epochs_trained'][-1]}",
        flush=True,
    )
    report(
        lambda xo: post.sample((1000,), x=xo, show_progress_bars=False), "fmpe default"
    )
    sys.exit()

if variant == "B":
    inf = NPE(prior, density_estimator=MeanFlowConfig(), show_progress_bars=False)
    est = inf.append_simulations(theta, x).train(
        learning_rate=1e-4, max_num_epochs=600, stop_after_epochs=10**6
    )
    print(f"B train {time.time() - t0:.0f}s", flush=True)
else:  # C/D: paper-style loop on the estimator alone
    est = MeanFlowConfig().build(theta, x)
    perm = torch.randperm(num_sims)
    n_val = num_sims // 10
    tr, va = perm[n_val:], perm[:n_val]
    epochs, bs = 500, 256
    steps_per_epoch = math.ceil(len(tr) / bs)
    total = epochs * steps_per_epoch
    warm = int(0.05 * total)
    opt = torch.optim.AdamW(est.parameters(), lr=5e-4, weight_decay=5e-3)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt,
        lambda k: (0.1 + 0.9 * k / warm)
        if k < warm
        else 0.5 * (1 + math.cos(math.pi * (k - warm) / (total - warm))),
    )
    for ep in range(epochs):
        est.train()
        for idx in tr[torch.randperm(len(tr))].split(bs):
            loss = est.loss(theta[idx], x[idx]).mean()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(est.parameters(), 1.5)
            opt.step()
            sched.step()
        if ep % 100 == 99:
            est.eval()
            with torch.no_grad():
                vl = est.loss(theta[va], x[va]).mean().item()
            print(
                f"{variant} ep {ep + 1} val {vl:.3f} ({time.time() - t0:.0f}s)",
                flush=True,
            )
est.eval()


def euler(xo, n=1000, steps=200):
    emb = est.embedding_net(xo).expand(n, -1)
    z = torch.randn(n, 2)
    ts = torch.linspace(1, 0, steps + 1)
    for a, b in zip(ts[:-1], ts[1:], strict=False):
        z = z + (b - a) * est.net(z, emb, a.expand(n), a.expand(n), "velocity")
    return z * est._std_input + est._mean_input


with torch.no_grad():
    for S in (1, 2, 4):
        report(
            lambda xo, S=S: est.sample((1000,), xo, num_steps=S)[:, 0],
            f"mean flow direct S={S}",
        )
    report(euler, "velocity head Euler200")
torch.save(est, f"mean_flow_variant_{variant}_{num_sims}.pt")
