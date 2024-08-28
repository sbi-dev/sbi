# What should I do when my 'posterior samples are outside the prior support' in SNPE?

When working with **multi-round** NPE (i.e., SNPE), you might have experienced the
following warning:

```python
Only x% posterior samples are within the prior support. It may take a long time to
collect the remaining 10000 samples. Consider interrupting (Ctrl-C) and switching to
'sample_with_mcmc=True'.
```

The reason for this issue is described in more detail
[here](https://arxiv.org/abs/2210.04815),
[here](https://arxiv.org/abs/2002.03712), and
[here](https://arxiv.org/abs/1905.07488). The following fixes are possible:

- use truncated proposals for SNPE (TSNPE)
```python
from sbi.inference import NPE
from sbi.utils import RestrictedPrior, get_density_thresholder

inference = NPE(prior)
proposal = prior
for _ in range(num_rounds):
    theta = proposal.sample((num_sims,))
    x = simulator(theta)
    _ = inference.append_simulations(theta, x).train(force_first_round_loss=True)
    posterior = inference.build_posterior().set_default_x(x_o)

    accept_reject_fn = get_density_thresholder(posterior, quantile=1e-4)
    proposal = RestrictedPrior(prior, accept_reject_fn, sample_with="rejection")
```

- sample with MCMC: `samples = posterior((num_samples,), x=x_o, sample_with_mcmc=True)`.
This approach will make sampling slower, but samples will not "leak".

- resort to single-round NPE and (if necessary) increase your simulation budget.

- if your prior is either Gaussian (torch.distributions.MultivariateNormal) or Uniform
(sbi.utils.BoxUniform), you can avoid leakage by using a mixture density network as
density estimator. I.e., set `density_estimator='mdn'` when creating the `SNPE`
inference object. When running inference, there should be a print statement "Using
SNPE-C with non-atomic loss".

- use a different algorithm, e.g., Sequential NRE and Sequential NLE. Note, however,
that these algorithms can have different issues and potential pitfalls.
