# What should I do when 'only x% proposal samples are accepted' in SNPE?

When working with **multi-round** NPE (i.e., SNPE), you might have experienced the
following warning:

```python
Only x% proposal samples are accepted. It may take a long time to collect the
remaining 10000 samples. You can prevent very long runtimes by setting
`max_sampling_time` to limit runtime, or disabling rejection sampling (e.g. via
`reject_outside_prior=False` in `posterior.sample()` when available). Alternatively,
consider interrupting (Ctrl-C) and switching to
`build_posterior(..., sample_with='mcmc')`.
```

Setting `reject_outside_prior=False` skips the rejection step. Sampling gets faster,
but the samples are no longer restricted to the prior bounds and can lie outside them.

The warning means that most of the posterior estimate lies outside the prior bounds.
This is called leakage. In multi-round NPE, leakage can grow with every round, and the
estimate then becomes poor inside the prior bounds as well. With a mixture density
network, training can eventually fail with NaN values. `log_prob()` shows the same
leakage: it estimates its normalizing constant from the fraction of samples within the
prior bounds, and warns when this fraction is small. The reason for leakage is
described in more detail
[here](https://arxiv.org/abs/2210.04815),
[here](https://arxiv.org/abs/2002.03712), and
[here](https://arxiv.org/abs/1905.07488). The following fixes are possible:

- use truncated proposals for SNPE (TSNPE), as shown in the
[guide on sequential methods](https://sbi.readthedocs.io/en/latest/how_to_guide/02_multiround_inference.html#leakage-and-truncated-proposals-tsnpe).

- sample with MCMC: `posterior = inference.build_posterior(sample_with="mcmc")`, then
`samples = posterior.sample((num_samples,), x=x_o)`.
MCMC samples only within the prior bounds, so samples will not "leak". At very low
acceptance rates, it can be faster than rejection sampling. It does not improve a poor
estimate, though.

- resort to single-round NPE and (if necessary) increase your simulation budget.

- use a different algorithm, e.g., Sequential NRE and Sequential NLE. Note, however,
that these algorithms can have different issues and potential pitfalls.

A mixture density network as density estimator does not avoid leakage. With a uniform
prior, multi-round NPE then uses a closed-form loss ("Using NPE_C with non-atomic
loss"), but its estimate can leak as well.
