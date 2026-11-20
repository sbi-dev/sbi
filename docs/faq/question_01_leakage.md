# What should I do when my 'posterior samples are outside the prior support' in SNPE?

When working with **multi-round** NPE (i.e., SNPE), you might have experienced the
following warning:

```python
Only x% posterior samples are within the prior support. It may take a long time to
collect the remaining 10000 samples. Consider interrupting (Ctrl-C) and switching to
'sample_with_mcmc=True'.
```

The warning means that most of the posterior estimate lies outside the prior bounds.
This is called leakage. In multi-round NPE, leakage can grow with every round, and the
estimate then becomes poor inside the prior bounds as well. With a mixture density
network, training can eventually fail with NaN values. The reason for leakage is
described in more detail
[here](https://arxiv.org/abs/2210.04815),
[here](https://arxiv.org/abs/2002.03712), and
[here](https://arxiv.org/abs/1905.07488). The following fixes are possible:

- use truncated proposals for SNPE (TSNPE), as shown in the
[guide on sequential methods](https://sbi.readthedocs.io/en/latest/how_to_guide/02_multiround_inference.html#leakage-and-truncated-proposals-tsnpe).

- sample with MCMC: `samples = posterior((num_samples,), x=x_o, sample_with_mcmc=True)`.
This approach will make sampling slower, but samples will not "leak". It does not
improve a poor estimate, though.

- resort to single-round NPE and (if necessary) increase your simulation budget.

- use a different algorithm, e.g., Sequential NRE and Sequential NLE. Note, however,
that these algorithms can have different issues and potential pitfalls.

A mixture density network as density estimator does not avoid leakage. With a uniform
prior, multi-round NPE then uses a closed-form loss ("Using NPE_C with non-atomic
loss"), but its estimate can leak as well.
