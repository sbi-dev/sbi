
# Can I use a custom prior with sbi?

As `sbi` works with torch distributions only, we recommend using those whenever
possible. For example, when you are used to using `scipy.stats` distributions as
priors, then we recommend using the corresponding `torch.distributions` instead.
Most `scipy` distributions are implemented in `PyTorch` as well.

In case you want to use a custom prior that is not in the set of common
distributions that's possible as well: You need to write a prior class that
mimicks the behaviour of a
[`torch.distributions.Distribution`](https://pytorch.org/docs/stable/_modules/torch/distributions/distribution.html#Distribution)
class. `sbi` will wrap this class to make it a fully functional torch
`Distribution`.

Essentially, the class needs two methods:

- `.sample(sample_shape)`, where sample_shape is a shape tuple, e.g., `(n,)`,
  and returns a batch of n samples, e.g., of shape (n, 2)` for a two dimenional
  prior.
- `.log_prob(value)` method that returns the "log probs" of parameters under the
  prior, e.g., for a batches of n parameters with shape `(n, ndims)` it should
  return a log probs array of shape `(n,)`.

For `sbi` > 0.17.2 this could look like the following:

```python
class CustomUniformPrior:
    """User defined numpy uniform prior.

    Custom prior with user-defined valid .sample and .log_prob methods.
    """

    def __init__(self, lower: Tensor, upper: Tensor, return_numpy: bool = False):
        self.lower = lower
        self.upper = upper
        self.dist = BoxUniform(lower, upper)
        self.return_numpy = return_numpy

    def sample(self, sample_shape=torch.Size([])):
        samples = self.dist.sample(sample_shape)
        return samples.numpy() if self.return_numpy else samples

    def log_prob(self, values):
        if self.return_numpy:
            values = torch.as_tensor(values)
        log_probs = self.dist.log_prob(values)
        return log_probs.numpy() if self.return_numpy else log_probs
```

Once you have such a class, you can wrap it into a `Distribution` using the
`process_prior` function `sbi` provides:

```python
from sbi.utils import process_prior

custom_prior = CustomUniformPrior(torch.zeros(2), torch.ones(2))
prior, *_ = process_prior(custom_prior)  # Keeping only the first return.
# use this wrapped prior in sbi...
```

In `sbi` it is sometimes required to check the support of the prior, e.g., when
the prior support is bounded and one wants to reject samples from the posterior
density estimator that lie outside the prior support. In torch `Distributions`
this is handled automatically. However, when using a custom prior, it is not.
Thus, if your prior has bounded support (like the one above), it makes sense to
pass the bounds to the wrapper function such that `sbi` can pass them to torch
`Distributions`:

```python
from sbi.utils import process_prior

custom_prior = CustomUniformPrior(torch.zeros(2), torch.ones(2))
prior = process_prior(custom_prior,
                      custom_prior_wrapper_kwargs=dict(lower_bound=torch.zeros(2),
                                                       upper_bound=torch.ones(2)))
# use this wrapped prior in sbi...
```

Note that in `custom_prior_wrapper_kwargs` you can pass additinal arguments for
the wrapper, e.g., `validate_args` or `arg_constraints` see the `Distribution`
documentation for more details.

If you are using `sbi` < 0.17.2 and use `NLE` the code above will produce a
`NotImplementedError` (see [#581](https://github.com/mackelab/sbi/issues/581)).
In this case, you need to update to a newer version of `sbi` or use `NPE`
instead.
