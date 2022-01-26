
# Can I use a custom prior with sbi?

Yes, if you pass a class that mimics the behaviour of a torch distribution, then sbi will wrap it as a torch distribution and can use it from there.
The `prepare_for_sbi` method takes care of the wrapping for the user. It is compatible with both numpy and scipy.

Essentially, the class needs two methods:
- `.sample(sample_shape)`, where sample_shape is a shape tuple, e.g., `(n,)`, and returns a batch of n samples, e.g., of shape (n, 2)` for a two dimenional prior.
- `.log_prob(value)` method that returns the "log probs" of parameters under the prior, e.g., for a batches of n parameters with shape `(n, ndims)` it should return a log probs array of shape `(n,)`.

For sbi > 0.17.2 this could look like the following:

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

If you are running sbi < 0.17.2 and use `SNLE` the code above will produce a `NotImplementedError` (see #581). In this case you need to update to a newer version of `sbi` or use `SNPE` instead. 
