
# Can I use a custom prior with sbi?

Yes, if you pass a class that mimics the behaviour of a torch distributions, then sbi will wrap it as a torch distribution and can use it from there.

Essentially, the class needs two methods:
- `.sample(sample_shape)`, where sample_shape is a shape tuple, e.g., `(m,n)`/`(n,)`, and returns m batches of n samples, e.g., of shape `(m, n, 2)`/`(n, 2)` for a two dimenional prior.
- `.log_prob(value)` method that returns the "log probs" of parameters under the prior, e.g., for m batches of n parameters with shape `(m, n, ndims)`/`(n, ndims)` it should return a log probs array of shape `(m,n)`/`(n,)`.

For sbi >> 0.17.2 this could look like the following:

```python
class CustomPrior:
    def __init__(self):
        pass

    def log_prob(self, X):
        pass

    def sample(self):
        pass
```

If you are using `SNLE` and are on sbi verion << 0.17.2 you need to update, for the feature to be fully supported. 