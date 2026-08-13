# Using the logit transformation
If you've ruled out simulator issues, you can try
training your density or ratio estimator in an unbounded space
using a logit transformation:

- **For NPE**: The transformation maps bounded parameters θ
to unbounded space before training, then applies the inverse (sigmoid)
after training to ensure posterior samples stay within prior bounds.

- **For NLE/NRE**: The transformation would need to map bounded
data x to unbounded space, which requires estimating data bounds
from simulations (more complex).

To enable this for NPE:

```python
from sbi.neural_nets import ZukoNSFConfig

density_estimator = ZukoNSFConfig(
    hidden_features=60,
    num_transforms=3,
    # Transforms parameters to unconstrained space.
    z_score_input="transform_to_unconstrained",
    # For NPE, this specifies the bounds of the parameters.
    x_dist=prior,
)
inference = NPE(prior, density_estimator=density_estimator)
```

This ensures that your density estimator operates in a
transformed space where it respects prior bounds,
improving the efficiency of rejection sampling.

Note: The `x_dist=prior` might seem confusing - internally,
sbi uses generic `x,y` notation where for NPE, `x` represents
parameters (θ) and `y` represents data.
This is why we pass the prior as `x_dist`.

Important:

- This transformation is currently only supported for zuko density estimators.
- For **NLE/NRE**, setting up this transformation is more
complex as it requires estimating bounds for the simulated data
rather than using prior bounds.
