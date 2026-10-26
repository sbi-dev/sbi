# Can I train my density estimator in an unconstrained space?

Yes. If posterior samples leak outside the prior bounds (see
[posterior samples outside the prior support](question_01_leakage.md)) and you have
ruled out simulator issues, you can train a supported density estimator in an
unbounded space using a logit transformation:

- **For NPE**: The transformation maps bounded parameters θ
to unbounded space before training, then applies the inverse (sigmoid)
after training to ensure posterior samples stay within prior bounds.

- **For NLE**: The modeled variable is the data x. Supply a distribution with
the appropriate data support as `x_dist`, rather than the parameter prior.

To enable this for NPE:

```python
import torch

from sbi.inference import NPE
from sbi.neural_nets import ZukoNSFConfig
from sbi.utils import BoxUniform

prior = BoxUniform(low=torch.zeros(2), high=torch.ones(2))

density_estimator = ZukoNSFConfig(
    hidden_features=60,
    num_transforms=3,
    z_score_input="transform_to_unconstrained",
    x_dist=prior,
)
inference = NPE(prior, density_estimator=density_estimator)
```

This ensures that your density estimator operates in a
transformed space where it respects prior bounds,
improving the efficiency of rejection sampling.

Despite its name, `x_dist` describes the support of the modeled variable.
For NPE, that variable is θ, so pass the prior.

Important:

- Zuko density configs and `MDNConfig` support this transformation.
- The nflows density configs, marginal configs, classifier configs, and
vector-field configs do not offer `"transform_to_unconstrained"` as an input
z-scoring mode.
