# The inference gets stuck?

If you've encountered the following warning:

```
WARNING:root:Only 0.002% proposal samples were accepted. It
                        may take a long time to collect the remaining 99980
                        samples. Consider interrupting (Ctrl-C) and switching to a
                        different sampling method with
                        `build_posterior(..., sample_with='mcmc')`. or
                        `build_posterior(..., sample_with='vi')`.
```

this indicates that a significant portion of the samples proposed by the density estimator fall outside the prior bounds. Several factors might be causing this issue:

1) Simulator Issues: Ensure that your simulator is functioning as expected and producing realistic outputs.
2) Insufficient Training Data: If the density estimator has been trained on too few simulations, it may lead to invalid estimations.
3) Problematic True Data: Check if there are inconsistencies or unexpected values in the observed data.


### Possible solutions

If you've ruled out these issues, you can try training your density estimator in an unbounded space using a logit transformation. This transformation maps your data to logit space before training and then applies the inverse logit (sigmoid function) to ensure that the trained density estimator remains within the prior bounds.

Instead of standardizing parameters using z-scoring, you can use the logit transformation. However, this requires providing a density estimation. The specific approach depends on the method you're using:

- For NPE (Neural Posterior Estimation): You can simply use the prior as the density estimation.
- For NLE/NRE (Neural Likelihood Estimation / Neural Ratio Estimation): A rough density approximation over data boundaries is needed, making the process more complex.


### How to apply the logit transformation

To enable logit transformation when defining your density estimator, use:

```
density_estimator_build_fun = posterior_nn(
    model="zuko_nsf", hidden_features=60, num_transforms=3, z_score_theta="logit", x_dist=prior
)
```
This ensures that your density estimator operates in a transformed space where it respects prior bounds, improving the efficiency of rejection sampling.

Note: The logit transformation is currently only supported for `zuko` density estimators.
