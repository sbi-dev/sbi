# Can the algorithms deal with invalid data, e.g., NaN or inf?

Yes. In single-round NPE, whenever a simulation returns at least one `NaN` or `inf`,
it is completely excluded from the training data by default. In other words, the
simulation is simply discarded.

In cases where a very large fraction of simulations return `NaN` or `inf`,
discarding many simulations can be wasteful. There are two options to deal with
this: Either you use the `RestrictionEstimator` to learn regions in parameter
space that do not produce `NaN` or `inf`, see
[here](https://sbi.readthedocs.io/en/latest/advanced_tutorials/06_restriction_estimator.html).
Alternatively, you can manually substitute the 'invalid' values with a
reasonable replacement. For example, at the end of your simulation code, you
search for invalid entries and replace them with a floating point number.
Importantly, in order for neural network training work well, the floating point
number should still be in a reasonable range, i.e., maybe a few standard
deviations outside of 'good' values.

NLE, NRE, and **multi-round** NPE are different: discarding invalid simulations
biases their training objective. These methods therefore raise an error when the
simulations contain `NaN` or `inf`:

```python
ValueError: Found 5 NaN simulations and 0 Inf simulations. Multiround NPE_C does
not allow invalid simulations. Replace the invalid values with an unreasonably
low or high value.
```

You can override this with `append_simulations(..., exclude_invalid_x=True)`, but
this gives systematically wrong results and is only recommended for expert users.
The safe options are the two described above: restrict the prior with the
`RestrictionEstimator`, or replace the invalid values in your simulation code.
