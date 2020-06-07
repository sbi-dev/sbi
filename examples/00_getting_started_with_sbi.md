## Getting started with *sbi*

### Making simulation-based inference easy

*sbi* provides a simple interface to run state-of-the-art algorithms for simulation-based inference.

In the simplest scenario, you only need to provide two things:

1) a prior distribution that allows to sample parameter sets.
2) a simulator that takes parameter sets and produces simulation outputs.

With these two things, *sbi* can run inference with just two lines of code:

```
infer = SNPE(prior=prior, simulator=simulator)
posterior = infer(num_rounds=1, num_simulations_per_round=1000)
```

Once you have the posterior, you can sample from it, evaluate the posterior log-probability, or plot the posterior.

```
posterior_samples = posterior.sample(1000, x=my_observation)
posterior_log_probability = posterior.log_prob(posterior_samples)
vis_posterior(posterior_samples)
```

### Requirements for the simulator and prior

So, for all algorithms, all you need to provide are a prior and a simulator. Let's talk about what requirements they need to satisfy:

**Prior**
A distribution that allows to sample parameter sets, e.g. a 5-dimensional uniform prior between [-1,1]: 
```
prior = utils.BoxUniform(-torch.ones(5), torch.ones(5))
```
Any datatype for the prior is allowed as long as it allows to call `prior.sample()` and `prior.log_prob()`.

**Simulator**
A python callable that takes in a parameter set and outputs data with some (even if very small) stochasticity, for example
```
def simulator(parameter_set):
    return 2.0 * parameter_set + torch.randn((parameters.shape))
```
Allowed data types and shapes for input and output:
- the input parameter set and the output have to be either a `np.ndarray` or a `torch.Tensor`. 
- the input parameter set should have either shape `(1,N)` or `(N)`, and the output must have shape `(1,M)` or `(M)`.


### Running different algorithms

*sbi* implements three classes of algorithms that can be used to obtain the posterior distribution: SNPE, SNL, and SRE. You can try the different algorithms by simply swapping out the name of the inference class:
```
infer = SNPE(prior=prior, simulator=simulator)
infer = SRE(prior=prior, simulator=simulator)
infer = SNL(prior=prior, simulator=simulator)
```
You can then infer, sample, evaluate, and plot the posterior as described above.


### FAQ

My simulator can process batches of parameters, can *sbi* harness this? 
- if your simulator is able to handle batches of `K` parameters, i.e. can take an input of shape `(K,N)` and then gives an output `(K,M)`, you need to set the `simulation_batch_size=K`.


   