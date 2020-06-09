# Getting started with *sbi*

### Making simulation-based inference easy

*sbi* provides a simple interface to run state-of-the-art algorithms for simulation-based inference.

You only need to provide two things:

1) a prior distribution that allows to sample parameter sets.
2) a simulator that takes parameter sets and produces simulation outputs.

With these two things, *sbi* can run inference with just two lines of code:

```
infer = SNPE(prior=prior, simulator=simulator)
posterior = infer(num_simulations=1000)
```

For some observation $x$, you can then sample from the posterior p(\theta|x), evaluate its log-probability, or plot it.

```
samples = posterior.sample(1000, x=observation)
log_probability = posterior.log_prob(samples, x=observation)
pairplot(samples)
```

### Requirements for the simulator, prior, and observation

For all algorithms, all you need to provide are a prior and a simulator. Let's talk about what requirements they need to satisfy:

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


**Observation**
An observation $x_o$ for which you want to infer the posterior $p(\theta|x_o)$, e.g.
```
observation = torch.zeros(5)
```
Allowed data types and shapes:
- either a `np.ndarray` or a `torch.Tensor`.
- shape must be either `(1,N)` or `(N)`.

### Running different algorithms

*sbi* implements three classes of algorithms that can be used to obtain the posterior distribution: SNPE, SNL, and SRE. You can try the different algorithms by simply swapping out the name of the inference class:
```
infer = SNPE(prior=prior, simulator=simulator)
infer = SNRE(prior=prior, simulator=simulator)
infer = SNLE(prior=prior, simulator=simulator)
```
You can then infer, sample, evaluate, and plot the posterior as described above.
