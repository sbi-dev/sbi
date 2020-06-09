# Getting started with `sbi`


```python
import torch
import sbi.utils as utils
from sbi.inference.base import infer
```

## Running the inference procedure

`sbi` provides a simple interface to run state-of-the-art algorithms for simulation-based inference.

For inference, you need to provide two things:

1) a prior distribution that allows to sample parameter sets.  
2) a simulator that takes parameter sets and produces simulation outputs.

For example, we can have a 3-dimensional parameter space with a uniform prior between [-1,1] and a simulator that adds 1.0 and some Gaussian noise to the parameter set:


```python
num_dim = 3
prior = utils.BoxUniform(-2*torch.ones(num_dim), 2*torch.ones(num_dim))

def simulator(parameter_set):
    return 1.0 + parameter_set + torch.randn(parameter_set.shape) * 0.1
```

`sbi` can then run inference:


```python
posterior = infer('SNPE', simulator, prior, num_simulations=1000)
```


    HBox(children=(FloatProgress(value=0.0, description='Running 1000 simulations.', max=1000.0, style=ProgressSty…


    
    Neural network successfully converged after 167 epochs.


Let's say we have made some observation $x$:


```python
observation = torch.zeros(3)
```

 Given this observation, we can then sample from the posterior $p(\theta|x)$, evaluate its log-probability, or plot it.


```python
samples = posterior.sample((10000,), x=observation)
log_probability = posterior.log_prob(samples, x=observation)
_ = utils.pairplot(samples, limits=[[-2,2],[-2,2],[-2,2]], fig_size=(6,6))
```


![png](00_getting_started_files/00_getting_started_10_0.png)


## Requirements for the simulator, prior, and observation

For all algorithms, all you need to provide are a prior and a simulator. Let's talk about what requirements they need to satisfy.


### Prior
A distribution that allows to sample parameter sets. Any datatype for the prior is allowed as long as it allows to call `prior.sample()` and `prior.log_prob()`.

### Simulator
A python callable that takes in a parameter set and outputs data with some (even if very small) stochasticity.

Allowed data types and shapes for input and output:
- the input parameter set and the output have to be either a `np.ndarray` or a `torch.Tensor`. 
- the input parameter set should have either shape `(1,N)` or `(N)`, and the output must have shape `(1,M)` or `(M)`.

### Observation
An observation $x_o$ for which you want to infer the posterior $p(\theta|x_o)$.

Allowed data types and shapes:
- either a `np.ndarray` or a `torch.Tensor`.
- shape must be either `(1,N)` or `(N)`.

## Running different algorithms

*sbi* implements three classes of algorithms that can be used to obtain the posterior distribution: SNPE, SNL, and SRE. You can try the different algorithms by simply swapping out the `method`:


```python
posterior = infer('SNPE', simulator, prior, num_simulations=1000)
posterior = infer('SNLE', simulator, prior, num_simulations=1000)
posterior = infer('SNRE', simulator, prior, num_simulations=1000)
```

You can then infer, sample, evaluate, and plot the posterior as described above.
