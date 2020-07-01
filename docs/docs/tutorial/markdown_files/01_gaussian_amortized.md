# Amortized posterior inference on Gaussian example

In this tutorial, we will demonstrate how `sbi` can infer an amortized posterior for a simple toy model with a uniform prior and Gaussian likelihood.


```python
import torch
import numpy as np

import sbi.utils as utils
from sbi.inference.base import infer
```

## Defining prior, simulator, and running inference
Say we have 3-dimensional parameter space, and the prior is uniformly distributed between `-2` and `2` in each dimension, i.e. $x\in [-2,2], y\in [-2,2], z \in [-2,2]$.


```python
num_dim = 3
prior = utils.BoxUniform(low=-2*torch.ones(num_dim), high=2*torch.ones(num_dim))
```

Our simulator takes the input parameters, adds `1.0` in each dimension, and then adds some Gaussian noise:


```python
def linear_gaussian(theta):
    return theta + 1.0 + torch.randn_like(theta) * 0.1
```

We can then run inference:


```python
posterior = infer(linear_gaussian, prior, 'SNPE', num_simulations=1000)
```

    Running 1000 simulations.: 100%|██████████| 1000/1000 [00:00<00:00, 44788.93it/s]


    Neural network successfully converged after 70 epochs.


## Amortized inference
Note that we have not yet provided an observation to the inference procedure. In fact, we can evaluate the posterior for different observations without having to re-run inference. This is called amortization. An amortized posterior is one that is not focused on any particular observation. Naturally, if the diversity of observations is large, any of the inference methods will need to run a sufficient number of simulations for the resulting posterior to perform well across these diverse observations.

Let's say we have two observations `x_o_1 = [0,0,0]` and `x_o_2 = [2,2,2]`:


```python
x_o_1 = torch.zeros(3,)
x_o_2 = 2.0*torch.ones(3,)
```

We can draw samples from the posterior given `x_o_1` and then plot them:


```python
posterior_samples_1 = posterior.sample((10000,), x=x_o_1)

# plot posterior samples
_ = utils.pairplot(posterior_samples_1, limits=[[-2,2],[-2,2],[-2,2]], fig_size=(5,5))
```


![png](01_gaussian_amortized_files/01_gaussian_amortized_13_0.png)


As it can be seen, the posterior samples are centered around `[-1,-1,-1]` in each dimension. 
This makes sense because the simulator always adds `1.0` in each dimension and we have observed `x_o_1 = [0,0,0]`.

Since the learned posterior is amortized, we can also draw samples from the posterior given the second observation without having to re-run inference:


```python
posterior_samples_2 = posterior.sample((10000,), x=x_o_2)

# plot posterior samples
_ = utils.pairplot(posterior_samples_2, limits=[[-2,2],[-2,2],[-2,2]], fig_size=(5,5))
```


![png](01_gaussian_amortized_files/01_gaussian_amortized_16_0.png)


So, if we have observed `x_o_2 = [2,2,2]`, the posterior is centered around `[1,1,1]` -- again, this makes sense because the simulator adds `1.0` in each dimension.
