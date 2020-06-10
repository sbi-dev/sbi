# User input checks

## What we require from the user

- **simulator Callablle** that takes and returns torch tensor or np arrays
- **prior object** with `.sample()` and `.log_prob()` methods, `batch_shape==1` (but it doesn't have to have attributes `batch_shape` and `event_shape`.  
  - `.sample()` returns a single parameter set, `.sample((batch_size,))` returns `batch_size` parameter sets. **Note**: the sample size here will turn into the `batch_size` of the simulator, hence our use of the terminology here.
  - alternatively we should allow for `np.random.` objects, and `scipy.stats.` objects.
- **observed data `x_o` ** as torch `Tensor` or `np.ndarray`, with non-nil batch shape, i.e., `x_o.shape = (1, obs_dims>=1)` 
  - if `obs dims > 1`, we will have to take care of that at the level of the NN in some cases (some architectures, e.g. convolutional embedding networks, will want to have the original data in its original shape).

## What we require in sbi

- simulator that takes and returns torch `Tensor`s, batched, e.g., it always assume a batch dimension, taking `(1, param dim)` for a single input and returning `(1, obs dim)`.
- prior object with behavior as torch `Distribution`.
- observed data as torch tensor, with batch dimension.

## What we do with the user input to get there

### Prior

- if it is a pytorch distribution
  - assert `batch_shape` is 1, informative error if not
  - get `parameter_dim=prior.event_shape`
  - DONE
- else
  - assert .sample and .log_prob methods.
  - assert return type is numpy. NOTE: or `torch.Tensor`?
  - set numpy flag True if it is the case
  - wrap with new class that inherits from `Distribution`
  - wraps `log_prob` and sample to get and return torch `Tensor`s. NOTE: let us be attentive to the impact of `as_tensor` on the trackability of the output.
  - DONE

### Observed data

- get data event shape
  - get single simulated data point by sampling once from the raw prior, then simulating. 
- make sure observation has `batch_dim >= 1`
  - assert that the squeezed simulated data number of dimensions is the same as the observed number of dimensions -1. **NOTE** : this rules out the multiple observations scenario, where we would like to to throw a `NotImplementedError` (see next point).
  - assert that the first observed data dimension (batch dim) is 1, say that multiple observed data points are not supported yet
- warn if the observed data is multi dimensional (explain that it will be interpreted as such).
- reshape to torch tensor with shape (1, data_dim), e.g., (1, 10), but also (1, 4, 4) (?). 
- DONE

### Simulator

- if numpy flag is `True`, wrap it to take and return torch tensors, float32. NOTE: downcasting of floats should elicit a warning, as the user might be using double precision for a good reason.
- try to simulate a batch sampled from the prior. NOTE: try separately the sample step and the simulate step, as you want to throw a differentiated exception.
- if needed, wrap the simulator using map (have a look at [`numpy.vectorize`](https://docs.scipy.org/doc/numpy/reference/generated/numpy.vectorize.html), *like the python map function, except it uses the broadcasting rules of numpy.* ), asserting batch dim. issue a warning. NOTE: here we're aiming at doing multiprocessing. 
- DONE

### Notes

- get sim input type from prior output
- use tensor float 32 in sbi
- cast simulators and priors accordingly 
- raise exception if log prob not scalar or .batch_shape > 1
- if data multi d and first d is not 1, error 
- check data by comparing to simulated data
- assert batched observed data
- replace prior with new class child, call super, or pass user prior as attribute 
- wrap the simulator to take and give torch tensor 
