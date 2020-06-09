# The advanced interface

In the previous tutorial, we have demonstrated how *sbi* can be used to run simulation-based inference with minimal code.

In addition to this simple interface, *sbi* also provides an **advanced interface** which gives the user more flexibility in designing the hyperparameters of the inference network.


### Features

The advanced interface allows you to customize the following:
- performing sequential posterior estimation by using `num_rounds>1`. This can decrease the number of simulations one has to run, but the inference procedure is no longer amortized.  
- specify your own density estimator, or change hyperparameters of existing ones (e.g. number of hidden units for 'NSF').
- use multiprocessing to run simulations on `num_workers` cores in parallel (or should we keep this in the easy interface?).  
- run simulations in batches, which can drastically speed up simulations.
- if available, choose between different methods to sample from the posterior (especially relevant for SNPE).
- use calibration kernels as proposed by Lueckmann et al. 2017
- many more small features, see docstrings...


### How to use it

To switch to the advanced mode, you need to call `set_mode('advanced')` before you import the inference algorithm:
```
sbi.utils.set_mode('advanced')
from sbi.inference.snpe.snpe import SNPE
```
In the advanced mode, you have to create an additional object called an `sbi_problem`:
```
my_sbi_problem = sbi_problem(simulator, prior)
```
You can then use this object to specify a custom density estimator:
```
my_density_estimator = get_nn_models(my_sbi_problem, model='nsf', num_hiddens=50, num_layers=10)
```
And you can run inference with additional features such as simulating in batches or sampling with MCMC.
```
infer = SNPE(sbi_problem=my_sbi_problem, density_estimator=my_density_estimator, simulation_batch_size=10, sample_with_mcmc=True)
posterior = infer(num_rounds=2, x_o=observation, num_simulations_per_round=1000)
```


### Switching back to the easy interface

```
sbi.utils.set_mode('easy')
from sbi.inference.snpe.snpe import SNPE
```