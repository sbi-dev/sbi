# When using multiple workers, I get a pickling error. Can I still use multiprocessing?

Yes, but you will have to make a few adjustments to your code. 

Some background: when using `num_workers > 1`, you might experience an error that a 
certain object from your simulator could not be pickled (an example can be found
[here](https://github.com/mackelab/sbi/issues/317)).

This can be fixed by forcing `sbi` to pickle with `dill` instead of the default 
`cloudpickle`. To do so, adjust your code as 
follows:

- Install `dill`:
```
pip install dill
```
- At the very beginning of your python script, set the pickler to `dill`:
```python
from joblib.externals.loky import set_loky_pickler
set_loky_pickler("dill")
```
- Move all imports required by your simulator into the simulator:
```python
# Imports specified outside of the simulator will break dill:
import torch
def my_simulator(parameters):
    return torch.ones(1,10)

# Therefore, move the imports into the simulator:
def my_simulator(parameters):
    import torch
    return torch.ones(1,10)
```

### Alternative: parallelize yourself

You can also write your own code to parallelize simulations with whatever 
multiprocessing framework you prefer. You can then simulate your data outside of `sbi` 
and pass the simulated data with the `provide_presimulated()` option described 
[here](https://www.mackelab.org/sbi/tutorial/06_provide_presimulated/).


### Some more background

`sbi` uses `joblib` to parallelize simulations, which in turn uses `pickle` or 
`cloudpickle` to serialize the simulator. Almost all simulators will be picklable with 
`cloudpickle`, but we have experienced issues e.g. with `neuron` simulators, see
[here](https://github.com/mackelab/sbi/issues/317).