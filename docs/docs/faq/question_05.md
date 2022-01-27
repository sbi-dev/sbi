
# How should I save and load objects in `sbi`?

`NeuralPosterior` objects are picklable.
```python
import pickle

# ... run inference
posterior = inference.build_posterior()

with open("/path/to/my_posterior.pkl", "wb") as handle:
    pickle.dump(posterior, handle)
```

Note: posterior objects that were saved under `sbi v0.17.2` or older can not be loaded under `sbi v0.18.0` or newer.

Note: if you try to load a posterior that was saved under `sbi v0.14.x` or earlier under `sbi v0.15.x` until `sbi v0.17.x`, you have to add:
```python
import sys
from sbi.utils import user_input_checks_utils

sys.modules["sbi.user_input.user_input_checks_utils"] = user_input_checks_utils
```
to your script before loading the posterior.


As of `sbi v0.18.0`, `NeuralInference` objects are also picklable.
```python
import pickle

# ... run inference
posterior = inference.build_posterior()

with open("/path/to/my_inference.pkl", "wb") as handle:
    pickle.dump(inference, handle)
```
However, saving and loading the `inference` object will slightly modify the object (in order to make it serializable). These modifications lead to the following two changes in behaviour:
1) Retraining from scratch is not supported, i.e. `.train(..., retrain_from_scratch=True)` does not work.
2) When the loaded object calls the `.train()` method, it generates a new tensorboard summary writer (instead of appending to the current one).