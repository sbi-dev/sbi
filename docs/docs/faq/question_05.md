
# How should I save and load objects in `sbi`?

`NeuralPosterior` objects are picklable.
```python
import pickle

# ... run inference
posterior = inference.build_posterior()

with open("my_path.pkl", "wb"):
    pickle.dump(posterior)
```

`NeuralInference` objects are not picklable. There are two workarounds:

- Pickle with `dill` (has to be installed with `pip install dill` first)
```python
import dill as pickle

inference = SNPE(prior)
# ... run inference

with open("my_path.pkl", "wb"):
    pickle.dump(inference)
```

- Delete un-picklable attributes and serialize with pickle. Using this option, you will not be able to use the `retrain_from_scratch` feature and you can only use the default `SummaryWriter`.
```python
import pickle

inference = SNPE(prior)
# ... run inference

posterior = inference.build_posterior()
inference._summary_writer = None
inference._build_neural_net = None
with open("my_path.pkl", "wb") as handle:
    pickle.dump(inference, handle)
```
Then, to load:
```python
with open("my_path.pkl", "rb") as handle:
    inference = pickle.load(handle)
inference._summary_writer = inference._default_summary_writer()
```