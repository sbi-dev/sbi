
# How should I save and load objects in `sbi`?

`NeuralPosterior` objects are picklable.
```python
import pickle

# ... run inference
posterior = inference.build_posterior()

with open("/path/to/my_posterior.pkl", "wb") as handle:
    pickle.dump(posterior, handle)
```

`NeuralInference` objects are not picklable. There are two workarounds:

- Pickle with [`dill`](https://pypi.org/project/dill/) (has to be installed with `pip install dill` first)
```python
import dill

inference = SNPE(prior)
# ... run inference

with open("path/to/my_inference.pkl", "wb") as handle:
    dill.dump(inference)
```

- Delete un-picklable attributes and serialize with pickle. Using this option, you will not be able to use the `retrain_from_scratch` feature and you can only use the default `SummaryWriter`.
```python
import pickle

inference = SNPE(prior)
# ... run inference

posterior = inference.build_posterior()
inference._summary_writer = None
inference._build_neural_net = None
with open("/path/to/my_inference.pkl", "wb") as handle:
    pickle.dump(inference, handle)
```
Then, to load:
```python
with open("/path/to/my_inference.pkl", "rb") as handle:
    inference_from_disk = pickle.load(handle)
inference_from_disk._summary_writer = inference_from_disk._default_summary_writer()
```
