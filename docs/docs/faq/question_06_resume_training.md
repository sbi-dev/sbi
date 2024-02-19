
# Can I stop neural network training and resume it later?

Many clusters have a time limit, and `sbi` might exceed this limit. You can
circumvent this problem by using the [flexible
interface](https://sbi-dev.github.io/sbi/tutorial/02_flexible_interface/). After
simulations are finished, `sbi` trains a neural network. If this process takes
too long, you can stop training and resume it later. The syntax is:

```python
inference = SNPE(prior=prior)
inference = inference.append_simulations(theta, x)
inference.train(max_num_epochs=300)  # Pick `max_num_epochs` such that it does not exceed the runtime.

with open("path/to/my/inference.pkl", "wb") as handle:
    pickle.dump(inference, handle)

# To resume training:
with open("path/to/my/inference.pkl", "rb") as handle:
    inference_from_disk = pickle.load(handle)
inference_from_disk.train(resume_training=True, max_num_epochs=600)  # Run epochs 301 until 600 (or stop early).
posterior = inference_from_disk.build_posterior()
```
