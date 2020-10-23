
# Can I use the GPU for training the density estimator?

TLDR; Yes, by passing `device="cuda"`. But no speed-ups for default density estimators.

Yes. When creating the inference object in the flexible interface, you can pass the
`device` as an argument, e.g.,

```python
inference = SNPE(simulator, prior, device="cuda", density_estimator="maf")
```

The device is set to `"cpu"` by default, and it can be set to anything, as long as it
maps to an existing PyTorch CUDA device. `sbi` will take care of copying the `net` and
the training data to and from the `device`. 

## Performance

Whether or not you reduce your training time when training on a GPU depends on the
problem at hand. We provide a couple of default density estimators for `SNPE`, `SNLE`
and `SNRE`, e.g., a mixture density network (`density_estimator="mdn"`) or a Masked
Autoregressive Flow (`density_estimator="maf"`). For those default density estimators
we do **not** expect a speed up. This is because the underlying neural networks are
quite shallow and not tall, e.g., they do not have many parameters or matrix
operations that profit a lot from being executed on the GPU. 

A speed up through training on the GPU will most likely become visible when you are
using convolutional modules in your neural networks. E.g., when passing an embedding
net for image processing like in this example: [https://github.com/mackelab/sbi/blob/main/tutorials/05_embedding_net.ipynb](https://github.com/mackelab/sbi/blob/main/tutorials/05_embedding_net.ipynb). 
