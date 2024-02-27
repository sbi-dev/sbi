
# Can I use the GPU for training the density estimator?

**TLDR**; Yes, by passing `device="cuda"` and by passing a prior that lives on the
device name you passed. But we expect no speed-ups for default density estimators.

## Setup

Yes, we support GPU training. When creating the inference object in the flexible
interface, you can pass the `device` as an argument, e.g.,

```python
inference = SNPE(prior, device="cuda", density_estimator="maf")
```

The device is set to `"cpu"` by default. But it can be set to anything, as long
as it maps to an existing PyTorch GPU device, e.g., `device="cuda"` or
`device="cuda:2"`. `sbi` will take care of copying the `net` and the training
data to and from the `device`.
We also support MPS as a GPU device for GPU-accelarated training on an Apple
Silicon chip, e.g., it is possible to pass `device="mps"`.

Note that the prior must be on the training device already, e.g., when passing
`device="cuda:0"`, make sure to pass a prior object that was created on that
device, e.g.,

```python
prior = torch.distributions.MultivariateNormal(loc=torch.zeros(2,
device="cuda:0"), covariance_matrix=torch.eye(2, device="cuda:0"))
```

## Performance

Whether or not you reduce your training time when training on a GPU depends on
the problem at hand. We provide a couple of default density estimators for
`SNPE`, `SNLE` and `SNRE`, e.g., a mixture density network
(`density_estimator="mdn"`) or a Masked Autoregressive Flow
(`density_estimator="maf"`). For these default density estimators, we do **not**
expect a speed-up. This is because the underlying neural networks are relatively
shallow and not tall, e.g., they do not have many parameters or matrix
operations that benefit from being executed on the GPU.

A speed-up through training on the GPU will most likely become visible when
using convolutional modules in your neural networks. E.g., when passing an
embedding net for image processing like in this example:
[https://github.com/sbi-dev/sbi/blob/main/tutorials/05_embedding_net.ipynb](https://github.com/sbi-dev/sbi/blob/main/tutorials/05_embedding_net.ipynb).
