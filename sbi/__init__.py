# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

"""sbi: simulation-based inference with PyTorch.

sbi estimates the posterior over a simulator's parameters from simulations, without
evaluating the simulator's likelihood. A minimal workflow with NPE::

    import torch

    from sbi.inference import NPE
    from sbi.utils import BoxUniform

    prior = BoxUniform(low=-2 * torch.ones(2), high=2 * torch.ones(2))
    theta = prior.sample((1000,))
    x = theta + 0.1 * torch.randn_like(theta)  # Replace with your simulator.

    inference = NPE(prior=prior)
    inference.append_simulations(theta, x).train()
    posterior = inference.build_posterior()
    samples = posterior.sample((1000,), x=torch.zeros(2))

NLE and NRE follow the same steps. The trainers' docstrings have more examples, e.g.
`help(sbi.inference.NPE)`.

Documentation of the current release: https://sbi.readthedocs.io/en/stable/
Index for coding agents: https://sbi.readthedocs.io/en/stable/llms.txt
"""

from sbi.__version__ import __version__  # noqa: F401
