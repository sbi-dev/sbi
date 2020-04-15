import os
from typing import Optional

import numpy as np
import scipy.stats
import torch
from matplotlib import pyplot as plt

import sbi.utils as utils
from sbi.mcmc import SliceSampler
from sbi.simulators.simulator import Simulator
from torch import Tensor

# fixed parameters for this simulator
num_xs_per_parameter = 4
theta_dim = 5
observation_dim = 8

# XXX: rewrite in PyTorch, maybe take from sbibm.
def non_linear_gaussian(theta: Tensor) -> Tensor:
    """Generate simulation outputs x from non-linear Gaussian model for the given batch
     of parameters theta.
        
    Arguments:
        theta: batch of parameters.
    
    Returns:
         batch of simulated data xs of shape
          (batch_size, 2 * num_xs_per_parameter)
    """
    # Run simulator in NumPy.
    if isinstance(theta, Tensor):
        theta = utils.tensor2numpy(theta)

    # If we have a single parameter then view it as a batch of one.
    if theta.ndim == 1:
        theta = theta[np.newaxis, :]

    num_simulations = theta.shape[0]

    # Run simulator to generate self._num_xs_per_parameter
    # observations from a 2D Gaussian parameterized by the 5 values in each theta.
    m0, m1, s0, s1, r = _unpack_params(theta)

    us = np.random.randn(num_simulations, num_xs_per_parameter, 2)
    xs = np.zeros_like(us)

    xs[:, :, 0] = s0 * us[:, :, 0] + m0
    xs[:, :, 1] = s1 * (r * us[:, :, 0] + np.sqrt(1.0 - r ** 2) * us[:, :, 1]) + m1

    # XXX: this doesnt have any effect remove?
    mean = torch.zeros(observation_dim)
    std = torch.ones(observation_dim)
    return (
        torch.tensor(
            xs.reshape([num_simulations, 2 * num_xs_per_parameter]), dtype=torch.float32
        )
        - mean.reshape(1, -1)
    ) / std.reshape(1, -1)


def _unpack_params(theta):
    """Unpack theta to m0, m1, s0, s1, r.
    
    Arguments:
        theta: batch of parameters of shape (batch_size, theta_dim)
    
    Returns:
        tuple(np.array) -- Tuple of parameters where each np.array holds a single
         theta for the batch.
    """
    assert theta.shape[1] == 5, "parameter dimension must be 5"

    m0 = theta[:, [0]]
    m1 = theta[:, [1]]
    s0 = theta[:, [2]] ** 2
    s1 = theta[:, [3]] ** 2
    r = np.tanh(theta[:, [4]])

    return m0, m1, s0, s1, r


def get_ground_truth_posterior_samples_nonlinear_gaussian(
    num_samples: Optional[int] = None,
) -> Tensor:
    """Get pseudo ground truth posterior samples from mcmc.
        
    We have pre-generated posterior samples using MCMC on the product of the analytic
    likelihood and a uniform prior on [-3, 3]^5.
    Thus they are ground truth as long as MCMC has behaved well.
    We load these once if samples have not been loaded before, store them for future
    use, and return as many as are requested.

    Keyword Arguments:
        num_samples: number of sample to return.
    
    Returns:
        posterior_samples: batch of posterior samples shaped [num_samples, theta_dim]
    """

    posterior_samples = torch.tensor(
        np.load(
            os.path.join(
                utils.get_data_root(),
                "nonlinear-gaussian",
                "true-posterior-samples.npy",
            )
        )
    )
    if num_samples is not None:
        return posterior_samples[:num_samples]
    else:
        return posterior_samples


# XXX This is a replication of the above function, no? needs refactoring.
class NonlinearGaussianSimulator(Simulator):
    """
    Implemenation of nonlinear Gaussian simulator as described in section 5.2 and
     appendix A.1 of 'Sequential Neural Likelihood', Papamakarios et al.
    """

    def __init__(self):
        """Set up simulator.
        """
        super().__init__()
        self._num_xs_per_parameter = 4
        self._posterior_samples = None

    def __call__(self, theta: Tensor) -> Tensor:
        """Generate observations from non-linear Gaussian model for the given batch of
         parameters theta.
        
        Args:
            theta: Batch of parameters.
        
        Returns:
            Batch of observations of shape
             (batch size, 2 * num_xs_per_parameter)
        """
        # Run simulator in NumPy.
        if isinstance(theta, Tensor):
            theta = utils.tensor2numpy(theta)

        # If we have a single theta then view it as a batch of one.
        if theta.ndim == 1:
            return self.simulate(theta[np.newaxis, :])[0]

        num_simulations = theta.shape[0]

        # Keep track of total simulations.
        self.num_total_simulations += num_simulations

        # Run simulator to generate self._num_xs_per_parameter
        # observations from a 2D Gaussian parameterized by the 5 values in theta.
        m0, m1, s0, s1, r = self._unpack_params(theta)

        us = np.random.randn(num_simulations, self._num_xs_per_parameter, 2)
        xs = np.empty_like(us)

        xs[:, :, 0] = s0 * us[:, :, 0] + m0
        xs[:, :, 1] = s1 * (r * us[:, :, 0] + np.sqrt(1.0 - r ** 2) * us[:, :, 1]) + m1

        mean, std = self._get_observation_normalization_parameters()
        return (
            torch.tensor(
                xs.reshape([num_simulations, 2 * self._num_xs_per_parameter]),
                dtype=torch.float32,
            )
            - mean.reshape(1, -1)
        ) / std.reshape(1, -1)

    def log_prob(self, x: Tensor, theta: Tensor):
        """Log likelihood of observations given parameter sets theta.
        
        Likelihood is proportional to a product of self._num_xs_per_parameter
         2D Gaussians and so log likelihood can be computed analytically.
        
        Args:
            x: Batch of observations of shape (batch_size, observation_dim).
            theta: Batch of parameters (batch_size, parameter_dim).
        
        Returns:
            L: Log likelihood log p(x | theta) for each item in the batch.
        """

        if isinstance(theta, Tensor):
            theta = utils.tensor2numpy(theta)

        if isinstance(x, Tensor):
            x = utils.tensor2numpy(x)

        if x.ndim == 1 and theta.ndim == 1:
            x, theta = (
                x.reshape(1, -1),
                theta.reshape(1, -1),
            )

        m0, m1, s0, s1, r = self._unpack_params(theta)
        logdet = np.log(s0) + np.log(s1) + 0.5 * np.log(1.0 - r ** 2)

        x = x.reshape([x.shape[0], self._num_xs_per_parameter, 2])
        us = np.empty_like(x)

        us[:, :, 0] = (x[:, :, 0] - m0) / s0
        us[:, :, 1] = (x[:, :, 1] - m1 - s1 * r * us[:, :, 0]) / (
            s1 * np.sqrt(1.0 - r ** 2)
        )
        us = us.reshape([us.shape[0], 2 * self._num_xs_per_parameter])

        L = (
            np.sum(scipy.stats.norm.logpdf(us), axis=1)
            - self._num_xs_per_parameter * logdet[:, 0]
        )

        return L
