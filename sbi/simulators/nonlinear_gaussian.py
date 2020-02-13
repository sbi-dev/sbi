import os

import numpy as np
import sbi.utils as utils
import scipy.stats
import torch
from matplotlib import pyplot as plt
from sbi.mcmc import SliceSampler
from sbi.simulators.simulator import Simulator

parameter_dim = 5
observation_dim = 8


class NonlinearGaussianSimulator(Simulator):
    """
    Implemenation of nonlinear Gaussian simulator as described in section 5.2 and appendix
    A.1 of 'Sequential Neural Likelihood', Papamakarios et al. 
    """

    def __init__(self):
        """Set up simulator. 
        """
        super().__init__()
        self._num_observations_per_parameter = 4
        self._posterior_samples = None

    def __call__(self, parameters):
        """Generate observations from non-linear Gaussian model for the given batch of parameters.
        
        Arguments:
            parameters {torch.Tensor} -- Batch of parameters.
        
        Returns:
            torch.Tensor [batch size, 2 * num_observations_per_parameter] -- Batch of observations.
        """
        # Run simulator in NumPy.
        if isinstance(parameters, torch.Tensor):
            parameters = utils.tensor2numpy(parameters)

        # If we have a single parameter then view it as a batch of one.
        if parameters.ndim == 1:
            return self.simulate(parameters[np.newaxis, :])[0]

        num_simulations = parameters.shape[0]

        # Keep track of total simulations.
        self.num_total_simulations += num_simulations

        # Run simulator to generate self._num_observations_per_parameter
        # observations from a 2D Gaussian parameterized by the 5 given parameters.
        m0, m1, s0, s1, r = self._unpack_params(parameters)

        us = np.random.randn(num_simulations, self._num_observations_per_parameter, 2)
        observations = np.empty_like(us)

        observations[:, :, 0] = s0 * us[:, :, 0] + m0
        observations[:, :, 1] = (
            s1 * (r * us[:, :, 0] + np.sqrt(1.0 - r ** 2) * us[:, :, 1]) + m1
        )

        mean, std = self._get_observation_normalization_parameters()
        return (
            torch.Tensor(
                observations.reshape(
                    [num_simulations, 2 * self._num_observations_per_parameter]
                )
            )
            - mean.reshape(1, -1)
        ) / std.reshape(1, -1)

    def log_prob(self, observations, parameters):
        """Log likelihood of observations given parameters. 
        
        Likelihood is proportional to a product of self._num_observations_per_parameter 2D
        Gaussians and so log likelihood can be computed analytically.
        
        Arguments:
            observations {torch.Tensor} [batch_size, observation_dim] -- Batch of observations.
            parameters {torch.Tensor} [batch_size, parameter_dim] -- Batch of parameters.
        
        Returns:
            torch.Tensor [batch_size] -- [Log likelihood log p(x | theta) for each item in the batch.
        """

        if isinstance(parameters, torch.Tensor):
            parameters = utils.tensor2numpy(parameters)

        if isinstance(observations, torch.Tensor):
            observations = utils.tensor2numpy(observations)

        if observations.ndim == 1 and parameters.ndim == 1:
            observations, parameters = (
                observations.reshape(1, -1),
                parameters.reshape(1, -1),
            )

        m0, m1, s0, s1, r = self._unpack_params(parameters)
        logdet = np.log(s0) + np.log(s1) + 0.5 * np.log(1.0 - r ** 2)

        observations = observations.reshape(
            [observations.shape[0], self._num_observations_per_parameter, 2]
        )
        us = np.empty_like(observations)

        us[:, :, 0] = (observations[:, :, 0] - m0) / s0
        us[:, :, 1] = (observations[:, :, 1] - m1 - s1 * r * us[:, :, 0]) / (
            s1 * np.sqrt(1.0 - r ** 2)
        )
        us = us.reshape([us.shape[0], 2 * self._num_observations_per_parameter])

        L = (
            np.sum(scipy.stats.norm.logpdf(us), axis=1)
            - self._num_observations_per_parameter * logdet[:, 0]
        )

        return L

    @staticmethod
    def _unpack_params(parameters):
        """Unpack parameters parameters to m0, m1, s0, s1, r.
        
        Arguments:
            parameters {np.array [batch_size, parameter_dim]} -- Batch of parameters.
        
        Returns:
            tuple(np.array) -- Tuple of parameters where each np.array holds a single parameter for the batch.
        """
        assert parameters.shape[1] == 5, "parameter dimension must be 5"

        m0 = parameters[:, [0]]
        m1 = parameters[:, [1]]
        s0 = parameters[:, [2]] ** 2
        s1 = parameters[:, [3]] ** 2
        r = np.tanh(parameters[:, [4]])

        return m0, m1, s0, s1, r

    def get_ground_truth_posterior_samples(self, num_samples=None):
        """Get pseudo ground truth posterior samples from mcmc.
        
        We have pre-generated posterior samples using MCMC on the product of the analytic
        likelihood and a uniform prior on [-3, 3]^5.
        Thus they are ground truth as long as MCMC has behaved well.
        We load these once if samples have not been loaded before, store them for future use,
        and return as many as are requested.

        Keyword Arguments:
            num_samples {int} -- Number of sample to return. (default: {None})
        
        Returns:
            torch.Tensor [num_samples, parameter_dim] -- Batch of posterior samples.
        """
        if self._posterior_samples is None:
            self._posterior_samples = torch.Tensor(
                np.load(
                    os.path.join(
                        utils.get_data_root(),
                        "nonlinear-gaussian",
                        "true-posterior-samples.npy",
                    )
                )
            )
        if num_samples is not None:
            return self._posterior_samples[:num_samples]
        else:
            return self._posterior_samples

    @property
    def parameter_dim(self):
        return 5

    @property
    def observation_dim(self):
        return 8

    @property
    def name(self):
        return "nonlinear-gaussian"

    @property
    def parameter_plotting_limits(self):
        return [-4, 4]

    @property
    def normalization_parameters(self):
        mean = torch.zeros(5)
        std = torch.ones(5)
        return mean, std

    def _get_observation_normalization_parameters(self):
        mean = torch.zeros(8)
        std = torch.ones(8)
        return mean, std
