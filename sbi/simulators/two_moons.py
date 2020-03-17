import numpy as np
import torch
from torch import float32

import sbi.utils as utils
from sbi.simulators.simulator import Simulator


class TwoMoonsSimulator(Simulator):
    """
    Implemenation of two moons simulator as described in section 4.1 and Appendix A.5.1 of
    'Automatic Posterior Transformation'.
    """

    def __init__(self):
        super().__init__()
        pass

    def __call__(self, parameters):
        """
        Generates observations for the given batch of parameters.

        :param parameters: torch.Tensor
            Batch of parameters.
        :return: torch.Tensor
            Batch of observations.
        """

        # Run simulator in NumPy.
        if isinstance(parameters, torch.Tensor):
            parameters = utils.tensor2numpy(parameters)

        # If we have a single parameter then view it as a batch of one.
        if parameters.ndim == 1:
            return self(parameters[None, ...])

        num_simulations = parameters.shape[0]

        # Keep track of total simulations.
        self.num_total_simulations += num_simulations

        # Run simulator.
        a = np.pi * np.random.rand(num_simulations) - np.pi / 2
        r = 0.01 * np.random.randn(num_simulations) + 0.1
        p = np.column_stack([r * np.cos(a) + 0.25, r * np.sin(a)])
        s = (1 / np.sqrt(2)) * np.column_stack(
            [
                -np.abs(parameters[:, 0] + parameters[:, 1]),
                (-parameters[:, 0] + parameters[:, 1]),
            ]
        )
        return torch.tensor(p + s, dtype=float32)

    @property
    def observation_dim(self):
        return 2

    @property
    def parameter_dim(self):
        return 2

    @property
    def name(self):
        return "two-moons"

    @property
    def parameter_plotting_limits(self):
        return [-1, 1]

    @property
    def normalization_parameters(self):
        mean = torch.zeros(2)
        std = torch.ones(2)
        return mean, std
