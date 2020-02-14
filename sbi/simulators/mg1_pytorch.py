import os
import pickle

import numpy as np
import sbi.utils as utils
import torch
from sbi.simulators.simulator import Simulator

from summarizers import MG1Summarizer


class MG1Simulator(Simulator):
    """
    The M/G/1 queue model.
    """

    def __init__(self, summarize_observations=True):

        super().__init__()
        self.n_sim_steparameters = 50
        self._summarize_observations = summarize_observations
        if summarize_observations:
            self._summarizer = MG1Summarizer()

    def simulate(self, parameters):

        parameters = utils.tensor2numpy(parameters)

        assert parameters.shape[1] == 3, "parameter must be 3-dimensional"
        p1, p2, p3 = parameters[:, 0:1], parameters[:, 1:2], parameters[:, 2:3]
        N = parameters.shape[0]

        # service times (uniformly distributed)
        sts = (p2 - p1) * np.random.rand(N, self.n_sim_steparameters) + p1

        # inter-arrival times (exponentially distributed)
        iats = -np.log(1.0 - np.random.rand(N, self.n_sim_steparameters)) / p3

        # arrival times
        ats = np.cumsum(iats, axis=1)

        # inter-departure times
        idts = np.empty([N, self.n_sim_steparameters], dtype=float)
        idts[:, 0] = sts[:, 0] + ats[:, 0]

        # departure times
        dts = np.empty([N, self.n_sim_steparameters], dtype=float)
        dts[:, 0] = idts[:, 0]

        for i in range(1, self.n_sim_steparameters):
            idts[:, i] = sts[:, i] + np.maximum(0.0, ats[:, i] - dts[:, i - 1])
            dts[:, i] = dts[:, i - 1] + idts[:, i]

        self.num_total_simulations += N

        if self._summarize_observations:
            idts = self._summarizer(idts)

        return torch.Tensor(idts)

    @property
    def parameter_dim(self):
        return 3

    @property
    def observation_dim(self):
        return 5

    @property
    def parameter_plotting_limits(self):
        return [[0, 10], [0, 20], [0, 1.0 / 3.0]]

    @property
    def name(self):
        return "mg1"
