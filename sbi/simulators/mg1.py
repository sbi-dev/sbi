import os
import pickle

import numpy as np
import sbi.utils as utils
import torch
from sbi.simulators.simulator import Simulator


class SimulatorModel:
    """
    Base class for a simulator model.
    """

    def __init__(self):

        self.n_sims = 0

    def sim(self, ps):

        raise NotImplementedError("simulator model must be implemented as a subclass")


class Model(SimulatorModel):
    """
    The M/G/1 queue model.
    """

    def __init__(self):

        SimulatorModel.__init__(self)
        self.n_sim_steps = 50

    def sim(self, ps, info=False, rng=np.random):

        ps = np.asarray(ps, float)

        if ps.ndim == 1:

            res = self.sim(ps[np.newaxis, :], info, rng)
            return tuple(map(lambda x: x[0], res)) if info else res[0]

        elif ps.ndim == 2:

            assert ps.shape[1] == 3, "parameter must be 3-dimensional"
            p1, p2, p3 = ps[:, 0:1], ps[:, 1:2], ps[:, 2:3]
            N = ps.shape[0]

            # service times (uniformly distributed)
            sts = (p2 - p1) * rng.rand(N, self.n_sim_steps) + p1

            # inter-arrival times (exponentially distributed)
            iats = -np.log(1.0 - rng.rand(N, self.n_sim_steps)) / p3

            # arrival times
            ats = np.cumsum(iats, axis=1)

            # inter-departure times
            idts = np.empty([N, self.n_sim_steps], dtype=float)
            idts[:, 0] = sts[:, 0] + ats[:, 0]

            # departure times
            dts = np.empty([N, self.n_sim_steps], dtype=float)
            dts[:, 0] = idts[:, 0]

            for i in range(1, self.n_sim_steps):
                idts[:, i] = sts[:, i] + np.maximum(0.0, ats[:, i] - dts[:, i - 1])
                dts[:, i] = dts[:, i - 1] + idts[:, i]

            self.n_sims += N

            return (sts, iats, ats, idts, dts) if info else idts

        else:
            raise TypeError("parameters must be either a 1-dim or a 2-dim array")


def whiten(xs, params):
    """
    Whitens a given dataset using the whitening transform provided.
    """

    means, U, istds = params

    ys = xs.copy()
    ys -= means
    ys = np.dot(ys, U)
    ys *= istds

    return ys


class Stats:
    """
    Summary statistics for the M/G/1 model: percentiles of the inter-departure times
    """

    def __init__(self):

        n_percentiles = 5
        self.perc = np.linspace(0.0, 100.0, n_percentiles)

        path = os.path.join(utils.get_data_root(), "mg1", "pilot_run_results.pkl")
        with open(path, "rb") as file:
            self.whiten_params = pickle.load(file, encoding="bytes")

    def calc(self, data):

        data = np.asarray(data, float)

        if data.ndim == 1:
            return self.calc(data[np.newaxis, :])[0]

        elif data.ndim == 2:

            stats = np.percentile(data, self.perc, axis=1).T
            stats = whiten(stats, self.whiten_params)

            return stats

        else:
            raise TypeError("data must be either a 1-dim or a 2-dim array")


def get_ground_truth_observation():
    path = os.path.join(utils.get_data_root(), "mg1", "observed_data.pkl")
    with open(path, "rb") as file:
        _, true_observation = pickle.load(file, encoding="bytes")
    return true_observation


class MG1Simulator(Simulator):
    """
    The M/G/1 queue model.
    """

    def __init__(self):

        super().__init__()
        self._simulator = Model()
        self._summarizer = Stats()

    def __call__(self, parameters):
        parameters = utils.tensor2numpy(parameters)
        observations = self._summarizer.calc(self._simulator.sim(parameters))
        return torch.Tensor(observations)

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

    @property
    def normalization_parameters(self):
        mean = torch.Tensor([5.0160, 10.0126, 0.1667])
        std = torch.Tensor([2.8934, 4.0788, 0.0961])
        return mean, std
