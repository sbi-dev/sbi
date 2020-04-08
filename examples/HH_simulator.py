"""HH_simulator.py: a Hodgekin-Huxley simulator"""

# GENERICS
from typing import List, Dict, Optional, Union

# IMPORT FROM SRC
from .HH_statistics import HHSummaryStats
from .HH_stimuli import CurrentClamp

# ANALYSIS
import numpy as np

# DEEP LEARNING
import torch

# SBI
from ..sbi.simulators.simulator import Simulator


class HHSimulator(Simulator):
    """Hodgkin-Huxley Simulator"""

    __name__ = 'Hodgkin-Huxley Simulator'

    params = {'g_leak': 0.1,  # mS/cm2
              'gbar_M': 0.07,  # mS/cm2
              'tau_max': 6e2,  # ms
              'Vt': -60,  # mV
              'nois_fact': 0.1,  # uA/cm2
              'E_leak': -70,  # mV
              'C': 1.,  # uF/cm2
              'E_Na': 53,  # mV
              'E_K': -107,  # mV
              'V0': -70,  # mV
              'dt': 0.01,  # s
              'gbar_Na': 20.,  # mS/cm2
              'gbar_K': 15.,  # mS/cm2
              'ref_period': 0.5,  # s, refractory period
              'stimulus': CurrentClamp(duration=120, stim_onset=10)  # Callable[[float], Tuple[np.ndarray, np.ndarray]]
              }

    def __init__(self, seed: Optional[int] = None, inference_parameters: List[str] = ['gbar_Na', 'gbar_K'], **params):
        """Initializes a Hodgkin-Huxley simulator"""
        super().__init__()
        self.params.update(params)
        self.seed = seed

        if not self.validate_inference_parameters(inference_parameters):
            raise ValueError('Passed inference parameters must match valid keys')
        self.inference_parameters = inference_parameters
        self._parameter_dim = len(inference_parameters)

        self.params['t'], self.params['I'] = self.params['stimulus'](self.params['dt'])

        self.summary_stats = HHSummaryStats(t_on=self.params['stimulus'].stim_onset,
                                            duration=self.params['stimulus'].duration,
                                            dt=self.params['dt'])

    def __call__(self, params: Union[np.ndarray, torch.Tensor] = np.array([[50., 4., 20.], [1., 1.5, 15.]]),
                 return_stats: bool = True) -> torch.FloatTensor:
        """Simulates the Hodgkin-Huxley model for a specified time duration and current and optionally computes summary
        statistics of the voltage trace

        Parameters
        ----------
        params : np.ndarray [batch_size, # inference parameters == len(self.inference_parameters)]

        Return
        ------
        torch.FloatTensor : batch of summary statistics of the voltage trace from a HH simulation
        """

        # PARSE INPUT
        if torch.is_tensor(params):
            params = params.detach().numpy()

        if params.ndim == 1:
            params = params[None, :]

        # simulate voltage trace
        V = self.simulate_voltage_trace(params)

        if not return_stats:
            return torch.from_numpy(V).float()

        stats = self.summary_stats(V)

        return torch.from_numpy(stats).float()

    def simulate_voltage_trace(self, params: Union[np.ndarray, torch.Tensor] = np.array([[50., 4., 20.], [1., 1.5, 15.]])) -> np.ndarray:
        """Simulates the Hodgkin-Huxley model for a specified time duration and current

        Parameters
        ----------
        params : np.ndarray [batch_size, # inference parameters == len(self.inference_parameters)]

        Return
        ------
        np.ndarray : batch of summary statistics of the voltage trace from a HH simulation
        """

        # PARSE INPUT
        if torch.is_tensor(params):
            params = params.detach().numpy()

        if params.ndim == 1:
            params = params[None, :]

        inference_params_values = params
        params = self.params
        params.update(self.batch_to_params_dict(inference_params_values))

        # PARAMETERS
        batch_size = inference_params_values.shape[0]

        params = self.params
        tstep = float(params['dt'])
        t = params['t']

        if self.seed is not None:
            rng = np.random.RandomState(seed=self.seed)
        else:
            rng = np.random.RandomState()

        ####################################
        # kinetics
        def efun(z) -> np.ndarray:
            condition = np.abs(z) < 1e-4
            return (condition.astype(float) * (1 - z / 2)) + (~condition).astype(float) * (z / (np.exp(z) - 1))

        def alpha_m(x):
            v1 = x - self.params['Vt'] - 13.
            return 0.32 * efun(-0.25 * v1) / 0.25

        def beta_m(x):
            v1 = x - self.params['Vt'] - 40
            return 0.28 * efun(0.2 * v1) / 0.2

        def alpha_h(x):
            v1 = x - self.params['Vt'] - 17.
            return 0.128 * np.exp(-v1 / 18.)

        def beta_h(x):
            v1 = x - self.params['Vt'] - 40.
            return 4.0 / (1 + np.exp(-0.2 * v1))

        def alpha_n(x):
            v1 = x - self.params['Vt'] - 15.
            return 0.032 * efun(-0.2 * v1) / 0.2

        def beta_n(x):
            v1 = x - self.params['Vt'] - 10.
            return 0.5 * np.exp(-v1 / 40)

        # steady-states and time constants
        def tau_n(x):
            return 1 / (alpha_n(x) + beta_n(x))

        def n_inf(x):
            return alpha_n(x) / (alpha_n(x) + beta_n(x))

        def tau_m(x):
            return 1 / (alpha_m(x) + beta_m(x))

        def m_inf(x):
            return alpha_m(x) / (alpha_m(x) + beta_m(x))

        def tau_h(x):
            return 1 / (alpha_h(x) + beta_h(x))

        def h_inf(x):
            return alpha_h(x) / (alpha_h(x) + beta_h(x))

        # slow non-inactivating K+
        def p_inf(x):
            v1 = x + 35.
            return 1.0 / (1. + np.exp(-0.1 * v1))

        def tau_p(x):
            v1 = x + 35.
            return self.params['tau_max'] / (3.3 * np.exp(0.05 * v1) + np.exp(-0.05 * v1))

        ####################################
        # simulation from initial point
        V = np.zeros((batch_size, t.shape[0]))  # voltage [batch_size, time]
        n = np.zeros_like(V)
        m = np.zeros_like(V)
        h = np.zeros_like(V)
        p = np.zeros_like(V)

        V[:, 0] = float(self.params['V0'])
        n[:, 0] = n_inf(V[:, 0])
        m[:, 0] = m_inf(V[:, 0])
        h[:, 0] = h_inf(V[:, 0])
        p[:, 0] = p_inf(V[:, 0])

        for i in range(1, t.shape[0]):
            tau_V_inv = ((m[:, i - 1] ** 3) * params['gbar_Na'] * h[:, i - 1] + (n[:, i - 1] ** 4) * params['gbar_K'] +
                         params['g_leak'] + params['gbar_M'] * p[:, i - 1]) / params['C']
            V_inf = ((m[:, i - 1] ** 3) * params['gbar_Na'] * h[:, i - 1] * params['E_Na'] +
                     (n[:, i - 1] ** 4) * params['gbar_K'] * params['E_K'] +
                     params['g_leak'] * params['E_leak'] +
                     params['gbar_M'] * p[:, i - 1] * params['E_K'] +
                     params['I'][i - 1] +
                     self.params['nois_fact'] * rng.randn() / (tstep ** 0.5)) / (tau_V_inv * self.params['C'])
            V[:, i] = V_inf + (V[:, i - 1] - V_inf) * np.exp(-tstep * tau_V_inv)
            n[:, i] = n_inf(V[:, i]) + (n[:, i - 1] - n_inf(V[:, i])) * np.exp(-tstep / tau_n(V[:, i]))
            m[:, i] = m_inf(V[:, i]) + (m[:, i - 1] - m_inf(V[:, i])) * np.exp(-tstep / tau_m(V[:, i]))
            h[:, i] = h_inf(V[:, i]) + (h[:, i - 1] - h_inf(V[:, i])) * np.exp(-tstep / tau_h(V[:, i]))
            p[:, i] = p_inf(V[:, i]) + (p[:, i - 1] - p_inf(V[:, i])) * np.exp(-tstep / tau_p(V[:, i]))

        return V

    def validate_inference_parameters(self, inference_parameters: List[str]) -> bool:
        """takes a list of inference parameters and checks whether they are specified model parameters

        Parameters
        ----------
        inference_parameters : List[str]
            List of HH model parameter names

        Returns
        -------
        bool
            validity specifier
        """
        checks = np.array([inference_parameter in self.params.keys() for inference_parameter in inference_parameters],
                          dtype=bool)
        return checks.all()

    def batch_to_params_dict(self, batch: np.ndarray) -> Dict[str, np.ndarray]:
        """takes a batch of inference parameters and organizes them in a dictionary. The keys are stored in the instance
        under self.inference_parameters

        Parameters
        ----------
        batch : np.ndarray
            Parameter values in batch form [batch_size, # of parameters]. The # of parameters must match the number of
            inference paramters (self.inference_parameters).

        Returns
        -------
        Dict[str, np.ndarray]
            dictionary of mapping inference parameters keys to inference parameter values
        """
        if not batch.shape[1] == len(self.inference_parameters):
            raise ValueError('Number of parameters in batch must match number of inference parameters specified in the '
                             'HHSimulator initiation. got : # parameters in batch == batch.shape[1] == \
                             {} != {} == len(self.inference_parameters)'.format(batch.shape[1],
                                                                                len(self.inference_parameters)))
        return {key: value for key, value in zip(self.inference_parameters, batch.transpose(1, 0))}

    @property
    def t(self) -> np.ndarray:
        return self.params['stimulus'].get_t(self.params['dt'])

    @property
    def parameter_dim(self):
        """
        Dimension of parameters for simulator.
        :return: int parameter_dim
        """
        return len(self.inference_parameters)

    @parameter_dim.setter
    def parameter_dim(self, value):
        print('(pseudo) setting parameter_dim to {}'.format(value))

    @property
    def observation_dim(self):
        """
        Dimension of observations for simulator.
        TODO: decide whether observation_dim always corresponds to dimension of summary.
        :return: int observation_dim
        """
        raise NotImplementedError

    @observation_dim.setter
    def observation_dim(self, value):
        print('(pseudo) setting observation_dim to {}'.format(value))

    @property
    def name(self):
        """
        Name of the simulator.
        :return: str name
        """
        return self.__name__

    @name.setter
    def name(self, value):
        self.__name__ = value
