"""HH_statistics.py: Summary statistics for HH simulated traces."""

# GENERICS
from typing import List, Tuple, Callable

# ANALYSIS
import numpy as np

# STATISTICS
from scipy import stats as spstats


class HHSummaryStats:

    def __init__(self, t_on: float = 10, duration: float = 120, num_moments: int = 4, dt: float = 0.1):
        """
        Parameters
        ----------
        t_on : float
            onset of stimulus in seconds
        duration : float
            duration of stimulus in seconds
        num_moments : int
            number of moments to include in summary statistics
        dt : float
            timestep size of simulation seconds
        Return
        ------
        """
        self.t_on = t_on
        self.duration = duration
        min_num_moments = 4
        self.num_moments = int(min([num_moments, min_num_moments]))
        self.dt = dt
        self.t = np.arange(0, duration + dt, dt)
        self.refractory_period = 0.5

    def __call__(self, v: np.ndarray) -> np.ndarray:
        """ compute summary statistics from the voltage traces v

        Parameters
        ----------
        v : np.ndarray [batch_size, # simulated timesteps]

        Return
        ------
        np.ndarray : [batch_size, # summary statistics] summary statistics of v
        """
        # PARSE INPUT
        v = v.copy()

        # PRAMETERS
        batch_size, num_dt = tuple(v.shape)

        # COMPUTATION
        stim_idcs = ((self.t_on <= self.t) & (self.t <= (self.t_on + self.duration))).nonzero()[0]
        assert stim_idcs.ndim == 1, 'Tensor of selections indices for stimulation time frame has to have ' \
                                    'dimensionality 1, got {}'.format(stim_idcs.ndim)
        # compute number of spikes
        num_spikes: np.ndarray = self.number_of_spikes(v[:, stim_idcs]).sum(axis=1).astype(float)  # [batch_size]

        # compute resting potential and std
        rest_pot, rest_pot_std = self.resting_potential(v)  # [batch_size], [batch_size]
        assert tuple(rest_pot.shape) == (batch_size,), '\'rest_pot\' has to be tensor of shape ' \
                                                       '[{}], got {}'.format(batch_size, list(rest_pot.shape))
        assert tuple(rest_pot_std.shape) == (batch_size,), '\'rest_pot_std\' has to be tensor of shape ' \
                                                           '[{}], got {}'.format(batch_size, list(rest_pot_std.shape))

        # compute mean and std of voltage during stimulation
        mean = v[:, stim_idcs].mean(axis=1)
        std = v[:, stim_idcs].mean(axis=1)
        assert tuple(mean.shape) == (batch_size,), '\'mean\' has to be tensor of shape [{}], got {}'.format(batch_size,
                                                                                                            list(
                                                                                                                mean.shape))
        assert tuple(std.shape) == (batch_size,), '\'std\' has to be tensor of shape [{}], got {}'.format(batch_size,
                                                                                                          list(
                                                                                                              std.shape))

        # compute normalized moments of order higher than 2
        v_moments = self.moments(v[:, stim_idcs])  # [batch_size, #moments]
        #         assert tuple(v_moments.shape) == (batch_size, self.num_moments - 2), '\'v_moments\' has to be tensor of shape {}, got {}'.format((batch_size, self.num_moments -2, ), list(std.shape))

        # collect
        summary_stats: np.ndarray = np.concatenate((num_spikes[:, None],
                                                    rest_pot[:, None],
                                                    rest_pot_std[:, None],
                                                    mean[:, None],
                                                    std[:, None],
                                                    v_moments), axis=1)

        return summary_stats

    def detect_spikes(self, v: np.ndarray,
                      threshold: Callable[[np.ndarray], np.ndarray] = lambda v: -10. * np.ones(
                          v.shape[0])) -> np.ndarray:
        """ detects spikes in the voltage traces

        Parameters
        ----------
        v : np.ndarray [batch_size, # simulated time steps] voltage traces
        threshold

        Returns
        -------
        np.ndarray : [batch_size, # simulated time steps] boolean mask of the voltage trace, where True indicates the
            existence of a spike in the respective simulated bin
        """

        # HELPER FUNCTIONS
        def remove_refractory_period_violations(spikes_idcs: List[int],
                                                refractory_period: float = self.refractory_period,
                                                timestep_size: float = self.dt) -> List[int]:
            """ removes those spike from a list that violate a given refractory period
            Parameters
            ----------
            spikes_idcs : List[int] list of spike indices
            refractory_period: float absolute refractory period in seconds
            timestep_size: float length of simulated timesteps in seconds

            Return
            ------
            List[int] : list of spike indices, where violating spikes are removed
            """
            ref_in_samples: float = refractory_period / timestep_size
            collect: List[int] = []
            for i, spike_idx in enumerate(spikes_idcs):
                if i == 0:  # always append the first spike
                    collect.append(spike_idx)
                    prev_spike_idx = spike_idx
                else:
                    dist = spike_idx - prev_spike_idx
                    if dist > ref_in_samples:  # add the next spike only if refractory period has passed
                        collect.append(spike_idx)
                        prev_spike_idx = spike_idx

            return collect

        def validate_up_down_crossings(up: np.ndarray, down: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
            """ checks the integrety of the upcrossings and downwncrossings
            Parameters
            ----------
            up : np.ndarray [# upcrossings] array of idices where the voltage trace upcrosses a certain
                threshold
            down : np.ndarray [# downcrossings] array of indices where the voltage trace downcrosses a certain
                threshold

            Return
            ------
            np.ndarray : [# upcrossings corrected] upcrossings
            np.ndarray : [# downcrossings corrected] downcrossings
            """
            # PARSE INPUT
            up, down = np.atleast_1d(up.squeeze()), np.atleast_1d(down.squeeze())
            assert up.ndim == 1 and down.ndim == 1, "Arrays of up- and downcrossings have to be of dimensionality 1, " \
                                                    "got \'up.ndim\' == {} and \'down.ndim\' == {}".format(up.ndim,
                                                                                                           down.ndim)

            # CORNER CASES
            if up.size == 0 or down.size == 0:  # corner case of no spikes
                return np.empty(0), np.empty(0)
            if up.size == 1:  # case only one upstroke
                if down.size == 1 and up < down:
                    return up, down
                if down.size == 1 and up > down:
                    return np.empty(0), np.empty(0)
                for i, d in enumerate(down):
                    if up[0] < d:
                        return up, np.ones(1, dtype=int) * d
                return np.empty(0), np.empty(0)
            # REGULAR CASES
            while (up - down > 0).any():
                up = up[1:]
                down = down[:up.shape[0]]

            return up, down

        # PARSE INPUT
        v = v.copy()

        # PARAMETERS
        thresh: np.ndarray = threshold(v).squeeze()
        if thresh.ndim == 1:
            thresh = thresh[:, None]
        assert thresh.ndim <= 2, 'Threshold has to be a scalar or a vector, expected \'threshold(v).ndim\' <= 2, ' \
                                 'got {}'.format(thresh.ndim)

        # COMPUTATION
        above_thresh: np.ndarray = np.sign(v - thresh).astype(float)  # True for all voltage values above threshold
        crossings = np.sign(np.diff(above_thresh, axis=1, append=above_thresh[:, -1][:, None]))
        assert ((crossings == -1) | (crossings == 0) | (
                    crossings == 1)).all(), 'Variable \'crossings\' contains values not in [-1, 0, 1], ' \
                                            'got \'crossings_i\' in {{}}'.format(crossings.unique())
        upcrossings: np.ndarray = crossings > 0
        downcrossings: np.ndarray = crossings < 0

        spikes = np.zeros_like(v, dtype=bool)
        # iterate over batches
        for t, (up, down) in enumerate(zip(upcrossings, downcrossings)):
            up_idcs, down_idcs = up.nonzero()[0], down.nonzero()[0]
            up_idcs, down_idcs = validate_up_down_crossings(up_idcs, down_idcs)
            assert up_idcs.size == down_idcs.size, 'up and down still have a different number of entries, got ' \
                                                   '\'up_idcs.size\' == ' \
                                                   '{} and \'down.size\' == {}'.format(up_idcs.size, down_idcs.size)
            assert (
                        up_idcs - down_idcs < 0).all(), 'There are still upstrokes before downstrokes contained in the ' \
                                                        'variable pair (\'up_idcs\', \'down_idcs\')'
            # get index of maximum value
            if up_idcs.size > 0:
                idcs: List[int] = [crossing[0] + v[t, crossing[0]:crossing[1]].argmax() for c, crossing in
                                   enumerate(zip(up_idcs, down_idcs))]
                idcs = remove_refractory_period_violations(idcs)
                spikes[t, idcs] = True  # mark spike in spikes mask

        return spikes

    def number_of_spikes(self, v: np.ndarray) -> np.ndarray:
        """ compute the number of spikes in the voltage traces
        Parameters
        ----------
        v : np.ndarray [batch_size, # simulated timesteps]

        Return
        ------
        np.ndarray : [batch_size] number of spikes in voltage trace
        """
        # PARSE INPUT
        v = v.copy()

        # PARAMETERS
        threshold: Callable[[np.ndarray], np.ndarray] = lambda trace: -10. * np.ones(trace.shape[0])

        # COMPUTATION
        spikes: np.ndarray = self.detect_spikes(v, threshold)
        spikes.sum(axis=1).squeeze()  # [batch_size]

        return spikes

    def resting_potential(self, v: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """ computes mean and std of resting potential in simulation
        Parameters
        ----------
        v : np.ndarray [batch_size, # simulated time steps]

        Return
        ------
        Tuple[np.ndarray, np.ndarray] : ([batch_size], [batch_size], ) resting potential and standard deviation of the
            resting potential
        """
        v = v.copy()
        assert v.shape[1] == self.t.shape[
            0], 'The time timesion of \'v\' and \'t\' have to be of equal length, expected \'v.shape[1]\' == {} != {} == \'len(t)\''.format(
            v.shape[1], self.t.shape[0])

        # compute mean
        rest_idcs = (self.t < self.t_on).nonzero()[0]
        assert rest_idcs.ndim == 1, 'Tensor of selections indices for rest time frame has to have dimenionality 1, got {}'.format(
            rest_idcs.ndim)
        mean = v[:, rest_idcs].mean(axis=1)

        # compute std
        std = v[:, int(.9 * self.t_on / self.dt):int(self.t_on / self.dt)].std(axis=1)

        return mean, std

    def moments(self, v: np.ndarray) -> List[float]:
        """ computes higher moments of the voltage trace in sumulation, see https://en.wikipedia.org/wiki/Moment_(mathematics)
        Parameters
        ----------
        v : np.ndarray [batch_size, # timesteps]

        Return
        ------
        np.ndarray : [# momentens] number of higher moments computes
        """
        v = v.copy()
        batch_size, _ = tuple(v.shape)

        exp = np.linspace(3, self.num_moments, self.num_moments - 2)
        std_pw = v.std(axis=1)[:, None] ** exp[None, :]
        std_pw = np.concatenate((np.ones((batch_size, 1)), std_pw), axis=1)

        central_moments = spstats.moment(v, np.linspace(2, self.num_moments, self.num_moments - 1), axis=1).T

        return central_moments / std_pw
