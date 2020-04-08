"""HH_stimuli.py: a Collection of stimuli that might be fed to HH simulator"""

# GENERICS
from typing import Optional, Tuple

# ANALYSIS
import numpy as np


class Stimulus:

    def __init__(self, duration: float = 120, stim_onset: float = 10, stim_end: Optional[float] = None):
        """Initializes the stimulus

        Parameters
        ----------
        duration: float
            total duration of stimulus in seconds. The duration must include the time until stimulus onset.
        stim_onset: float
            onset of stimulus in seconds
        stim_end: float
            end of stimulus in seconds, default: duration
        """
        # PARSE INPUT
        if stim_onset > duration:
            raise ValueError('The stimulus onset may not lie later than the total stimulation time.')
        if stim_end is None:
            stim_end = duration
        if stim_end > duration:
            raise ValueError('The stimulus end may not lie later than the total stimulation time.')
        if stim_onset > stim_end:
            raise ValueError('The stimulus end may lie prior to the stimulus onset.')

        # setup time axis
        self._stim_onset = stim_onset
        self._duration = duration
        self._stim_end = stim_end

    def get_t(self, dt) -> np.ndarray:
        """
        Returns
        -------
        np.ndarray: [num time steps] time array
        """
        return np.arange(0, self._duration + dt, dt)

    @property
    def stim_onset(self):
        """
        Returns
        -------
        float: onset of stimulus in seconds
        """
        return self._stim_onset

    @property
    def duration(self):
        """
        Returns
        -------
        float: duration of stimulus in seconds
        """
        return self._duration

    @property
    def stim_end(self):
        """
        Returns
        -------
        float: end of stimulus in seconds
        """
        return self._stim_end


class CurrentClamp(Stimulus):

    def __init__(self, stim_onset: float = 10, duration: float = 120,
                 current_level: float = 5e-4, axon_radius: float = 70.*1e-4):
        """Initializes the stimulus. The stimulus end is set to duration - stim_onset. Thus, duration >= 2 * stim_onset

        Parameters
        ----------
        stim_onset: float
            onset of stimulus in seconds
        duration: float
            Duration of stimulus in seconds. The duration must include the time until stimulus onset.
        current_level: float
            level of current injection in siemens
        axon_radius: float
            axon radius in cm

        Returns
        -------
        np.ndarray: [num time steps] array of time values
        np.ndarray: [num time steps] array of current value. Current clamp is 'on' for
            stim_onset < t < duration - stim_onset
        """
        super().__init__(stim_onset=stim_onset, duration=duration, stim_end=duration-stim_onset)

        # setup
        self.current_level = current_level
        self.axon_radius = axon_radius

    def __call__(self, dt: float = 0.01) -> Tuple[np.ndarray, np.ndarray]:
        """

        Returns
        -------
        np.ndarray: [num time steps]  array of current values
        """
        t = self.get_t(dt)

        # external current
        area = np.pi*(self.axon_radius**2)  # cm2
        I = np.zeros_like(t)
        I[int(np.round(self.stim_onset/dt)):int(np.round(self.stim_end/dt))] = self.current_level/area  # muA/cm2

        return t, I

    @property
    def membrane_area(self) -> float:
        """return membrane area used in calculation of this stimulus
        Return
        ------
        float: membrane in cm^2
        """
        return np.pi*(self.axon_radius**2)
