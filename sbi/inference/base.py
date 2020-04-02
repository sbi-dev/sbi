
from abc import ABC
import os.path
from typing import Callable, Optional
import warnings

import torch
from torch import Tensor
from torch.distributions import Uniform
from torch.utils.tensorboard import SummaryWriter

from sbi.simulators.simutils import set_simulator_attributes
from sbi.utils import get_log_root, get_timestamp
from sbi.utils.torchutils import atleast_2d, get_default_device


class NeuralInference(ABC):
    """Abstract base class for neural inference methods.
    """

    def __init__(
        self,
        simulator: Callable,
        prior,
        true_observation: Tensor,
        simulation_batch_size: int = 1,
        device: Optional[torch.device] = None,
        summary_writer: Optional[SummaryWriter] = None
    ):

        """
        Args:

            simulator: a regular function parameter->result
                Both parameters and result can be multi-dimensional.         
            prior: distribution-like object with `log_prob`and `sample` methods.
            true_observation: tensor containing the observation x_o.
                If it has more than one dimension, the leading dimension will be interpreted as a batch dimension but *currently* only the first batch element will be used to condition on.
            simulation_batch_size: number of parameter sets that the 
                simulator accepts and converts to data x at once. If -1, we simulate all parameter sets at the same time. If >= 1, the simulator has to process data of shape (simulation_batch_size, parameter_dimension).
            summary_writer: an optional SummaryWriter to control, among others, log     
                file location (default is <current working directory>/logs.)
            device: torch.device on which to compute (optional).
            mcmc_method: MCMC method to use for posterior sampling, one of 
                ['slice', 'hmc', 'nuts'].
        """

        self._warn_on_possibly_batched_observations(true_observation)
        
        # XXX want self._true_observation (atleast_2d) as attribute instead?
        self._simulator = set_simulator_attributes(simulator, prior,
                                                   true_observation)
        self._true_observation = atleast_2d(true_observation)

        self._warn_on_batch_reinterpretation_extra_d_uniform(prior)
        
        self._simulation_batch_size = simulation_batch_size

        self._prior = prior
        
        self._device = get_default_device() if device is None else device

        # Initialize roundwise (parameter, observation) storage.
        self._parameter_bank, self._observation_bank = [], []

        # XXX We could instantiate here the Posterior for all children. Two problems:
        # XXX 1. We must dispatch to right PotentialProvider for mcmc based on name
        # XXX 2. `alg_family` cannot be resolved only from `self.__class__.__name__`,
        # XXX     since SRE, AALR demand different handling but are both in SRE class.

        if summary_writer is None:
            log_dir = os.path.join(
                get_log_root(),
                self.__class__.__name__,
                simulator.name,
                get_timestamp(),
            )
            self._summary_writer = SummaryWriter(log_dir)
        else:
            self._summary_writer = summary_writer

        # Logging during training (by SummaryWriter).
        self._summary = dict(
            mmds=[],
            median_observation_distances=[],
            negative_log_probs_true_parameters=[],
            neural_net_fit_times= [], #XXX unused elsewhere
            epochs=[],
            best_validation_log_probs=[],
        )
        
    def _warn_on_batch_reinterpretation_extra_d_uniform(self, prior):
        """Warn when prior is a batch of scalar Uniforms.

        Most likely the user wants to specify a prior on a multi-dimensional parameter
        rather than several 1D priors at once.
        """
        
        # Note .batch_shape will always work because and is short-circuiting. 
        if isinstance(prior, Uniform) and prior.batch_shape.numel() > 1: 
            warnings.warn(
                f"""The specified Uniform prior is a prior on *several scalar parameters* (i.e. a batch), not a prior on a multi-dimensional parameter.

                Please use utils.torchutils.BoxUniform if you'd rather put a prior on a multi-dimensional parameter.
                """
            )
            
    def _warn_on_possibly_batched_observations(self, true_observation):
                
        if true_observation.squeeze().ndim > 1:
            warnings.warn(
                """`true_observation` has D>1 dimensions. SBI interprets the leading dimension as a batch dimension, but it *currently* only processes a single observation, i.e. the first element of the batch.
                
                For example:
                
                > true_observation = [ [1,2,3], [4,5,6] ] 
                
                is interpreted as two vector observations, only the first of which is currently used to condition inference.
                
                Use rather:
                
                > true_observation = [ [[1,2,3], [4,5,6]] ]
                > true_observation = [ [1], [2], [3]]
                
                if your single observation is matrix-shaped or scalar-shaped.
                
                Finally:
                
                > true_observation = [1]
                > true_observation = [1, 2, 3]
                
                will be interpreted as one scalar observation or one vector observation and don't require wrapping (unsqueezing).
                """
            )
