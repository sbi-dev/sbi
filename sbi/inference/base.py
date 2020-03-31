
from abc import ABC
from typing import Optional, Callable
import warnings
import os.path
import torch
from torch.utils.tensorboard import SummaryWriter
from torch import Tensor
from torch.distributions import Uniform
from sbi.utils.torchutils import atleast_2d, get_default_device
from sbi.simulators.simutils import set_simulator_attributes
from sbi.utils import get_log_root, get_timestamp
from copy import deepcopy
from sbi.inference.posteriors.sbi_posterior import Posterior


class NeuralInference(ABC):
    """Abstract base class for neural inference methods.
    """

    def __init__(
        self,
        simulator: Callable,
        prior: torch.distributions.Distribution,
        true_observation: Tensor,
        simulation_batch_size: int = 1,
        device: Optional[torch.device] = None,
        summary_writer: Optional[SummaryWriter] = None
    ):

        """
        Args:
            simulator: function that takes (vector) parameter and returns one  
                (vector) result per parameter vector, with shape (num_parameters, result_dimension)
            
            prior: distribution-like object with `log_prob`and `sample` methods.
            
            true_observation: tensor containing the observation x0.
            
            simulation_batch_size: number of parameter vectors that the 
                 simulator accepts and converts to data x at once. If -1, we simulate all parameter sets at the same time. If >= 1, the simulator has to process data of shape (sim_batch_size, parameter_dimension).
            
            summary_writer: an optional SummaryWriter to control log file 
                location (default is <current working directory>/logs.)
             
            device: torch.device on which to compute (optional).
            
            mcmc_method: MCMC method to use for posterior sampling, one of 
                ['slice', 'hmc', 'nuts'].
        """

        # check inputs: observation
        self._true_observation = atleast_2d(true_observation)

        # warn if prior is Uniform and dim_input > 1 (inferred by drawing once)
        # XXX if prior is a torch Distribution, then it should support
        # XXX event_shape, no need to sample
        # XXX also, why use 'input' here? -> parms are input to the simulator
        scalar_input = prior.sample().numel() == 1

        if isinstance(prior, Uniform) and not scalar_input:
            warnings.warn(
                f"""The parameter dimension of the simulator we expect 
                rom the provided prior's `event_shape` is >1,
                and the prior is a PyTorch Uniform distribution. 
                Beware that you are using a `batch_shape` >1 implicitly 
                and `event_shape` 1, because PyTorch does not support 
                multivariate Uniform. Please consider using a BoxUniform prior instead."""
            )
        # warning if observed data is multidimensional.
        # XXX we want numel here?
        if true_observation.squeeze().ndim > 1:
            warnings.warn(
                """`true_observation` has several dimensions. 
                It could be e.g. a matrix (D=2) of observed data or a batch of
                1D vector observations. SBI currently does not support batches."""
            )

        # XXX do simulator attributes get used outside of us accessing them from 
        # XXX inference classes? In that case, we can store them in a dict
        # XXX here, e.g. self._simspec()...
        self._simulator = set_simulator_attributes(simulator, prior, true_observation)

        self._prior = prior
        
        self._device = get_default_device() if device is None else device

        self._simulation_batch_size = simulation_batch_size

        # initialize roundwise (parameter, observation) storage
        self._parameter_bank, self._observation_bank = [], []

        # XXX we could instantiate here the Posterior
        # XXX solve first dispatching to the right PotentialProvider
        # XXX f"Class name: {self.__class__.__name__}" -> use in dispatching
        # XXX additional complication now alg_family is not just class name
        # XXX (distinction introduced between SRE, AALR, not reflected in class)
        # XXX  could this be done here, instead of in the child class?
        # XXX  have to pass string or can access subclass name (for alg family)
       
        # XXX set embedding net for all?

        # each run has an associated log directory for TensorBoard output.
        # XXX pass / set up log_dir, not SummaryWriter. Cleaner, allows manual override
        # XXX but then, **kwargs_summary_writer?
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

        # XXX maybe use dict() or a dataclass -- requires renaming of keys
        # Each run also has a dictionary of summary statistics which are
        # populated over the course of training.

        self._summary = {
            "mmds": [],
            "median-observation-distances": [],
            "negative-log-probs-true-parameters": [],
            "neural-net-fit-times": [],
            "epochs": [],
            "best-validation-log-probs": [],
        }