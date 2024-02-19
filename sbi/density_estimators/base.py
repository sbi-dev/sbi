from pyknos.nflows.flows import Flow
import torch
from torch import Tensor, nn
from typing import Optional,Union
from warnings import warn

class DensityEstimator:
    r"""Base class for density estimators.
    The density estimator class is a wrapper around neural networks that 
    allows to evaluate the log_probs, sample, and provide the loss of $theta,x$ pairs.

    """
    def __init__(self,
                 net: Union[Flow, nn.Module],
                 ):
        r"""Base class for density estimators.

        Args:
            net: Neural network.
        """


        self.net = net

    def log_prob(self, input: Tensor, condition: Tensor) -> Tensor:
        r"""Return the batched log probabilities of the inputs given the conditions.

        Args:
            input: Inputs to evaluate the log probability of.
            x: Conditions.

        Returns:
            Sample-wise log probabilities.
        """
        raise NotImplementedError
    
    def loss(self, input: Tensor, condition: Tensor) -> Tensor:
        r"""Return the loss for training the density estimator.

        Args:
            input: Inputs to evaluate the loss on.
            condition: Conditions.
            
        Returns:
            Loss.
        """
        raise NotImplementedError
    
    def sample(self, num_samples: int, condition: Tensor) -> Tensor:
        r"""Return samples from the density estimator.

        Args:
            num_samples: Number of samples to return.

        Returns:
            Samples.
        """
        raise NotImplementedError