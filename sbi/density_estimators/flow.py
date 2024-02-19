from pyknos.nflows import flows
from sbi.density_estimators import DensityEstimator
import torch
from torch import Tensor, nn
from typing import Optional,Union
from warnings import warn


class FlowDensityEstimator(DensityEstimator):
    r"""Flow-based density estimators.
    Flow type objects already have a .log_prob() and .sample() method, so here we just wrap them and add the .loss() method.
    """
    def __init__(self,
                 net: flows.Flow,
                 ):

        super().__init__(net)

    def log_prob(self, input: Tensor, condition: Tensor) -> Tensor:
        r"""Return the log probability of the data given the parameters.

        Args:
            theta: Parameters of the model.
            x: Data.

        Returns:
            Log probability of the data given the parameters.
        """
        return self.net.log_prob(input, condition)
    
    def loss(self, input: Tensor, condition: Tensor) -> Tensor:
        r"""Return the loss for training the density estimator.

        Args:
            theta: Parameters of the model.
            x: Data.

        Returns:
            Loss.
        """
        #We return the mean of the batch.
        return self.log_prob(input, condition).mean()
    
    def sample(self, num_samples: int, condition: Tensor) -> Tensor:
        r"""Return samples from the density estimator.

        Args:
            num_samples: Number of samples to return.

        Returns:
            Samples.
        """
        raise self.net.sample(num_samples, context=condition)