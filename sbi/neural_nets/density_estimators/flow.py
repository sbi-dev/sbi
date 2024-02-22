from typing import Optional, Union
from warnings import warn

import torch
from pyknos.nflows import flows
from torch import Tensor, nn

from sbi.neural_nets.density_estimators.base import DensityEstimator


class NFlowsNSF(DensityEstimator):
    r"""`nflows`- based neural spline flow density estimator.
    Flow type objects already have a .log_prob() and .sample() method, so here we just wrap them and add the .loss() method.
    """

    def __init__(
        self,
        net: flows.Flow,
    ):

        super().__init__(net)

    def log_prob(self, input: Tensor, condition: Tensor) -> Tensor:
        r"""Return the batched log probabilities of the inputs given the conditions.

        Args:
            input: Inputs to evaluate the log probability of. Must have a batch dimension.
            condition: Conditions. Must have a batch dimension.

        Returns:
            Sample-wise log probabilities.
        """
        return self.net.log_prob(input, context=condition)

    def loss(self, input: Tensor, condition: Tensor) -> Tensor:
        r"""Return the loss for training the density estimator.

        Args:
            input: Inputs to evaluate the loss on. Must be batched.
            condition: Conditions. Must be batched.

        Returns:
            Average negative log-probability.
        """
        # We return the mean of the batch.
        return -self.log_prob(input, condition).mean()

    def sample(self, sample_shape: torch.Size, condition: Tensor) -> Tensor:
        r"""Return samples from the density estimator.

        Args:
            sample_shape: Shape of the samples to return.

        Returns:
            Samples.
        """
        num_samples = sample_shape[0]
        if len(condition.shape) == 1:
            condition = condition.unsqueeze(0)
        return self.net.sample(num_samples, context=condition)
