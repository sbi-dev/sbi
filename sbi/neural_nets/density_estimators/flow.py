import torch
from pyknos.nflows import flows
from torch import Tensor

from sbi.neural_nets.density_estimators.base import DensityEstimator
from sbi.types import Shape


class NFlowsFlow(DensityEstimator):
    r"""`nflows`- based normalizing flow density estimator.

    Flow type objects already have a .log_prob() and .sample() method, so here we just
    wrap them and add the .loss() method.
    """

    def __init__(self, net: flows.Flow):
        super().__init__(net)

    def log_prob(self, input: Tensor, condition: Tensor) -> Tensor:
        r"""Return the batched log probabilities of the inputs given the conditions.

        Args:
            input: Inputs to evaluate the log probability of. Must have batch dimension.
            condition: Conditions. Must have batch dimension.

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
            Negative log-probability.
        """

        return -self.log_prob(input, condition)

    def sample(self, sample_shape: Shape, condition: Tensor) -> Tensor:
        r"""Return samples from the density estimator.

        Args:
            sample_shape: Batch dimensions of the samples to return
            condition: Condition.

        Returns:
            Samples.
        """

        num_samples = torch.Size(sample_shape).numel()

        # nflows.sample() expects conditions to be batched.
        if len(condition.shape) == 1:
            condition = condition.unsqueeze(0)
        return self.net.sample(num_samples, context=condition).reshape((
            *sample_shape,
            -1,
        ))
