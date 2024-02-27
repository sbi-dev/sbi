import torch
from torch import Tensor, nn


class DensityEstimator:
    r"""Base class for density estimators.

    The density estimator class is a wrapper around neural networks that
    allows to evaluate the `log_prob`, `sample`, and provide the `loss` of $theta,x$
    pairs.
    """

    def __init__(self, net: nn.Module) -> None:
        r"""Base class for density estimators.

        Args:
            net: Neural network.
        """

        self.net = net

    def log_prob(self, input: Tensor, condition: Tensor, **kwargs) -> Tensor:
        r"""Return the batched log probabilities of the inputs given the conditions.

        Args:
            input: Inputs to evaluate the log probability of. Must have batch dimension.
            x: Conditions. Must have batch dimension.

        Returns:
            Sample-wise log probabilities.
        """

        raise NotImplementedError

    def loss(self, input: Tensor, condition: Tensor, **kwargs) -> Tensor:
        r"""Return the loss for training the density estimator.

        Args:
            input: Inputs to evaluate the loss on.
            condition: Conditions.

        Returns:
            Loss.
        """

        raise NotImplementedError

    def sample(self, sample_shape: torch.Size, condition: Tensor, **kwargs) -> Tensor:
        r"""Return samples from the density estimator.

        Args:
            sample_shape: Shape of the samples to return.

        Returns:
            Samples.
        """

        raise NotImplementedError
